"""Pure synthetic demo: generate Gaussians on SMPL surface, animate with skeleton.

No external dataset needed. Gaussians are sampled directly from the SMPL
rest-pose mesh, inheriting vertex skinning weights. Then SMPL pose parameters
drive the Gaussians via LBS deformation.

This avoids the coordinate-system alignment problem of mapping SMPL to an
external scene — everything is in SMPL's native coordinate system.

Usage:
    export PYTHONUTF8=1
    python run_synthetic_demo.py \\
        --smpl_model_path E:/Code/mmforall/model/assets/SMPL/SMPL_NEUTRAL.pkl \\
        --output_dir synthetic_demo \\
        --n_frames 120 --fps 30
"""

import argparse
import os
import sys
import math
import time

import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skinned_gaussian.smpl_model import SMPLWrapper
from skinned_gaussian.skinned_gaussians import SkinnedGaussians
from skinned_gaussian.skinning import lbs_deform_gaussians
from skinned_gaussian.pose_io import generate_demo_poses, save_pose_sequence
from skinned_gaussian.animation import render_animation, render_animation_comparison

SH_C0 = 0.28209479177387814


def sample_gaussians_on_smpl(smpl, n_gaussians=50000, device='cuda'):
    """Sample Gaussians on the SMPL mesh surface with skinning weights.

    Each Gaussian is placed at a random mesh face (barycentric sample),
    inheriting interpolated skinning weights from the face vertices.

    Returns: (means, scales_raw, quats, opacities_raw, sh0, skin_weights)
    """
    # Get rest-pose mesh.
    v, f = smpl.get_rest_mesh()  # (6890, 3), (13776, 3)
    weights = smpl.get_skinning_weights()  # (6890, 24)
    n_verts = v.shape[0]
    n_faces = f.shape[0]

    # Compute face areas for area-weighted sampling.
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    areas = 0.5 * cross.norm(dim=-1)  # (n_faces,)
    probs = areas / areas.sum()

    # Sample faces by area.
    face_indices = torch.multinomial(probs, n_gaussians, replacement=True)  # (N,)

    # Barycentric coordinates (uniform within each triangle).
    u = torch.rand(n_gaussians, 1, device=device)
    v_ = torch.rand(n_gaussians, 1, device=device)
    # Ensure u + v <= 1.
    mask = (u + v_) > 1
    u[mask] = 1 - u[mask]
    v_[mask] = 1 - v_[mask]
    w = 1 - u - v_

    # Interpolate positions.
    f_verts = v[f[face_indices]]  # (N, 3, 3)
    means = (u * f_verts[:, 0] + v_ * f_verts[:, 1] + w * f_verts[:, 2]).squeeze(-1)  # (N, 3)

    # Interpolate skinning weights.
    f_weights = weights[f[face_indices]]  # (N, 3, 24)
    skin_weights = (u * f_weights[:, 0] + v_ * f_weights[:, 1] + w * f_weights[:, 2]).squeeze(-1)  # (N, 24)
    # Normalize.
    skin_weights = skin_weights / skin_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    # Add small random offset along normal for thickness.
    normals = cross / cross.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    f_normals = normals[face_indices]  # (N, 3)
    offsets = (torch.rand(n_gaussians, 1, device=device) - 0.5) * 0.02  # ±1cm
    means = means + offsets * f_normals

    # Scales: isotropic, ~10% of mesh size for good surface coverage.
    mesh_size = (v.max(0).values - v.min(0).values).norm().item()
    init_scale = mesh_size * 0.1  # 10% of mesh size
    scales_raw = torch.log(torch.full((n_gaussians, 3), init_scale, device=device))

    # Quaternions: identity.
    quats = torch.zeros(n_gaussians, 4, device=device)
    quats[:, 0] = 1.0

    # Opacities: logit(0.9) — high opacity.
    opacities_raw = torch.full((n_gaussians,), math.log(0.9 / 0.1), device=device)

    # SH DC: assign random skin-ish colors (warm tones).
    r = 0.6 + torch.rand(n_gaussians, device=device) * 0.2  # 0.6-0.8
    g = 0.4 + torch.rand(n_gaussians, device=device) * 0.2  # 0.4-0.6
    b = 0.3 + torch.rand(n_gaussians, device=device) * 0.2  # 0.3-0.5
    colors = torch.stack([r, g, b], dim=-1)  # (N, 3)
    sh0 = ((colors - 0.5) / SH_C0)[:, None, :]  # (N, 1, 3)

    print(f"  Sampled {n_gaussians} Gaussians on SMPL mesh")
    print(f"  Means range: [{means.min():.3f}, {means.max():.3f}]")
    print(f"  Skin weights: {skin_weights.shape}, sum={skin_weights.sum(-1).min():.4f}-{skin_weights.sum(-1).max():.4f}")

    return means, scales_raw, quats, opacities_raw, sh0, skin_weights


def make_camera(smpl, device='cuda'):
    """Create a camera looking at the SMPL model from a good viewpoint."""
    from multiview_refine.cameras import Camera, invert_se3

    # SMPL is centred at origin, Y-up, ~1.7m tall.
    # Place camera at [2, 1, 4] looking at [0, 0, 0].
    # In OpenCV convention (+Y down): camera at [2, -1, 4] (slightly above).
    pos = torch.tensor([2.0, -1.0, 4.0], device=device)
    look_at = torch.tensor([0.0, 0.0, 0.0], device=device)
    up = torch.tensor([0.0, -1.0, 0.0], device=device)  # OpenCV: -Y is up

    forward = look_at - pos
    forward = forward / forward.norm()
    right = torch.cross(forward, up)
    right = right / right.norm()
    down = torch.cross(right, forward)
    down = down / down.norm()

    c2w = torch.eye(4, device=device)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = pos
    w2c = invert_se3(c2w)

    W, H = 1024, 768
    fx = 800.0
    return Camera(
        w2c=w2c, c2w=c2w, position=pos,
        fx=fx, fy=fx, cx=W / 2, cy=H / 2,
        width=W, height=H,
    )


def main():
    parser = argparse.ArgumentParser(description="Synthetic skinned Gaussian demo.")
    parser.add_argument('--smpl_model_path', required=True)
    parser.add_argument('--output_dir', default='synthetic_demo')
    parser.add_argument('--n_gaussians', type=int, default=50000)
    parser.add_argument('--n_frames', type=int, default=120)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── 1. Load SMPL ──
    print("=" * 60)
    print("Synthetic Skinned Gaussian Demo")
    print("=" * 60)
    print("\nStep 1: Load SMPL model")
    smpl = SMPLWrapper(args.smpl_model_path, device=args.device)

    # ── 2. Sample Gaussians on SMPL surface ──
    print(f"\nStep 2: Sample {args.n_gaussians} Gaussians on SMPL mesh surface")
    means, scales_raw, quats, opacities_raw, sh0, skin_weights = \
        sample_gaussians_on_smpl(smpl, n_gaussians=args.n_gaussians, device=args.device)

    # ── 3. Create SkinnedGaussians ──
    print(f"\nStep 3: Create SkinnedGaussians")
    skinned = SkinnedGaussians.from_attributes(
        means=means, scales_raw=scales_raw, quats=quats,
        opacities_raw=opacities_raw, sh0=sh0,
        skin_weights=skin_weights,
        n_joints=24, sh_degree=0, device=args.device,
    )

    # ── 4. Verify: rest pose = identity ──
    print(f"\nStep 4: Verify rest pose = identity")
    rest_T = smpl.get_joint_transforms(betas=torch.zeros(10, device=device))
    means_rest, _, _ = skinned.deform(rest_T)
    diff = (means_rest - means).abs().max()
    print(f"  Rest pose diff: {diff:.6f} ({'✓ identity' if diff < 0.01 else '✗ NOT identity'})")

    # ── 5. Generate demo poses ──
    print(f"\nStep 5: Generate {args.n_frames} demo pose frames")
    pose_seq = generate_demo_poses(n_frames=args.n_frames, device=args.device)
    save_pose_sequence(os.path.join(args.output_dir, 'demo_poses.json'), pose_seq)

    # ── 6. Create camera ──
    print(f"\nStep 6: Set up camera")
    cam = make_camera(smpl, device=args.device)
    print(f"  Camera: {cam.width}x{cam.height}, position={cam.position.tolist()}")

    # ── 7. Render rest pose (canonical) ──
    print(f"\nStep 7: Render rest pose")
    from multiview_refine.gsplat_wrapper import render_gaussians
    bg = torch.zeros(3, device=device)
    with torch.no_grad():
        rgb_rest, alpha_rest, _ = render_gaussians(
            means_rest, skinned.scales, skinned.quats,
            skinned.get_colors(), skinned.opacities, cam,
            background=bg, sh_degree=0)
    print(f"  Rest render: rgb_mean={rgb_rest.mean():.4f}, alpha={alpha_rest.mean():.4f}")
    # Save rest pose image.
    rest_img = cv2.cvtColor(
        (rgb_rest.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(args.output_dir, 'rest_pose.png'), rest_img)

    # ── 8. Render animation ──
    print(f"\nStep 8: Render animation ({args.n_frames} frames)")
    anim_path = os.path.join(args.output_dir, 'animation.mp4')
    render_animation(
        skinned, smpl, pose_seq, cam,
        output_path=anim_path, fps=args.fps,
    )

    # Comparison video.
    comp_path = os.path.join(args.output_dir, 'comparison.mp4')
    render_animation_comparison(
        skinned, smpl, pose_seq, cam,
        output_path=comp_path, fps=args.fps,
    )

    # ── 9. Save skinned Gaussians ──
    print(f"\nStep 9: Save skinned Gaussians")
    skinned_ply = os.path.join(args.output_dir, 'skinned_gaussians.ply')
    skinned.save_ply(skinned_ply)

    # ── 10. Verify animation effect ──
    print(f"\nStep 10: Verify animation")
    # Frame 30 (max arm raise).
    bp = pose_seq['body_pose']
    go = pose_seq['global_orient']
    tr = pose_seq['transl']
    T30 = smpl.get_joint_transforms(
        betas=pose_seq['betas'], body_pose=bp[30],
        global_orient=go[30], transl=tr[30])
    means_30, _, _ = skinned.deform(T30)
    diff30 = (means_30 - means_rest).abs().max()
    print(f"  Frame 30 vs rest: max_diff={diff30:.4f}")

    # Check which joints moved.
    per_gs = (means_30 - means_rest).norm(dim=-1)
    top = per_gs.topk(5)
    w = skinned.skin_weights
    joint_names = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee',
                   'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot',
                   'Neck', 'L_Collar', 'R_Collar', 'L_Shoulder', 'R_Shoulder',
                   'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hand', 'R_Hand', 'Head']
    print(f"  Top 5 moved Gaussians:")
    for idx in top.indices[:5]:
        dj = w[idx].topk(2)
        j = dj.indices[0].item()
        print(f"    GS {idx}: move={per_gs[idx]:.4f}, joint={joint_names[j]} (w={dj.values[0]:.3f})")

    print(f"\nDone! Outputs in {args.output_dir}/:")
    print(f"  animation.mp4       — SMPL-driven animation")
    print(f"  comparison.mp4      — canonical vs animated")
    print(f"  rest_pose.png       — rest pose render")
    print(f"  skinned_gaussians.ply — rigged Gaussians")
    print(f"  demo_poses.json     — pose sequence")


if __name__ == "__main__":
    main()
