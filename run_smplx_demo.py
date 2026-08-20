"""Generate a skinned Gaussian demo with rotating camera + varied actions.

Actions:
  - Arm wave (right arm raises and lowers)
  - Body turn (rotation left/right)
  - Jaw open/close (talking motion)
  - Expression change (smile/frown)
  - Breathing (spine subtle movement)

Camera:
  - Slow orbit around the body (360° over the full sequence)

Usage:
    export PYTHONUTF8=1
    python run_smplx_demo.py \\
        --smpl_model_path E:/Code/mmforall/model/assets/SMPLX/SMPLX_NEUTRAL_2020.npz \\
        --output_dir smplx_demo --n_frames 240 --fps 30
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

SH_C0 = 0.28209479177387814


def sample_gaussians_on_smplx(smplx, n_gaussians=100000, device='cuda'):
    """Sample Gaussians on SMPL-X mesh surface with skinning weights + expression bases."""
    v, f = smplx.get_rest_mesh()
    w = smplx.get_skinning_weights()  # (10475, 55)

    # Area-weighted face sampling.
    v0, v1, v2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    areas = 0.5 * cross.norm(dim=-1)
    probs = areas / areas.sum()
    face_idx = torch.multinomial(probs, n_gaussians, replacement=True)

    # Barycentric.
    u = torch.rand(n_gaussians, 1, device=device)
    v_ = torch.rand(n_gaussians, 1, device=device)
    mask = (u + v_) > 1
    u[mask] = 1 - u[mask]
    v_[mask] = 1 - v_[mask]
    w_bc = 1 - u - v_

    # Positions + normals for offset.
    normals = cross / cross.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    f_verts = v[f[face_idx]]
    f_normals = normals[face_idx]
    offsets = (torch.rand(n_gaussians, 1, device=device) - 0.5) * 0.02
    means = (u * f_verts[:, 0] + v_ * f_verts[:, 1] + w_bc * f_verts[:, 2]).squeeze(-1)
    means = means + offsets * f_normals

    # Skinning weights.
    f_w = w[f[face_idx]]
    skin_w = (u * f_w[:, 0] + v_ * f_w[:, 1] + w_bc * f_w[:, 2]).squeeze(-1)
    skin_w = skin_w / skin_w.sum(-1, keepdim=True).clamp_min(1e-8)

    # Scales, quats, opacity.
    mesh_size = (v.max(0).values - v.min(0).values).norm().item()
    scales_raw = torch.log(torch.full((n_gaussians, 3), mesh_size * 0.08, device=device))
    quats = torch.zeros(n_gaussians, 4, device=device)
    quats[:, 0] = 1.0
    opac_raw = torch.full((n_gaussians,), math.log(0.9 / 0.1), device=device)

    # Colors: skin tone with slight variation.
    r = 0.65 + torch.rand(n_gaussians, device=device) * 0.15
    g = 0.45 + torch.rand(n_gaussians, device=device) * 0.15
    b = 0.35 + torch.rand(n_gaussians, device=device) * 0.15
    colors = torch.stack([r, g, b], dim=-1)
    sh0 = ((colors - 0.5) / SH_C0)[:, None, :]

    print(f"  Sampled {n_gaussians} Gaussians on SMPL-X mesh ({mesh_size:.2f}m)")
    return means, scales_raw, quats, opac_raw, sh0, skin_w


def make_orbit_cameras(smplx, n_frames, device='cuda'):
    """Generate a slow orbit camera trajectory around the SMPL-X body."""
    from multiview_refine.cameras import Camera, invert_se3

    v, _ = smplx.get_rest_mesh()
    centre = v.median(0).values
    radius = 4.0
    elev = math.radians(10)  # slightly above

    cameras = []
    for i in range(n_frames):
        az = 2 * math.pi * i / n_frames  # full 360° over all frames
        pos = centre + torch.tensor([
            radius * math.cos(elev) * math.sin(az),
            -radius * math.sin(elev),  # OpenCV +Y down → negative = above
            radius * math.cos(elev) * math.cos(az),
        ], device=device)

        forward = centre - pos
        forward = forward / forward.norm()
        up = torch.tensor([0., -1., 0.], device=device)  # OpenCV: -Y is up
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
        cameras.append(Camera(
            w2c=w2c, c2w=c2w, position=pos,
            fx=fx, fy=fx, cx=W / 2, cy=H / 2,
            width=W, height=H))
    return cameras


def generate_rich_poses(n_frames=240, device='cuda'):
    """Generate a rich pose sequence with multiple actions."""
    dev = torch.device(device)
    t = torch.linspace(0, 2 * math.pi, n_frames, device=dev)

    # Body pose (21 joints × 3 = 63).
    body_pose = torch.zeros(n_frames, 63, device=dev)

    # R_Shoulder (SMPL-X joint 16 → slot 15): arm wave.
    body_pose[:, 15 * 3 + 0] = torch.sin(t) * 0.8       # raise/lower
    body_pose[:, 15 * 3 + 2] = torch.cos(t * 0.7) * 0.3  # rotate

    # R_Elbow (joint 17 → slot 16): bend.
    body_pose[:, 16 * 3 + 0] = 0.4 + torch.sin(t) * 0.3

    # L_Shoulder (joint 15 → slot 14): counter-wave.
    body_pose[:, 14 * 3 + 0] = torch.sin(t + math.pi * 0.7) * 0.5
    body_pose[:, 14 * 3 + 2] = torch.cos(t * 0.8) * 0.2

    # L_Elbow (joint 18 → slot 17): bend.
    body_pose[:, 17 * 3 + 0] = 0.3 + torch.sin(t + 1) * 0.2

    # Spine breathing.
    body_pose[:, 2 * 3 + 0] = torch.sin(t * 2) * 0.05   # Spine1
    body_pose[:, 5 * 3 + 0] = torch.sin(t * 2) * 0.05   # Spine2

    # Head slight nod (joint 12 → slot 11).
    body_pose[:, 11 * 3 + 0] = torch.sin(t * 0.5) * 0.15

    # Global rotation: slow turn.
    global_orient = torch.zeros(n_frames, 3, device=dev)
    global_orient[:, 1] = torch.sin(t * 0.3) * 0.15

    transl = torch.zeros(n_frames, 3, device=dev)

    # Jaw: talking motion (open/close).
    jaw_pose = torch.zeros(n_frames, 3, device=dev)
    jaw_pose[:, 0] = (torch.sin(t * 3) * 0.2 + 0.2).clamp(min=0)  # fast jaw movement

    # Eyes: gaze shift.
    leye_pose = torch.zeros(n_frames, 3, device=dev)
    leye_pose[:, 1] = torch.sin(t * 0.4) * 0.15
    reye_pose = torch.zeros(n_frames, 3, device=dev)
    reye_pose[:, 1] = torch.sin(t * 0.4) * 0.15

    # Expression: smile/frown + brow.
    expression = torch.zeros(n_frames, 100, device=dev)
    expression[:, 0] = torch.sin(t * 0.5) * 0.8   # smile cycle
    expression[:, 1] = torch.sin(t * 0.7) * 0.4   # brow
    expression[:, 2] = torch.sin(t * 1.2) * 0.3   # cheek

    print(f"  Generated {n_frames} frames: arm wave + body turn + jaw + expression + head nod")
    return {
        'betas': torch.zeros(300, device=dev),
        'body_pose': body_pose,
        'global_orient': global_orient,
        'transl': transl,
        'jaw_pose': jaw_pose,
        'leye_pose': leye_pose,
        'reye_pose': reye_pose,
        'expression': expression,
        'n_frames': n_frames,
    }


def main():
    parser = argparse.ArgumentParser(description="SMPL-X skinned Gaussian demo with rotating camera.")
    parser.add_argument('--smpl_model_path', required=True)
    parser.add_argument('--output_dir', default='smplx_demo')
    parser.add_argument('--n_gaussians', type=int, default=100000)
    parser.add_argument('--n_frames', type=int, default=240)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ── 1. Load SMPL-X ──
    print("=" * 70)
    print("SMPL-X Skinned Gaussian Demo — rotating camera + rich actions")
    print("=" * 70)
    print("\nStep 1: Load SMPL-X model")
    from skinned_gaussian.smplx_model import SMPLXWrapper
    smplx = SMPLXWrapper(args.smpl_model_path, device=args.device)

    # ── 2. Sample Gaussians ──
    print(f"\nStep 2: Sample {args.n_gaussians} Gaussians on SMPL-X mesh")
    means, scales_raw, quats, opac_raw, sh0, skin_w = \
        sample_gaussians_on_smplx(smplx, args.n_gaussians, device)

    # ── 3. Rig to SMPL-X (with expression bases + optimizable weights) ──
    print(f"\nStep 3: Rig to SMPL-X (55 joints + expression)")
    from skinned_gaussian.rigging import rig_gaussians_to_smplx
    skinned = rig_gaussians_to_smplx(
        means, scales_raw, quats, opac_raw, sh0, smplx,
        n_neighbours=4, temperature=0.05, device=args.device,
        weights_optimizable=True, n_expression_bases=100)

    # ── 4. Generate poses ──
    print(f"\nStep 4: Generate {args.n_frames} pose frames")
    pose_seq = generate_rich_poses(args.n_frames, device)

    # ── 5. Generate orbit cameras ──
    print(f"\nStep 5: Generate orbit camera trajectory")
    cameras = make_orbit_cameras(smplx, args.n_frames, device)

    # ── 6. Verify rest pose = identity ──
    print(f"\nStep 6: Verify rest pose = identity")
    rest_T = smplx.get_joint_transforms()
    means_rest, _, _ = skinned.deform(rest_T)
    diff = (means_rest - means).abs().max()
    print(f"  Rest diff: {diff:.6f} ({'✓' if diff < 0.01 else '✗'})")

    # ── 7. Render animation with rotating camera ──
    print(f"\nStep 7: Render {args.n_frames} frames (orbit camera + animation)")
    from multiview_refine.gsplat_wrapper import render_gaussians
    from multiview_refine.video_io import create_video_writer

    bg = torch.zeros(3, device=device)
    W, H = cameras[0].width, cameras[0].height
    vw, _ = create_video_writer(
        os.path.join(args.output_dir, 'smplx_demo.mp4'), 'mp4v', args.fps, (W, H))

    t0 = time.time()
    for i in range(args.n_frames):
        # Joint transforms with jaw/eye/expression.
        T = smplx.get_joint_transforms(
            betas=pose_seq['betas'],
            body_pose=pose_seq['body_pose'][i],
            global_orient=pose_seq['global_orient'][i],
            transl=pose_seq['transl'][i],
            jaw_pose=pose_seq['jaw_pose'][i],
            leye_pose=pose_seq['leye_pose'][i],
            reye_pose=pose_seq['reye_pose'][i],
            expression=pose_seq['expression'][i])

        expr = pose_seq['expression'][i]

        with torch.no_grad():
            rgb, alpha, meta = skinned.render_posed(
                T, cameras[i], background=bg, sh_degree=0, expression=expr)

        frame = cv2.cvtColor(
            (rgb.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR)
        vw.write(frame)

        if i % 30 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  frame {i:4d}/{args.n_frames}: rgb={rgb.mean():.4f}, "
                  f"alpha={alpha.mean():.4f}, {elapsed:.1f}s")

    vw.release()
    elapsed = time.time() - t0
    print(f"\n  Video saved: {os.path.getsize(args.output_dir + '/smplx_demo.mp4') // 1024} KB "
          f"({elapsed:.1f}s)")

    # ── 8. Save key frames ──
    print(f"\nStep 8: Save key frames")
    # Re-render a few key frames at higher quality for screenshots.
    key_frames = [0, 60, 120, 180]
    for idx in key_frames:
        T = smplx.get_joint_transforms(
            betas=pose_seq['betas'],
            body_pose=pose_seq['body_pose'][idx],
            global_orient=pose_seq['global_orient'][idx],
            transl=pose_seq['transl'][idx],
            jaw_pose=pose_seq['jaw_pose'][idx],
            expression=pose_seq['expression'][idx])
        with torch.no_grad():
            rgb, _, _ = skinned.render_posed(
                T, cameras[idx], background=bg, sh_degree=0,
                expression=pose_seq['expression'][idx])
        frame = cv2.cvtColor(
            (rgb.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR)
        path = os.path.join(args.output_dir, f'frame_{idx:03d}.png')
        cv2.imwrite(path, frame)
        print(f"    {path}")

    # ── 9. Save skinned Gaussians ──
    skinned.save_ply(os.path.join(args.output_dir, 'skinned_gaussians.ply'))

    # ── 10. Verify animation effect ──
    print(f"\nStep 10: Verify animation")
    per_gs_diffs = []
    for idx in [30, 60, 120, 180]:
        T = smplx.get_joint_transforms(
            betas=pose_seq['betas'],
            body_pose=pose_seq['body_pose'][idx],
            jaw_pose=pose_seq['jaw_pose'][idx],
            expression=pose_seq['expression'][idx])
        means_i, _, _ = skinned.deform(T, expression=pose_seq['expression'][idx])
        d = (means_i - means_rest).abs().max().item()
        per_gs_diffs.append(d)
        print(f"  Frame {idx}: max deformation = {d:.4f}")

    print(f"\n{'=' * 70}")
    print(f"Done! Outputs in {args.output_dir}/:")
    print(f"  smplx_demo.mp4          — {args.n_frames} frames, orbit camera + actions")
    print(f"  frame_000/060/120/180.png — key frame screenshots")
    print(f"  skinned_gaussians.ply   — rigged Gaussians (55 joints + expression)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
