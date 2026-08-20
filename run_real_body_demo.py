"""Drive real reconstructed Gaussians with SMPL-X skeleton.

Reads Gaussians from scratch_output/refined_pruned.ply, rigs them to
SMPL-X (55 joints + expression), and renders an animation with a
rotating camera showing various body actions.

Usage:
    export PYTHONUTF8=1
    python run_real_body_demo.py \\
        --gaussians scratch_output/refined_pruned.ply \\
        --smpl_model_path E:/Code/mmforall/model/assets/SMPLX/SMPLX_NEUTRAL_2020.npz \\
        --output_dir real_body_demo --n_frames 240 --fps 30
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


def make_orbit_cameras(centre, n_frames, device='cuda', radius=6.0, elev_deg=10):
    """Generate orbit cameras around a centre point."""
    from multiview_refine.cameras import Camera, invert_se3

    elev = math.radians(elev_deg)
    cameras = []
    for i in range(n_frames):
        az = 2 * math.pi * i / n_frames
        pos = centre + torch.tensor([
            radius * math.cos(elev) * math.sin(az),
            -radius * math.sin(elev),
            radius * math.cos(elev) * math.cos(az),
        ], device=device)
        forward = centre - pos
        forward = forward / forward.norm()
        up = torch.tensor([0., -1., 0.], device=device)
        right = torch.cross(forward, up)
        right = right / right.norm()
        down = torch.cross(right, forward)
        down = down / down.norm()
        c2w = torch.eye(4, device=device)
        c2w[:3, 0] = right; c2w[:3, 1] = down; c2w[:3, 2] = forward; c2w[:3, 3] = pos
        w2c = invert_se3(c2w)
        W, H = 1024, 768
        fx = 700.0
        cameras.append(Camera(w2c=w2c, c2w=c2w, position=pos,
                              fx=fx, fy=fx, cx=W/2, cy=H/2, width=W, height=H))
    return cameras


def generate_body_actions(n_frames=240, device='cuda'):
    """Generate natural body actions: wave, turn, reach, breathe."""
    dev = torch.device(device)
    t = torch.linspace(0, 2 * math.pi, n_frames, device=dev)

    # Split into 4 action phases:
    # 0-25%: right arm wave
    # 25-50%: both arms raise
    # 50-75%: body turn + reach
    # 75-100%: return to rest

    body_pose = torch.zeros(n_frames, 63, device=dev)
    global_orient = torch.zeros(n_frames, 3, device=dev)
    jaw_pose = torch.zeros(n_frames, 3, device=dev)
    expression = torch.zeros(n_frames, 100, device=dev)

    phase1 = t < math.pi / 2      # wave
    phase2 = (t >= math.pi / 2) & (t < math.pi)   # both arms
    phase3 = (t >= math.pi) & (t < 3 * math.pi / 2)  # turn + reach
    phase4 = t >= 3 * math.pi / 2  # return

    # Phase 1: Right arm wave (R_Shoulder=16→slot15, R_Elbow=17→slot16)
    t1 = t[phase1]
    body_pose[phase1, 15*3+0] = torch.sin(t1 * 4) * 0.8   # wave up/down
    body_pose[phase1, 15*3+2] = 0.2                        # arm out
    body_pose[phase1, 16*3+0] = 0.5 + torch.sin(t1 * 4) * 0.3  # elbow bend

    # Phase 2: Both arms raise
    t2 = t[phase2] - math.pi / 2
    body_pose[phase2, 15*3+0] = torch.sin(t2 * 2) * 0.6   # R arm
    body_pose[phase2, 14*3+0] = torch.sin(t2 * 2 + 0.5) * 0.6  # L arm
    body_pose[phase2, 16*3+0] = 0.3                        # R elbow
    body_pose[phase2, 17*3+0] = 0.3                        # L elbow

    # Phase 3: Body turn + reach forward
    t3 = t[phase3] - math.pi
    global_orient[phase3, 1] = torch.sin(t3 * 2) * 0.3     # turn body
    body_pose[phase3, 15*3+0] = 0.4                         # R arm forward
    body_pose[phase3, 14*3+0] = 0.4                         # L arm forward
    body_pose[phase3, 11*3+0] = torch.sin(t3 * 2) * 0.1    # head turn

    # Phase 4: Return to rest (smooth decay)
    t4 = t[phase4] - 3 * math.pi / 2
    decay = torch.exp(-t4 * 2)
    body_pose[phase4, 15*3+0] = torch.sin(t4 * 3) * 0.3 * decay
    body_pose[phase4, 14*3+0] = torch.sin(t4 * 3 + 0.5) * 0.3 * decay

    # Continuous: breathing + jaw + expression
    body_pose[:, 2*3+0] = torch.sin(t * 2) * 0.03   # spine breathing
    body_pose[:, 5*3+0] = torch.sin(t * 2) * 0.03
    jaw_pose[:, 0] = (torch.sin(t * 4) * 0.15 + 0.1).clamp(min=0)  # talking
    expression[:, 0] = torch.sin(t * 0.8) * 0.5      # smile
    expression[:, 1] = torch.sin(t * 1.1) * 0.3      # brow

    transl = torch.zeros(n_frames, 3, device=dev)
    leye = torch.zeros(n_frames, 3, device=dev)
    reye = torch.zeros(n_frames, 3, device=dev)
    leye[:, 1] = torch.sin(t * 0.5) * 0.1
    reye[:, 1] = torch.sin(t * 0.5) * 0.1

    print(f"  Actions: wave → both arms raise → body turn + reach → return")
    return {
        'betas': torch.zeros(300, device=dev),
        'body_pose': body_pose, 'global_orient': global_orient,
        'transl': transl, 'jaw_pose': jaw_pose,
        'leye_pose': leye, 'reye_pose': reye,
        'expression': expression, 'n_frames': n_frames,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussians', default='scratch_output/refined_pruned.ply')
    parser.add_argument('--smpl_model_path', required=True)
    parser.add_argument('--output_dir', default='real_body_demo')
    parser.add_argument('--n_frames', type=int, default=240)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ═══ 1. Load real reconstructed Gaussians ═══
    print("=" * 70)
    print("Real Body Demo — drive reconstructed Gaussians with SMPL-X")
    print("=" * 70)
    print(f"\nStep 1: Load Gaussians from {args.gaussians}")
    from multiview_refine.ply_io import load_gaussians_ply
    means, scales_raw, quats, opac_raw, sh0, shN = load_gaussians_ply(args.gaussians, device)
    opacities = torch.sigmoid(opac_raw)
    print(f"  {len(means)} Gaussians")
    print(f"  Y range: [{means[:,1].min():.3f}, {means[:,1].max():.3f}]")

    # Prune dark opaque floaters.
    colors_eval = (sh0[:, 0, :] * SH_C0 + 0.5).clamp(0, 1)
    keep = ~((colors_eval.max(dim=-1).values < 0.01) & (opacities > 0.5))
    print(f"  Pruning {(~keep).sum().item()} dark floaters → {keep.sum().item()} remaining")
    means, scales_raw, quats, opac_raw, sh0 = [x[keep] for x in [means, scales_raw, quats, opac_raw, sh0]]

    # ═══ 2. Load SMPL-X + align ═══
    print(f"\nStep 2: Load SMPL-X + align to Gaussian cloud")
    from skinned_gaussian.smplx_model import SMPLXWrapper
    smplx = SMPLXWrapper(args.smpl_model_path, device=args.device)
    rest_verts, _ = smplx.get_rest_mesh()
    smpl_h = (rest_verts[:, 1].max() - rest_verts[:, 1].min()).item()
    gs_h = (means[:, 1].max() - means[:, 1].min()).item()
    scale = gs_h / smpl_h
    gc = means.median(0).values
    smplx.align_to_scene(scale=scale, transl=gc)
    print(f"  Scale: {scale:.3f}, Centroid: {gc.tolist()}")

    # ═══ 3. Rig Gaussians to SMPL-X ═══
    print(f"\nStep 3: Rig {len(means)} Gaussians to SMPL-X (55 joints + expression)")
    from skinned_gaussian.rigging import rig_gaussians_to_smplx
    skinned = rig_gaussians_to_smplx(
        means, scales_raw, quats, opac_raw, sh0, smplx,
        n_neighbours=4, temperature=0.1, device=args.device,
        weights_optimizable=False, n_expression_bases=100)

    # ═══ 4. Verify rest = identity ═══
    rest_T = smplx.get_joint_transforms()
    means_rest, _, _ = skinned.deform(rest_T)
    diff = (means_rest - means).abs().max()
    print(f"\nStep 4: Rest pose diff = {diff:.6f} ({'✓' if diff < 0.01 else '✗'})")

    # ═══ 5. Generate actions + orbit cameras ═══
    print(f"\nStep 5: Generate {args.n_frames} action frames + orbit cameras")
    pose_seq = generate_body_actions(args.n_frames, device)
    cameras = make_orbit_cameras(gc, args.n_frames, device)

    # ═══ 6. Render ═══
    print(f"\nStep 6: Render {args.n_frames} frames (orbit + actions)")
    from multiview_refine.gsplat_wrapper import render_gaussians
    from multiview_refine.video_io import create_video_writer

    bg = torch.zeros(3, device=device)
    W, H = cameras[0].width, cameras[0].height
    vw, _ = create_video_writer(
        os.path.join(args.output_dir, 'real_body_demo.mp4'), 'mp4v', args.fps, (W, H))

    t0 = time.time()
    for i in range(args.n_frames):
        T = smplx.get_joint_transforms(
            betas=pose_seq['betas'],
            body_pose=pose_seq['body_pose'][i],
            global_orient=pose_seq['global_orient'][i],
            transl=pose_seq['transl'][i],
            jaw_pose=pose_seq['jaw_pose'][i],
            leye_pose=pose_seq['leye_pose'][i],
            reye_pose=pose_seq['reye_pose'][i],
            expression=pose_seq['expression'][i])

        with torch.no_grad():
            rgb, alpha, meta = skinned.render_posed(
                T, cameras[i], background=bg, sh_degree=0,
                expression=pose_seq['expression'][i])

        frame = cv2.cvtColor(
            (rgb.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8),
            cv2.COLOR_RGB2BGR)
        vw.write(frame)

        if i % 30 == 0 or i == 0:
            elapsed = time.time() - t0
            # Measure deformation from rest
            means_i, _, _ = skinned.deform(T, expression=pose_seq['expression'][i])
            deform = (means_i - means_rest).abs().max().item()
            print(f"  frame {i:4d}/{args.n_frames}: rgb={rgb.mean():.4f}, "
                  f"deform={deform:.4f}, {elapsed:.1f}s")

    vw.release()
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Video: {os.path.getsize(os.path.join(args.output_dir, 'real_body_demo.mp4'))//1024} KB")

    # ═══ 7. Save key frames ═══
    print(f"\nStep 7: Save key frames")
    for idx in [0, 60, 120, 180]:
        T = smplx.get_joint_transforms(
            body_pose=pose_seq['body_pose'][idx],
            global_orient=pose_seq['global_orient'][idx],
            jaw_pose=pose_seq['jaw_pose'][idx],
            expression=pose_seq['expression'][idx])
        with torch.no_grad():
            rgb, _, _ = skinned.render_posed(
                T, cameras[idx], background=bg, sh_degree=0,
                expression=pose_seq['expression'][idx])
        frame = cv2.cvtColor(
            (rgb.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8),
            cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.output_dir, f'frame_{idx:03d}.png'), frame)
        print(f"    frame_{idx:03d}.png")

    print(f"\n{'=' * 70}")
    print(f"Done! Outputs in {args.output_dir}/:")
    print(f"  real_body_demo.mp4 — {args.n_frames} frames, rotating camera + body actions")
    print(f"  frame_000/060/120/180.png — key frame screenshots")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
