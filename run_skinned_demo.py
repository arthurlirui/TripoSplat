"""Demo: rig static Gaussians to SMPL skeleton and render an animation.

Pipeline:
  1. Load static Gaussians from a PLY (e.g. from-scratch reconstruction).
  2. Load SMPL model.
  3. Rig: bind Gaussians to SMPL skeleton (inherit skinning weights).
  4. Generate a demo pose sequence (arm wave + rotation).
  5. Render animation video (LBS deformation + gsplat rendering).

Usage:
    export PYTHONUTF8=1
    python run_skinned_demo.py \\
        --gaussians scratch_output/refined_pruned.ply \\
        --smpl_model_path E:/Code/mmforall/model/assets/SMPL/SMPL_NEUTRAL.pkl \\
        --transforms static/000001/mvs/transforms_test.json \\
        --output_dir skinned_demo \\
        --n_frames 120 --fps 30

    # With a custom pose sequence:
    python run_skinned_demo.py \\
        --gaussians scratch_output/refined_pruned.ply \\
        --smpl_model_path E:/Code/mmforall/model/assets/SMPL/SMPL_NEUTRAL.pkl \\
        --transforms static/000001/mvs/transforms_test.json \\
        --pose_file my_poses.json \\
        --output_dir skinned_demo
"""

import argparse
import os
import sys
import math

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Skinned Gaussian splatting demo — SMPL-driven animation.")
    parser.add_argument('--gaussians', required=True,
                        help='Path to static Gaussian PLY file.')
    parser.add_argument('--smpl_model_path', required=True,
                        help='Path to SMPL .pkl file.')
    parser.add_argument('--transforms', default=None,
                        help='Path to transforms.json for camera setup. '
                             'If None, uses a default orbit camera.')
    parser.add_argument('--init_view', default='00.png',
                        help='Init view filename for camera selection.')
    parser.add_argument('--pose_file', default=None,
                        help='Path to custom pose sequence (JSON/NPZ). '
                             'If None, generates a demo pose sequence.')
    parser.add_argument('--output_dir', default='skinned_demo')
    parser.add_argument('--n_frames', type=int, default=120)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--image_scale', type=float, default=0.5,
                        help='Image resolution scale for rendering.')
    parser.add_argument('--n_neighbours', type=int, default=4,
                        help='Number of nearest SMPL vertices for weight init.')
    parser.add_argument('--temperature', type=float, default=0.05,
                        help='Softmax temperature for distance weighting.')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load static Gaussians ──
    print("=" * 60)
    print("Step 1: Load static Gaussians")
    print("=" * 60)
    from multiview_refine.ply_io import load_gaussians_ply
    means, scales_raw, quats, opacities_raw, sh0, _ = load_gaussians_ply(
        args.gaussians, device=args.device)
    print(f"  Loaded {len(means)} Gaussians from {args.gaussians}")
    print(f"  Means range: [{means.min():.3f}, {means.max():.3f}]")

    # ── 2. Load SMPL ──
    print(f"\n{'=' * 60}")
    print("Step 2: Load SMPL model")
    print("=" * 60)
    from skinned_gaussian.smpl_model import SMPLWrapper
    smpl = SMPLWrapper(args.smpl_model_path, gender='neutral', device=args.device)

    # ── 3. Rig: bind Gaussians to SMPL ──
    print(f"\n{'=' * 60}")
    print("Step 3: Rig Gaussians to SMPL skeleton")
    print("=" * 60)
    from skinned_gaussian.rigging import rig_gaussians_to_smpl

    # We need to align the Gaussian coordinate system with SMPL's.
    # SMPL rest pose is centred at origin, ~2m tall.
    # The static Gaussians are in the scene coordinate system.
    # For rigging, we work in the Gaussian's coordinate system directly —
    # the SMPL mesh will be placed to match the Gaussian cloud.
    # (In a real pipeline, you'd fit SMPL to the images first, but for
    #  this demo we just use the Gaussian centroid and scale.)

    # Estimate SMPL placement: centroid and scale of the Gaussian cloud.
    gc = means.median(0).values  # Gaussian centroid
    # Compute actual SMPL rest-pose height from the mesh bounding box.
    rest_verts, _ = smpl.get_rest_mesh()
    smpl_height = (rest_verts[:, 1].max() - rest_verts[:, 1].min()).item()
    gs_height = (means[:, 1].max() - means[:, 1].min()).item()
    smpl_scale = gs_height / smpl_height if smpl_height > 0.01 else 1.0

    print(f"  Gaussian centroid: {gc.tolist()}")
    print(f"  Gaussian height (Y range): {gs_height:.3f}")
    print(f"  SMPL scale to fit: {smpl_scale:.3f}")

    # Scale and translate SMPL rest mesh to match Gaussian cloud.
    # align_to_scene modifies smpl in-place.
    smpl.align_to_scene(scale=smpl_scale, transl=gc)

    skinned = rig_gaussians_to_smpl(
        means, scales_raw, quats, opacities_raw, sh0,
        smpl, betas=torch.zeros(10, device=args.device),
        n_neighbours=args.n_neighbours,
        temperature=args.temperature,
        device=args.device,
    )

    # ── 4. Generate or load pose sequence ──
    print(f"\n{'=' * 60}")
    print("Step 4: Prepare pose sequence")
    print("=" * 60)
    from skinned_gaussian.pose_io import generate_demo_poses, load_pose_sequence, save_pose_sequence

    if args.pose_file:
        pose_seq = load_pose_sequence(args.pose_file, device=args.device)
        print(f"  Loaded from {args.pose_file}")
    else:
        pose_seq = generate_demo_poses(
            n_frames=args.n_frames, device=args.device)
        # Save the generated poses for reference.
        save_pose_sequence(
            os.path.join(args.output_dir, 'demo_poses.json'), pose_seq)

    # ── 5. Set up camera ──
    print(f"\n{'=' * 60}")
    print("Step 5: Set up camera")
    print("=" * 60)

    if args.transforms:
        from multiview_refine.cameras import load_cameras, find_camera_by_filename
        cameras = load_cameras(args.transforms, device=args.device,
                               image_scale=args.image_scale, load_image=False)
        cam = cameras[0]  # Use first camera.
        print(f"  Using camera from {args.transforms}: {cam.width}x{cam.height}")
    else:
        # Default orbit camera.
        from multiview_refine.cameras import Camera, invert_se3
        W, H = 1024, 750
        fx = 700.0
        cam = Camera(
            w2c=invert_se3(torch.eye(4, device=args.device)),
            c2w=torch.eye(4, device=args.device),
            position=torch.tensor([0., 0., 5.], device=args.device),
            fx=fx, fy=fx, cx=W / 2, cy=H / 2, width=W, height=H,
        )
        print(f"  Default camera: {W}x{H}, fx={fx}")

    # ── 6. Render animation ──
    print(f"\n{'=' * 60}")
    print("Step 6: Render animation")
    print("=" * 60)
    from skinned_gaussian.animation import render_animation, render_animation_comparison

    # Main animation video.
    anim_path = os.path.join(args.output_dir, 'animation.mp4')
    render_animation(
        skinned, smpl, pose_seq, cam,
        output_path=anim_path, fps=args.fps,
    )

    # Comparison video (canonical vs animated).
    comp_path = os.path.join(args.output_dir, 'comparison.mp4')
    render_animation_comparison(
        skinned, smpl, pose_seq, cam,
        output_path=comp_path, fps=args.fps,
    )

    # ── 7. Save skinned Gaussians ──
    print(f"\n{'=' * 60}")
    print("Step 7: Save skinned Gaussians")
    print("=" * 60)
    skinned_ply = os.path.join(args.output_dir, 'skinned_gaussians.ply')
    skinned.save_ply(skinned_ply)
    print(f"\nDone! Outputs in {args.output_dir}/:")
    print(f"  animation.mp4      — SMPL-driven animation")
    print(f"  comparison.mp4     — canonical vs animated side-by-side")
    print(f"  skinned_gaussians.ply — rigged Gaussians with bone weights")
    print(f"  demo_poses.json    — generated pose sequence")


if __name__ == "__main__":
    main()
