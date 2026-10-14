"""Step-by-step pipeline for multi-view Gaussian refinement.

Each phase is independent and saves intermediate results, so you can inspect
the coordinate-system alignment at each stage.

Usage:
    # Step 1: Generate initial Gaussians from a single image
    python run_pipeline.py step1 --init_image static/000001/mvs/images/00.png \\
        --output_dir output/pipeline_output --num_gaussians 131072

    # Step 2: Align (rotate + scale) the Gaussians to the scene coordinate system
    python run_pipeline.py step2 --input_ply pipeline_output/initial_gaussians.ply \\
        --transforms static/000001/mvs/transforms_train.json \\
        --init_view 00.png --output_dir output/pipeline_output

    # Step 3: Render the aligned Gaussians from all training camera poses
    python run_pipeline.py step3 --input_ply pipeline_output/aligned_gaussians.ply \\
        --transforms static/000001/mvs/transforms_train.json \\
        --output_dir output/pipeline_output --image_scale 0.5

    # Run all steps in sequence
    python run_pipeline.py all --init_image static/000001/mvs/images/00.png \\
        --transforms static/000001/mvs/transforms_train.json \\
        --init_view 00.png --output_dir output/pipeline_output
"""

import argparse
import os
import sys
import math
import json

import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiview_refine.ply_io import load_gaussians_ply, save_gaussians_ply
from multiview_refine.video_io import create_video_writer

SH_C0 = 0.28209479177387814


# ── Step 1: Generate initial Gaussians ────────────────────────────────────

def step1_generate(args):
    """Generate initial Gaussians from a single image using TripoSplat."""
    from triposplat import TripoSplatPipeline

    print("=" * 60)
    print("Step 1: Generate initial Gaussians from single image")
    print("=" * 60)

    pipe = TripoSplatPipeline(
        ckpt_path=args.ckpt_path,
        decoder_path=args.decoder_path,
        dinov3_path=args.dinov3_path,
        flux2_vae_encoder_path=args.flux2_vae_encoder_path,
        rmbg_path=args.rmbg_path,
        device=args.device,
    )

    gaussian, prepared = pipe.run(
        args.init_image,
        num_gaussians=args.num_gaussians,
        remove_background=True,
        show_progress=True,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Save the prepared (preprocessed) image for reference.
    prepared.save(os.path.join(args.output_dir, "init_prepared.png"))

    # Extract raw Gaussian attributes (in TripoSplat's AABB coordinate frame).
    means = gaussian.get_xyz.detach().float()  # (N, 3) in [-0.5, 0.5]
    scales_activated = gaussian.get_scaling.detach().float()  # (N, 3)
    scales_raw = torch.log(scales_activated.clamp_min(1e-8))
    quats = (gaussian._rotation + gaussian.rots_bias[None, :]).detach().float()  # (N, 4)
    opacities_activated = gaussian.get_opacity.detach().squeeze(-1).float()  # (N,)
    opacities_raw = torch.log(opacities_activated.clamp(1e-6, 1 - 1e-6))
    sh0 = gaussian._features_dc.detach().float()  # (N, 1, 3)
    shN = torch.zeros(means.shape[0], 0, 3, dtype=torch.float32)

    print(f"  Gaussians: {means.shape[0]}")
    print(f"  Means range: x=[{means[:,0].min():.3f}, {means[:,0].max():.3f}]")
    print(f"              y=[{means[:,1].min():.3f}, {means[:,1].max():.3f}]")
    print(f"              z=[{means[:,2].min():.3f}, {means[:,2].max():.3f}]")
    print(f"  Centroid: {means.mean(dim=0).tolist()}")

    ply_path = os.path.join(args.output_dir, "initial_gaussians.ply")
    save_gaussians_ply(ply_path, means, scales_raw, quats, opacities_raw, sh0, shN)
    print(f"\nStep 1 complete. Initial Gaussians saved to {ply_path}")
    print("  These are in TripoSplat's normalized AABB [-0.5, 0.5] coordinate frame.")


# ── Step 2: Align to scene coordinate system ──────────────────────────────

def step2_align(args):
    """Align (rotate + scale) initial Gaussians to the scene coordinate system.

    Uses the shared ``estimate_similarity_transform_multiview`` from the
    alignment module, initialised with the yaw of the init camera's c2w
    rotation for a faster and more targeted search.
    """
    from multiview_refine.cameras import load_cameras, find_camera_by_filename
    from multiview_refine.quaternion import quaternion_compose, rotation_matrix_to_quaternion
    from multiview_refine.alignment import estimate_similarity_transform_multiview
    from multiview_refine.coord_registry import MODEL2SCENE

    print("=" * 60)
    print("Step 2: Align Gaussians to scene coordinate system")
    print("  Method: multiview_alignment_module")
    print("=" * 60)

    means, scales_raw, quats, opacities_raw, sh0, shN = load_gaussians_ply(
        args.input_ply, device=args.device)
    print(f"  Loaded {means.shape[0]} Gaussians from {args.input_ply}")
    print(f"  Means range: [{means.min():.3f}, {means.max():.3f}]")

    cameras = load_cameras(args.transforms, device=args.device,
                           image_scale=0.25, load_image=True)
    init_cam = find_camera_by_filename(cameras, args.init_view)
    print(f"  Init camera: cam_idx={init_cam.cam_idx}, position={init_cam.position.tolist()}")

    # ── Pre-transform: apply model→scene rotation ──
    # The PLY stores Gaussians in model space (AABB [-0.5,0.5]³).  We rotate
    # to the Y-up scene frame using MODEL2SCENE from coord_registry.  This
    # maps (x,y,z) → (x, z, −y), putting the model in the same Y-up world
    # frame as the SfM cameras.
    R_model2scene = MODEL2SCENE.to(args.device)

    means_scene = means @ R_model2scene.T  # rotate means to Y-up scene frame
    quats_scene = quaternion_compose(rotation_matrix_to_quaternion(R_model2scene), quats)

    # ── Extract init_yaw from the init camera's c2w rotation ──
    forward = init_cam.c2w[:3, 2]
    init_yaw = math.atan2(float(forward[0]), float(forward[2]))

    print(f"\n  R_model2scene applied to means (model → Y-up scene)")
    print(f"  init_yaw = {math.degrees(init_yaw):.1f}° (from c2w forward)")

    # ── Delegate to the shared alignment module ──
    scales_activated = scales_raw.exp()
    opacities = torch.sigmoid(opacities_raw)
    colors = sh0  # (N, 1, 3) SH DC

    scale, R, t = estimate_similarity_transform_multiview(
        tensors=(means_scene, scales_activated, quats_scene, opacities, colors),
        cameras=cameras,
        init_yaw=init_yaw,
        alpha_threshold=0.1,
        verbose=True,
    )

    # ── Apply final transform ──
    # The alignment module returns transform relative to the scene-space
    # means.  Total: xyz_scene = scale * (R @ means_scene) + t
    # which is equivalent to: scale * (R @ R_model2scene @ means_orig) + t
    means_aligned = scale * (means_scene @ R.T) + t
    scales_aligned_raw = scales_raw + math.log(scale)
    R_total = R @ R_model2scene  # model → scene
    q_total = rotation_matrix_to_quaternion(R_total)
    quats_aligned = quaternion_compose(q_total, quats)

    print(f"\n  Final: scale={scale:.4f}")
    print(f"  Aligned means range: [{means_aligned.min():.3f}, {means_aligned.max():.3f}]")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    ply_path = os.path.join(args.output_dir, "aligned_gaussians.ply")
    save_gaussians_ply(ply_path, means_aligned, scales_aligned_raw, quats_aligned,
                       opacities_raw, sh0, shN)

    params = {
        'scale': float(scale), 'rotation': R_total.cpu().numpy().tolist(),
        'translation': t.cpu().numpy().tolist(),
        'R_model2scene': R_model2scene.cpu().numpy().tolist(),
        'init_yaw_deg': math.degrees(init_yaw),
        'method': 'multiview_alignment_module',
    }
    with open(os.path.join(args.output_dir, "alignment_params.json"), 'w') as f:
        json.dump(params, f, indent=2)
    print(f"\n  Saved {ply_path}")
    print(f"Step 2 complete.")


# ── Step 3: Render from all training poses ────────────────────────────────

def step3_render(args):
    """Render the aligned Gaussians from all training camera poses and save images."""
    from multiview_refine.cameras import load_cameras
    from multiview_refine.gsplat_wrapper import render_gaussians

    print("=" * 60)
    print("Step 3: Render aligned Gaussians from all training poses")
    print("=" * 60)

    means, scales_raw, quats, opacities_raw, sh0, shN = load_gaussians_ply(
        args.input_ply, device=args.device)
    scales = scales_raw.exp()
    opacities = torch.sigmoid(opacities_raw)
    print(f"  Loaded {means.shape[0]} Gaussians from {args.input_ply}")
    print(f"  Means range: [{means.min():.3f}, {means.max():.3f}]")

    cameras = load_cameras(args.transforms, device=args.device,
                           image_scale=args.image_scale, load_image=True)
    print(f"  Loaded {len(cameras)} cameras at {cameras[0].width}x{cameras[0].height}")

    # Create output directories.
    render_dir = os.path.join(args.output_dir, "renders")
    gt_dir = os.path.join(args.output_dir, "gt")
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    bg = torch.zeros(3, device=args.device)

    print(f"\n  Rendering {len(cameras)} views...")

    # Also create a side-by-side comparison image grid.
    comparison_frames = []

    for i, cam in enumerate(cameras):
        with torch.no_grad():
            rgb, alpha, meta = render_gaussians(
                means, scales, quats, sh0, opacities, cam,
                background=bg, sh_degree=0,
            )

        # Save rendered image.
        render_np = (rgb.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        render_bgr = cv2.cvtColor(render_np, cv2.COLOR_RGB2BGR)
        fname = os.path.basename(cam.file_path) or f"cam_{cam.cam_idx}"
        cv2.imwrite(os.path.join(render_dir, f"{fname}.png"), render_bgr)

        # Save GT image.
        if cam.image is not None:
            gt_np = (cam.image.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(gt_dir, f"{fname}.png"), gt_bgr)
            # Side-by-side for video.
            side_by_side = np.concatenate([render_bgr, gt_bgr], axis=1)
        else:
            side_by_side = render_bgr
        comparison_frames.append(side_by_side)

        # Print per-camera stats.
        cam_z = (cam.w2c[:3, :3] @ means.mean(dim=0) + cam.w2c[:3, 3])[2].item()
        gt_mean = cam.image.mean().item() if cam.image is not None else 0.0
        if i < 5 or i % 6 == 0:
            print(f"    cam {i:2d} (idx={cam.cam_idx:2d}, {cam.file_path:12s}): "
                  f"z={cam_z:.2f}, render_mean={rgb.mean():.4f}, "
                  f"alpha={alpha.mean():.4f}, gt_mean={gt_mean:.4f}")

    # Save comparison video.
    video_path = os.path.join(args.output_dir, "alignment_check.mp4")
    H, W = comparison_frames[0].shape[:2]
    vw, video_path = create_video_writer(video_path, "mp4v", 5, (W, H))
    for frame in comparison_frames:
        vw.write(frame)
    vw.release()
    print(f"\n  Comparison video: {video_path}")
    print(f"  Rendered images: {render_dir}/")
    print(f"  GT images: {gt_dir}/")

    print(f"\nStep 3 complete. Inspect the renders to verify coordinate alignment.")
    print(f"  If the renders match the GT images, the alignment is correct.")
    print(f"  If renders are black or misaligned, check the alignment parameters.")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step-by-step multi-view Gaussian pipeline.")
    subparsers = parser.add_subparsers(dest='step', required=True)

    # Common arguments.
    def add_common_args(p):
        p.add_argument('--output_dir', default='output/pipeline_output')
        p.add_argument('--device', default='cuda')

    # Step 1.
    p1 = subparsers.add_parser('step1', help='Generate initial Gaussians from single image')
    p1.add_argument('--init_image', required=True)
    p1.add_argument('--num_gaussians', type=int, default=131072)
    p1.add_argument('--ckpt_path', default='ckpts/diffusion_models/triposplat_fp16.safetensors')
    p1.add_argument('--decoder_path', default='ckpts/vae/triposplat_vae_decoder_fp16.safetensors')
    p1.add_argument('--dinov3_path', default='ckpts/clip_vision/dino_v3_vit_h.safetensors')
    p1.add_argument('--flux2_vae_encoder_path', default='ckpts/vae/flux2-vae.safetensors')
    p1.add_argument('--rmbg_path', default='ckpts/background_removal/birefnet.safetensors')
    add_common_args(p1)

    # Step 2.
    p2 = subparsers.add_parser('step2', help='Align Gaussians to scene coordinate system')
    p2.add_argument('--input_ply', required=True)
    p2.add_argument('--transforms', required=True)
    p2.add_argument('--init_view', required=True, help='e.g. 00.png')
    add_common_args(p2)

    # Step 3.
    p3 = subparsers.add_parser('step3', help='Render aligned Gaussians from all training poses')
    p3.add_argument('--input_ply', required=True)
    p3.add_argument('--transforms', required=True)
    p3.add_argument('--image_scale', type=float, default=0.5)
    add_common_args(p3)

    # All steps.
    pa = subparsers.add_parser('all', help='Run all steps in sequence')
    pa.add_argument('--init_image', required=True)
    pa.add_argument('--transforms', required=True)
    pa.add_argument('--init_view', required=True)
    pa.add_argument('--num_gaussians', type=int, default=131072)
    pa.add_argument('--image_scale', type=float, default=0.5)
    pa.add_argument('--ckpt_path', default='ckpts/diffusion_models/triposplat_fp16.safetensors')
    pa.add_argument('--decoder_path', default='ckpts/vae/triposplat_vae_decoder_fp16.safetensors')
    pa.add_argument('--dinov3_path', default='ckpts/clip_vision/dino_v3_vit_h.safetensors')
    pa.add_argument('--flux2_vae_encoder_path', default='ckpts/vae/flux2-vae.safetensors')
    pa.add_argument('--rmbg_path', default='ckpts/background_removal/birefnet.safetensors')
    add_common_args(pa)

    args = parser.parse_args()

    if args.step == 'step1':
        step1_generate(args)
    elif args.step == 'step2':
        step2_align(args)
    elif args.step == 'step3':
        step3_render(args)
    elif args.step == 'all':
        # Run all steps.
        args.input_ply = os.path.join(args.output_dir, "initial_gaussians.ply")
        step1_generate(args)

        args.input_ply = os.path.join(args.output_dir, "initial_gaussians.ply")
        step2_align(args)

        args.input_ply = os.path.join(args.output_dir, "aligned_gaussians.ply")
        step3_render(args)


if __name__ == "__main__":
    main()
