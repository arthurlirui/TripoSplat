"""Step-by-step pipeline for multi-view Gaussian refinement.

Each phase is independent and saves intermediate results, so you can inspect
the coordinate-system alignment at each stage.

Usage:
    # Step 1: Generate initial Gaussians from a single image
    python run_pipeline.py step1 --init_image static/000001/mvs/images/00.png \\
        --output_dir pipeline_output --num_gaussians 131072

    # Step 2: Align (rotate + scale) the Gaussians to the scene coordinate system
    python run_pipeline.py step2 --input_ply pipeline_output/initial_gaussians.ply \\
        --transforms static/000001/mvs/transforms_train.json \\
        --init_view 00.png --output_dir pipeline_output

    # Step 3: Render the aligned Gaussians from all training camera poses
    python run_pipeline.py step3 --input_ply pipeline_output/aligned_gaussians.ply \\
        --transforms static/000001/mvs/transforms_train.json \\
        --output_dir pipeline_output --image_scale 0.5

    # Run all steps in sequence
    python run_pipeline.py all --init_image static/000001/mvs/images/00.png \\
        --transforms static/000001/mvs/transforms_train.json \\
        --init_view 00.png --output_dir pipeline_output
"""

import argparse
import os
import sys
import math

import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SH_C0 = 0.28209479177387814


# ── PLY I/O ────────────────────────────────────────────────────────────────

def save_gaussians_ply(path, means, scales_raw, quats, opacities_raw, sh0, shN=None):
    """Save Gaussians to a 3DGS-compatible PLY (raw/pre-activation values)."""
    import struct
    from plyfile import PlyData, PlyElement

    N = means.shape[0]
    means_np = means.cpu().numpy()
    sh0_np = sh0.cpu().numpy()  # (N, 1, 3)
    f_dc = sh0_np.transpose(0, 2, 1).reshape(N, -1)  # (N, 3)

    if shN is not None and shN.numel() > 0:
        shN_np = shN.cpu().numpy()
        f_rest = shN_np.transpose(0, 2, 1).reshape(N, -1)
    else:
        f_rest = np.zeros((N, 0), dtype=np.float32)

    normals = np.zeros_like(means_np)
    opacities_np = opacities_raw.cpu().numpy().reshape(-1, 1)
    scales_np = scales_raw.cpu().numpy()
    rotations_np = torch.nn.functional.normalize(quats, dim=-1).cpu().numpy()

    data = np.concatenate([means_np, normals, f_dc, f_rest, opacities_np, scales_np, rotations_np], axis=1)

    names = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(f_dc.shape[1]):
        names.append(f'f_dc_{i}')
    for i in range(f_rest.shape[1]):
        names.append(f'f_rest_{i}')
    names.append('opacity')
    for i in range(3):
        names.append(f'scale_{i}')
    for i in range(4):
        names.append(f'rot_{i}')

    dtype = [(n, 'f4') for n in names]
    elements = np.empty(N, dtype=dtype)
    elements[:] = list(map(tuple, data))

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el], byte_order='<', text=False).write(path)
    print(f"  Saved {N} Gaussians to {path}")


def load_gaussians_ply(path, device='cuda'):
    """Load Gaussians from PLY. Returns (means, scales_raw, quats, opacities_raw, sh0, shN)."""
    from plyfile import PlyData
    plydata = PlyData.read(path)
    v = plydata['vertex']
    data = np.stack([v[name] for name in v.data.dtype.names], axis=1)
    names = list(v.data.dtype.names)

    def col(name):
        return data[:, names.index(name)]

    means = np.stack([col('x'), col('y'), col('z')], axis=-1)
    f_dc = np.stack([col('f_dc_0'), col('f_dc_1'), col('f_dc_2')], axis=-1)
    sh0 = f_dc[:, None, :]
    f_rest_names = sorted([n for n in names if n.startswith('f_rest_')])
    if f_rest_names:
        f_rest = np.stack([col(n) for n in f_rest_names], axis=-1)
        shN = f_rest.reshape(-1, len(f_rest_names) // 3, 3)
    else:
        shN = np.zeros((means.shape[0], 0, 3), dtype=np.float32)

    opacities = col('opacity')
    scales = np.stack([col('scale_0'), col('scale_1'), col('scale_2')], axis=-1)
    quats = np.stack([col('rot_0'), col('rot_1'), col('rot_2'), col('rot_3')], axis=-1)

    dev = torch.device(device)
    return (
        torch.from_numpy(means).float().to(dev),
        torch.from_numpy(scales).float().to(dev),
        torch.from_numpy(quats).float().to(dev),
        torch.from_numpy(opacities).float().to(dev),
        torch.from_numpy(sh0).float().to(dev),
        torch.from_numpy(shN).float().to(dev),
    )


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

    Algorithm:
      1. Convert TripoSplat model coords → OpenCV via _DEFAULT_TRANSFORM:
         model→PLY (OpenGL): T_ply = [[1,0,0],[0,0,-1],[0,1,0]]
         PLY→OpenCV:         T_cv  = diag(1,-1,-1)
         R_model2cv = T_cv @ T_ply
      2. Use init camera's c2w rotation as R_align (aligns virtual cam to real cam).
         Total rotation: R = c2w[:3,:3] @ R_model2cv
      3. Scene centre = median of all camera positions.
      4. Search scale via silhouette IoU on init camera.
      5. Fine-tune yaw/pitch/roll using multi-view IoU.
    """
    from multiview_refine.cameras import load_cameras, find_camera_by_filename
    from multiview_refine.gaussian_model import _rotation_matrix_to_quaternion, _quaternion_compose
    from multiview_refine.gsplat_wrapper import render_alpha
    from multiview_refine.alignment import _euler_to_matrix, _compute_iou_batch

    print("=" * 60)
    print("Step 2: Align Gaussians to scene coordinate system")
    print("  Method: c2w rotation + scale search + multi-view fine-tune")
    print("=" * 60)

    means, scales_raw, quats, opacities_raw, sh0, shN = load_gaussians_ply(
        args.input_ply, device=args.device)
    print(f"  Loaded {means.shape[0]} Gaussians from {args.input_ply}")
    print(f"  Means range: [{means.min():.3f}, {means.max():.3f}]")

    cameras = load_cameras(args.transforms, device=args.device,
                           image_scale=0.25, load_image=True)
    init_cam = find_camera_by_filename(cameras, args.init_view)
    print(f"  Init camera: cam_idx={init_cam.cam_idx}, position={init_cam.position.tolist()}")

    # Step 1: model → OpenCV rotation
    T_ply = torch.tensor([[1,0,0],[0,0,-1],[0,1,0]], dtype=torch.float32, device=args.device)
    T_cv  = torch.tensor([[1,0,0],[0,-1,0],[0,0,-1]], dtype=torch.float32, device=args.device)
    R_model2cv = T_cv @ T_ply
    print(f"\n  R_model2cv (model→OpenCV via PLY):")
    print(f"    {R_model2cv.cpu().numpy().round(4).tolist()}")

    # Step 2: align virtual camera direction to real camera
    R_align = init_cam.c2w[:3, :3]
    R = R_align @ R_model2cv
    print(f"  R = c2w[:3,:3] @ R_model2cv:")
    print(f"    {R.cpu().numpy().round(4).tolist()}")

    # Step 3: scene centre = median of camera positions
    positions = torch.stack([c.position for c in cameras])
    scene_centre = positions.median(0).values
    print(f"  Scene centre (camera median): {scene_centre.tolist()}")

    # Step 4: search scale via init-camera silhouette IoU
    scales_activated = scales_raw.exp()
    opacities = torch.sigmoid(opacities_raw)
    gt_mask = (init_cam.image.mean(0) > 0.05).float()

    print(f"\n  Searching scale...")
    best_iou = -1.0
    best_scale = 1.0
    for scale in np.linspace(0.1, 5.0, 50):
        means_aligned = scale * (means @ R.T) + scene_centre
        scales_aligned = scales_activated * scale
        q_R = _rotation_matrix_to_quaternion(R)
        quats_aligned = _quaternion_compose(q_R, quats)
        with torch.no_grad():
            alpha_r = render_alpha(means_aligned, scales_aligned, quats_aligned, opacities, init_cam)
        r_mask = (alpha_r.squeeze(0) > 0.05).float()
        inter = (r_mask * gt_mask).sum()
        union = ((r_mask + gt_mask) > 0).float().sum()
        iou = (inter / max(union, 1)).item()
        if iou > best_iou:
            best_iou = iou
            best_scale = scale
    print(f"    Best scale={best_scale:.4f}, init-cam IoU={best_iou:.4f}")

    # Step 5: fine-tune rotation using multi-view IoU
    print(f"\n  Fine-tuning rotation (multi-view IoU)...")
    subset_cams = [cameras[i] for i in range(0, len(cameras), 4)]
    subset_masks = [(cameras[i].image.mean(0) > 0.05).float() for i in range(0, len(cameras), 4)]

    best_iou_fine = best_iou
    best_R_fine = R.clone()
    best_scale_fine = best_scale

    for scale in np.linspace(max(best_scale * 0.5, 0.1), best_scale * 2, 10):
        for yaw in np.linspace(-0.5, 0.5, 7):
            for pitch in np.linspace(-0.5, 0.5, 7):
                for roll in np.linspace(-0.2, 0.2, 5):
                    R_fine = _euler_to_matrix(pitch, yaw, roll).to(args.device) @ R
                    means_aligned = scale * (means @ R_fine.T) + scene_centre
                    scales_aligned = scales_activated * scale
                    q_fine = _rotation_matrix_to_quaternion(R_fine)
                    quats_aligned = _quaternion_compose(q_fine, quats)
                    sils = []
                    for cam in subset_cams:
                        with torch.no_grad():
                            alpha_r = render_alpha(means_aligned, scales_aligned, quats_aligned, opacities, cam)
                        sils.append((alpha_r.squeeze(0) > 0.05).float())
                    iou = _compute_iou_batch(sils, subset_masks)
                    if iou > best_iou_fine:
                        best_iou_fine = iou
                        best_R_fine = R_fine.clone()
                        best_scale_fine = scale
                        print(f"    scale={scale:.2f} yaw={np.degrees(yaw):.0f}° pitch={np.degrees(pitch):.0f}° roll={np.degrees(roll):.0f}° IoU={iou:.4f} ★")

    # Apply final transform
    R = best_R_fine
    scale = best_scale_fine
    means_aligned = scale * (means @ R.T) + scene_centre
    scales_aligned_raw = scales_raw + math.log(scale)
    q_R = _rotation_matrix_to_quaternion(R)
    quats_aligned = _quaternion_compose(q_R, quats)

    print(f"\n  Final: scale={scale:.4f}, multi-view IoU={best_iou_fine:.4f}")
    print(f"  Aligned means range: [{means_aligned.min():.3f}, {means_aligned.max():.3f}]")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    ply_path = os.path.join(args.output_dir, "aligned_gaussians.ply")
    save_gaussians_ply(ply_path, means_aligned, scales_aligned_raw, quats_aligned,
                       opacities_raw, sh0, shN)

    import json
    params = {
        'scale': float(scale), 'rotation': R.cpu().numpy().tolist(),
        'translation': (scene_centre - scale * (R @ means.mean(0))).cpu().numpy().tolist(),
        'scene_centre': scene_centre.cpu().numpy().tolist(),
        'R_model2cv': R_model2cv.cpu().numpy().tolist(),
        'R_align': R_align.cpu().numpy().tolist(),
        'best_iou': float(best_iou_fine),
        'method': 'c2w_rotation + scale_search + multiview_finetune',
    }
    with open(os.path.join(args.output_dir, "alignment_params.json"), 'w') as f:
        json.dump(params, f, indent=2)
    print(f"\n  Saved {ply_path}")
    print(f"Step 2 complete. IoU = {best_iou_fine:.4f}")


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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(video_path, fourcc, 5, (W, H))
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
        p.add_argument('--output_dir', default='pipeline_output')
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
