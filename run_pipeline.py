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

    Uses the init-view camera pose from transforms.json to determine the
    alignment. The key insight: TripoSplat generates Gaussians as if viewed
    from the init camera. We need to find the transform that places the
    Gaussians into the real scene coordinate system such that rendering from
    the init camera pose reproduces the init image.
    """
    from multiview_refine.cameras import load_cameras, find_camera_by_filename, invert_se3
    from multiview_refine.gaussian_model import _rotation_matrix_to_quaternion, _quaternion_compose

    print("=" * 60)
    print("Step 2: Align Gaussians to scene coordinate system")
    print("=" * 60)

    # Load initial Gaussians.
    means, scales_raw, quats, opacities_raw, sh0, shN = load_gaussians_ply(
        args.input_ply, device=args.device)
    print(f"  Loaded {means.shape[0]} Gaussians from {args.input_ply}")
    print(f"  Means range: [{means.min():.3f}, {means.max():.3f}]")
    print(f"  Centroid: {means.mean(dim=0).tolist()}")

    # Load cameras.
    cameras = load_cameras(args.transforms, device=args.device,
                           image_scale=1.0, load_image=False)
    init_cam = find_camera_by_filename(cameras, args.init_view)
    print(f"\n  Init camera: cam_idx={init_cam.cam_idx}, file={init_cam.file_path}")
    print(f"  Init camera position: {init_cam.position.tolist()}")
    print(f"  Init camera c2w (OpenCV):\n{init_cam.c2w.cpu().numpy().round(4)}")

    # ── Alignment strategy ──
    # TripoSplat generates Gaussians in a normalized frame where:
    #   - The camera looks down +Z (OpenCV-like) at the object
    #   - The object is centered at origin, size ~1 (AABB [-0.5, 0.5])
    #   - The camera is at roughly [0, 0, D] looking at origin
    #
    # The real init camera is at `init_cam.position` looking at the scene.
    # We need to:
    #   1. Rotate the Gaussians so their "front" matches the init camera's view
    #   2. Scale them to match the real scene size
    #   3. Translate to match the scene centre
    #
    # Since TripoSplat uses the convention that the camera is at [0, 0, +Z]
    # looking at the origin (OpenCV +Z forward), the Gaussians' "front" is
    # towards -Z. The init camera's "forward" direction is its c2w[:3, 2]
    # (OpenCV +Z = forward).

    # The init camera's forward direction (where it looks) in world space.
    cam_forward = init_cam.c2w[:3, 2]  # OpenCV: +Z = forward
    cam_position = init_cam.c2w[:3, 3]

    # The object centre in the real scene: we estimate it as a point along
    # the camera's forward direction, at a reasonable distance.
    # The distance can be estimated from the camera radius (distance to origin).
    cam_distance = cam_position.norm().item()
    scene_centre = cam_position + cam_forward * cam_distance * 0.7  # 70% towards scene
    print(f"\n  Camera forward: {cam_forward.tolist()}")
    print(f"  Camera distance from origin: {cam_distance:.3f}")
    print(f"  Estimated scene centre: {scene_centre.tolist()}")

    # TripoSplat's object is at origin, camera at [0, 0, +D].
    # In the real scene, the object should be at scene_centre, and the init
    # camera looks at it from cam_position.
    #
    # The rotation R maps TripoSplat's object frame to the scene frame.
    # TripoSplat camera direction: [0, 0, 1] (looking +Z)
    # Real camera direction: cam_forward
    # R should map [0, 0, 1] → cam_forward (approximately)
    #
    # But actually, the Gaussians' "viewing direction" in TripoSplat's frame
    # is from the camera at [0, 0, D] towards the origin, which is [0, 0, -1].
    # The real camera looks from cam_position towards scene_centre, direction
    # = (scene_centre - cam_position) = cam_forward * dist * 0.7.
    # So we need R such that R @ [0, 0, -1] ∝ -cam_forward, i.e., R @ [0, 0, 1] ∝ cam_forward.

    # Build R: rotation that maps [0, 0, 1] → cam_forward.
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=means.device)
    forward = cam_forward / cam_forward.norm()
    # Find rotation from z_axis to forward using Rodrigues' formula.
    cross = torch.cross(z_axis, forward, dim=0)
    sin_angle = cross.norm()
    cos_angle = torch.dot(z_axis, forward)
    if sin_angle < 1e-6:
        # Already aligned or anti-aligned.
        if cos_angle > 0:
            R = torch.eye(3, device=means.device)
        else:
            # 180° rotation around any perpendicular axis.
            R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                             dtype=torch.float32, device=means.device)
    else:
        # Rodrigues formula.
        axis = cross / sin_angle
        K = torch.zeros(3, 3, device=means.device)
        K[0, 1] = -axis[2]; K[0, 2] = axis[1]
        K[1, 0] = axis[2]; K[1, 2] = -axis[0]
        K[2, 0] = -axis[1]; K[2, 1] = axis[0]
        angle = torch.atan2(sin_angle, cos_angle)
        R = torch.eye(3, device=means.device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

    print(f"\n  Rotation matrix R:\n{R.cpu().numpy().round(4)}")

    # Scale: TripoSplat object size ~1, real scene size estimated from camera distance.
    # The object should occupy a reasonable portion of the camera's view.
    # Scale = scene_radius / tripo_object_radius
    tripo_radius = float(means.std(dim=0).norm().item() * 2)  # ~2σ
    scene_radius = cam_distance * 0.3  # rough: object is 30% of camera distance
    scale = scene_radius / max(tripo_radius, 1e-6)
    print(f"  Tripo object radius: {tripo_radius:.4f}")
    print(f"  Estimated scene radius: {scene_radius:.4f}")
    print(f"  Scale: {scale:.4f}")

    # Translation: place the rotated+scaled centroid at the scene centre.
    centroid_tripo = means.mean(dim=0)
    t = scene_centre - scale * (R @ centroid_tripo)
    print(f"  Translation: {t.tolist()}")

    # Apply the transform.
    means_aligned = scale * (means @ R.T) + t
    scales_aligned_raw = scales_raw + math.log(scale)  # add log(scale) to log-scales
    q_R = _rotation_matrix_to_quaternion(R)
    quats_aligned = _quaternion_compose(q_R, quats)
    print(f"\n  Aligned means range: [{means_aligned.min():.3f}, {means_aligned.max():.3f}]")
    print(f"  Aligned centroid: {means_aligned.mean(dim=0).tolist()}")

    # Verify: transform the init camera position to camera space and check
    # that the aligned centroid is in front of the camera (z > 0).
    centroid_cam = init_cam.w2c[:3, :3] @ means_aligned.mean(dim=0) + init_cam.w2c[:3, 3]
    print(f"\n  Verification:")
    print(f"    Aligned centroid in init camera space: {centroid_cam.tolist()}")
    print(f"    z = {centroid_cam[2].item():.4f} (must be > 0 for OpenCV/gsplat)")

    # Save aligned Gaussians.
    os.makedirs(args.output_dir, exist_ok=True)
    ply_path = os.path.join(args.output_dir, "aligned_gaussians.ply")
    save_gaussians_ply(ply_path, means_aligned, scales_aligned_raw, quats_aligned,
                       opacities_raw, sh0, shN)

    # Save alignment parameters for reference.
    import json
    params = {
        'scale': float(scale),
        'rotation': R.cpu().numpy().tolist(),
        'translation': t.cpu().numpy().tolist(),
        'scene_centre': scene_centre.cpu().numpy().tolist(),
        'init_camera_position': cam_position.cpu().numpy().tolist(),
    }
    params_path = os.path.join(args.output_dir, "alignment_params.json")
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"\n  Alignment params saved to {params_path}")

    print(f"\nStep 2 complete. Aligned Gaussians saved to {ply_path}")


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
