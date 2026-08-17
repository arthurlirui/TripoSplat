"""Render a turntable (orbit) video from refined Gaussians.

Two modes:
  1. Orbit mode (default): fits a circle to real camera positions and renders
     a smooth 360° orbit around the Gaussian centroid.
  2. Real-camera mode (--use_real_cameras): renders the Gaussians from the
     actual multi-view camera poses in order, with optional GT image side-by-side.

Usage (orbit):
    python render_turntable.py \\
        --ply refine_output/refined.ply \\
        --transforms static/000001/mvs/transforms_train.json \\
        --output refine_output/turntable.mp4 \\
        --n_frames 120 --fps 30 --resolution 1024

Usage (real cameras with GT comparison):
    python render_turntable.py \\
        --ply refine_output/refined.ply \\
        --transforms static/000001/mvs/transforms_train.json \\
        --output refine_output/comparison.mp4 \\
        --use_real_cameras --compare_gt --image_scale 0.5
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))

from multiview_refine.cameras import Camera, load_cameras, invert_se3
from multiview_refine.gsplat_wrapper import render_gaussians
from multiview_refine.ply_io import load_gaussians_ply
from multiview_refine.video_io import create_video_writer

SH_C0 = 0.28209479177387814


# ── Orbit trajectory ───────────────────────────────────────────────────────

def fit_orbit_cameras(cameras, n_frames, look_at=None, elevation=None):
    """Generate a smooth orbit camera trajectory from real camera positions."""
    positions = torch.stack([c.position for c in cameras]).cpu().numpy()
    if look_at is None:
        centre = np.median(positions, axis=0)
    else:
        centre = np.asarray(look_at, dtype=np.float32)
    radii = np.linalg.norm(positions - centre, axis=1)
    radius = float(np.mean(radii))
    if elevation is None:
        elevations = np.arcsin((positions[:, 1] - centre[1]) / np.clip(radii, 1e-8, None))
        elevation = float(np.mean(elevations))
    up = np.array([0.0, -1.0, 0.0])
    azims = np.linspace(0, 2 * math.pi, n_frames, endpoint=False)
    c2w_list = []
    for az in azims:
        pos = centre + np.array([
            radius * math.cos(elevation) * math.sin(az),
            -radius * math.sin(elevation),
            radius * math.cos(elevation) * math.cos(az),
        ], dtype=np.float32)
        forward = centre - pos
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up); right /= np.linalg.norm(right)
        down = np.cross(right, forward); down /= np.linalg.norm(down)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right; c2w[:3, 1] = down; c2w[:3, 2] = forward; c2w[:3, 3] = pos
        c2w_list.append(c2w)
    return c2w_list, radius


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Render a turntable video from refined Gaussians.")
    parser.add_argument("--ply", required=True)
    parser.add_argument("--transforms", default=None)
    parser.add_argument("--output", default="refine_output/turntable.mp4")
    parser.add_argument("--n_frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--elevation", type=float, default=None)
    parser.add_argument("--background", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sh_degree", type=int, default=0)
    parser.add_argument("--scale_boost", type=float, default=1.0,
                        help="Multiply all Gaussian scales by this factor.")
    parser.add_argument("--use_real_cameras", action="store_true",
                        help="Render from real camera poses instead of a fitted orbit.")
    parser.add_argument("--compare_gt", action="store_true",
                        help="Side-by-side render vs GT (only with --use_real_cameras).")
    parser.add_argument("--image_scale", type=float, default=0.5,
                        help="Image scale for real camera mode.")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load Gaussians ──
    print(f"Loading Gaussians from {args.ply}...")
    means, scales_raw, quats, opacities_raw, sh0, shN = load_gaussians_ply(args.ply, device)
    n = means.shape[0]
    print(f"  {n} Gaussians, means range [{means.min():.3f}, {means.max():.3f}]")
    print(f"  scale_boost = {args.scale_boost}")

    scales = scales_raw.exp() * args.scale_boost
    opacities = torch.sigmoid(opacities_raw)
    if shN.numel() > 0 and args.sh_degree > 0:
        colors = torch.cat([sh0, shN], dim=1)
        sh_degree = args.sh_degree
    else:
        colors = sh0
        sh_degree = 0

    bg = torch.tensor(args.background, dtype=torch.float32, device=device)

    # ── Build camera list ──
    if args.use_real_cameras and args.transforms:
        print("\nUsing real camera poses...")
        real_cameras = load_cameras(args.transforms, device=device,
                                    image_scale=args.image_scale, load_image=True)
        cameras_to_render = real_cameras
        W, H = real_cameras[0].width, real_cameras[0].height
        print(f"  {len(cameras_to_render)} cameras, {W}x{H}")
    elif args.transforms:
        print("\nBuilding orbit trajectory...")
        real_cameras = load_cameras(args.transforms, device=device,
                                    image_scale=1.0, load_image=False)
        gc = means.median(dim=0).values.cpu().numpy()
        print(f"  Gaussian centroid: {gc.tolist()}")
        c2w_list, radius = fit_orbit_cameras(
            real_cameras, args.n_frames,
            look_at=gc,
            elevation=math.radians(args.elevation) if args.elevation is not None else None,
        )
        aspect = real_cameras[0].width / real_cameras[0].height
        W = args.resolution
        H = int(W / aspect)
        fx = args.fx if args.fx is not None else 0.7 * W
        fy = fx
        cx, cy = W / 2.0, H / 2.0
        print(f"  Orbit radius={radius:.3f}, {W}x{H}, fx={fx:.1f}")
        cameras_to_render = []
        for c2w_np in c2w_list:
            c2w = torch.from_numpy(c2w_np).float().to(device)
            w2c = invert_se3(c2w)
            cameras_to_render.append(Camera(
                w2c=w2c, c2w=c2w, position=c2w[:3, 3],
                fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H,
            ))
    else:
        # Default orbit
        spread = float(means.std(dim=0).norm().item())
        radius = max(spread * 3.0, 2.0)
        centre = means.median(dim=0).values.cpu().numpy()
        elev = math.radians(args.elevation) if args.elevation is not None else math.radians(15)
        up = np.array([0.0, -1.0, 0.0])
        azims = np.linspace(0, 2 * math.pi, args.n_frames, endpoint=False)
        W, H = args.resolution, int(args.resolution * 3 / 4)
        fx = args.fx if args.fx is not None else 0.7 * W
        fy, cx, cy = fx, W / 2.0, H / 2.0
        cameras_to_render = []
        for az in azims:
            pos = centre + np.array([
                radius * math.cos(elev) * math.sin(az),
                -radius * math.sin(elev),
                radius * math.cos(elev) * math.cos(az),
            ], dtype=np.float32)
            forward = centre - pos; forward /= np.linalg.norm(forward)
            right = np.cross(forward, up); right /= np.linalg.norm(right)
            down = np.cross(right, forward); down /= np.linalg.norm(down)
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, 0] = right; c2w[:3, 1] = down; c2w[:3, 2] = forward; c2w[:3, 3] = pos
            c2w_t = torch.from_numpy(c2w).float().to(device)
            cameras_to_render.append(Camera(
                w2c=invert_se3(c2w_t), c2w=c2w_t, position=c2w_t[:3, 3],
                fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H,
            ))

    # ── Render frames ──
    n_frames = len(cameras_to_render)
    print(f"\nRendering {n_frames} frames...")

    # Determine output dimensions (double width if compare_gt)
    out_W = W * 2 if args.compare_gt and args.use_real_cameras else W
    out_H = H

    video_writer, args.output = create_video_writer(
        args.output, "mp4v", args.fps, (out_W, out_H))

    for i, cam in enumerate(cameras_to_render):
        with torch.no_grad():
            rgb, alpha, _ = render_gaussians(
                means, scales, quats, colors, opacities, cam,
                background=bg, sh_degree=sh_degree,
            )

        frame_np = (rgb.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        if args.compare_gt and args.use_real_cameras and cam.image is not None:
            gt_np = (cam.image.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)
            if gt_bgr.shape[:2] != frame_bgr.shape[:2]:
                gt_bgr = cv2.resize(gt_bgr, (W, H))
            frame_bgr = np.concatenate([frame_bgr, gt_bgr], axis=1)

        video_writer.write(frame_bgr)

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Frame {i+1}/{n_frames}  mean={frame_np.mean():.1f}")

    video_writer.release()
    print(f"\nSaved: {args.output} ({n_frames} frames, {out_W}x{out_H}, {args.fps} fps)")
    print("Done!")


if __name__ == "__main__":
    main()
