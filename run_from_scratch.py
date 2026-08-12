"""Train 3D Gaussians from scratch (no TripoSplat initialization).

Initializes Gaussians from a sparse random point cloud within the camera
bounding box, then optimizes via gsplat differentiable rasterization with
adaptive density control (DefaultStrategy: clone/split/prune + opacity reset).

This is the classic 3DGS approach — no single-image prior needed, just
multi-view images + camera poses.

Usage:
    export PYTHONUTF8=1
    python run_from_scratch.py \\
        --transforms static/000001/mvs/transforms_test.json \\
        --init_view 00.png \\
        --num_iterations 10000 \\
        --image_scale 0.5 \\
        --output_dir scratch_output
"""

import argparse
import os
import sys
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiview_refine.cameras import load_cameras, compute_cameras_extent, split_train_test, find_camera_by_filename
from multiview_refine.gaussian_model import RefineableGaussians
from multiview_refine.gsplat_wrapper import render_gaussians
from multiview_refine.losses import compute_loss, compute_psnr
from gsplat.strategy import DefaultStrategy

SH_C0 = 0.28209479177387814


def init_random_point_cloud(cameras, n_points=100000, device='cuda'):
    """Initialize a random point cloud within the camera bounding box.

    Points are uniformly sampled in a sphere centred at the scene centre
    (median of camera positions), with radius = camera extent.
    """
    positions = torch.stack([c.position for c in cameras])
    centre = positions.median(0).values
    extent = compute_cameras_extent(cameras)
    radius = extent * 0.5

    # Uniform points in a sphere.
    r = torch.rand(n_points, device=device) ** (1/3) * radius
    theta = torch.rand(n_points, device=device) * 2 * math.pi
    phi = torch.acos(2 * torch.rand(n_points, device=device) - 1)
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    points = torch.stack([x, y, z], dim=-1) + centre.to(device)

    # Colors: random.
    colors = torch.rand(n_points, 3, device=device) * 0.5 + 0.3  # [0.3, 0.8]
    return points, colors


def main():
    parser = argparse.ArgumentParser(description="Train 3D Gaussians from scratch.")
    parser.add_argument('--transforms', required=True, help='Path to transforms.json')
    parser.add_argument('--init_view', default=None, help='Init view for train/test split')
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--image_scale', type=float, default=0.5)
    parser.add_argument('--n_init_points', type=int, default=100000)
    parser.add_argument('--output_dir', default='scratch_output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sh_degree', type=int, default=0)
    parser.add_argument('--save_interval', type=int, default=2000)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ── Load cameras ──
    print("=" * 60)
    print("From-scratch 3D Gaussian Splatting")
    print("=" * 60)
    print("\nLoading cameras...")
    cameras = load_cameras(args.transforms, device=device,
                           image_scale=args.image_scale, load_image=True)
    print(f"  {len(cameras)} cameras at {cameras[0].width}x{cameras[0].height}")

    extent = compute_cameras_extent(cameras)
    print(f"  Scene extent: {extent:.4f}")

    # ── Initialize random point cloud ──
    print(f"\nInitializing random point cloud ({args.n_init_points} points)...")
    points, colors = init_random_point_cloud(cameras, args.n_init_points, device)
    print(f"  Points range: [{points.min():.3f}, {points.max():.3f}]")

    # ── Create Gaussian model ──
    model = RefineableGaussians(sh_degree=args.sh_degree, device=device)

    # Initialize from point cloud (like 3DGS).
    means = points
    # Scales: log(distance to nearest neighbour) — use a fixed small value for random init.
    init_scale = extent * 0.001  # small relative to scene
    scales_raw = torch.log(torch.full((len(means), 3), init_scale, device=device))
    # Quaternions: identity.
    quats = torch.zeros(len(means), 4, device=device)
    quats[:, 0] = 1.0
    # Opacities: logit(0.1) — low opacity, let optimization increase it.
    opacities_raw = torch.full((len(means),), math.log(0.1 / 0.9), device=device)
    # SH DC: from colors (f_dc = (rgb - 0.5) / C0).
    sh0 = ((colors - 0.5) / SH_C0)[:, None, :]  # (N, 1, 3)
    shN = torch.zeros(len(means), 0, 3, device=device)

    model._means = torch.nn.Parameter(means.contiguous())
    model._scales = torch.nn.Parameter(scales_raw.contiguous())
    model._quats = torch.nn.Parameter(quats.contiguous())
    model._opacities = torch.nn.Parameter(opacities_raw.contiguous())
    model._sh0 = torch.nn.Parameter(sh0.contiguous())
    model._shN = torch.nn.Parameter(shN, requires_grad=False)
    print(f"  {model.n_gaussians} Gaussians initialized")

    # ── Training setup ──
    strategy = DefaultStrategy(
        prune_opa=0.005,
        grow_grad2d=0.0002,
        grow_scale3d=0.01,
        grow_scale2d=0.05,
        prune_scale3d=0.1,
        prune_scale2d=0.15,
        refine_start_iter=500,
        refine_stop_iter=15000,
        refine_every=100,
        reset_every=3000,
        verbose=False,
    )
    model.training_setup(scene_scale=extent, strategy=strategy)

    # ── Train/test split ──
    if args.init_view:
        train_cams, test_cams = split_train_test(cameras, args.init_view)
    else:
        train_cams = cameras
        test_cams = [cameras[0]]
    print(f"  Train: {len(train_cams)}, Test: {len(test_cams)}")

    bg = torch.zeros(3, device=device)

    # ── Optimization loop ──
    print(f"\nOptimizing for {args.num_iterations} iterations...")
    t0 = time.time()

    for it in range(1, args.num_iterations + 1):
        model.train()
        model.update_learning_rate(it)
        cam = random.choice(train_cams)
        rgb, alpha, meta = render_gaussians(
            model.means, model.scales, model.raw_quats,
            model.get_colors(), model.opacities, cam,
            background=bg, sh_degree=args.sh_degree,
        )
        loss = compute_loss(rgb, cam.image)
        info = dict(meta)
        info['step'] = it
        model.step_pre_backward(info)
        loss.backward()
        model.step_post_backward(info, step=it, packed=True)
        model.optimizer_step()
        model.optimizer_zero_grad()

        if it % 1000 == 0 or it == 1:
            with torch.no_grad():
                # Eval on a bright test camera.
                eval_cam = max(test_cams, key=lambda c: c.image.mean().item())
                er, _, _ = render_gaussians(
                    model.means, model.scales, model.raw_quats,
                    model.get_colors(), model.opacities, eval_cam,
                    background=bg, sh_degree=args.sh_degree,
                )
                psnr = compute_psnr(er, eval_cam.image).item()
                # Also eval on a bright train camera.
                bright_cam = max(train_cams, key=lambda c: c.image.mean().item())
                br, _, _ = render_gaussians(
                    model.means, model.scales, model.raw_quats,
                    model.get_colors(), model.opacities, bright_cam,
                    background=bg, sh_degree=args.sh_degree,
                )
                psnr_bright = compute_psnr(br, bright_cam.image).item()
            elapsed = time.time() - t0
            print(f"  iter {it:5d}: loss={loss.item():.4f}, "
                  f"psnr_test={psnr:.2f}, psnr_train={psnr_bright:.2f}, "
                  f"n_gs={model.n_gaussians}, {elapsed:.1f}s")

        if args.save_interval > 0 and it % args.save_interval == 0:
            model.save_ply(os.path.join(args.output_dir, f"scratch_iter{it}.ply"))

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Final: {model.n_gaussians} Gaussians")

    # ── Export ──
    model.save_ply(os.path.join(args.output_dir, "refined.ply"))
    model.save_splat(os.path.join(args.output_dir, "refined.splat"))
    print(f"  Saved refined.ply + refined.splat")

    # ── Per-camera PSNR ──
    print(f"\nPer-camera PSNR:")
    psnrs = []
    for i, cam in enumerate(cameras):
        with torch.no_grad():
            rgb, _, _ = render_gaussians(
                model.means, model.scales, model.raw_quats,
                model.get_colors(), model.opacities, cam,
                background=bg, sh_degree=args.sh_degree,
            )
        p = compute_psnr(rgb, cam.image).item()
        psnrs.append(p)
        if i % 8 == 0:
            print(f"  cam {i:2d} (idx={cam.cam_idx:2d}): PSNR={p:.2f}")
    print(f"  Mean PSNR: {np.mean(psnrs):.2f}, Max: {np.max(psnrs):.2f}, Min: {np.min(psnrs):.2f}")


if __name__ == "__main__":
    main()
