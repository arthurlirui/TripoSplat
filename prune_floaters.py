"""Prune floater Gaussians based on multi-view spatial consistency.

A floater is a Gaussian that:
  1. Is rendered as visible (high alpha) in many views, BUT
  2. In those views, the GT image at the projected location is background
     (no foreground content), OR
  3. Has high opacity but is spatially isolated (few neighbours within
     a scale-proportional radius).

The filter renders all Gaussians from every camera, accumulates per-Gaussian
"visibility vs GT mismatch" scores, and prunes Gaussians with high mismatch.

Usage:
    python prune_floaters.py \\
        --input_ply scratch_output/refined.ply \\
        --transforms static/000001/mvs/transforms_test.json \\
        --output_ply scratch_output/refined_pruned.ply \\
        --image_scale 0.5
"""

import argparse
import os
import sys
import math

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiview_refine.cameras import load_cameras
from multiview_refine.gsplat_wrapper import render_gaussians
from multiview_refine.ply_io import load_gaussians_ply, save_gaussians_ply
from multiview_refine.quaternion import quaternion_compose, rotation_matrix_to_quaternion


def prune_floaters(
    means, scales_raw, quats, opacities_raw, sh0, shN,
    cameras, device='cuda',
    alpha_threshold=0.1,      # Gaussian is "visible" if rendered alpha > this
    gt_threshold=0.05,         # GT pixel is "background" if intensity < this
    mismatch_ratio=0.5,        # Prune if visible in >50% views but GT is background
    min_visibility=3,          # Must be visible in at least N views to be considered
    knn_k=10,                  # KNN isolation: number of neighbours to check
    knn_dist_factor=3.0,       # Isolated if KNN distance > factor * median scale
):
    """Prune floater Gaussians using multi-view consistency + KNN isolation.

    Returns a boolean keep_mask (True = keep, False = prune).
    """
    import gsplat

    N = len(means)
    scales = scales_raw.exp()
    opacities = torch.sigmoid(opacities_raw)

    # ── 1. Multi-view mismatch: render from each camera, check if visible
    #    Gaussians project to background regions in GT ──
    print("  Multi-view mismatch scoring...")
    visible_count = torch.zeros(N, device=device)
    bg_mismatch_count = torch.zeros(N, device=device)

    for i, cam in enumerate(cameras):
        with torch.no_grad():
            rgb, alpha, meta = render_gaussians(
                means, scales, quats, sh0, opacities, cam, sh_degree=0)

        # GT foreground mask
        gt_mask = (cam.image.mean(0) > gt_threshold)  # (H, W)

        # Per-Gaussian visibility from meta
        radii = meta['radii']
        if radii.dim() > 1:
            r_max = radii.max(dim=-1).values
        else:
            r_max = radii

        # In packed mode, meta has gaussian_ids mapping
        gaussian_ids = meta.get('gaussian_ids')
        if gaussian_ids is not None:
            visible_mask = r_max > 0  # (nnz,)
            vis_ids = gaussian_ids[visible_mask]

            # For visible Gaussians, check where they project in GT
            # Use means2d to find their pixel positions
            means2d = meta['means2d']  # (nnz, 2) or (N, 2)
            if means2d.shape[0] == len(gaussian_ids):
                # Packed mode: means2d is (nnz, 2)
                vis_means2d = means2d[visible_mask]  # (M, 2)
            else:
                vis_means2d = means2d[r_max > 0]

            # Sample GT at projected positions
            H, W = cam.height, cam.width
            px = vis_means2d[:, 0].long().clamp(0, W - 1)
            py = vis_means2d[:, 1].long().clamp(0, H - 1)
            gt_vals = gt_mask[py, px]  # (M,) bool: True = foreground

            # Accumulate
            visible_count[vis_ids] += 1
            bg_mismatch_count[vis_ids] += (~gt_vals).float()

    # Mismatch ratio: of the views where this Gaussian is visible,
    # what fraction have background at the projected location?
    visibility = visible_count.clamp_min(1)
    mismatch_ratio_per_gs = bg_mismatch_count / visibility

    # Floaters: visible in many views but mostly project to background
    multi_view_floaters = (
        (visible_count >= min_visibility) &
        (mismatch_ratio_per_gs > mismatch_ratio) &
        (opacities > 0.1)  # only consider visible Gaussians
    )
    print(f"    Multi-view floaters: {multi_view_floaters.sum().item()} "
          f"(visible≥{min_visibility}, mismatch>{mismatch_ratio})")

    # ── 2. KNN isolation: Gaussians far from their K nearest neighbours ──
    print("  KNN isolation scoring...")
    from scipy.spatial import cKDTree

    means_np = means.cpu().numpy()
    scale_max_np = scales.max(dim=-1).values.cpu().numpy()
    tree = cKDTree(means_np)

    # For each Gaussian, find distance to K-th nearest neighbour
    dists, _ = tree.query(means_np, k=knn_k + 1)  # +1 because self is included
    knn_dist = dists[:, -1]  # distance to K-th neighbour

    # Isolated if KNN distance is much larger than the Gaussian's own scale
    median_scale = np.median(scale_max_np)
    isolated = knn_dist > (knn_dist_factor * scale_max_np + knn_dist_factor * median_scale)
    isolated_floaters = torch.from_numpy(isolated).to(device) & (opacities > 0.1)
    print(f"    KNN-isolated floaters: {isolated_floaters.sum().item()} "
          f"(K={knn_k}, dist>{knn_dist_factor}x scale)")

    # ── 3. Combined floater mask ──
    prune_mask = multi_view_floaters | isolated_floaters
    keep_mask = ~prune_mask

    print(f"\n  Total floaters to prune: {prune_mask.sum().item()} "
          f"({100 * prune_mask.float().mean():.1f}%)")
    print(f"  Remaining: {keep_mask.sum().item()}")

    # Print stats of pruned vs kept
    if prune_mask.sum() > 0:
        pruned_opac = opacities[prune_mask]
        kept_opac = opacities[keep_mask]
        print(f"  Pruned opacity: mean={pruned_opac.mean():.3f}, max={pruned_opac.max():.3f}")
        print(f"  Kept opacity:   mean={kept_opac.mean():.3f}")

    return keep_mask


def main():
    parser = argparse.ArgumentParser(description="Prune floater Gaussians.")
    parser.add_argument('--input_ply', required=True)
    parser.add_argument('--transforms', required=True)
    parser.add_argument('--output_ply', required=True)
    parser.add_argument('--image_scale', type=float, default=0.5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--alpha_threshold', type=float, default=0.1)
    parser.add_argument('--gt_threshold', type=float, default=0.05)
    parser.add_argument('--mismatch_ratio', type=float, default=0.5)
    parser.add_argument('--min_visibility', type=int, default=3)
    parser.add_argument('--knn_k', type=int, default=10)
    parser.add_argument('--knn_dist_factor', type=float, default=3.0)
    args = parser.parse_args()

    print("=" * 60)
    print("Pruning floater Gaussians (multi-view consistency + KNN isolation)")
    print("=" * 60)

    device = torch.device(args.device)
    means, scales_raw, quats, opacities_raw, sh0, shN = load_gaussians_ply(
        args.input_ply, device)
    print(f"  Loaded {len(means)} Gaussians from {args.input_ply}")

    cameras = load_cameras(args.transforms, device=device,
                           image_scale=args.image_scale, load_image=True)
    print(f"  Loaded {len(cameras)} cameras")

    keep_mask = prune_floaters(
        means, scales_raw, quats, opacities_raw, sh0, shN,
        cameras, device=device,
        alpha_threshold=args.alpha_threshold,
        gt_threshold=args.gt_threshold,
        mismatch_ratio=args.mismatch_ratio,
        min_visibility=args.min_visibility,
        knn_k=args.knn_k,
        knn_dist_factor=args.knn_dist_factor,
    )

    # Save pruned Gaussians
    means_p = means[keep_mask]
    scales_p = scales_raw[keep_mask]
    quats_p = quats[keep_mask]
    opacities_p = opacities_raw[keep_mask]
    sh0_p = sh0[keep_mask]
    shN_p = shN[keep_mask] if shN.numel() > 0 else shN

    save_gaussians_ply(args.output_ply, means_p, scales_p, quats_p,
                       opacities_p, sh0_p, shN_p)
    print(f"\n  Saved {len(means_p)} pruned Gaussians to {args.output_ply}")

    # Quick PSNR comparison
    from multiview_refine.losses import compute_psnr
    bg = torch.zeros(3, device=device)

    print("\n  PSNR comparison (before vs after pruning):")
    scales_full = scales_raw.exp()
    opac_full = torch.sigmoid(opacities_raw)
    scales_pruned = scales_p.exp()
    opac_pruned = torch.sigmoid(opacities_p)

    for i in range(0, len(cameras), 8):
        cam = cameras[i]
        with torch.no_grad():
            rgb_before, _, _ = render_gaussians(
                means, scales_full, quats, sh0, opac_full, cam, sh_degree=0)
            rgb_after, _, _ = render_gaussians(
                means_p, scales_pruned, quats_p, sh0_p, opac_pruned, cam, sh_degree=0)
        p_before = compute_psnr(rgb_before, cam.image).item()
        p_after = compute_psnr(rgb_after, cam.image).item()
        print(f"    cam {i:2d}: before={p_before:.2f} → after={p_after:.2f} "
              f"({'↑' if p_after > p_before else '↓'}{abs(p_after - p_before):.2f})")


if __name__ == "__main__":
    main()
