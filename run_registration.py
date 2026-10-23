"""Register a generated Gaussian avatar into a photographed scene (closed form).

This is the runnable entry point for :mod:`multiview_refine.registration`.  It
takes a TripoSplat-generated PLY (raw model frame, AABB) plus the capture's
calibrated cameras, recovers the 7-DoF similarity transform that places the
avatar in world coordinates, and writes:

  * ``registered.ply``          — Gaussians in scene/world coordinates
  * ``registration_params.json`` — the transform, its matrix, and diagnostics
  * ``registration_metrics.json``— IoU / orientation / height-consistency
  * ``compare_*.png``           — GT | registered render, per sample view
  * ``compare_all.mp4``         — the same comparison as a turntable video

Usage
-----
    conda activate rtg132
    python run_registration.py
    python run_registration.py --ply output/gaussian_output/000001/gaussians.ply \
        --transforms static/000001/mvs/transforms_train.json
    # With the optional differentiable polish (slower, uses shading):
    python run_registration.py --refine_iters 300
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Preload the cached gsplat CUDA extension before any gsplat import
# (torch 2.12+ breaks gsplat's JIT path; see pointsplat_bridge/_gsplat_preload).
from pointsplat_bridge._gsplat_preload import preload_gsplat_cuda

preload_gsplat_cuda()

from multiview_refine.cameras import load_cameras
from multiview_refine.gsplat_wrapper import render_gaussians
from multiview_refine.ply_io import load_gaussians_ply, save_gaussians_ply
from multiview_refine.registration import (
    register_gaussians_to_scene,
    verify_registration,
)
from multiview_refine.video_io import create_video_writer


def tensor_to_bgr(rgb: torch.Tensor, alpha: torch.Tensor | None = None) -> np.ndarray:
    """(3, H, W) float in [0,1] → BGR uint8, optionally with an alpha strip."""
    import cv2

    img = (rgb.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if alpha is not None:
        a = (alpha.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        img = np.concatenate([img, np.stack([a] * 3, axis=-1)], axis=1)
    return img


def hstack_labeled(panels: list[tuple[np.ndarray, str]]) -> np.ndarray:
    """Concatenate images horizontally with a label bar above each."""
    import cv2

    labelled = []
    for img, label in panels:
        bar = np.zeros((34, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)
        labelled.append(np.concatenate([bar, img], axis=0))
    h = max(p.shape[0] for p in labelled)
    out = []
    for p in labelled:
        if p.shape[0] < h:
            p = np.concatenate(
                [p, np.zeros((h - p.shape[0], p.shape[1], 3), dtype=np.uint8)], axis=0)
        out.append(p)
    return np.concatenate(out, axis=1)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Closed-form registration of generated Gaussians to a scene")
    ap.add_argument("--ply", default="output/gaussian_output/000001/gaussians.ply",
                    help="TripoSplat PLY in the raw model (AABB) frame")
    ap.add_argument("--transforms", default="static/000001/mvs/transforms_train.json")
    ap.add_argument("--output_dir", default="output/registration")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image_scale", type=float, default=0.5,
                    help="Downscale loaded images (and intrinsics) for speed")
    ap.add_argument("--alpha_threshold", type=float, default=0.5,
                    help="Foreground threshold for the real masks (the render "
                         "uses the same value; see registration.py)")
    ap.add_argument("--min_valid_views", type=int, default=6,
                    help="Uncropped views required before falling back to the "
                         "least-cropped subset")
    ap.add_argument("--refine_iters", type=int, default=0,
                    help="Differentiable pose polish iterations (0 = closed form only)")
    ap.add_argument("--n_yaw_coarse", type=int, default=36)
    ap.add_argument("--n_yaw_fine", type=int, default=24)
    ap.add_argument("--no_calibrate_scale", action="store_true",
                    help="Skip the rendering-consistent scale calibration")
    ap.add_argument("--save_ply", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 68)
    print("Closed-form registration: generated Gaussians → photographed scene")
    print("=" * 68)
    print(f"  PLY:        {args.ply}")
    print(f"  Cameras:    {args.transforms}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Device:     {device}")

    # ── Load cameras ──
    t0 = time.time()
    cameras = load_cameras(
        args.transforms,
        base_dir=os.path.dirname(os.path.abspath(args.transforms)),
        device=device,
        image_scale=args.image_scale,
        load_image=True,
    )
    print(f"  Loaded {len(cameras)} cameras in {time.time() - t0:.1f}s "
          f"({cameras[0].width}x{cameras[0].height}, "
          f"fx={cameras[0].fx:.0f})")

    # ── Load Gaussians (raw model frame) ──
    means, scales_raw, quats_raw, opac_raw, sh0, _shN = load_gaussians_ply(
        args.ply, device=device)
    scales = scales_raw.exp()
    opacities = torch.sigmoid(opac_raw)
    quats = torch.nn.functional.normalize(quats_raw, dim=-1)
    print(f"  Loaded {means.shape[0]} Gaussians (model frame), "
          f"extent {np.round((means.max(0).values - means.min(0).values).cpu().numpy(), 3).tolist()}")

    # ── Register ──
    print("\n" + "-" * 68)
    print("Estimating the 7-DoF similarity transform")
    print("-" * 68)
    t0 = time.time()
    result = register_gaussians_to_scene(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        cameras=cameras,
        colors=sh0,
        alpha_threshold=args.alpha_threshold,
        min_valid_views=args.min_valid_views,
        n_yaw_coarse=args.n_yaw_coarse,
        n_yaw_fine=args.n_yaw_fine,
        calibrate_scale=not args.no_calibrate_scale,
        refine_iters=args.refine_iters,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  scale = {result.scale:.5f}")
    print(f"  R     =\n{np.round(result.R.detach().cpu().numpy(), 5)}")
    print(f"  t     = {np.round(result.t.detach().cpu().numpy(), 5).tolist()}")
    print(f"  subject centre (world) = {np.round(result.subject_centre, 4).tolist()}")
    print(f"  metric height          = {result.metric_height:.3f} m")
    print(f"  yaw                    = {math.degrees(result.yaw):.1f}° "
          f"±{result.yaw_std_deg:.0f}°")

    # ── Metrics ──
    print("\n" + "-" * 68)
    print("Verifying against the real photographs")
    print("-" * 68)
    metrics = verify_registration(
        result, means, scales, quats, opacities, cameras,
        alpha_threshold=args.alpha_threshold,
    )
    print(f"  views={metrics['n_views']}  "
          f"mean IoU={metrics['mean_iou']:.4f}  min IoU={metrics['min_iou']:.4f}")
    print(f"  implied height: {metrics['implied_height_median']:.3f} m "
          f"(std {metrics['height_consistency_std']:.4f})")
    print(f"  orientation agreement={metrics['orientation_agreement']:.2f}  "
          f"flip detected={metrics['flip_detected']}")

    if metrics["flip_detected"]:
        print("  ❌ The registered model appears upside down or back-to-front "
              "relative to the photographs.")
    elif metrics["mean_iou"] < 0.5:
        print("  ⚠️  Low overlap — check the masks, or whether the PLY is from "
              "the same capture as these cameras.")

    # ── Save the world-coordinate PLY ──
    if args.save_ply:
        mt, st, qt = result.apply(means, scales, quats)
        ply_out = os.path.join(args.output_dir, "registered.ply")
        # PLY stores raw (pre-activation) scales and opacities.
        save_gaussians_ply(ply_out, mt, torch.log(st.clamp_min(1e-8)), qt,
                           opac_raw, sh0, None)
        print(f"\n  Saved world-coordinate PLY: {ply_out}")

    params = result.to_dict()
    params["source_ply"] = args.ply
    params["transforms"] = args.transforms
    params["elapsed_seconds"] = elapsed
    params["convention"] = ("xyz_scene = scale * (R @ xyz_model) + t; "
                            "R already folds in MODEL2SCENE")
    with open(os.path.join(args.output_dir, "registration_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    with open(os.path.join(args.output_dir, "registration_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved registration_params.json / registration_metrics.json")

    # ── Comparison renders ──
    print("\n" + "-" * 68)
    print("Rendering comparisons")
    print("-" * 68)
    import cv2

    mt, st, qt = result.apply(means, scales, quats)
    bg = torch.zeros(3, device=device)
    n = len(cameras)
    sample_idx = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    frames = []

    for i in range(n):
        cam = cameras[i]
        with torch.no_grad():
            rgb, alpha, _ = render_gaussians(
                mt, st, qt, sh0, opacities, cam, background=bg, sh_degree=0)
        render_img = tensor_to_bgr(rgb)
        gt_img = (tensor_to_bgr(cam.image) if cam.image is not None
                  else np.zeros_like(render_img))
        panel = hstack_labeled([
            (gt_img, f"GT {os.path.basename(cam.file_path)}"),
            (render_img, f"registered cam{i}"),
        ])
        frames.append(panel)
        if i in sample_idx:
            cv2.imwrite(os.path.join(args.output_dir, f"compare_cam{i:02d}.png"), panel)

    if frames:
        H, W = frames[0].shape[:2]
        path = os.path.join(args.output_dir, "compare_all.mp4")
        writer, path = create_video_writer(path, "mp4v", 4, (W, H))
        for f in frames:
            writer.write(f if f.shape[:2] == (H, W)
                         else cv2.resize(f, (W, H)))
        writer.release()
        print(f"  Saved {len(frames)} comparison frames and {path}")

    print("\n" + "=" * 68)
    print("Registration complete")
    print("=" * 68)
    return 0 if not metrics["flip_detected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
