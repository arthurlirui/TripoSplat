"""End-to-end test of differentiable alignment refinement.

Loads real multi-view cameras from static/000000/mvs/ and initial Gaussians
from demo_output/initial_gaussians.ply, runs the coarse alignment grid search
followed by the differentiable refinement, and saves intermediate results
(transforms, IoU/PSNR metrics, rendered comparison images, aligned PLY) to
output/alignment_refine_test/.

Usage:
    conda run -n rtg132 python run_alignment_refine_test.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np
import torch

# Preload gsplat CUDA extension before any gsplat import (torch 2.12+ JIT workaround).
from pointsplat_bridge._gsplat_preload import preload_gsplat_cuda
preload_gsplat_cuda()

# ── Configuration ──
TRANSFORMS_JSON = "static/000000/mvs/transforms_train.json"
INIT_PLY = "output/demo_output/initial_gaussians.ply"
OUTPUT_DIR = "output/alignment_refine_test"
IMAGE_SCALE = 0.25  # downscale for speed
REFINE_ITERS = 500
SH_DEGREE = 0
INIT_VIEW = "00.png"  # full-body, uncropped generation view


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Step 1: Load cameras ──
    print("\n" + "=" * 60)
    print("Step 1: Loading multi-view cameras")
    print("=" * 60)

    from multiview_refine.cameras import load_cameras, compute_cameras_extent

    cameras = load_cameras(
        TRANSFORMS_JSON,
        base_dir=os.path.dirname(os.path.abspath(TRANSFORMS_JSON)),
        device=device,
        image_scale=IMAGE_SCALE,
        load_image=True,
    )
    print(f"  Loaded {len(cameras)} cameras")
    print(f"  Image resolution: {cameras[0].width}x{cameras[0].height}")

    cameras_extent = compute_cameras_extent(cameras)
    print(f"  Cameras extent: {cameras_extent:.4f}")

    # ── Step 2: Load initial Gaussians from PLY ──
    print("\n" + "=" * 60)
    print("Step 2: Loading initial Gaussians")
    print("=" * 60)

    from multiview_refine.ply_io import load_gaussians_ply
    from multiview_refine.coord_registry import MODEL2SCENE
    from multiview_refine.quaternion import quaternion_compose, rotation_matrix_to_quaternion

    means_raw, scales_raw, quats_raw, opac_raw, sh0, shN = load_gaussians_ply(INIT_PLY, device=device)

    # Activate: scales = exp(raw), opacities = sigmoid(raw)
    scales = scales_raw.exp()
    quats = torch.nn.functional.normalize(quats_raw, dim=-1)
    opacities = torch.sigmoid(opac_raw)
    colors = sh0  # (N, 1, 3)

    # ── Pre-transform: model → Y-up scene ──
    # Apply MODEL2SCENE to bring the Gaussians from the model AABB frame to
    # the Y-up scene frame, consistent with run_pipeline.py and
    # run_pipeline_verify.py.  The alignment module then only needs to search
    # for corrections on top of this canonical orientation.
    R_m2s = MODEL2SCENE.to(device)
    means = means_raw @ R_m2s.T
    quats = quaternion_compose(rotation_matrix_to_quaternion(R_m2s), quats)

    print(f"  Loaded {means.shape[0]} Gaussians from {INIT_PLY}")
    print(f"  Applied MODEL2SCENE (model → Y-up scene)")
    print(f"  Means range: [{means.min():.3f}, {means.max():.3f}]")

    # ── Step 3: Coarse alignment ──
    # Prefer the analytic init-view alignment: it is closed-form, and avoids
    # the silhouette search's two failure modes on this data (it inflates the
    # scale and cannot distinguish front from back).
    print("\n" + "=" * 60)
    print("Step 3: Coarse alignment")
    print("=" * 60)

    from multiview_refine.alignment import (
        estimate_similarity_transform_multiview,
        align_from_init_view,
    )
    from multiview_refine.cameras import find_camera_by_filename

    t0 = time.time()
    init_cam = None
    for split in ("val", "test", "train"):
        p = re.sub(r"transforms_(train|val|test)\.json$", f"transforms_{split}.json",
                   TRANSFORMS_JSON)
        if not os.path.exists(p):
            continue
        try:
            _c = load_cameras(p, base_dir=os.path.dirname(os.path.abspath(p)),
                              device=device, image_scale=IMAGE_SCALE, load_image=True)
            init_cam = find_camera_by_filename(_c, INIT_VIEW)
            break
        except ValueError:
            continue

    if init_cam is not None:
        print(f"  Analytic init-view alignment using {INIT_VIEW} from {split} split")
        # means_raw is the raw AABB frame; MODEL2SCENE is folded into R.
        scale_coarse, R_coarse, t_coarse_val = align_from_init_view(
            means_raw, init_cam, cameras, init_mask=init_cam.alpha, verbose=True)
    else:
        print("  Init view uncalibrated — falling back to silhouette search")
        scale_coarse, R_coarse, t_coarse_val = estimate_similarity_transform_multiview(
            tensors=(means, scales, quats, opacities, colors),
            cameras=cameras,
            rmbg_fn=None,
            n_rotation_samples=24,
            n_scale_samples=10,
            scale_range=(0.05, 2.0),
            alpha_threshold=0.1,
            init_yaw=None,
            verbose=True,
        )
        # The fallback search returns R in the MODEL2SCENE-rotated frame;
        # fold MODEL2SCENE in so both paths agree.
        R_coarse = R_coarse @ MODEL2SCENE.to(device)
    time_coarse = time.time() - t0
    print(f"\n  Coarse alignment done in {time_coarse:.1f}s")
    print(f"  scale={scale_coarse:.4f}")
    print(f"  R=\n{R_coarse}")
    print(f"  t={t_coarse_val.tolist()}")

    # Both paths now return R that maps the RAW model frame to the scene, so
    # every downstream call uses means_raw (not the MODEL2SCENE-rotated copy).
    means = means_raw

    # Save coarse transform
    coarse_result = {
        "scale": scale_coarse,
        "R": R_coarse.cpu().tolist(),
        "t": t_coarse_val.cpu().tolist(),
        "time_seconds": time_coarse,
    }
    with open(os.path.join(OUTPUT_DIR, "coarse_transform.json"), "w") as f:
        json.dump(coarse_result, f, indent=2)
    print(f"  Saved coarse_transform.json")

    # ── Step 4: Differentiable refinement ──
    print("\n" + "=" * 60)
    print("Step 4: Differentiable alignment refinement")
    print("=" * 60)

    from multiview_refine.differentiable_refine import refine_alignment

    t0 = time.time()
    scale_refined, R_refined, t_refined = refine_alignment(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        colors=colors,
        cameras=cameras,
        coarse_transform=(scale_coarse, R_coarse, t_coarse_val),
        rmbg_fn=None,
        num_iterations=REFINE_ITERS,
        lr_scale=5e-3,
        lr_rotation=5e-4,
        lr_translation=5e-3,
        lambda_rgb=1.0,
        lambda_sil=0.3,
        lambda_reg=0.01,
        sh_degree=SH_DEGREE,
        background_color=(0.0, 0.0, 0.0),
        view_subset=8,
        verbose=True,
    )
    time_refine = time.time() - t0
    print(f"\n  Refinement done in {time_refine:.1f}s")
    print(f"  scale: {scale_coarse:.4f} → {scale_refined:.4f}")
    print(f"  t: {t_coarse_val.tolist()} → {t_refined.tolist()}")

    # Save refined transform
    refined_result = {
        "scale": scale_refined,
        "R": R_refined.cpu().tolist(),
        "t": t_refined.cpu().tolist(),
        "time_seconds": time_refine,
        "coarse_scale": scale_coarse,
        "scale_ratio": scale_refined / scale_coarse,
        "t_delta": (t_refined - t_coarse_val).cpu().tolist(),
    }
    with open(os.path.join(OUTPUT_DIR, "refined_transform.json"), "w") as f:
        json.dump(refined_result, f, indent=2)
    print(f"  Saved refined_transform.json")

    # ── Step 5: Render comparison images and compute metrics ──
    print("\n" + "=" * 60)
    print("Step 5: Computing metrics and rendering comparisons")
    print("=" * 60)

    from multiview_refine.gsplat_wrapper import render_gaussians, render_alpha
    from multiview_refine.alignment import _apply_transform, _get_real_masks, _compute_iou
    from PIL import Image

    real_masks = _get_real_masks(cameras, rmbg_fn=None)
    bg = torch.zeros(3, device=device)

    def compute_metrics(scale_val, R_val, t_val, label):
        means_t, scales_t, quats_t = _apply_transform(
            means, scales, quats, scale_val, R_val, t_val)
        ious, psnrs = [], []
        for i, cam in enumerate(cameras):
            with torch.no_grad():
                alpha = render_alpha(means_t, scales_t, quats_t, opacities, cam)
                sil = (alpha.squeeze(0) > 0.1).float()
                real_mask = real_masks[i]
                if sil.shape != real_mask.shape:
                    sil = torch.nn.functional.interpolate(
                        sil.unsqueeze(0).unsqueeze(0), size=real_mask.shape,
                        mode="bilinear", align_corners=False).squeeze()
                ious.append(_compute_iou(sil, real_mask))

                rgb, _, _ = render_gaussians(
                    means_t, scales_t, quats_t, colors, opacities,
                    cam, background=bg, sh_degree=SH_DEGREE)
                mse = torch.nn.functional.mse_loss(rgb, cam.image)
                psnr = 100.0 if mse < 1e-10 else 10.0 * torch.log10(1.0 / mse).item()
                psnrs.append(psnr)

        mean_iou = float(np.mean(ious))
        mean_psnr = float(np.mean(psnrs))
        print(f"  {label}: mean IoU={mean_iou:.4f}, mean PSNR={mean_psnr:.2f}")
        return mean_iou, mean_psnr, ious, psnrs

    iou_c, psnr_c, ious_c, psnrs_c = compute_metrics(
        scale_coarse, R_coarse, t_coarse_val, "Coarse ")
    iou_r, psnr_r, ious_r, psnrs_r = compute_metrics(
        scale_refined, R_refined, t_refined, "Refined")

    metrics = {
        "coarse": {"mean_iou": iou_c, "mean_psnr": psnr_c,
                    "per_view_iou": ious_c, "per_view_psnr": psnrs_c},
        "refined": {"mean_iou": iou_r, "mean_psnr": psnr_r,
                     "per_view_iou": ious_r, "per_view_psnr": psnrs_r},
        "improvement": {
            "iou_delta": iou_r - iou_c,
            "psnr_delta": psnr_r - psnr_c,
        }
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Δ IoU={iou_r-iou_c:+.4f}, Δ PSNR={psnr_r-psnr_c:+.2f}")

    # Render comparison images: GT | Coarse | Refined
    def to_uint8(img):
        return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    for idx in [0, len(cameras) // 4, len(cameras) // 2, 3 * len(cameras) // 4]:
        cam = cameras[idx]
        m_c, s_c, q_c = _apply_transform(means, scales, quats, scale_coarse, R_coarse, t_coarse_val)
        m_r, s_r, q_r = _apply_transform(means, scales, quats, scale_refined, R_refined, t_refined)
        with torch.no_grad():
            rgb_c, _, _ = render_gaussians(m_c, s_c, q_c, colors, opacities, cam, background=bg, sh_degree=SH_DEGREE)
            rgb_r, _, _ = render_gaussians(m_r, s_r, q_r, colors, opacities, cam, background=bg, sh_degree=SH_DEGREE)

        combined = np.concatenate([
            to_uint8(cam.image), to_uint8(rgb_c), to_uint8(rgb_r)
        ], axis=1)
        fname = f"compare_view{idx:02d}.png"
        Image.fromarray(combined).save(os.path.join(OUTPUT_DIR, fname))
        print(f"  Saved {fname}")

    # ── Step 6: Save aligned PLY ──
    print("\n" + "=" * 60)
    print("Step 6: Saving aligned Gaussians (refined)")
    print("=" * 60)

    from multiview_refine.ply_io import save_gaussians_ply

    m_final, s_final, q_final = _apply_transform(
        means, scales, quats, scale_refined, R_refined, t_refined)
    # Convert back to raw for saving.
    scales_final_raw = torch.log(s_final.clamp_min(1e-8))
    opac_final_raw = torch.log(opacities.clamp(1e-6, 1 - 1e-6) /
                                (1 - opacities.clamp(1e-6, 1 - 1e-6)))
    save_gaussians_ply(
        os.path.join(OUTPUT_DIR, "refined_aligned.ply"),
        m_final, scales_final_raw, q_final, opac_final_raw, sh0,
        shN if shN.numel() > 0 else None)
    print(f"  Saved refined_aligned.ply ({m_final.shape[0]} Gaussians)")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Output directory: {OUTPUT_DIR}/")
    print(f"  Files saved:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"    {f} ({size:,} bytes)")
    print(f"  Coarse:  IoU={iou_c:.4f}, PSNR={psnr_c:.2f}  ({time_coarse:.1f}s)")
    print(f"  Refined: IoU={iou_r:.4f}, PSNR={psnr_r:.2f}  (+{time_refine:.1f}s)")
    print(f"  Δ IoU={iou_r-iou_c:+.4f}, Δ PSNR={psnr_r-psnr_c:+.2f}")
    print("\nDone!")


if __name__ == "__main__":
    main()
