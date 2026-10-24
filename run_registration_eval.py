"""Multi-view visual evaluation of a registered Gaussian model.

Renders the registered Gaussians from every calibrated camera and lays the
result next to the real photograph, so the registration can be judged by eye as
well as by number.

For each view it writes a four-panel strip:

    | real photo | rendered model | 50/50 blend | silhouette overlay |

* **blend** — the real photo and the render at 50 % each.  Alignment error shows
  up as doubling/ghosting of edges; a correct registration reads as one image.
* **silhouette overlay** — real-mask boundary in green, rendered-silhouette
  boundary in red.  Overlapping stretches look yellow, so gaps are visible
  directly rather than inferred from an IoU scalar.

It also writes:

* ``view_grid.png`` — contact sheet of every view (GT over render).
* ``metrics.json``  — per-view IoU, centroid offset, height and orientation.
* a printed table with the same numbers.

Usage
-----
    conda activate rtg132
    python run_registration_eval.py
    python run_registration_eval.py --ply output/registration/registered.ply
    python run_registration_eval.py --view_stride 2      # every 2nd camera
    python run_registration_eval.py --columns 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pointsplat_bridge._gsplat_preload import preload_gsplat_cuda

preload_gsplat_cuda()

import cv2

from multiview_refine.cameras import load_cameras
from multiview_refine.gsplat_wrapper import render_alpha, render_gaussians
from multiview_refine.orientation import silhouette_orientation_stats
from multiview_refine.ply_io import load_gaussians_ply

LABEL_H = 30
PANEL_GAP = 4


# ── image helpers ───────────────────────────────────────────────────────────

def _to_bgr(img: torch.Tensor) -> np.ndarray:
    """(3, H, W) float in [0,1] → BGR uint8."""
    arr = (img.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _label(img: np.ndarray, text: str, color=(0, 255, 255)) -> np.ndarray:
    bar = np.full((LABEL_H, img.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(bar, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                color, 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def _stack_h(panels: list[np.ndarray], gap: int = PANEL_GAP) -> np.ndarray:
    """Concatenate horizontally, padding shorter panels at the bottom."""
    h = max(p.shape[0] for p in panels)
    out = []
    for i, p in enumerate(panels):
        if p.shape[0] < h:
            p = np.concatenate(
                [p, np.zeros((h - p.shape[0], p.shape[1], 3), np.uint8)], axis=0)
        if i:
            out.append(np.full((h, gap, 3), 45, np.uint8))
        out.append(p)
    return np.concatenate(out, axis=1)


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    """1-pixel contour of a binary mask."""
    m = mask.astype(np.uint8)
    er = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
    return (m - er).astype(bool)


def _overlay_silhouettes(real: np.ndarray, pred: np.ndarray,
                         real_bgr: np.ndarray) -> np.ndarray:
    """Drape both mask boundaries over a dimmed copy of the real photo."""
    base = (real_bgr.astype(np.float32) * 0.35).astype(np.uint8)
    # Real boundary green, rendered boundary red; coincident pixels turn yellow.
    base[_mask_boundary(real)] = (0, 220, 0)
    red = _mask_boundary(pred)
    only_red = red & ~_mask_boundary(real)
    base[red] = (60, 60, 255)
    base[only_red] = (0, 90, 255)
    return base


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-view comparison of a registered model")
    ap.add_argument("--ply", default="output/registration/registered.ply",
                    help="Gaussians already in world coordinates")
    ap.add_argument("--transforms", default="static/000001/mvs/transforms_train.json")
    ap.add_argument("--output_dir", default="output/registration_eval")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image_scale", type=float, default=0.5)
    ap.add_argument("--alpha_threshold", type=float, default=0.5,
                    help="Threshold for both the real mask and the render")
    ap.add_argument("--view_stride", type=int, default=1,
                    help="Use every Nth camera (1 = all)")
    ap.add_argument("--columns", type=int, default=4,
                    help="Columns in the contact sheet")
    ap.add_argument("--grid_thumb_width", type=int, default=320,
                    help="Thumbnail width per cell in the contact sheet")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 74)
    print("Multi-view registration evaluation")
    print("=" * 74)
    print(f"  Model:   {args.ply}")
    print(f"  Cameras: {args.transforms}")
    print(f"  Output:  {args.output_dir}")

    cameras = load_cameras(
        args.transforms,
        base_dir=os.path.dirname(os.path.abspath(args.transforms)),
        device=device,
        image_scale=args.image_scale,
        load_image=True,
    )
    means, scales_raw, quats_raw, opac_raw, sh0, _ = load_gaussians_ply(
        args.ply, device=device)
    scales = scales_raw.exp()
    opacities = torch.sigmoid(opac_raw)
    quats = torch.nn.functional.normalize(quats_raw, dim=-1)
    print(f"  {len(cameras)} cameras, {means.shape[0]} Gaussians "
          f"({cameras[0].width}x{cameras[0].height})")

    bg = torch.zeros(3, device=device)
    thr = args.alpha_threshold
    idxs = list(range(0, len(cameras), max(1, args.view_stride)))

    rows = []
    thumbs_gt, thumbs_pred = [], []

    print("\n" + "-" * 74)
    print(f"{'view':>5} {'file':<12} {'clip':<5} {'IoU':>6} "
          f"{'d_centroid(px)':>15} {'height(m)':>10} {'orient':>7}")
    print("-" * 74)

    for i in idxs:
        cam = cameras[i]
        with torch.no_grad():
            rgb, alpha, _ = render_gaussians(
                means, scales, quats, sh0, opacities, cam,
                background=bg, sh_degree=0)
            if cam.alpha is None:
                continue
            real_mask = (cam.alpha.squeeze(0) > thr).detach().cpu().numpy()

        pred_mask = (alpha.squeeze(0) > thr).detach().cpu().numpy()
        gt_bgr = _to_bgr(cam.image)
        pred_bgr = _to_bgr(rgb)
        blend = ((gt_bgr.astype(np.float32) * 0.5
                  + pred_bgr.astype(np.float32) * 0.5)).astype(np.uint8)
        sil = _overlay_silhouettes(real_mask, pred_mask, gt_bgr)

        iou = _iou(real_mask, pred_mask)
        ys_p, xs_p = np.nonzero(pred_mask)
        ys_r, xs_r = np.nonzero(real_mask)
        clipped = bool(real_mask[0].any() or real_mask[-1].any()
                       or real_mask[:, 0].any() or real_mask[:, -1].any())

        if len(ys_p) > 20 and len(ys_r) > 20:
            d = (float(xs_p.mean() - xs_r.mean()), float(ys_p.mean() - ys_r.mean()))
        else:
            d = (float("nan"), float("nan"))

        # Implied metric height from the render.  Only meaningful when the
        # silhouette is whole: a cropped view has no real head/feet, so its
        # apparent extent understates the subject and the number is noise.
        height = float("nan")
        if len(ys_p) > 20 and not clipped:
            centre_cam = (cam.w2c[:3, :3] @ means.mean(dim=0) + cam.w2c[:3, 3])
            z = float(centre_cam[2])
            if z > 1e-6:
                height = float((ys_p.max() - ys_p.min() + 1) * z / cam.fy)

        p_st = silhouette_orientation_stats(pred_mask)
        r_st = silhouette_orientation_stats(real_mask)
        if (p_st["valid"] and r_st["valid"]
                and not p_st.get("ambiguous", False)
                and not r_st.get("ambiguous", False)):
            orient = "ok" if (p_st["band_ratio"] < 1.0) == (r_st["band_ratio"] < 1.0) else "FLIP"
        else:
            orient = "n/a"

        rows.append(dict(
            view=i, file=cam.file_path, clipped=clipped, iou=iou,
            centroid_dx=d[0], centroid_dy=d[1],
            implied_height_m=height, orientation=orient,
            mask_area_real=int(real_mask.sum()), mask_area_pred=int(pred_mask.sum()),
        ))

        print(f"{i:5d} {os.path.basename(cam.file_path):<12} "
              f"{str(clipped):<5} {iou:6.3f} "
              f"({d[0]:+6.1f},{d[1]:+6.1f}) {height:10.3f} {orient:>7}")

        # Four-panel strip for this view.
        panel = _stack_h([
            _label(gt_bgr, f"real  {os.path.basename(cam.file_path)}"),
            _label(pred_bgr, f"render  IoU={iou:.3f}"),
            _label(blend, "50/50 blend"),
            _label(sil, "silhouette: real=green render=red"),
        ])
        cv2.imwrite(os.path.join(args.output_dir, f"view{i:02d}_compare.png"), panel)

        thumbs_gt.append(_label(gt_bgr, f"real cam{i}"))
        thumbs_pred.append(_label(pred_bgr, f"render cam{i} IoU={iou:.2f}"))

    # ── Contact sheet: real on top row(s), render below ──
    def make_grid(items: list[np.ndarray], cols: int) -> np.ndarray:
        cell_w = args.grid_thumb_width
        cells = []
        for im in items:
            s = cell_w / im.shape[1]
            cells.append(cv2.resize(im, (cell_w, max(1, int(im.shape[0] * s)))))
        cell_h = max(c.shape[0] for c in cells)
        n_rows = math.ceil(len(cells) / cols)
        sheet = np.full((n_rows * (cell_h + PANEL_GAP), cols * (cell_w + PANEL_GAP), 3),
                        35, np.uint8)
        for k, c in enumerate(cells):
            r, col = divmod(k, cols)
            y = r * (cell_h + PANEL_GAP)
            x = col * (cell_w + PANEL_GAP)
            sheet[y:y + c.shape[0], x:x + c.shape[1]] = c
        return sheet

    if thumbs_gt:
        grid = np.concatenate([
            _label(make_grid(thumbs_gt, args.columns), "REAL PHOTOGRAPHS", (0, 255, 0)),
            np.full((PANEL_GAP * 3, make_grid(thumbs_gt, args.columns).shape[1], 3),
                    35, np.uint8),
            _label(make_grid(thumbs_pred, args.columns), "REGISTERED RENDER", (0, 200, 255)),
        ], axis=0)
        cv2.imwrite(os.path.join(args.output_dir, "view_grid.png"), grid)
        print(f"\n  Contact sheet: view_grid.png  "
              f"({grid.shape[1]}x{grid.shape[0]})")

    # ── Summary ──
    clean = [r for r in rows if not r["clipped"]]
    def mean(key, rs):
        vals = [r[key] for r in rs if np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "n_views": len(rows),
        "n_uncropped": len(clean),
        "alpha_threshold": thr,
        "mean_iou_all": mean("iou", rows),
        "mean_iou_uncropped": mean("iou", clean),
        "min_iou_uncropped": min((r["iou"] for r in clean), default=float("nan")),
        "max_iou_uncropped": max((r["iou"] for r in clean), default=float("nan")),
        "mean_abs_centroid_dx_uncropped": float(np.mean([abs(r["centroid_dx"]) for r in clean])) if clean else float("nan"),
        "mean_abs_centroid_dy_uncropped": float(np.mean([abs(r["centroid_dy"]) for r in clean])) if clean else float("nan"),
        "mean_implied_height_m": mean("implied_height_m", clean),
        "height_std_m": float(np.std([r["implied_height_m"] for r in clean if np.isfinite(r["implied_height_m"])])) if clean else float("nan"),
        "n_flipped": sum(1 for r in clean if r["orientation"] == "FLIP"),
        "n_orientation_judged": sum(1 for r in clean if r["orientation"] in ("ok", "FLIP")),
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"summary": summary, "per_view": rows}, f, indent=2)

    print("\n" + "=" * 74)
    print("Summary")
    print("=" * 74)
    print(f"  views: {summary['n_views']} ({summary['n_uncropped']} uncropped)")
    print(f"  mean IoU:  {summary['mean_iou_all']:.4f} all  |  "
          f"{summary['mean_iou_uncropped']:.4f} uncropped  |  "
          f"min {summary['min_iou_uncropped']:.4f}  max {summary['max_iou_uncropped']:.4f}")
    print(f"  centroid offset (uncropped): "
          f"{summary['mean_abs_centroid_dx_uncropped']:.1f} px, "
          f"{summary['mean_abs_centroid_dy_uncropped']:.1f} px")
    print(f"  implied height: {summary['mean_implied_height_m']:.3f} m "
          f"(std {summary['height_std_m']:.4f})")
    print(f"  upright: {summary['n_orientation_judged'] - summary['n_flipped']}"
          f"/{summary['n_orientation_judged']} judged views, "
          f"{summary['n_flipped']} flipped")
    print(f"\n  Wrote per-view strips, view_grid.png and metrics.json to "
          f"{args.output_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
