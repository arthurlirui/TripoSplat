"""Multi-view alignment: estimate rotation + scale for initial Gaussians.

Instead of using a single-view depth ICP, this module renders the initial
Gaussians from **all** training camera viewpoints, compares the rendered
silhouettes (alpha maps) with the real images' foreground masks, and
searches for the rotation R and scale s that maximise silhouette overlap
across all views.

The transform maps: ``xyz_scene = s * (R @ xyz_tripo) + t`` where t is
determined by centroid alignment once R and s are fixed.

Strategy:
  1. Extract foreground masks from real images (BiRefNet or alpha channel).
  2. For each candidate (R, s), render all views and compute IoU between
     rendered alpha and real mask.
  3. Optimise (R, s) to maximise mean IoU via grid search + gradient-free
     refinement (Nelder-Mead or exhaustive rotation search).
  4. Compute t from centroid alignment.
"""

from __future__ import annotations

import math
import itertools
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn.functional as F

from .cameras import Camera
from .gsplat_wrapper import render_gaussians, render_alpha

SH_C0 = 0.28209479177387814


def _tripo_gaussian_to_tensors(tripo_gaussian, device):
    """Extract fp32 tensors from a TripoSplat Gaussian object."""
    means = tripo_gaussian.get_xyz.detach().to(device).float()
    scales = tripo_gaussian.get_scaling.detach().to(device).float()
    quats = (tripo_gaussian._rotation + tripo_gaussian.rots_bias[None, :]).detach().to(device).float()
    opacities = tripo_gaussian.get_opacity.detach().squeeze(-1).to(device).float()
    colors = tripo_gaussian._features_dc.detach().to(device).float()
    return means, scales, quats, opacities, colors


def _apply_transform(means, scales, quats, scale, R, t):
    """Apply a similarity transform to Gaussian attributes.

    means: s * (R @ means) + t
    scales: s * scales
    quats: compose R's quaternion with each Gaussian's quaternion
    """
    from .quaternion import quaternion_compose, rotation_matrix_to_quaternion

    means_new = scale * (means @ R.T) + t
    scales_new = scales * scale
    q_R = rotation_matrix_to_quaternion(R)  # (4,)
    quats_new = quaternion_compose(q_R, quats)  # (N, 4)
    return means_new, scales_new, quats_new


def _get_real_masks(cameras, rmbg_fn=None, alpha_threshold=0.5):
    """Extract foreground masks from real camera images.

    Returns a list of (H, W) boolean tensors, one per camera.
    """
    from PIL import Image as PILImage

    masks = []
    for cam in cameras:
        if cam.alpha is not None:
            mask = (cam.alpha.squeeze(0) > alpha_threshold).float()
        elif rmbg_fn is not None:
            # Use BiRefNet to get the mask.
            img_np = (cam.image.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            img_pil = PILImage.fromarray(img_np)
            rgba = rmbg_fn(img_pil)
            alpha = np.asarray(rgba, dtype=np.float32)
            if alpha.ndim == 3 and alpha.shape[-1] == 4:
                alpha = alpha[..., 3] / 255.0
            else:
                alpha = np.ones(alpha.shape[:2], dtype=np.float32)
            if alpha.shape[:2] != (cam.height, cam.width):
                alpha_pil = PILImage.fromarray((alpha * 255).astype(np.uint8), mode="L")
                alpha_pil = alpha_pil.resize((cam.width, cam.height), PILImage.BILINEAR)
                alpha = np.array(alpha_pil, dtype=np.float32) / 255.0
            mask = torch.from_numpy(alpha).to(cam.device)
        else:
            # Fallback: use image brightness as mask (non-black = foreground).
            mask = (cam.image.mean(dim=0) > 0.05).float()
        masks.append(mask)
    return masks


def _render_silhouettes(means, scales, quats, opacities, cameras, alpha_threshold=0.1):
    """Render binary silhouette masks from the given Gaussians for each camera.

    Returns list of (H, W) float tensors.
    """
    silhouettes = []
    for cam in cameras:
        with torch.no_grad():
            alpha = render_alpha(means, scales, quats, opacities, cam)
        sil = (alpha.squeeze(0) > alpha_threshold).float()
        silhouettes.append(sil)
    return silhouettes


def _compute_iou(pred_mask, real_mask):
    """Compute Intersection-over-Union of two binary masks."""
    intersection = (pred_mask * real_mask).sum()
    union = ((pred_mask + real_mask) > 0).float().sum()
    if union < 1:
        return 0.0
    return float((intersection / union).item())


def _compute_iou_batch(pred_masks, real_masks):
    """Compute mean IoU across multiple view pairs."""
    ious = []
    for pred, real in zip(pred_masks, real_masks):
        # Resize pred to match real if needed
        if pred.shape != real.shape:
            pred = F.interpolate(
                pred.unsqueeze(0).unsqueeze(0), size=real.shape,
                mode="bilinear", align_corners=False
            ).squeeze()
        ious.append(_compute_iou(pred, real))
    return np.mean(ious)


def _euler_to_matrix(rx, ry, rz):
    """Euler angles (radians) → rotation matrix (XYZ extrinsic)."""
    Rx = torch.tensor([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)],
    ], dtype=torch.float32)
    Ry = torch.tensor([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)],
    ], dtype=torch.float32)
    Rz = torch.tensor([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1],
    ], dtype=torch.float32)
    return Rz @ Ry @ Rx


def _estimate_translation(means, cameras, real_masks):
    """Estimate translation t by aligning the centroid of visible Gaussians
    with the centroid of the real scene (derived from mask centroids).

    The real scene centre is estimated as the mean of camera positions
    weighted by mask coverage.
    """
    # Gaussian centroid (in current frame).
    gc = means.mean(dim=0)

    # Scene centre: mean of camera positions that have significant foreground.
    positions = []
    for i, (cam, mask) in enumerate(zip(cameras, real_masks)):
        coverage = mask.mean().item()
        if coverage > 0.02:  # at least 2% foreground
            positions.append(cam.position.cpu().numpy())
    if len(positions) < 2:
        scene_centre = np.zeros(3, dtype=np.float32)
    else:
        scene_centre = np.mean(positions, axis=0)

    t = torch.from_numpy(scene_centre).float().to(means.device) - gc
    return t


def estimate_similarity_transform_multiview(
    tripo_gaussian=None,
    tensors: tuple | None = None,
    cameras: list[Camera] | None = None,
    rmbg_fn: Callable | None = None,
    n_rotation_samples: int = 24,
    n_scale_samples: int = 10,
    scale_range: tuple[float, float] = (0.05, 2.0),
    alpha_threshold: float = 0.1,
    init_yaw: float | None = None,
    verbose: bool = True,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Estimate 7-DoF similarity transform via multi-view silhouette matching.

    Renders the initial Gaussians from all camera viewpoints for each
    candidate (R, s) and picks the one that maximises mean IoU between
    rendered silhouettes and real foreground masks.

    Args:
        tripo_gaussian: TripoSplat ``Gaussian`` object (in AABB coordinates).
            Ignored if *tensors* is provided.
        tensors: Optional ``(means, scales, quats, opacities, colors)`` tuple.
            When given, bypasses the Gaussian extraction step — useful when
            the data has already been loaded from PLY and pre-transformed.
        cameras: List of multi-view cameras (with loaded images).
        rmbg_fn: Background removal function (PIL → RGBA). If None, uses
            image brightness as foreground mask.
        n_rotation_samples: Number of rotation samples per axis (coarse search).
        n_scale_samples: Number of scale samples.
        scale_range: (min_scale, max_scale) to search.
        alpha_threshold: Threshold for rendered alpha → binary silhouette.
        init_yaw: If given (radians), the coarse yaw search is centered around
            this value instead of doing a full 360° sweep. Improves speed when
            a good initial estimate is available (e.g. from c2w alignment).
        verbose: Print progress.

    Returns:
        (scale, R (3,3), t (3,)) such that ``xyz_scene = s * (R @ xyz_tripo) + t``.
    """
    if tensors is not None:
        means, scales, quats, opacities, colors = tensors
    else:
        assert tripo_gaussian is not None, "Either tripo_gaussian or tensors must be provided"
        device = cameras[0].device
        means, scales, quats, opacities, colors = _tripo_gaussian_to_tensors(tripo_gaussian, device)

    device = cameras[0].device

    if verbose:
        print(f"  Gaussians: {means.shape[0]}, means range [{means.min():.3f}, {means.max():.3f}]")

    # ── Step 1: Extract real foreground masks ──
    if verbose:
        print("  Extracting foreground masks from real images...")
    real_masks = _get_real_masks(cameras, rmbg_fn=rmbg_fn)

    # ── Step 2: Coarse grid search over rotation and scale ──
    if verbose:
        print(f"  Grid search: {n_rotation_samples} rotations × {n_scale_samples} scales...")

    # Rotation: sample yaw (around Y), pitch (around X), roll (around Z).
    # For efficiency, first search yaw (most important for orbit datasets),
    # then refine pitch and roll.
    if init_yaw is not None:
        half_window = math.pi / 4  # narrow window around the hint
        yaw_samples = np.linspace(init_yaw - half_window, init_yaw + half_window,
                                  n_rotation_samples, endpoint=False)
    else:
        yaw_samples = np.linspace(0, 2 * math.pi, n_rotation_samples, endpoint=False)
    scale_samples = np.linspace(scale_range[0], scale_range[1], n_scale_samples)

    best_iou = -1.0
    best_params = (1.0, 0.0, 0.0, 0.0, scale_range[0])  # (yaw, pitch, roll, _, scale)

    # Initial translation: align Gaussian centroid to scene centre.
    t_init = _estimate_translation(means, cameras, real_masks)

    # Phase 1: Coarse yaw + scale search (pitch=0, roll=0).
    for yaw in yaw_samples:
        for s in scale_samples:
            R = _euler_to_matrix(0.0, yaw, 0.0).to(device)
            t = t_init.to(device)
            means_t, scales_t, quats_t = _apply_transform(means, scales, quats, s, R, t)
            # Render silhouettes for a subset of cameras (every 3rd for speed).
            subset_cams = cameras[::3]
            subset_masks = real_masks[::3]
            sils = _render_silhouettes(means_t, scales_t, quats_t, opacities, subset_cams, alpha_threshold)
            iou = _compute_iou_batch(sils, subset_masks)
            if iou > best_iou:
                best_iou = iou
                best_params = (yaw, 0.0, 0.0, s)
                if verbose:
                    print(f"    yaw={math.degrees(yaw):.0f}°, scale={s:.3f}, IoU={iou:.4f} ★")

    if verbose:
        print(f"  Coarse best: yaw={math.degrees(best_params[0]):.1f}°, scale={best_params[3]:.3f}, IoU={best_iou:.4f}")

    # Phase 2: Fine search around the best yaw and scale, adding pitch/roll.
    best_yaw, _, _, best_scale = best_params
    yaw_fine = np.linspace(best_yaw - math.pi / n_rotation_samples,
                           best_yaw + math.pi / n_rotation_samples, 7)
    scale_fine = np.linspace(max(best_scale * 0.7, scale_range[0]),
                             min(best_scale * 1.3, scale_range[1]), 5)
    pitch_samples = np.linspace(-math.pi / 6, math.pi / 6, 5)  # ±30°
    roll_samples = np.linspace(-math.pi / 6, math.pi / 6, 5)   # ±30°

    if verbose:
        print(f"  Fine search: {len(yaw_fine)}×{len(pitch_samples)}×{len(roll_samples)}×{len(scale_fine)}...")

    for yaw in yaw_fine:
        for pitch in pitch_samples:
            for roll in roll_samples:
                for s in scale_fine:
                    R = _euler_to_matrix(pitch, yaw, roll).to(device)
                    t = t_init.to(device)
                    means_t, scales_t, quats_t = _apply_transform(means, scales, quats, s, R, t)
                    sils = _render_silhouettes(means_t, scales_t, quats_t, opacities, cameras, alpha_threshold)
                    iou = _compute_iou_batch(sils, real_masks)
                    if iou > best_iou:
                        best_iou = iou
                        best_params = (yaw, pitch, roll, s)
                        if verbose:
                            print(f"    yaw={math.degrees(yaw):.0f}° pitch={math.degrees(pitch):.0f}° "
                                  f"roll={math.degrees(roll):.0f}° scale={s:.3f} IoU={iou:.4f} ★")

    best_yaw, best_pitch, best_roll, best_scale = best_params
    R_best = _euler_to_matrix(best_pitch, best_yaw, best_roll).to(device)

    # ── Step 3: Final translation estimation ──
    means_final = best_scale * (means @ R_best.T) + t_init.to(device)
    t_best = _estimate_translation(means_final, cameras, real_masks)

    if verbose:
        print(f"  Final: yaw={math.degrees(best_yaw):.1f}°, pitch={math.degrees(best_pitch):.1f}°, "
              f"roll={math.degrees(best_roll):.1f}°, scale={best_scale:.4f}, IoU={best_iou:.4f}")
        print(f"  Translation: {t_best.tolist()}")

    return best_scale, R_best, t_best


# Keep backward-compatible alias
def estimate_similarity_transform(
    tripo_gaussian,
    init_camera: Camera,
    init_image: torch.Tensor,
    depth_estimator=None,
    rmbg_fn=None,
    cameras: list[Camera] | None = None,
    n_icp_iterations: int = 10,
    max_points: int = 5000,
    alpha_threshold: float = 0.1,
    verbose: bool = True,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Estimate similarity transform — delegates to multi-view method if cameras
    are provided, otherwise falls back to single-view depth ICP.
    """
    if cameras is not None and len(cameras) > 1:
        return estimate_similarity_transform_multiview(
            tripo_gaussian, cameras, rmbg_fn=rmbg_fn,
            alpha_threshold=alpha_threshold, verbose=verbose,
        )

    # Fallback: simple centroid + scale estimation.
    device = init_camera.device
    means, scales, quats, opacities, colors = _tripo_gaussian_to_tensors(tripo_gaussian, device)

    # Centroid alignment.
    gc = means.mean(dim=0)
    scene_centre = init_camera.position * 0.7  # rough estimate: 70% towards camera
    t = scene_centre - gc

    # Scale: ratio of camera distance to Gaussian spread.
    cam_dist = init_camera.position.norm().item()
    gs_spread = float(means.std(dim=0).norm().item())
    scale = max(cam_dist / (gs_spread * 4), 0.01)

    R = torch.eye(3, device=device)
    if verbose:
        print(f"  Fallback alignment: scale={scale:.4f}, t={t.tolist()}")
    return scale, R, t
