"""Depth + silhouette ICP alignment for initializing Gaussians to the scene.

Estimates a 7-DoF similarity transform (scale, rotation, translation) that maps
TripoSplat-generated Gaussians (in their normalised AABB coordinate frame) to
the real multi-view camera coordinate system (OpenCV convention).

Strategy:
  1. Render the init Gaussians from the init camera → depth + alpha maps.
  2. Estimate monocular depth on the real init image (Depth Anything V2).
  3. Back-project both depth maps to 3D point clouds using the init camera.
  4. Mask by silhouette (render alpha for the synthetic; BiRefNet for real).
  5. Estimate the similarity transform via the Umeyama method (closed-form),
     followed by a few ICP iterations for refinement.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from .cameras import Camera
from .gsplat_wrapper import render_depth, render_alpha


def backproject_depth(
    depth: torch.Tensor,
    camera: Camera,
    mask: Optional[torch.Tensor] = None,
    max_points: int = 10000,
) -> torch.Tensor:
    """Back-project a depth map to 3D points in camera space.

    Uses the pinhole model: ``x = (u - cx) * z / fx``,
    ``y = (v - cy) * z / fy``, ``z = depth``.

    Args:
        depth: (H, W) or (1, H, W) depth map (camera-space z).
        camera: The camera with matching intrinsics.
        mask: Optional (H, W) boolean/float mask; only masked pixels are kept.
        max_points: Subsample to at most this many points (random).

    Returns:
        (M, 3) 3D points in **camera** space.
    """
    if depth.ndim == 3:
        depth = depth.squeeze(0)  # (H, W)
    H, W = depth.shape
    device = depth.device

    # Create pixel coordinate grid.
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    z = depth  # (H, W)
    x = (u - camera.cx) * z / camera.fx
    y = (v - camera.cy) * z / camera.fy

    points = torch.stack([x, y, z], dim=-1)  # (H, W, 3)

    # Apply mask.
    if mask is not None:
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        valid = mask > 0.5
    else:
        valid = torch.ones_like(z, dtype=torch.bool)

    # Also filter out zero / invalid depths.
    valid = valid & (z > 0.01) & torch.isfinite(z)

    points = points[valid]  # (M, 3)

    # Subsample if too many points.
    if points.shape[0] > max_points:
        idx = torch.randperm(points.shape[0], device=device)[:max_points]
        points = points[idx]

    return points


def camera_to_world(points_cam: torch.Tensor, camera: Camera) -> torch.Tensor:
    """Transform points from camera space to world space using c2w."""
    # points_cam: (N, 3); c2w: (4, 4)
    R = camera.c2w[:3, :3]
    t = camera.c2w[:3, 3]
    return (points_cam @ R.T) + t[None, :]


def umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    with_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Estimate a similarity transform (s, R, t) mapping src → dst.

    Uses the closed-form Umeyama method.

    Args:
        src: (N, 3) source points.
        dst: (N, 3) destination points (same N).
        with_scale: If True, estimate a uniform scale; else scale=1.

    Returns:
        (scale, R (3x3), t (3,)) such that ``dst ≈ s * src @ R.T + t``.
    """
    assert src.shape == dst.shape
    n = src.shape[0]

    mu_src = src.mean(axis=0)  # (3,)
    mu_dst = dst.mean(axis=0)

    src_centered = src - mu_src  # (N, 3)
    dst_centered = dst - mu_dst

    # Cross-covariance matrix.
    cross_cov = dst_centered.T @ src_centered / n  # (3, 3)

    U, S, Vt = np.linalg.svd(cross_cov)

    # Handle reflection.
    d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    D = np.diag([1, 1, d])
    R = U @ D @ Vt

    if with_scale:
        var_src = np.sum(src_centered ** 2) / n
        scale = np.trace(np.diag(S) @ D) / var_src
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)

    return float(scale), R, t


def icp_refine(
    src: np.ndarray,
    dst: np.ndarray,
    init_scale: float,
    init_R: np.ndarray,
    init_t: np.ndarray,
    n_iterations: int = 10,
    max_corr_dist: float = float("inf"),
) -> tuple[float, np.ndarray, np.ndarray]:
    """Refine the similarity transform via ICP (nearest-neighbour matching).

    Args:
        src: (N, 3) source points.
        dst: (M, 3) destination points.
        init_scale, init_R, init_t: initial transform.
        n_iterations: number of ICP iterations.
        max_corr_dist: max correspondence distance (points farther are rejected).

    Returns:
        Refined (scale, R, t).
    """
    from scipy.spatial import cKDTree

    scale, R, t = init_scale, init_R.copy(), init_t.copy()

    tree = cKDTree(dst)

    for _ in range(n_iterations):
        # Transform source.
        src_transformed = scale * (src @ R.T) + t  # (N, 3)

        # Find nearest neighbours in dst.
        dists, indices = tree.query(src_transformed, k=1)

        # Filter by max correspondence distance.
        if max_corr_dist != float("inf"):
            valid = dists < max_corr_dist
            if valid.sum() < 10:
                break
            corr_dst = dst[indices[valid]]
            corr_src = src[valid]
        else:
            corr_dst = dst[indices]
            corr_src = src

        # Re-estimate transform.
        scale, R, t = umeyama(corr_src, corr_dst, with_scale=True)

    return scale, R, t


def estimate_similarity_transform(
    tripo_gaussian,
    init_camera: Camera,
    init_image: torch.Tensor,
    depth_estimator=None,
    rmbg_fn=None,
    n_icp_iterations: int = 10,
    max_points: int = 5000,
    alpha_threshold: float = 0.3,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Estimate the 7-DoF similarity transform from TripoSplat to scene coords.

    Args:
        tripo_gaussian: TripoSplat ``Gaussian`` object (in AABB coordinates).
        init_camera: The camera corresponding to the init view.
        init_image: (3, H, W) real image for depth estimation.
        depth_estimator: A callable that takes a PIL image and returns a
            depth map (H, W) normalised to [0, 1]. If None, uses
            ``TripoSplatPipeline.depth_estimator`` (must be passed separately).
        rmbg_fn: Optional callable that takes a PIL image and returns an RGBA
            image with alpha mask. Used for silhouette extraction. If None,
            uses the rendered alpha only.
        n_icp_iterations: Number of ICP refinement iterations.
        max_points: Max points for back-projection (subsampled).
        alpha_threshold: Threshold for the rendered alpha silhouette.

    Returns:
        (scale, R (3,3), t (3,)) such that ``xyz_scene = scale * (R @ xyz_tripo) + t``.
    """
    device = init_camera.device

    # ── Step 1: Render depth + alpha from the init Gaussians ──
    # Convert to float32 — TripoSplat outputs may be fp16 (from fp16 checkpoints),
    # but gsplat 1.5.x requires fp32.
    means = tripo_gaussian.get_xyz.detach().to(device).float()
    scales = tripo_gaussian.get_scaling.detach().to(device).float()
    quats = (tripo_gaussian._rotation + tripo_gaussian.rots_bias[None, :]).detach().to(device).float()
    opacities = tripo_gaussian.get_opacity.detach().squeeze(-1).to(device).float()
    colors = tripo_gaussian._features_dc.detach().to(device).float()

    render_depth_map = render_depth(
        means, scales, quats, colors, opacities, init_camera, sh_degree=0
    )  # (1, H, W)

    render_alpha_map = render_alpha(
        means, scales, quats, opacities, init_camera
    )  # (1, H, W)

    # ── Step 2: Estimate depth on the real init image ──
    # Convert tensor to PIL for the depth estimator.
    from PIL import Image as PILImage
    img_np = (init_image.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    img_pil = PILImage.fromarray(img_np)

    if depth_estimator is not None:
        depth_result = depth_estimator(img_pil)
        real_depth_raw = depth_result["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
        # Normalise to [0, 1].
        dmin, dmax = real_depth_raw.min(), real_depth_raw.max()
        if dmax > dmin:
            real_depth_raw = (real_depth_raw - dmin) / (dmax - dmin)
        # Resize to camera resolution if needed.
        if real_depth_raw.shape[:2] != (init_camera.height, init_camera.width):
            depth_pil = PILImage.fromarray((real_depth_raw * 255).astype(np.uint8), mode="L")
            depth_pil = depth_pil.resize(
                (init_camera.width, init_camera.height), PILImage.BILINEAR
            )
            real_depth_raw = np.array(depth_pil, dtype=np.float32) / 255.0
        real_depth = torch.from_numpy(real_depth_raw).to(device)
    else:
        # Fallback: no depth estimator, use rendered depth as a proxy.
        real_depth = render_depth_map.squeeze(0)

    # ── Step 3: Build silhouette masks ──
    # Synthetic: rendered alpha.
    synth_mask = (render_alpha_map.squeeze(0) > alpha_threshold).float()

    # Real: BiRefNet alpha if available, else use the whole image.
    if rmbg_fn is not None:
        rgba = rmbg_fn(img_pil)
        real_alpha = np.asarray(rgba, dtype=np.float32)
        if real_alpha.ndim == 3 and real_alpha.shape[-1] == 4:
            real_alpha = real_alpha[..., 3] / 255.0
        else:
            real_alpha = np.ones(real_alpha.shape[:2], dtype=np.float32)
        # Resize to camera resolution.
        if real_alpha.shape[:2] != (init_camera.height, init_camera.width):
            alpha_pil = PILImage.fromarray((real_alpha * 255).astype(np.uint8), mode="L")
            alpha_pil = alpha_pil.resize(
                (init_camera.width, init_camera.height), PILImage.BILINEAR
            )
            real_alpha = np.array(alpha_pil, dtype=np.float32) / 255.0
        real_mask = torch.from_numpy(real_alpha).to(device)
    else:
        real_mask = torch.ones(
            (init_camera.height, init_camera.width), dtype=torch.float32, device=device
        )

    # ── Step 4: Back-project to 3D point clouds ──
    # Synthetic points (in TripoSplat AABB coords, since the Gaussians are there).
    # We render using the init camera, which has world-to-camera for the *scene*
    # coordinate system, but the Gaussians are in AABB. So the rendered depth
    # is in the *camera* space of the init camera, but the camera's w2c assumes
    # the Gaussians are in scene coords.
    #
    # We need to be careful: the init camera's w2c transforms *scene* world
    # coordinates to camera space. The TripoSplat Gaussians are in AABB coords
    # (not scene coords). So when we render, gsplat uses the camera's w2c
    # directly on the Gaussian means — the resulting depth is "as if the
    # Gaussians were in scene coords". This is fine for alignment: we
    # back-project the rendered depth to get the AABB points in the camera
    # frame, and back-project the real depth to get the scene points in the
    # camera frame. The transform between them is what we want.

    # Synthetic: back-project rendered depth to camera space, then to "AABB world".
    # Since we rendered with the init camera's w2c, the camera-space points
    # correspond to AABB-world points via the camera's c2w. We transform back
    # to get AABB-world coords (which is what the Gaussians are in).
    synth_cam_pts = backproject_depth(
        render_depth_map, init_camera, mask=synth_mask, max_points=max_points
    )  # (M, 3) in camera space
    # Transform camera → world (AABB world, since the camera was applied to AABB means).
    synth_world_pts = camera_to_world(synth_cam_pts, init_camera)  # (M, 3) in AABB coords

    # Real: back-project real depth to camera space, then to scene world.
    real_cam_pts = backproject_depth(
        real_depth, init_camera, mask=real_mask, max_points=max_points
    )  # (M, 3) in camera space
    real_world_pts = camera_to_world(real_cam_pts, init_camera)  # (M, 3) in scene coords

    if synth_world_pts.shape[0] < 10 or real_world_pts.shape[0] < 10:
        # Not enough points; return identity transform.
        print(
            f"Warning: not enough points for alignment "
            f"(synth={synth_world_pts.shape[0]}, real={real_world_pts.shape[0]}). "
            "Using identity transform."
        )
        return 1.0, torch.eye(3, device=device), torch.zeros(3, device=device)

    # ── Step 5: Estimate similarity transform ──
    # We need to match synth (AABB) → real (scene).
    # Umeyama requires equal-length correspondences, so we subsample the larger.
    n = min(synth_world_pts.shape[0], real_world_pts.shape[0])
    synth_np = synth_world_pts[:n].cpu().numpy()
    real_np = real_world_pts[:n].cpu().numpy()

    # Initial estimate via Umeyama (treating the first n points as correspondences —
    # this is a rough initialisation since they're not true correspondences).
    # A better initialisation: align centroids and set scale by std ratio.
    centroid_synth = synth_np.mean(axis=0)
    centroid_real = real_np.mean(axis=0)
    scale_init = float(
        np.std(real_np - centroid_real) / max(np.std(synth_np - centroid_synth), 1e-8)
    )
    R_init = np.eye(3)
    t_init = centroid_real - scale_init * centroid_synth

    # ICP refinement: find true nearest-neighbour correspondences.
    # We use all synth points as source and all real points as destination.
    all_synth_np = synth_world_pts.cpu().numpy()
    all_real_np = real_world_pts.cpu().numpy()

    scale, R_np, t_np = icp_refine(
        all_synth_np,
        all_real_np,
        init_scale=scale_init,
        init_R=R_init,
        init_t=t_init,
        n_iterations=n_icp_iterations,
        max_corr_dist=scale_init * 2.0,  # generous threshold
    )

    R = torch.from_numpy(R_np).float().to(device)
    t = torch.from_numpy(t_np).float().to(device)

    print(
        f"Alignment: scale={scale:.4f}, "
        f"R={R.diagonal().tolist()}, "
        f"t={t.tolist()}"
    )

    return scale, R, t
