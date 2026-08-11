"""gsplat 1.5.x rendering wrapper.

Encapsulates the high-level :func:`gsplat.rasterization` API into convenient
functions that take a :class:`~multiview_refine.cameras.Camera` and return
images in the ``(3, H, W)`` layout expected by the loss functions.

gsplat 1.5.x conventions:
  - ``means``:     (N, 3) world-space centres.
  - ``quats``:     (N, 4) raw quaternions (w, x, y, z); normalised internally.
  - ``scales``:    (N, 3) **activated** scales (not log-space).
  - ``opacities``: (N,)   **activated** opacities in [0, 1] (not logit).
  - ``colors``:    (N, K, 3) SH coefficients when ``sh_degree`` is set;
                   (N, D) raw colours when ``sh_degree`` is None.
  - ``viewmats``:  (C, 4, 4) world-to-camera (OpenCV convention).
  - ``Ks``:        (C, 3, 3) intrinsics matrix ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``.
  - Returns ``(render_colors, render_alphas, meta)`` with colours in ``(C, H, W, X)``.
"""

from __future__ import annotations

from typing import Optional

import torch

import gsplat

from .cameras import Camera


def _intrinsics_matrix(camera: Camera) -> torch.Tensor:
    """Build a 3x3 pinhole intrinsics matrix from the camera's fx/fy/cx/cy."""
    K = torch.zeros((3, 3), dtype=torch.float32, device=camera.device)
    K[0, 0] = camera.fx
    K[1, 1] = camera.fy
    K[0, 2] = camera.cx
    K[1, 2] = camera.cy
    K[2, 2] = 1.0
    return K


def render_gaussians(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    camera: Camera,
    background: Optional[torch.Tensor] = None,
    sh_degree: Optional[int] = 0,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    rasterize_mode: str = "classic",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Render Gaussians from a single camera viewpoint.

    Args:
        means:     (N, 3)  world-space Gaussian centres.
        scales:    (N, 3)  **activated** scales.
        quats:     (N, 4)  raw quaternions (normalised internally).
        colors:    (N, K, 3) SH coefficients if ``sh_degree`` is set, or
                   (N, 3) raw RGB if ``sh_degree`` is None.
        opacities: (N,)    **activated** opacities in [0, 1].
        camera:    Target viewpoint.
        background: (3,) background colour, defaults to black.
        sh_degree:  SH degree to activate. Use ``None`` for raw colours,
                    ``0`` for DC-only SH.
        near_plane, far_plane: clip planes in camera-space z.
        rasterize_mode: ``"classic"`` or ``"antialiased"``.

    Returns:
        (rendered_rgb (3, H, W), rendered_alpha (1, H, W), meta_dict) —
        rgb/alpha are float32; meta_dict contains gsplat intermediate results
        (including ``means2d`` needed by the densification strategy).
    """
    device = means.device
    H, W = camera.height, camera.width

    if background is None:
        background = torch.zeros(3, dtype=torch.float32, device=device)
    background = background.to(device)

    # gsplat expects batched inputs: (C, 4, 4) and (C, 3, 3).
    viewmats = camera.w2c.to(device).unsqueeze(0)  # (1, 4, 4)
    Ks = _intrinsics_matrix(camera).unsqueeze(0)   # (1, 3, 3)

    means = means.contiguous().to(device)
    scales = scales.contiguous().to(device)
    quats = quats.contiguous().to(device)
    colors = colors.contiguous().to(device)
    opacities = opacities.contiguous().to(device)

    render_colors, render_alphas, meta = gsplat.rasterization(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        W,
        H,
        near_plane=near_plane,
        far_plane=far_plane,
        sh_degree=sh_degree,
        backgrounds=background,
        render_mode="RGB",
        rasterize_mode=rasterize_mode,
    )
    # render_colors: (1, H, W, 3) -> (3, H, W)
    rgb = render_colors[0].permute(2, 0, 1).contiguous()
    # render_alphas: (1, H, W, 1) -> (1, H, W)
    alpha = render_alphas[0].permute(2, 0, 1).contiguous()
    return rgb, alpha, meta


@torch.no_grad()
def render_rgb(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    camera: Camera,
    background: Optional[torch.Tensor] = None,
    sh_degree: Optional[int] = 0,
) -> torch.Tensor:
    """Convenience wrapper that returns only the RGB image (3, H, W)."""
    rgb, _, _ = render_gaussians(
        means, scales, quats, colors, opacities, camera,
        background=background, sh_degree=sh_degree,
    )
    return rgb


@torch.no_grad()
def render_depth(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    camera: Camera,
    sh_degree: Optional[int] = 0,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
) -> torch.Tensor:
    """Render an accumulated z-depth map (1, H, W).

    Uses gsplat's ``render_mode='RGB+D'`` which appends the accumulated
    z-depth as an extra channel after the RGB channels.
    """
    device = means.device
    H, W = camera.height, camera.width

    viewmats = camera.w2c.to(device).unsqueeze(0)
    Ks = _intrinsics_matrix(camera).unsqueeze(0)

    render_colors, _render_alphas, _meta = gsplat.rasterization(
        means.contiguous().to(device),
        quats.contiguous().to(device),
        scales.contiguous().to(device),
        opacities.contiguous().to(device),
        colors.contiguous().to(device),
        viewmats,
        Ks,
        W,
        H,
        near_plane=near_plane,
        far_plane=far_plane,
        sh_degree=sh_degree,
        backgrounds=None,  # no background for depth (defaults to zeros)
        render_mode="RGB+D",
    )
    # render_colors: (1, H, W, 4) — last channel is depth.
    depth = render_colors[0, ..., 3:4].permute(2, 0, 1).contiguous()  # (1, H, W)
    return depth


@torch.no_grad()
def render_alpha(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    camera: Camera,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
) -> torch.Tensor:
    """Render a silhouette/alpha map (1, H, W) in [0, 1]."""
    device = means.device
    H, W = camera.height, camera.width

    viewmats = camera.w2c.to(device).unsqueeze(0)
    Ks = _intrinsics_matrix(camera).unsqueeze(0)

    # Use a dummy colour; we only care about the alpha channel.
    dummy_colors = torch.zeros((means.shape[0], 3), dtype=torch.float32, device=device)

    _render_colors, render_alphas, _meta = gsplat.rasterization(
        means.contiguous().to(device),
        quats.contiguous().to(device),
        scales.contiguous().to(device),
        opacities.contiguous().to(device),
        dummy_colors,
        viewmats,
        Ks,
        W,
        H,
        near_plane=near_plane,
        far_plane=far_plane,
        sh_degree=None,
    )
    alpha = render_alphas[0].permute(2, 0, 1).contiguous()  # (1, H, W)
    return alpha
