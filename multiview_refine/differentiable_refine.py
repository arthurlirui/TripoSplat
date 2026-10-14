"""Differentiable alignment refinement.

After the coarse silhouette-IoU grid search (:mod:`alignment`) produces an
initial 7-DoF similarity transform ``(scale, R, t)``, this module runs a short
**gradient-based** refinement that optimises the global transform directly
against the multi-view RGB images via gsplat's differentiable rasterizer.

The Gaussian attributes (means, scales, quats, opacities, SH) are **frozen** —
only four parameters are optimised:

* ``log_scale``  (scalar) — log of the uniform scale factor.
* ``q_delta``    (4,)     — quaternion of an incremental rotation applied
  on top of the coarse rotation ``R_coarse``.
* ``t_delta``    (3,)     — additive translation correction.

The effective transform at any point is::

    R   = R_delta(q_delta) @ R_coarse
    s   = exp(log_scale) * scale_coarse
    t   = t_coarse + t_delta

    means_new = s * (R @ means_frozen) + t

This keeps the optimisation well-conditioned: the parameters start at zero
(or identity for the quaternion), so the initial transform equals the coarse
result, and gradients push it toward better RGB alignment.

Loss = ``lambda_rgb`` * (L1 + DSSIM on rendered RGB vs. GT)
     + ``lambda_sil`` * (Dice loss on rendered alpha vs. real mask)

Only ~500–1000 iterations are needed because the coarse alignment already
brings the silhouettes into rough correspondence; the differentiable pass
corrects sub-degree rotations and residual scale/translation error that the
grid search cannot resolve.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cameras import Camera
from .gsplat_wrapper import render_gaussians
from .losses import compute_loss
from .quaternion import rotation_matrix_to_quaternion, quaternion_compose


class SimilarityTransform(nn.Module):
    """Learnable 7-DoF similarity transform applied to frozen Gaussians.

    The transform is parameterised as an incremental correction on top of a
    coarse ``(scale, R, t)``:

    * ``log_scale``  — additive log-scale correction (init 0 ⇒ scale unchanged).
    * ``q_delta``    — incremental rotation quaternion (init = identity).
    * ``t_delta``    — additive translation correction (init 0).

    The effective transform is::

        s = exp(log_scale) * scale_coarse
        R = R(q_delta) @ R_coarse
        t = t_coarse + t_delta
        means_new = s * (R @ means_frozen) + t
    """

    def __init__(
        self,
        scale: float,
        R: torch.Tensor,   # (3, 3)
        t: torch.Tensor,   # (3,)
        device: torch.device | str = "cuda",
    ):
        super().__init__()
        self._scale_coarse = float(scale)
        self.register_buffer("_R_coarse", R.to(device).float())          # (3, 3)
        self.register_buffer("_t_coarse", t.to(device).float())          # (3,)

        # Learnable parameters (initialised at identity / zero correction).
        self.log_scale = nn.Parameter(torch.tensor(0.0, device=device))
        self.q_delta = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)  # identity quat
        )
        self.t_delta = nn.Parameter(torch.zeros(3, device=device))

    # ── Effective transform ─────────────────────────────────────────────

    @property
    def scale(self) -> torch.Tensor:
        """Effective uniform scale (scalar tensor)."""
        return torch.exp(self.log_scale) * self._scale_coarse

    @property
    def R(self) -> torch.Tensor:
        """Effective rotation matrix (3, 3).

        ``R = R(q_delta) @ R_coarse``
        """
        q_norm = F.normalize(self.q_delta, dim=-1)
        # Quaternion to rotation matrix.
        w, x, y, z = q_norm.unbind(-1)
        R_delta = torch.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ]).reshape(3, 3)
        return R_delta @ self._R_coarse

    @property
    def t(self) -> torch.Tensor:
        """Effective translation (3,)."""
        return self._t_coarse + self.t_delta

    def transform_means(self, means: torch.Tensor) -> torch.Tensor:
        """Apply the similarity transform to Gaussian centres.

        ``means_new = s * (R @ means.T).T + t = s * (means @ R.T) + t``
        """
        s = self.scale
        R = self.R
        t = self.t
        return s * (means @ R.T) + t

    def transform_quats(self, quats: torch.Tensor) -> torch.Tensor:
        """Rotate Gaussian quaternions by the effective rotation R.

        ``q_new = q(R) ⊗ q_old``
        """
        q_R = rotation_matrix_to_quaternion(self.R)  # (4,)
        return quaternion_compose(q_R, quats)          # (N, 4)

    def transform_scales(self, scales: torch.Tensor) -> torch.Tensor:
        """Scale Gaussian scales by the effective uniform scale."""
        return scales * self.scale

    def get_params_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(scale, R, t)`` as detached tensors for downstream use."""
        return (
            self.scale.detach(),
            self.R.detach(),
            self.t.detach(),
        )


def _get_real_masks(cameras, rmbg_fn=None, alpha_threshold=0.5):
    """Extract foreground masks from real camera images (reused from alignment)."""
    import numpy as np
    from PIL import Image as PILImage

    masks = []
    for cam in cameras:
        if cam.alpha is not None:
            mask = (cam.alpha.squeeze(0) > alpha_threshold).float()
        elif rmbg_fn is not None:
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
            mask = (cam.image.mean(dim=0) > 0.05).float()
        masks.append(mask)
    return masks


def refine_alignment(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    cameras: list[Camera],
    coarse_transform: tuple[float, torch.Tensor, torch.Tensor],
    rmbg_fn: Callable | None = None,
    num_iterations: int = 800,
    lr_scale: float = 5e-3,
    lr_rotation: float = 5e-4,
    lr_translation: float = 5e-3,
    lambda_rgb: float = 1.0,
    lambda_sil: float = 0.3,
    lambda_l1: float = 0.8,
    lambda_dssim: float = 0.2,
    lambda_reg: float = 0.01,
    sh_degree: int = 0,
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    view_subset: int = 8,
    verbose: bool = True,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Refine a coarse similarity transform via differentiable rendering.

    Optimises ``(log_scale, q_delta, t_delta)`` to minimise a combined
    RGB + silhouette loss across multiple views, with Gaussian attributes
    frozen.

    A regularisation term keeps the incremental parameters close to their
    initial values (identity rotation, zero translation/scale correction),
    preventing the optimiser from collapsing to a degenerate solution
    (e.g. shrinking all Gaussians out of view to trivially minimise a
    silhouette BCE).

    Args:
        means:     (N, 3) frozen Gaussian centres (in the coarse-aligned frame
                   *before* applying the coarse transform — i.e. the original
                   TripoSplat AABB frame).
        scales:    (N, 3) activated scales (frozen).
        quats:     (N, 4) raw quaternions (frozen).
        opacities: (N,) activated opacities (frozen).
        colors:    (N, K, 3) SH coefficients (frozen).
        cameras:   Multi-view cameras with loaded images.
        coarse_transform: ``(scale, R, t)`` from the grid search.
        rmbg_fn:   Background removal function for silhouette masks.
        num_iterations: Refinement iterations.
        lr_scale, lr_rotation, lr_translation: Learning rates.
        lambda_rgb: Weight for the RGB (L1 + DSSIM) loss term.
        lambda_sil: Weight for the silhouette loss term (Dice loss).
        lambda_l1, lambda_dssim: Sub-weights within the RGB loss.
        lambda_reg: Weight for the regularisation term that penalises
                    deviation of (log_scale, q_delta, t_delta) from their
                    initial values.
        sh_degree: SH degree for rendering.
        background_color: Background colour for rendering.
        view_subset: Number of cameras to render per iteration (random subset
                     for efficiency). If 0, uses all cameras.
        verbose: Print progress.

    Returns:
        ``(scale, R, t)`` — the refined similarity transform, same format as
        the coarse transform.
    """
    device = cameras[0].device

    # Freeze Gaussian attributes.
    means = means.detach().to(device).float()
    scales = scales.detach().to(device).float()
    quats = quats.detach().to(device).float()
    opacities = opacities.detach().to(device).float()
    colors = colors.detach().to(device).float()

    # Learnable transform.
    sim = SimilarityTransform(
        coarse_transform[0],
        coarse_transform[1],
        coarse_transform[2],
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": [sim.log_scale], "lr": lr_scale},
            {"params": [sim.q_delta], "lr": lr_rotation},
            {"params": [sim.t_delta], "lr": lr_translation},
        ],
        eps=1e-15,
    )

    bg = torch.tensor(background_color, dtype=torch.float32, device=device)

    # Pre-extract real masks for silhouette loss.
    real_masks = _get_real_masks(cameras, rmbg_fn=rmbg_fn)

    if verbose:
        print(f"  Differentiable refinement: {num_iterations} iters, "
              f"views/iter={view_subset if view_subset > 0 else len(cameras)}")
        s0, R0, t0 = sim.get_params_tuple()
        print(f"  Initial: scale={s0.item():.4f}, t={t0.tolist()}")

    n_cams = len(cameras)
    use_all = view_subset <= 0 or view_subset >= n_cams

    # Initial parameter values for regularisation.
    log_scale_init = sim.log_scale.detach().clone()
    q_delta_init = sim.q_delta.detach().clone()    # [1, 0, 0, 0]
    t_delta_init = sim.t_delta.detach().clone()    # [0, 0, 0]

    for iteration in range(1, num_iterations + 1):
        sim.train()
        optimizer.zero_grad()

        # Transform Gaussians.
        means_t = sim.transform_means(means)
        scales_t = sim.transform_scales(scales)
        quats_t = sim.transform_quats(quats)

        # Select cameras for this iteration.
        if use_all:
            cam_indices = list(range(n_cams))
        else:
            cam_indices = torch.randperm(n_cams)[:view_subset].tolist()

        total_rgb_loss = 0.0
        total_sil_loss = 0.0
        n_views = 0

        for ci in cam_indices:
            cam = cameras[ci]
            rgb, alpha, _ = render_gaussians(
                means_t, scales_t, quats_t, colors, opacities,
                cam, background=bg, sh_degree=sh_degree,
            )

            # RGB loss (L1 + DSSIM).
            gt = cam.image  # (3, H, W)
            rgb_loss = compute_loss(rgb, gt, lambda_l1=lambda_l1, lambda_dssim=lambda_dssim)
            total_rgb_loss = total_rgb_loss + rgb_loss

            # Silhouette loss: Dice loss (not BCE — BCE has a degenerate
            # minimum at alpha=0 when the background dominates the image).
            # Dice = 1 - 2*|α∩m| / (|α| + |m|)  is minimised at full overlap
            # and has no incentive to collapse alpha to zero.
            real_mask = real_masks[ci]  # (H, W) float
            alpha_2d = alpha.squeeze(0)  # (H, W)
            if alpha_2d.shape != real_mask.shape:
                alpha_2d = F.interpolate(
                    alpha_2d.unsqueeze(0).unsqueeze(0),
                    size=real_mask.shape,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()
                real_mask = real_mask.to(alpha_2d.device)
            intersection = (alpha_2d * real_mask).sum()
            denom = alpha_2d.sum() + real_mask.sum() + 1e-6
            dice_loss = 1.0 - 2.0 * intersection / denom
            total_sil_loss = total_sil_loss + dice_loss
            n_views += 1

        # Regularisation: penalise deviation from initial parameters.
        # For the quaternion, use the rotation-invariant geodesic distance
        # ``1 - dot(q, q_init)^2`` instead of raw 4-vector L2, because q and
        # −q represent the same rotation (double cover).  A naive L2 on the
        # 4-vector would incorrectly penalise the sign-flipped identity.
        q_delta_n = F.normalize(sim.q_delta, dim=-1)
        q_init_n = F.normalize(q_delta_init, dim=-1)
        quat_dist = 1.0 - (q_delta_n * q_init_n).sum() ** 2
        reg_loss = (
            (sim.log_scale - log_scale_init) ** 2
            + quat_dist
            + ((sim.t_delta - t_delta_init) ** 2).sum()
        )

        loss = (
            lambda_rgb * total_rgb_loss
            + lambda_sil * total_sil_loss
            + lambda_reg * reg_loss
        ) / max(n_views, 1)
        loss.backward()
        optimizer.step()

        if verbose and (iteration % 100 == 0 or iteration == 1):
            s_val = sim.scale.item()
            t_val = sim.t.detach().tolist()
            print(
                f"    iter {iteration:4d}: loss={loss.item():.4f}, "
                f"rgb={total_rgb_loss.item() / max(n_views, 1):.4f}, "
                f"sil={total_sil_loss.item() / max(n_views, 1):.4f}, "
                f"reg={reg_loss.item():.4f}, "
                f"scale={s_val:.4f}"
            )

    s_final, R_final, t_final = sim.get_params_tuple()
    if verbose:
        print(f"  Refined: scale={s_final.item():.4f}, t={t_final.tolist()}")

    return s_final.item(), R_final, t_final
