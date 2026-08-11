"""Loss functions and quality metrics for Gaussian splatting refinement.

Implements the standard 3DGS loss: ``lambda_l1 * L1 + lambda_dssim * (1 - SSIM)``
and PSNR for evaluation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# SH degree-0 constant: color = f_dc * C0 + 0.5
SH_C0 = 0.28209479177387814


def compute_loss(
    rendered: torch.Tensor,
    gt: torch.Tensor,
    lambda_l1: float = 0.8,
    lambda_dssim: float = 0.2,
) -> torch.Tensor:
    """Combined L1 + DSSIM loss (the standard 3DGS formulation).

    Args:
        rendered: (3, H, W) or (C, 3, H, W) rendered image.
        gt:       (3, H, W) or (C, 3, H, W) ground-truth image.
        lambda_l1:    weight for L1 term.
        lambda_dssim: weight for DSSIM (1 - SSIM) term.

    Returns:
        Scalar loss tensor.
    """
    l1 = F.l1_loss(rendered, gt)
    ssim_val = ssim(rendered, gt)
    dssim = 1.0 - ssim_val
    return lambda_l1 * l1 + lambda_dssim * dssim


def ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    win_size: int = 11,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Structural Similarity Index (SSIM) between two images.

    Uses a Gaussian-windowed SSIM implementation. Accepts (3, H, W) or
    (C, 3, H, W) tensors.

    Args:
        img1, img2: images to compare, float32 in [0, data_range].
        win_size:   size of the Gaussian window.
        data_range: dynamic range of the images (1.0 for [0,1]).

    Returns:
        Scalar SSIM value (higher is better, max = 1).
    """
    # Ensure 4D: (C, channels, H, W)
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    # Ensure both on same device.
    img2 = img2.to(img1.device)
    # torchmetrics SSIM expects (C, channels, H, W) with float in [0, 1].
    try:
        from torchmetrics.functional.image import structural_similarity_index_measure
    except ImportError:
        from torchmetrics.functional import structural_similarity_index_measure
    return structural_similarity_index_measure(
        img1, img2,
        kernel_size=win_size,
        data_range=data_range,
        reduction="elementwise_mean",
    )


def compute_psnr(
    rendered: torch.Tensor,
    gt: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio between rendered and GT images.

    Args:
        rendered: (3, H, W) rendered image.
        gt:       (3, H, W) ground-truth image.
        data_range: dynamic range (1.0 for [0,1] images).

    Returns:
        Scalar PSNR in dB (higher is better).
    """
    if rendered.ndim == 3:
        rendered = rendered.unsqueeze(0)
        gt = gt.unsqueeze(0)
    gt = gt.to(rendered.device)
    try:
        from torchmetrics.functional.image import peak_signal_noise_ratio
    except ImportError:
        from torchmetrics.functional import peak_signal_noise_ratio
    return peak_signal_noise_ratio(rendered, gt, data_range=data_range)


def sh_dc_to_rgb(sh_dc: torch.Tensor) -> torch.Tensor:
    """Convert SH degree-0 DC coefficients to RGB colours.

    ``rgb = f_dc * C0 + 0.5`` where ``C0 = 0.28209479...``

    Args:
        sh_dc: (N, 1, 3) or (N, 3) SH DC coefficients.

    Returns:
        (N, 3) RGB colours in [0, ∞) (not clamped).
    """
    if sh_dc.ndim == 3:
        sh_dc = sh_dc.squeeze(1)  # (N, 1, 3) -> (N, 3)
    return sh_dc * SH_C0 + 0.5


def rgb_to_sh_dc(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB colours to SH degree-0 DC coefficients.

    Inverse of :func:`sh_dc_to_rgb`: ``f_dc = (rgb - 0.5) / C0``.

    Args:
        rgb: (N, 3) RGB colours.

    Returns:
        (N, 3) SH DC coefficients.
    """
    return (rgb - 0.5) / SH_C0
