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
    (C, 3, H, W) tensors. Implemented directly to avoid torchmetrics
    dependency issues (torchmetrics.text imports transformers).
    """
    # Ensure 4D: (C, channels, H, W)
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    img2 = img2.to(img1.device)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Create Gaussian kernel.
    sigma = 1.5
    coords = torch.arange(win_size, dtype=torch.float32, device=img1.device) - win_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    # 2D kernel: (win_size, win_size)
    kernel2d = g[:, None] * g[None, :]
    # Expand to (channels, 1, win_size, win_size)
    channels = img1.shape[1]
    kernel = kernel2d[None, None].expand(channels, 1, win_size, win_size).contiguous()

    # Compute SSIM maps.
    pad = win_size // 2
    mu1 = F.conv2d(img1, kernel, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=channels)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def compute_psnr(
    rendered: torch.Tensor,
    gt: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio between rendered and GT images.

    Implemented directly to avoid torchmetrics dependency issues.
    """
    gt = gt.to(rendered.device)
    mse = F.mse_loss(rendered, gt)
    if mse < 1e-10:
        return torch.tensor(100.0, device=rendered.device)
    psnr = 10.0 * torch.log10(data_range ** 2 / mse)
    return psnr


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
