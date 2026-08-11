"""Multi-view refinement of TripoSplat-initialized 3D Gaussians.

This package takes a single-view TripoSplat generation, aligns it to a set of
real multi-view cameras (Nerfstudio transforms.json), and refines the Gaussians
via differentiable rasterization (gsplat) optimization.
"""

from .cameras import (
    Camera,
    load_cameras,
    find_camera_by_filename,
    compute_cameras_extent,
    split_train_test,
)
from .gsplat_wrapper import render_gaussians, render_rgb, render_depth, render_alpha
from .gaussian_model import RefineableGaussians
from .losses import compute_loss, compute_psnr, ssim, sh_dc_to_rgb, rgb_to_sh_dc
from .alignment import estimate_similarity_transform
from .refine import refine_from_multiview

__all__ = [
    "Camera",
    "load_cameras",
    "find_camera_by_filename",
    "compute_cameras_extent",
    "split_train_test",
    "render_gaussians",
    "render_rgb",
    "render_depth",
    "render_alpha",
    "RefineableGaussians",
    "compute_loss",
    "compute_psnr",
    "ssim",
    "sh_dc_to_rgb",
    "rgb_to_sh_dc",
    "estimate_similarity_transform",
    "refine_from_multiview",
]
