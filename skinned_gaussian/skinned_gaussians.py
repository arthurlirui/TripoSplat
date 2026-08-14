"""Skinned Gaussians — 3D Gaussians with SMPL skinning weights.

Stores Gaussian attributes in canonical (rest-pose) space and per-Gaussian
skinning weights. Given joint transformations, deforms Gaussians to posed
space via LBS and renders with gsplat.
"""

from __future__ import annotations

import os
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .skinning import lbs_deform_gaussians, _quaternion_compose_batch, _rotation_matrix_to_quaternion_batch

SH_C0 = 0.28209479177387814


class SkinnedGaussians(nn.Module):
    """3D Gaussians with SMPL skinning weights, defined in canonical space.

    Parameters (all in canonical / rest-pose space):
      _means:         (N, 3)  Gaussian centres.
      _scales:        (N, 3)  log-space scales.
      _quats:         (N, 4)  raw quaternions (w, x, y, z).
      _opacities:     (N,)    logit opacities.
      _sh0:           (N, 1, 3)  SH degree-0 DC coefficients.
      _skin_weights:  (N, K)  raw skinning weights (softmax-normalized on use).
    """

    def __init__(self, n_joints: int = 24, sh_degree: int = 0, device: str = 'cuda'):
        super().__init__()
        self.n_joints = n_joints
        self.sh_degree = sh_degree
        self.device = torch.device(device)

        # Parameters (allocated in from_static / from_ply).
        self._means: nn.Parameter | None = None
        self._scales: nn.Parameter | None = None
        self._quats: nn.Parameter | None = None
        self._opacities: nn.Parameter | None = None
        self._sh0: nn.Parameter | None = None
        self._skin_weights: nn.Parameter | None = None

    # ── Properties ────────────────────────────────────────────────────

    @property
    def means(self) -> torch.Tensor:
        return self._means

    @property
    def scales(self) -> torch.Tensor:
        return self._scales.exp()

    @property
    def raw_scales(self) -> torch.Tensor:
        return self._scales

    @property
    def quats(self) -> torch.Tensor:
        return F.normalize(self._quats, dim=-1)

    @property
    def raw_quats(self) -> torch.Tensor:
        return self._quats

    @property
    def opacities(self) -> torch.Tensor:
        return torch.sigmoid(self._opacities)

    @property
    def skin_weights(self) -> torch.Tensor:
        """Normalized skinning weights (sum to 1 per Gaussian)."""
        return F.softmax(self._skin_weights, dim=-1)

    @property
    def n_gaussians(self) -> int:
        return self._means.shape[0] if self._means is not None else 0

    # ── Initialization ────────────────────────────────────────────────

    @classmethod
    def from_attributes(
        cls,
        means: torch.Tensor,         # (N, 3)
        scales_raw: torch.Tensor,    # (N, 3) log-space
        quats: torch.Tensor,         # (N, 4)
        opacities_raw: torch.Tensor, # (N,) logit
        sh0: torch.Tensor,           # (N, 1, 3)
        skin_weights: torch.Tensor,  # (N, K) raw (will be stored as logits)
        n_joints: int = 24,
        sh_degree: int = 0,
        device: str = 'cuda',
    ) -> "SkinnedGaussians":
        """Create from explicit attribute tensors."""
        model = cls(n_joints=n_joints, sh_degree=sh_degree, device=device)
        model._means = nn.Parameter(means.float().to(device))
        model._scales = nn.Parameter(scales_raw.float().to(device))
        model._quats = nn.Parameter(quats.float().to(device))
        model._opacities = nn.Parameter(opacities_raw.float().to(device))
        model._sh0 = nn.Parameter(sh0.float().to(device))
        # Store skin weights as logits (so softmax gives normalized weights).
        weights_norm = skin_weights.float().to(device)
        weights_norm = weights_norm / weights_norm.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        # log(w) as raw parameter → softmax(log(w)) = w (approximately).
        model._skin_weights = nn.Parameter(torch.log(weights_norm.clamp_min(1e-8)))
        return model

    # ── Deformation ───────────────────────────────────────────────────

    def deform(self, joint_transforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deform canonical Gaussians to posed space via LBS.

        Args:
            joint_transforms: (K, 4, 4) per-joint world transforms.

        Returns:
            means_p (N, 3), scales_p (N, 3 activated), quats_p (N, 4 normalized).
        """
        return lbs_deform_gaussians(
            self.means,
            self.scales,        # activated
            self.raw_quats,     # raw (will be normalized in deform)
            self.skin_weights,  # softmax-normalized
            joint_transforms.to(self.device),
        )

    def get_colors(self) -> torch.Tensor:
        """Return SH coefficients for rendering."""
        return self._sh0  # (N, 1, 3)

    def render_posed(
        self,
        joint_transforms: torch.Tensor,
        camera,
        background: Optional[torch.Tensor] = None,
        sh_degree: int = 0,
    ):
        """Deform + render in one call.

        Returns (rgb, alpha, meta) from gsplat.
        """
        from multiview_refine.gsplat_wrapper import render_gaussians

        means_p, scales_p, quats_p = self.deform(joint_transforms)
        return render_gaussians(
            means_p, scales_p, quats_p,
            self.get_colors(), self.opacities,
            camera, background=background, sh_degree=sh_degree,
        )

    # ── Export ────────────────────────────────────────────────────────

    @torch.no_grad()
    def save_ply(self, path: str):
        """Save to PLY with skinning weights as extra vertex properties.

        Extended attribute list:
          x, y, z, nx, ny, nz, f_dc_0..2, opacity, scale_0..2, rot_0..3,
          bone_weight_0..K-1
        """
        from plyfile import PlyData, PlyElement

        N = self.n_gaussians
        means_np = self._means.cpu().numpy()
        normals = np.zeros_like(means_np)
        f_dc = self._sh0[:, 0, :].cpu().numpy()  # (N, 3)
        opacities = self._opacities.cpu().numpy()
        scales = self._scales.cpu().numpy()
        rotations = F.normalize(self._quats, dim=-1).cpu().numpy()
        weights = self.skin_weights.cpu().numpy()  # (N, K)

        # Build attribute names.
        names = ['x', 'y', 'z', 'nx', 'ny', 'nz',
                 'f_dc_0', 'f_dc_1', 'f_dc_2',
                 'opacity', 'scale_0', 'scale_1', 'scale_2',
                 'rot_0', 'rot_1', 'rot_2', 'rot_3']
        for k in range(self.n_joints):
            names.append(f'bone_weight_{k}')

        # Build data array.
        data = np.concatenate([
            means_np, normals, f_dc, opacities.reshape(-1, 1),
            scales, rotations, weights,
        ], axis=1)

        dtype = [(n, 'f4') for n in names]
        elements = np.empty(N, dtype=dtype)
        elements[:] = list(map(tuple, data))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el], byte_order='<', text=False).write(path)
        print(f"  Saved {N} skinned Gaussians to {path}")

    @torch.no_grad()
    def save_splat(self, path: str, joint_transforms: Optional[torch.Tensor] = None):
        """Save to .splat format (posed if joint_transforms given, canonical otherwise).

        Args:
            path: output .splat file path.
            joint_transforms: (K, 4, 4) if saving posed Gaussians, None for canonical.
        """
        import struct

        if joint_transforms is not None:
            means_p, scales_p, quats_p = self.deform(joint_transforms)
        else:
            means_p = self.means
            scales_p = self.scales
            quats_p = self.quats

        means_np = means_p.cpu().numpy()
        scales_np = scales_p.cpu().numpy()
        opacities_np = self.opacities.cpu().numpy()
        quats_np = F.normalize(quats_p, dim=-1).cpu().numpy()
        rgb = (self._sh0[:, 0, :].cpu().numpy() * SH_C0 + 0.5).clip(0, 1)

        # Sort by descending opacity * prod(scale).
        sort_key = opacities_np * scales_np.prod(axis=-1)
        order = np.argsort(-sort_key)
        means_np = means_np[order]
        scales_np = scales_np[order]
        opacities_np = opacities_np[order]
        quats_np = quats_np[order]
        rgb = rgb[order]

        rgba = (np.concatenate([rgb, opacities_np.reshape(-1, 1)], axis=1)
                .clip(0, 1) * 255).astype(np.uint8)
        quats_u8 = np.clip(quats_np * 128 + 128, 0, 255).astype(np.uint8)

        with open(path, 'wb') as f:
            for i in range(len(means_np)):
                f.write(means_np[i].astype(np.float32).tobytes())  # 12B
                f.write(scales_np[i].astype(np.float32).tobytes())  # 12B
                f.write(rgba[i].tobytes())  # 4B
                f.write(quats_u8[i].tobytes())  # 4B
        print(f"  Saved {len(means_np)} splats to {path}")
