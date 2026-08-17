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

from .skinning import lbs_deform_gaussians
from multiview_refine.quaternion import quaternion_compose, rotation_matrix_to_quaternion

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

    def __init__(self, n_joints: int = 24, sh_degree: int = 0, device: str = 'cuda',
                 n_expression: int = 0):
        super().__init__()
        self.n_joints = n_joints
        self.sh_degree = sh_degree
        self.n_expression = n_expression  # number of expression blendshape bases
        self.device = torch.device(device)

        # Parameters (allocated in from_attributes / from_ply).
        self._means: nn.Parameter | None = None
        self._scales: nn.Parameter | None = None
        self._quats: nn.Parameter | None = None
        self._opacities: nn.Parameter | None = None
        self._sh0: nn.Parameter | None = None
        self._skin_weights: nn.Parameter | None = None
        # Expression blendshape basis: (N, n_expression, 3)
        # Means deformation: Δμ = Σ α_i * basis_i
        self._expr_basis: nn.Parameter | None = None
        # Whether skin weights are optimizable (requires_grad=True for joint optimization).
        self._weights_optimizable = False
        # Reference weights for regularization (fixed, from SMPL).
        self._weights_ref: torch.Tensor | None = None

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
        expr_basis: torch.Tensor | None = None,  # (N, n_expr, 3) expression displacement bases
        weights_optimizable: bool = False,         # if True, skin weights can be fine-tuned
    ) -> "SkinnedGaussians":
        """Create from explicit attribute tensors.

        Args:
            expr_basis: (N, n_expression, 3) per-Gaussian expression displacement bases.
                When provided, expression coefficients α can deform means:
                μ_posed = μ_canonical + Σ α_i * expr_basis_i
            weights_optimizable: if True, skin weights are trainable and a
                regularization loss against the initial SMPL weights is used.
        """
        n_expr = expr_basis.shape[1] if expr_basis is not None else 0
        model = cls(n_joints=n_joints, sh_degree=sh_degree, device=device,
                     n_expression=n_expr)
        model._means = nn.Parameter(means.float().to(device))
        model._scales = nn.Parameter(scales_raw.float().to(device))
        model._quats = nn.Parameter(quats.float().to(device))
        model._opacities = nn.Parameter(opacities_raw.float().to(device))
        model._sh0 = nn.Parameter(sh0.float().to(device))
        # Store skin weights as logits (so softmax gives normalized weights).
        weights_norm = skin_weights.float().to(device)
        weights_norm = weights_norm / weights_norm.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        model._skin_weights = nn.Parameter(torch.log(weights_norm.clamp_min(1e-8)))
        # Expression basis (fixed, not optimized in MVP).
        if expr_basis is not None:
            model._expr_basis = nn.Parameter(expr_basis.float().to(device),
                                              requires_grad=False)
        # Weight optimization setup.
        model._weights_optimizable = weights_optimizable
        if weights_optimizable:
            # Store reference weights for regularization (detached copy).
            model._weights_ref = model.skin_weights.detach().clone()
        return model

    # ── Deformation ───────────────────────────────────────────────────

    def deform(self, joint_transforms: torch.Tensor,
               expression: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deform canonical Gaussians to posed space via LBS + expression.

        Args:
            joint_transforms: (K, 4, 4) per-joint relative transforms.
            expression: (n_expression,) expression coefficients. If provided
                and _expr_basis exists, adds expression displacement to means.

        Returns:
            means_p (N, 3), scales_p (N, 3 activated), quats_p (N, 4 normalized).
        """
        # Start with canonical means.
        means_c = self.means

        # Apply expression blendshape displacement (if available).
        if expression is not None and self._expr_basis is not None and self._expr_basis.numel() > 0:
            # Δμ = Σ α_i * basis_i  →  (N, 3)
            expr_disp = torch.einsum('nei,e->ni', self._expr_basis, expression)
            means_c = means_c + expr_disp

        return lbs_deform_gaussians(
            means_c,
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
        expression: torch.Tensor | None = None,
    ):
        """Deform + render in one call.

        Args:
            expression: (n_expression,) expression coefficients for facial animation.

        Returns (rgb, alpha, meta) from gsplat.
        """
        from multiview_refine.gsplat_wrapper import render_gaussians

        means_p, scales_p, quats_p = self.deform(joint_transforms, expression)
        return render_gaussians(
            means_p, scales_p, quats_p,
            self.get_colors(), self.opacities,
            camera, background=background, sh_degree=sh_degree,
        )

    def weight_regularization_loss(self, lambda_reg: float = 0.01) -> torch.Tensor:
        """L2 regularization on skin weights deviation from SMPL prior.

        Encourages optimized weights to stay close to the initial SMPL-inherited
        weights. Only effective when _weights_optimizable=True.

        Returns:
            Scalar loss tensor.
        """
        if not self._weights_optimizable or self._weights_ref is None:
            return torch.tensor(0.0, device=self.device)
        w = self.skin_weights  # (N, K) softmax-normalized, differentiable
        return lambda_reg * ((w - self._weights_ref) ** 2).mean()

    def training_setup_weights(self, lr: float = 1e-3):
        """Add skin weights to the optimizer for joint weight optimization.

        Call this after training_setup() if _weights_optimizable=True.
        """
        if self._weights_optimizable and self._skin_weights is not None:
            # Add skin weights to a separate optimizer group.
            self.optimizers['skin_weights'] = torch.optim.Adam(
                [{'params': [self._skin_weights], 'lr': lr, 'name': 'skin_weights'}],
                eps=1e-15)

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
