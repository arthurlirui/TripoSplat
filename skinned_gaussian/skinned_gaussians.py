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
        """Create an optimizer over the skin weights.

        Only meaningful when ``_weights_optimizable=True``; otherwise this is a
        no-op.  Kept separate from the Gaussian optimizer so the two can use
        different learning rates.
        """
        if not self._weights_optimizable or self._skin_weights is None:
            return None
        if not hasattr(self, "optimizers"):
            self.optimizers = {}
        opt = torch.optim.Adam(
            [{'params': [self._skin_weights], 'lr': lr, 'name': 'skin_weights'}],
            eps=1e-15)
        self.optimizers['skin_weights'] = opt
        return opt

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

        # Expression basis is part of the rig: without it a reloaded PLY can
        # move the body but cannot animate the face.
        has_expr = self._expr_basis is not None and self._expr_basis.numel() > 0
        if has_expr:
            expr = self._expr_basis.cpu().numpy()  # (N, n_expr, 3)
            n_expr = expr.shape[1]
            # Flatten as (N, 3*n_expr); interleaved so column (j*3+c) is
            # the j-th basis' c-th component.
            expr_flat = expr.reshape(N, -1)

        # Build attribute names.
        names = ['x', 'y', 'z', 'nx', 'ny', 'nz',
                 'f_dc_0', 'f_dc_1', 'f_dc_2',
                 'opacity', 'scale_0', 'scale_1', 'scale_2',
                 'rot_0', 'rot_1', 'rot_2', 'rot_3']
        for k in range(self.n_joints):
            names.append(f'bone_weight_{k}')
        if has_expr:
            for j in range(n_expr):
                for c in range(3):
                    names.append(f'expr_{j}_{c}')

        # Build data array.
        blocks = [means_np, normals, f_dc, opacities.reshape(-1, 1),
                  scales, rotations, weights]
        if has_expr:
            blocks.append(expr_flat)
        data = np.concatenate(blocks, axis=1)

        dtype = [(n, 'f4') for n in names]
        elements = np.empty(N, dtype=dtype)
        elements[:] = list(map(tuple, data))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el], byte_order='<', text=False).write(path)
        print(f"  Saved {N} skinned Gaussians to {path}")

    @classmethod
    def load_ply(cls, path: str, device: str = 'cuda',
                 n_joints: int | None = None,
                 weights_optimizable: bool = False) -> "SkinnedGaussians":
        """Load a rig previously written by :meth:`save_ply`.

        Args:
            path: PLY written by :meth:`save_ply`.
            device: torch device.
            n_joints: joint count, inferred from ``bone_weight_*`` when None.
            weights_optimizable: allow fine-tuning the loaded weights.
        """
        from plyfile import PlyData
        import re

        v = PlyData.read(path)['vertex']
        names = list(v.data.dtype.names)

        def col(name):
            return np.asarray(v[name], dtype=np.float32)

        means = np.stack([col('x'), col('y'), col('z')], axis=-1)
        f_dc = np.stack([col('f_dc_0'), col('f_dc_1'), col('f_dc_2')], axis=-1)
        sh0 = f_dc[:, None, :]
        opacities = col('opacity')
        scales = np.stack([col('scale_0'), col('scale_1'), col('scale_2')], axis=-1)
        quats = np.stack([col('rot_0'), col('rot_1'), col('rot_2'), col('rot_3')],
                         axis=-1)

        bone_cols = [n for n in names if re.fullmatch(r'bone_weight_\d+', n)]
        bone_cols.sort(key=lambda n: int(n.rsplit('_', 1)[1]))
        n_detected = len(bone_cols)
        if n_joints is not None and n_joints > n_detected:
            raise ValueError(
                f"{path} carries {n_detected} bone weights but n_joints={n_joints} "
                "was requested.")
        n_joints = n_detected if n_joints is None else n_joints
        if n_detected == 0:
            raise ValueError(f"{path} has no bone_weight_* properties; "
                             "not a skinned Gaussian PLY.")
        weights = np.stack([col(n) for n in bone_cols[:n_joints]], axis=-1)

        # Expression bases, if present.
        expr_cols = [n for n in names if re.fullmatch(r'expr_\d+_\d+', n)]
        expr_basis = None
        if expr_cols:
            expr_cols.sort(key=lambda n: (int(n.split('_')[1]), int(n.split('_')[2])))
            expr_flat = np.stack([col(n) for n in expr_cols], axis=-1)  # (N, 3*n_expr)
            expr_basis = torch.from_numpy(
                expr_flat.reshape(len(means), -1, 3)).float().to(device)

        return cls.from_attributes(
            means=torch.from_numpy(means).float(),
            scales_raw=torch.from_numpy(scales).float(),
            quats=torch.from_numpy(quats).float(),
            opacities_raw=torch.from_numpy(opacities).float(),
            sh0=torch.from_numpy(sh0).float(),
            skin_weights=torch.from_numpy(weights).float(),
            n_joints=n_joints,
            sh_degree=0,
            device=device,
            expr_basis=expr_basis,
            weights_optimizable=weights_optimizable,
        )

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
