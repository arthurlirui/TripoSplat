"""Linear Blend Skinning (LBS) for 3D Gaussian deformation.

Given Gaussians in canonical (rest-pose) space and per-joint transformation
matrices, LBS deforms each Gaussian's position, rotation, and scale to the
posed space.

Core formulas (following GauHuman):
  G_blend = Σ_k w_k * G_k              (blended affine transform, 3×4)
  μ^p = G_blend[:3,:3] @ μ^c + G_blend[:3,3]   (position)
  Σ^p = G_blend[:3,:3] @ Σ^c @ G_blend[:3,:3]^T  (covariance)

For rotation and scale, we polar-decompose G_blend[:3,:3] = R̃ @ S̃:
  q^p = q(R̃) ⊗ q^c     (rotation: compose blended rotation with canonical)
  s^p = diag(S̃) * s^c   (scale: multiply blended scale with canonical)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from multiview_refine.quaternion import quaternion_compose, rotation_matrix_to_quaternion

# SH degree-0 constant (for colour conversion).
SH_C0 = 0.28209479177387814


def lbs_deform_gaussians(
    means_c: torch.Tensor,      # (N, 3) canonical positions
    scales_c: torch.Tensor,     # (N, 3) canonical activated scales
    quats_c: torch.Tensor,      # (N, 4) canonical quaternions (w,x,y,z)
    weights: torch.Tensor,      # (N, K) skinning weights, K=24
    joint_transforms: torch.Tensor,  # (K, 4, 4) per-joint world transforms
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deform canonical Gaussians to posed space via LBS.

    Args:
        means_c: (N, 3) canonical-space Gaussian centres.
        scales_c: (N, 3) canonical-space **activated** scales (not log).
        quats_c: (N, 4) canonical-space quaternions (will be normalized).
        weights: (N, K) skinning weights (should sum to 1 per row).
        joint_transforms: (K, 4, 4) per-joint homogeneous world transforms.

    Returns:
        means_p: (N, 3) posed-space Gaussian centres.
        scales_p: (N, 3) posed-space **activated** scales.
        quats_p: (N, 4) posed-space quaternions (normalized).
    """
    N, K = weights.shape
    device = means_c.device

    # ── 1. Blended transformation: G_blend = Σ_k w_k * G_k ──
    # joint_transforms: (K, 4, 4) → (K, 3, 4) (take 3×4 affine part)
    G_k = joint_transforms[:, :3, :]  # (K, 3, 4)
    # weights: (N, K) → (N, K, 1, 1) for broadcasting
    # G_blend: (N, 3, 4)
    G_blend = torch.einsum('nk,kij->nij', weights, G_k)  # (N, 3, 4)

    # Linear part (rotation + scale combined).
    L = G_blend[:, :, :3]  # (N, 3, 3)
    # Translation part.
    t = G_blend[:, :, 3]   # (N, 3)

    # ── 2. Position deformation ──
    # μ^p = L @ μ^c + t
    means_p = torch.bmm(L, means_c.unsqueeze(-1)).squeeze(-1) + t  # (N, 3)

    # ── 3. Polar decomposition: L = R̃ @ S̃ ──
    # R̃ is the closest rotation matrix to L (orthogonal part).
    # S̃ = R̃^T @ L (symmetric positive-definite, contains scale).
    R_tilde, S_tilde = _polar_decompose(L)  # (N, 3, 3), (N, 3, 3)

    # ── 4. Rotation deformation ──
    # q^p = q(R̃) ⊗ q^c  (compose blended rotation with canonical rotation)
    q_R = rotation_matrix_to_quaternion(R_tilde)  # (N, 4)
    q_c = F.normalize(quats_c, dim=-1)  # (N, 4)
    quats_p = quaternion_compose(q_R, q_c)  # (N, 4)

    # ── 5. Scale deformation ──
    # Take the diagonal of S̃ as the anisotropic scale factor,
    # then multiply with canonical scale.
    # For robustness, we use the full diagonal (not just the norm).
    scale_factor = torch.diagonal(S_tilde, dim1=-2, dim2=-1)  # (N, 3)
    # Clamp to avoid negative or zero scales.
    scale_factor = scale_factor.clamp_min(0.01)
    scales_p = scale_factor * scales_c  # (N, 3)

    return means_p, scales_p, quats_p


def _polar_decompose(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Polar decomposition A = R @ S where R is orthogonal, S is SPD.

    Uses iterative Newton-Schulz method (differentiable, no SVD needed).

    Args:
        A: (N, 3, 3) general matrices.

    Returns:
        R: (N, 3, 3) orthogonal rotation matrices.
        S: (N, 3, 3) symmetric positive-definite scale matrices.
    """
    N = A.shape[0]
    device = A.device

    # Initialize R with A, then iterate.
    R = A.clone()
    for _ in range(10):  # Newton-Schulz iterations
        # R_new = 0.5 * (R + R^T^{-1})
        # For 3×3, use the transpose of inverse.
        R_inv = torch.linalg.inv(R)
        R_new = 0.5 * (R + R_inv.transpose(-2, -1))
        # Check convergence.
        diff = (R_new - R).abs().max()
        R = R_new
        if diff < 1e-6:
            break

    # S = R^T @ A
    S = torch.bmm(R.transpose(-2, -1), A)

    # Ensure S is symmetric (numerical stability).
    S = 0.5 * (S + S.transpose(-2, -1))

    return R, S
