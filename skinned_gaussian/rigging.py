"""Rigging — bind static Gaussians to the SMPL skeleton.

For each Gaussian, finds the nearest SMPL mesh vertices and inherits
their skinning weights, weighted by inverse distance.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from .smpl_model import SMPLWrapper
from .skinned_gaussians import SkinnedGaussians
from multiview_refine.ply_io import load_gaussians_ply

def rig_gaussians_to_smpl(
    means: torch.Tensor,          # (N, 3) static Gaussian positions
    scales_raw: torch.Tensor,     # (N, 3) log-space scales
    quats: torch.Tensor,          # (N, 4) quaternions
    opacities_raw: torch.Tensor,  # (N,) logit opacities
    sh0: torch.Tensor,            # (N, 1, 3) SH DC
    smpl: SMPLWrapper,
    betas: torch.Tensor | None = None,
    n_neighbours: int = 4,
    temperature: float = 0.05,
    device: str = 'cuda',
) -> SkinnedGaussians:
    """Bind static Gaussians to the SMPL skeleton.

    For each Gaussian:
      1. Find the n_neighbours closest SMPL rest-pose vertices.
      2. Compute distance-based weights: w_i = exp(-dist_i / temperature).
      3. Inherit skinning weights: W_gs = Σ_i w_i * W_smpl[nearest_i].
      4. Normalize so that Σ_k W_gs[k] = 1.

    Args:
        means: (N, 3) static Gaussian centres (world space).
        scales_raw, quats, opacities_raw, sh0: Gaussian attributes.
        smpl: SMPLWrapper instance.
        betas: (10,) shape parameters for rest-pose mesh.
        n_neighbours: number of nearest SMPL vertices to consider.
        temperature: softmax temperature for distance weighting.
        device: torch device.

    Returns:
        SkinnedGaussians with initialized skinning weights.
    """
    print("Rigging Gaussians to SMPL skeleton...")

    # Get SMPL rest-pose mesh.
    rest_verts, _ = smpl.get_rest_mesh(betas)  # (6890, 3)
    smpl_weights = smpl.get_skinning_weights()  # (6890, 24)

    N = means.shape[0]
    rest_verts_np = rest_verts.cpu().numpy()
    smpl_weights_np = smpl_weights.cpu().numpy()
    means_np = means.cpu().numpy()

    # Build KD-tree on SMPL vertices.
    tree = cKDTree(rest_verts_np)

    # Find n_neighbours nearest vertices for each Gaussian.
    dists, indices = tree.query(means_np, k=n_neighbours)  # (N, K_nn), (N, K_nn)

    # Distance-based weighting: closer vertices contribute more.
    if n_neighbours == 1:
        dists = dists.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    # Softmax over inverse distances.
    logits = -dists / max(temperature, 1e-8)  # (N, K_nn)
    logits = logits - logits.max(axis=-1, keepdims=True)  # numerical stability
    exp_logits = np.exp(logits)
    neighbour_weights = exp_logits / exp_logits.sum(axis=-1, keepdims=True)  # (N, K_nn)

    # Inherit skinning weights: weighted average of neighbour vertex weights.
    # smpl_weights[indices]: (N, K_nn, 24)
    neighbour_smpl_weights = smpl_weights_np[indices]  # (N, K_nn, 24)
    # Weighted sum: (N, 24) = Σ_i w_i * smpl_w_i
    skin_weights = np.einsum('nk,nkj->nj', neighbour_weights, neighbour_smpl_weights)  # (N, 24)

    # Normalize (should already sum to 1, but enforce for numerical stability).
    row_sums = skin_weights.sum(axis=-1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    skin_weights = skin_weights / row_sums

    # Stats.
    n_rigid = (skin_weights.max(axis=-1) > 0.99).sum()
    print(f"  {N} Gaussians rigged to {smpl.NUM_JOINTS} joints")
    print(f"  Neighbours: {n_neighbours}, temperature: {temperature}")
    print(f"  Rigid (single-joint) Gaussians: {n_rigid} ({100*n_rigid/N:.1f}%)")
    print(f"  Mean max weight: {skin_weights.max(axis=-1).mean():.4f}")

    # Create SkinnedGaussians.
    skin_weights_t = torch.from_numpy(skin_weights).float().to(device)
    return SkinnedGaussians.from_attributes(
        means=means.to(device),
        scales_raw=scales_raw.to(device),
        quats=quats.to(device),
        opacities_raw=opacities_raw.to(device),
        sh0=sh0.to(device),
        skin_weights=skin_weights_t,
        n_joints=smpl.NUM_JOINTS,
        sh_degree=0,
        device=device,
    )


def rig_gaussians_to_smplx(
    means: torch.Tensor,
    scales_raw: torch.Tensor,
    quats: torch.Tensor,
    opacities_raw: torch.Tensor,
    sh0: torch.Tensor,
    smplx,                          # SMPLXWrapper
    betas: torch.Tensor | None = None,
    expression: torch.Tensor | None = None,
    n_neighbours: int = 4,
    temperature: float = 0.05,
    device: str = 'cuda',
    weights_optimizable: bool = True,
    n_expression_bases: int = 100,
) -> SkinnedGaussians:
    """Bind Gaussians to SMPL-X skeleton with expression blendshape support.

    In addition to skinning weights (55 joints), this also computes per-Gaussian
    expression displacement bases by interpolating from SMPL-X mesh vertices.

    Args:
        smplx: SMPLXWrapper instance.
        expression: (100,) expression coefficients for rest-pose mesh.
        n_expression_bases: number of expression blendshape bases to store.
        weights_optimizable: if True, skin weights can be fine-tuned during
            optimization (with L2 regularization to SMPL-X prior).

    Returns:
        SkinnedGaussians with 55-joint skinning weights + expression bases.
    """
    print("Rigging Gaussians to SMPL-X skeleton (55 joints + expression)...")

    # Get SMPL-X rest-pose mesh (with expression applied if given).
    rest_verts, _ = smplx.get_rest_mesh(betas, expression)  # (10475, 3)
    smpl_weights = smplx.get_skinning_weights()  # (10475, 55)

    N = means.shape[0]
    rest_verts_np = rest_verts.cpu().numpy()
    smpl_weights_np = smpl_weights.cpu().numpy()
    means_np = means.cpu().numpy()

    # Build KD-tree on SMPL-X vertices.
    tree = cKDTree(rest_verts_np)
    dists, indices = tree.query(means_np, k=n_neighbours)

    if n_neighbours == 1:
        dists = dists.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    # Softmax over inverse distances.
    logits = -dists / max(temperature, 1e-8)
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    neighbour_weights = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    # Inherit 55-joint skinning weights.
    neighbour_smpl_weights = smpl_weights_np[indices]  # (N, K_nn, 55)
    skin_weights = np.einsum('nk,nkj->nj', neighbour_weights, neighbour_smpl_weights)
    row_sums = np.maximum(skin_weights.sum(axis=-1, keepdims=True), 1e-8)
    skin_weights = skin_weights / row_sums

    # Interpolate expression blendshape bases from SMPL-X mesh.
    # shapedirs_expr: (10475, 3, 100) → per-vertex expression displacement.
    # For each Gaussian, interpolate the expression bases from nearest vertices.
    smpl_expr_bases = smplx.shapedirs_expr.cpu().numpy()  # (10475, 3, 100)
    # neighbour expr bases: (N, K_nn, 3, 100)
    neighbour_expr = smpl_expr_bases[indices]  # (N, K_nn, 3, n_expr)
    # Weighted interpolation: (N, 3, n_expr) = Σ w_i * expr_i
    n_expr = min(n_expression_bases, smpl_expr_bases.shape[2])
    expr_basis = np.einsum('nk,nkje->nje', neighbour_weights, neighbour_expr[:, :, :, :n_expr])
    # Transpose to (N, n_expr, 3) for storage.
    expr_basis = expr_basis.transpose(0, 2, 1)  # (N, n_expr, 3)

    # Stats.
    n_rigid = (skin_weights.max(axis=-1) > 0.99).sum()
    print(f"  {N} Gaussians rigged to {smplx.NUM_JOINTS} joints")
    print(f"  Expression bases: {n_expr}")
    print(f"  Rigid (single-joint) Gaussians: {n_rigid} ({100*n_rigid/N:.1f}%)")
    print(f"  Mean max weight: {skin_weights.max(axis=-1).mean():.4f}")
    print(f"  Expression basis norm: mean={np.linalg.norm(expr_basis, axis=(1,2)).mean():.6f}")
    print(f"  Weights optimizable: {weights_optimizable}")

    # Create SkinnedGaussians with expression + optimizable weights.
    skin_weights_t = torch.from_numpy(skin_weights).float().to(device)
    expr_basis_t = torch.from_numpy(expr_basis).float().to(device)
    return SkinnedGaussians.from_attributes(
        means=means.to(device),
        scales_raw=scales_raw.to(device),
        quats=quats.to(device),
        opacities_raw=opacities_raw.to(device),
        sh0=sh0.to(device),
        skin_weights=skin_weights_t,
        n_joints=smplx.NUM_JOINTS,
        sh_degree=0,
        device=device,
        expr_basis=expr_basis_t,
        weights_optimizable=weights_optimizable,
    )
