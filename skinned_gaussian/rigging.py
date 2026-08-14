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


def load_static_gaussians(ply_path: str, device: str = 'cuda'):
    """Load a standard 3DGS PLY and return raw attributes.

    Returns: (means, scales_raw, quats, opacities_raw, sh0)
    """
    from plyfile import PlyData

    plydata = PlyData.read(ply_path)
    v = plydata['vertex']
    data = np.stack([v[name] for name in v.data.dtype.names], axis=1)
    names = list(v.data.dtype.names)

    def col(name):
        return data[:, names.index(name)]

    means = np.stack([col('x'), col('y'), col('z')], axis=-1)
    f_dc = np.stack([col('f_dc_0'), col('f_dc_1'), col('f_dc_2')], axis=-1)
    sh0 = f_dc[:, None, :]
    opacities = col('opacity')
    scales = np.stack([col('scale_0'), col('scale_1'), col('scale_2')], axis=-1)
    quats = np.stack([col('rot_0'), col('rot_1'), col('rot_2'), col('rot_3')], axis=-1)

    dev = torch.device(device)
    return (
        torch.from_numpy(means).float().to(dev),
        torch.from_numpy(scales).float().to(dev),
        torch.from_numpy(quats).float().to(dev),
        torch.from_numpy(opacities).float().to(dev),
        torch.from_numpy(sh0).float().to(dev),
    )
