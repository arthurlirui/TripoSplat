"""Warm-start: single monocular frame -> a drivable 3DGS avatar.

This stage corresponds to Sec. 3.2 of the paper. It (1) runs a feed-forward
Gaussian generator on the first frame to obtain an initial Gaussian set,
(2) estimates SMPL parameters and registers the skeleton, and (3) rigs the
Gaussians to the SMPL skeleton via nearest-vertex skinning-weight
inheritance. The result is a ``SkinnedGaussians`` avatar that can be posed
and rendered immediately.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from skinned_gaussian.smpl_model import SMPLWrapper
from skinned_gaussian.skinned_gaussians import SkinnedGaussians
from multiview_refine.ply_io import load_gaussians_ply


# A feed-forward generator maps an image path (or PIL image) to a Gaussian
# result object exposing .get_xyz / .get_scaling / .get_rotation /
# .get_opacity / .get_features_dc (the TripoSplat ``Gaussian`` interface),
# or directly to a PLY path.
FeedForwardGenerator = Callable[..., object]


def _extract_gaussian_attributes(gen_output) -> tuple:
    """Extract (means, scales_raw, quats, opacities_raw, sh0) from a
    TripoSplat-style Gaussian object, or load from a PLY path string.
    """
    if isinstance(gen_output, str):
        # PLY path.
        means, scales_raw, quats, opacities_raw, sh0 = load_gaussians_ply(gen_output)
        return means, scales_raw, quats, opacities_raw, sh0

    # TripoSplat Gaussian object.
    means = gen_output.get_xyz.detach().float()
    scales = gen_output.get_scaling.detach().float()  # activated
    # invert exp activation to get raw log-scales.
    import torch as _t
    scales_raw = _t.log(scales.clamp_min(1e-6))
    quats = (gen_output._rotation + gen_output.rots_bias[None, :]).detach().float()
    opacities = gen_output.get_opacity.detach().squeeze(-1).float()  # activated
    opacities_raw = _t.log(opacities.clamp_min(1e-6) / (1 - opacities).clamp_min(1e-6))
    sh0 = gen_output._features_dc.detach().float()
    if sh0.ndim == 2:
        sh0 = sh0.unsqueeze(1)  # (N, 1, 3)
    return means, scales_raw, quats, opacities_raw, sh0


def warm_start_avatar(
    generator: FeedForwardGenerator,
    generator_kwargs: dict,
    smpl: SMPLWrapper,
    device: str = "cuda",
    n_neighbours: int = 4,
    temperature: float = 0.05,
    weights_optimizable: bool = True,
) -> tuple[SkinnedGaussians, dict]:
    """Warm-start a drivable avatar from a single frame.

    Args:
        generator:           callable producing a TripoSplat Gaussian or a PLY path.
        generator_kwargs:    kwargs forwarded to ``generator`` (e.g. image path,
                             num_gaussians, remove_background).
        smpl:                an :class:`SMPLWrapper` with the subject's shape.
        device:              torch device.
        n_neighbours:        #nearest SMPL vertices for skinning-weight inheritance.
        temperature:         softmax temperature for distance weighting.
        weights_optimizable: if True, skin weights are trainable (with L2 reg).

    Returns:
        (avatar, info) where ``avatar`` is a :class:`SkinnedGaussians` and
        ``info`` holds diagnostics (n_gaussians, rigging stats).
    """
    # ── 1. Feed-forward generation ──
    gen_output = generator(**generator_kwargs)
    means, scales_raw, quats, opacities_raw, sh0 = _extract_gaussian_attributes(gen_output)
    means = means.to(device)
    scales_raw = scales_raw.to(device)
    quats = quats.to(device)
    opacities_raw = opacities_raw.to(device)
    sh0 = sh0.to(device)

    # ── 2. SMPL rest mesh + skinning weights ──
    rest_verts, _ = smpl.get_rest_mesh()       # (6890, 3)
    smpl_weights = smpl.get_skinning_weights() # (6890, 24)

    # ── 3. Rig: inherit skinning weights from nearest rest-pose vertices ──
    from scipy.spatial import cKDTree
    import torch.nn.functional as F

    rest_np = rest_verts.cpu().numpy()
    means_np = means.cpu().numpy()
    tree = cKDTree(rest_np)
    dist, idx = tree.query(means_np, k=n_neighbours)  # (N, k)

    # distance-softmax weights over neighbours.
    dist_t = torch.as_tensor(dist, dtype=torch.float32, device=device)
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=device)
    nbr_weights = F.softmax(-dist_t / temperature, dim=-1)  # (N, k)

    # gather neighbour skinning weights and blend.
    # smpl_weights: (V, 24) -> gather (N, k, 24)
    nbr_skin = smpl_weights[idx_t]  # (N, k, 24)
    skin_weights = torch.einsum("nk,nkj->nj", nbr_weights, nbr_skin)  # (N, 24)
    skin_weights = skin_weights / skin_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    # ── 4. Build skinned avatar ──
    avatar = SkinnedGaussians.from_attributes(
        means=means,
        scales_raw=scales_raw,
        quats=quats,
        opacities_raw=opacities_raw,
        sh0=sh0,
        skin_weights=skin_weights,
        n_joints=smpl.NUM_JOINTS,
        sh_degree=0,
        device=device,
        weights_optimizable=weights_optimizable,
    )

    info = {
        "n_gaussians": avatar.n_gaussians,
        "n_neighbours": n_neighbours,
        "temperature": temperature,
        "rest_verts": rest_verts.shape[0],
    }
    return avatar, info
