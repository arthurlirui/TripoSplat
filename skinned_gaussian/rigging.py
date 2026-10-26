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
from .joints import BODY_POSE_DIM
from multiview_refine.ply_io import load_gaussians_ply


def _posed_vertices(model, betas=None, pose=None, expression=None) -> torch.Tensor:
    """Model vertices after LBS with *pose*, in the model's own frame.

    Used as the nearest-neighbour reference for rigging.  The shipped
    templates are T-posed, so matching a photographed, arms-down subject
    against them assigns the hanging arms to whichever torso vertex happens to
    be closest — every such Gaussian ends up on the spine and the elbow/wrist
    joints receive none.
    """
    if pose is None:
        verts, _ = model.get_rest_mesh(betas)
        if expression is not None and hasattr(model, 'shapedirs_expr'):
            verts, _ = model.get_rest_mesh(betas, expression)
        return verts

    with torch.no_grad():
        if hasattr(model, 'shapedirs_expr') and expression is not None:
            T = model.get_joint_transforms(
                betas=betas, body_pose=pose, expression=expression)
            verts, _ = model.get_rest_mesh(betas, expression)
        else:
            T = model.get_joint_transforms(betas=betas, body_pose=pose)
            verts, _ = model.get_rest_mesh(betas)
        w = model.get_skinning_weights().to(verts.device).float()
        T = T.to(verts.device)
        T_blend = torch.einsum('vk,kij->vij', w, T)
        vh = torch.cat(
            [verts, torch.ones(verts.shape[0], 1, device=verts.device)], dim=-1)
        return torch.einsum('vij,vj->vi', T_blend, vh)[:, :3]


def estimate_rig_pose(
    model,
    means: torch.Tensor,
    betas: torch.Tensor | None = None,
    expression: torch.Tensor | None = None,
    swings: tuple[float, ...] = (-1.1, -0.8, 0.0, 0.8, 1.1),
    n_sample: int = 4000,
) -> tuple[torch.Tensor, dict]:
    """Pick the body pose whose mesh best matches the Gaussian cloud.

    Shoulder swing about Z is the one degree of freedom that matters for
    rigging: the templates are T-posed while a photographed subject stands
    arms-down, so the arm vertices can sit half a metre from any Gaussian.
    Nearest-neighbour weight transfer then hands the hanging arms to torso
    joints and the elbow/wrist joints receive none, so the arms cannot be
    animated at all.

    :func:`~.fit_body_to_gaussians` already searches a few swings, but which
    sign lowers the arms depends on how the model was rotated into the scene
    frame, so the answer cannot be a constant.  This measures it instead: for
    each candidate, pose the mesh, and score the *arm* vertices (those whose
    dominant bone is an elbow or wrist) by distance to the nearest Gaussian.
    Scoring the arms rather than the whole mesh keeps the torso — which
    dominates a full-body vertex count — from deciding the result.

    Args:
        model: SMPL or SMPL-X wrapper, already placed in the scene frame.
        means: (N, 3) Gaussian centres.
        betas: shape parameters for the reference mesh.
        expression: (100,) SMPL-X expression, or None.
        swings: candidate shoulder angles, in radians.
        n_sample: cap on Gaussians used for the distance queries.

    Returns:
        (best body pose (63-dim), diagnostics)
    """
    from .joints import L_SHOULDER, R_SHOULDER, body_pose_slot

    scene = np.asarray(means.detach().cpu().numpy(), dtype=np.float64)
    if len(scene) > n_sample:
        rng = np.random.default_rng(0)
        scene = scene[rng.choice(len(scene), n_sample, replace=False)]
    tree = cKDTree(scene)

    weights = model.get_skinning_weights().detach().cpu().numpy()
    dominant = weights.argmax(axis=1)
    # Elbow and wrist joints; SMPL-X adds no extra arm joints, so these
    # indices are valid for both skeletons.
    arm_joints = (18, 19, 20, 21)
    arm_mask = np.isin(dominant, arm_joints)
    if arm_mask.sum() < 20:
        # No arm vertices found — fall back to the whole mesh.
        arm_mask = np.ones(len(dominant), dtype=bool)

    best = (float("inf"), 0.0)
    scores = {}
    for sw in swings:
        bp = torch.zeros(BODY_POSE_DIM, device=model.device)
        bp[body_pose_slot(L_SHOULDER) + 2] = sw
        bp[body_pose_slot(R_SHOULDER) + 2] = -sw
        verts = _posed_vertices(model, betas, bp, expression).detach().cpu().numpy()
        d, _ = tree.query(verts[arm_mask])
        score = float(np.median(d))
        scores[sw] = score
        if score < best[0]:
            best = (score, sw)

    score, swing = best
    bp = torch.zeros(BODY_POSE_DIM, device=model.device)
    bp[body_pose_slot(L_SHOULDER) + 2] = swing
    bp[body_pose_slot(R_SHOULDER) + 2] = -swing

    diag = {
        "swing_rad": float(swing),
        "swing_deg": float(np.degrees(swing)),
        "arm_median_distance": score,
        "scores": {f"{k:+.2f}": float(v) for k, v in scores.items()},
        "n_arm_vertices": int(arm_mask.sum()),
        "method": "arm_distance_sweep",
    }
    return bp, diag


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
    rig_pose: torch.Tensor | None = None,
    auto_rig_pose: bool = True,
) -> SkinnedGaussians:
    """Bind static Gaussians to the SMPL skeleton.

    **Precondition**: ``smpl.align_to_scene()`` must have been called so that
    the SMPL rest-pose mesh is in the same coordinate frame as ``means``.
    For Gaussians in the OpenCV scene frame (Y-down), pass
    ``rotation=coord_registry.SMPL2OPENCV`` to ``align_to_scene``.

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
        rig_pose: optional body pose (63/69-dim) used to deform the mesh
            before the nearest-neighbour search.  Pass the same pose the body
            was fitted with (e.g. ``joints.a_pose_body_pose()``) so the search
            compares like with like; with the T-pose template a subject
            standing arms-down gets its arms assigned to the torso.
            Skinning weights are a property of the mesh and are unaffected.
        auto_rig_pose: when *rig_pose* is None, measure the best shoulder
            swing with :func:`estimate_rig_pose` instead of using the T-pose.

    Returns:
        SkinnedGaussians with initialized skinning weights.
    """
    print("Rigging Gaussians to SMPL skeleton...")

    if rig_pose is None and auto_rig_pose:
        rig_pose, pose_diag = estimate_rig_pose(smpl, means, betas)
        print(f"  Auto rig pose: shoulders {pose_diag['swing_deg']:+.0f}° "
              f"(arm distance {pose_diag['arm_median_distance']:.4f})")

    # Reference mesh for the nearest-vertex search.
    rest_verts = _posed_vertices(smpl, betas, rig_pose)  # (6890, 3)
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
    rig_pose: torch.Tensor | None = None,
    auto_rig_pose: bool = True,
) -> SkinnedGaussians:
    """Bind Gaussians to SMPL-X skeleton with expression blendshape support.

    **Precondition**: ``smplx.align_to_scene()`` must have been called so that
    the SMPL-X rest-pose mesh is in the same coordinate frame as ``means``.

    In addition to skinning weights (55 joints), this also computes per-Gaussian
    expression displacement bases by interpolating from SMPL-X mesh vertices.

    Args:
        smplx: SMPLXWrapper instance.
        expression: (100,) expression coefficients for rest-pose mesh.
        n_expression_bases: number of expression blendshape bases to store.
        weights_optimizable: if True, skin weights can be fine-tuned during
            optimization (with L2 regularization to SMPL-X prior).
        rig_pose: optional body pose (63/69-dim) deforming the mesh before the
            nearest-neighbour search; see :func:`rig_gaussians_to_smpl`.

    Returns:
        SkinnedGaussians with 55-joint skinning weights + expression bases.
    """
    print("Rigging Gaussians to SMPL-X skeleton (55 joints + expression)...")

    if rig_pose is None and auto_rig_pose:
        rig_pose, pose_diag = estimate_rig_pose(smplx, means, betas, expression)
        print(f"  Auto rig pose: shoulders {pose_diag['swing_deg']:+.0f}° "
              f"(arm distance {pose_diag['arm_median_distance']:.4f})")

    # Reference mesh for the nearest-vertex search.
    rest_verts = _posed_vertices(smplx, betas, rig_pose, expression)  # (10475, 3)
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
