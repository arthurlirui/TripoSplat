"""Fit a SMPL/SMPL-X body model to a Gaussian point cloud.

Rigging needs the body model to sit **inside** the Gaussians, in the same
frame.  Downstream, ``rig_gaussians_to_smpl`` assigns skinning weights by
nearest-vertex lookup, so a mis-placed skeleton silently produces weights for
the wrong limb — the animation runs, but the mesh falls apart.

This module recovers the similarity transform::

    xyz_scene = scale * (R @ xyz_model) + t

from the Gaussians alone, in four stages:

1. **Up axis** — the point cloud's principal axis weighted by how much it is
   elongated (a standing body is tall).  Cameras, when supplied, fix the sign,
   because "up" in the world is ambiguous from geometry alone.
2. **Scale** — the model's height along its own up axis against the cloud's
   extent along the estimated up axis.
3. **Yaw** — a brute-force sweep minimising trimmed Chamfer distance.  A
   sweep, not an eigenvector, because the RGB-D cloud of a person is close to
   a shell of revolution and its horizontal PCA axes do not track shoulder
   orientation.
4. **ICP** — trimmed nearest-neighbour correspondences with an Umeyama
   similarity fit, which removes the residual error of the three closed-form
   stages.

Every stage reports diagnostics; :func:`fit_body_to_gaussians` returns the
transform together with a Chamfer-distance report so callers can gate on fit
quality instead of assuming it worked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from scipy.spatial import cKDTree

from .joints import (
    SMPL_NUM_JOINTS,
    a_pose_body_pose,
    L_FOOT,
    R_FOOT,
    HEAD,
)

# The body model's own up axis is +Y in the SMPL/SMPL-X rest frame.
SMPL_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)


# ── Result ──────────────────────────────────────────────────────────────────

@dataclass
class BodyFitResult:
    """Similarity transform placing a body model inside a Gaussian cloud.

    Attributes:
        scale: uniform scale factor.
        R: (3, 3) rotation, model frame → scene frame.
        t: (3,) translation, in scene coordinates.
        up_axis: (3,) estimated up direction in scene coordinates.
        chamfer_model_to_scene: mean nearest-neighbour distance (scene units)
            from transformed model vertices to the Gaussian cloud.
        chamfer_scene_to_model: the reverse direction, which is larger when
            the cloud carries things the body model does not (hair, loose
            clothing).
        diagnostics: per-stage values, for logging or gating.
    """

    scale: float
    R: torch.Tensor
    t: torch.Tensor
    up_axis: np.ndarray
    chamfer_model_to_scene: float = float("nan")
    chamfer_scene_to_model: float = float("nan")
    diagnostics: dict = field(default_factory=dict)

    def matrix(self) -> torch.Tensor:
        """4×4 homogeneous similarity matrix (scale folded into the rotation)."""
        M = torch.eye(4, dtype=self.R.dtype, device=self.R.device)
        M[:3, :3] = self.scale * self.R
        M[:3, 3] = self.t
        return M

    def apply(self, points: torch.Tensor) -> torch.Tensor:
        """Map *points* from the model frame to the scene frame."""
        return self.scale * (points @ self.R.T) + self.t

    def to_dict(self) -> dict:
        return {
            "scale": float(self.scale),
            "R": self.R.detach().cpu().tolist(),
            "t": self.t.detach().cpu().tolist(),
            "matrix": self.matrix().detach().cpu().tolist(),
            "up_axis": np.asarray(self.up_axis).tolist(),
            "chamfer_model_to_scene": float(self.chamfer_model_to_scene),
            "chamfer_scene_to_model": float(self.chamfer_scene_to_model),
            "diagnostics": self.diagnostics,
            "convention": "xyz_scene = scale * (R @ xyz_model) + t",
        }


# ── Geometry primitives ─────────────────────────────────────────────────────

def _subsample(points: np.ndarray, n_max: int, seed: int = 0) -> np.ndarray:
    """Uniform random subset, for bounding the cost of repeated kNN queries."""
    if len(points) <= n_max:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), n_max, replace=False)
    return points[idx]


def _pca_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Principal axes of a point cloud.

    Returns:
        (mean, axes, eigenvalues) where *axes* are columns sorted by
        descending eigenvalue.
    """
    mu = points.mean(axis=0)
    X = points - mu
    C = (X.T @ X) / max(len(X) - 1, 1)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    return mu, evecs[:, order], evals[order]


def estimate_up_axis(
    means: torch.Tensor | np.ndarray,
    cameras: Sequence | None = None,
    n_sample: int = 20000,
) -> tuple[np.ndarray, dict]:
    """Estimate the subject's up direction in scene coordinates.

    The long axis of a standing body is its vertical axis, so the cloud's
    principal axis is the candidate.  Geometry alone cannot say which end is
    the head, so when *cameras* are available the sign is chosen by projecting
    the two ends: the end that lands higher in the image is the head.

    Args:
        means: (N, 3) Gaussian centres, scene frame.
        cameras: optional cameras used to disambiguate the sign.  Pointing both
            up and down is degenerate when the capture is side-on, so agreement
            is reported and a low value means the sign is unreliable.
        n_sample: cap on points used for the PCA.

    Returns:
        (unit up vector, diagnostics)

    Raises:
        ValueError: if the cloud is degenerate (fewer than 3 points) or nearly
            isotropic, in which case there is no dominant axis to find.
    """
    pts = np.asarray(means.detach().cpu().numpy() if torch.is_tensor(means) else means,
                     dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        raise ValueError(f"need at least 3 points, got shape {pts.shape}")

    pts = _subsample(pts, n_sample)
    mu, axes, evals = _pca_axes(pts)
    up = axes[:, 0]

    elongation = math.sqrt(max(evals[0], 0.0) / max(evals[1], 1e-12))
    diag: dict = {
        "elongation": float(elongation),
        "eigenvalues": evals.tolist(),
        "n_points": int(len(pts)),
        "method": "pca_longest_axis",
    }
    if elongation < 1.15:
        diag["warning"] = (
            f"Point cloud is nearly isotropic (elongation={elongation:.2f}); "
            "the long axis may not be the body's vertical axis. "
            "Pass an explicit up axis if the subject is not standing."
        )

    if cameras:
        # Project both candidate ends; the head sits higher in image space
        # (smaller v in the OpenCV convention used by multiview_refine).
        half = 0.5 * float(np.ptp(pts @ up))
        ends = np.stack([mu + half * up, mu - half * up])
        votes = 0.0
        valid = 0
        for cam in cameras:
            w2c = cam.w2c.detach().cpu().numpy()
            hom = np.concatenate([ends, np.ones((2, 1))], axis=1)
            cam_pts = hom @ w2c.T
            z = cam_pts[:, 2]
            if np.any(z <= 1e-6):
                continue
            v = cam.fy * cam_pts[:, 1] / z + cam.cy
            # v[0] is the +up end.  Negative difference → +up end is higher.
            votes += float(np.sign(-(v[0] - v[1])))
            valid += 1
        if valid:
            agreement = abs(votes) / valid
            if votes < 0:
                up = -up
            diag["up_sign_votes"] = f"{int(abs(votes))}/{valid}"
            diag["up_sign_agreement"] = float(agreement)
            if agreement < 0.6:
                diag["warning_sign"] = (
                    f"Cameras disagree on which end is up "
                    f"(agreement={agreement:.2f}); the body may be inverted."
                )

    up = up / np.linalg.norm(up)
    diag["up_axis"] = up.tolist()
    return up, diag


def _extent_along(points: np.ndarray, direction: np.ndarray) -> float:
    """Peak-to-peak extent of the cloud projected onto *direction*."""
    proj = points @ direction
    return float(proj.max() - proj.min())


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit vector *a* onto unit vector *b*."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:  # antiparallel: rotate 180° about any orthogonal axis
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 0.0, 1.0])
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    K = np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def _rotation_y(angle: float) -> np.ndarray:
    """Rotation about the world Y axis."""
    ca, sa = math.cos(angle), math.sin(angle)
    return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])


def umeyama(src: np.ndarray, dst: np.ndarray,
            with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform mapping *src* onto *dst*.

    Closed-form (Umeyama 1991), including the reflection fix so the result is
    always a proper rotation.

    Returns:
        (scale, R, t) with ``dst ≈ scale * R @ src + t``.
    """
    n = len(src)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    S = src - mu_s
    D = dst - mu_d

    cov = (D.T @ S) / n
    U, sv, Vt = np.linalg.svd(cov)

    S_fix = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S_fix[2, 2] = -1.0
    R = U @ S_fix @ Vt

    var_s = float((S ** 2).sum() / n)
    if with_scale:
        scale = float(np.trace(np.diag(sv) @ S_fix) / max(var_s, 1e-12))
    else:
        scale = 1.0
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


def trimmed_chamfer(
    model_pts: np.ndarray,
    scene_pts: np.ndarray,
    trim_frac: float = 0.8,
    symmetric: bool = False,
    _scene_tree: cKDTree | None = None,
) -> float:
    """Mean nearest-neighbour distance after discarding the worst points.

    The body model cannot cover hair, loose clothing or background stragglers,
    so the raw mean is dominated by a few outliers.  Trimming makes the score
    reflect the fit of the bulk of the subject.

    Args:
        symmetric: also measure scene→model.  A one-sided model→scene score is
            minimised by *shrinking* the model until it hides inside the
            densest part of the cloud, which collapses the scale; the reverse
            term penalises the model for failing to cover the subject and
            cancels that bias.  Use this for the fit objective.
        _scene_tree: prebuilt KD-tree over *scene_pts*, to avoid rebuilding it
            on every iteration of a sweep.
    """
    tree = _scene_tree if _scene_tree is not None else cKDTree(scene_pts)
    d, _ = tree.query(model_pts, k=1)
    k = max(1, int(round(trim_frac * len(d))))
    score = float(np.sort(d)[:k].mean())

    if symmetric:
        d2, _ = cKDTree(model_pts).query(scene_pts, k=1)
        k2 = max(1, int(round(trim_frac * len(d2))))
        score = 0.5 * (score + float(np.sort(d2)[:k2].mean()))
    return score


# ── Main entry point ────────────────────────────────────────────────────────

def _posed_vertices(body_model, betas=None, body_pose=None) -> np.ndarray:
    """Body-model vertices posed by *body_pose*, in the model's own frame.

    Uses LBS on the mesh vertices (``v_posed = Σ_k w_vk · T_k · [v, 1]``),
    which works identically for SMPL and SMPL-X and needs no per-class helper.
    Pose blend shapes are skipped: they are a sub-centimetre correction and
    irrelevant to a coarse similarity fit.
    """
    rest_verts, _ = body_model.get_rest_mesh(betas)
    if body_pose is None:
        return rest_verts.detach().cpu().numpy().astype(np.float64)

    with torch.no_grad():
        T = body_model.get_joint_transforms(betas=betas, body_pose=body_pose)
        w = body_model.get_skinning_weights()  # (V, K)
        T = T.to(rest_verts.device)
        w = w.to(rest_verts.device).float()
        T_blend = torch.einsum('vk,kij->vij', w, T)  # (V, 4, 4)
        vh = torch.cat(
            [rest_verts, torch.ones(rest_verts.shape[0], 1, device=rest_verts.device)],
            dim=-1)
        posed = torch.einsum('vij,vj->vi', T_blend, vh)[:, :3]
    return posed.detach().cpu().numpy().astype(np.float64)


def fit_body_to_gaussians(
    means: torch.Tensor,
    body_model,
    betas: torch.Tensor | None = None,
    cameras: Sequence | None = None,
    up_axis: np.ndarray | None = None,
    fit_pose=None,
    n_yaw: int = 36,
    icp_iters: int = 30,
    trim_frac: float = 0.8,
    n_sample_model: int = 3000,
    n_sample_scene: int = 20000,
    refine_scale: bool = True,
    verbose: bool = True,
) -> BodyFitResult:
    """Fit *body_model* (SMPL or SMPL-X) into a Gaussian cloud.

    The returned transform satisfies ``xyz_scene = scale*(R @ xyz_model) + t``
    and can be fed straight to ``body_model.align_to_scene(scale, R, t)``.

    Args:
        means: (N, 3) Gaussian centres in scene coordinates.
        body_model: :class:`SMPLWrapper` or :class:`SMPLXWrapper`.  Must still
            be in its native (un-aligned) frame — calling ``align_to_scene``
            first would compound the two transforms.
        betas: shape parameters for the rest mesh.
        cameras: optional views, used only to disambiguate the up-axis sign.
        up_axis: explicit (3,) up direction; skips stage 1 when given.
        fit_pose: body pose used for the *fitting* mesh only (63- or 69-dim).
            Defaults to :func:`~.joints.a_pose_body_pose`, because the shipped
            SMPL/SMPL-X templates are T-posed and a photographed subject is
            not — fitting a T-pose shrinks the model to fit its arms inside
            the torso.  Pass ``None`` explicitly to fit the template as-is.
        n_yaw: number of yaw hypotheses in the sweep.
        icp_iters: maximum ICP iterations.
        trim_frac: fraction of best correspondences kept per ICP step.
        n_sample_model: model surface points used for the fit.
        n_sample_scene: Gaussian points used for the fit.
        refine_scale: allow ICP to update the scale.  Disable to keep the
            height-derived scale fixed, which is better when the cloud has a
            thick shell (heavy clothing) that would bias the fit outward.
        verbose: print each stage.

    Returns:
        :class:`BodyFitResult` with the transform and a Chamfer report.
    """
    scene_full = np.asarray(means.detach().cpu().numpy(), dtype=np.float64)
    scene = _subsample(scene_full, n_sample_scene)

    # ── Stage 1: up axis ──
    if up_axis is not None:
        up = np.asarray(up_axis, dtype=np.float64)
        up = up / np.linalg.norm(up)
        up_diag = {"method": "user_supplied"}
    else:
        up, up_diag = estimate_up_axis(scene_full, cameras=cameras)

    # ── Reference mesh: the model posed like the subject ──
    # The swing sign depends on how the model is rotated into the scene, so a
    # default pose is used for the initial fit and the swing is re-measured
    # afterwards (see ``choose_rig_pose``); this keeps the two stages from
    # disagreeing about where the arms are.
    if fit_pose is None:
        fit_pose = a_pose_body_pose(device=body_model.device)
        pose_note = "a_pose_default"
    else:
        pose_note = "explicit"

    model_pts = _posed_vertices(body_model, betas, fit_pose)

    # ── Stage 2: scale from height along the up axis ──
    model_up = SMPL_UP
    model_height = _extent_along(model_pts, model_up)
    scene_height = _extent_along(scene, up)

    if model_height < 1e-6:
        raise ValueError(f"body model has zero height (model_height={model_height:.2e})")
    scale = scene_height / model_height

    # ── Stage 3: rotation up + yaw sweep ──
    R0 = _rotation_between(model_up, up)

    model_fit = _subsample(model_pts, n_sample_model)
    scene_tree_pts = scene

    best = (float("inf"), 0.0, np.eye(3))
    tree_scene = cKDTree(scene_tree_pts)
    for i in range(max(1, n_yaw)):
        yaw = 2.0 * math.pi * i / max(1, n_yaw)
        R = _rotation_y(yaw) @ R0
        t = scene.mean(axis=0) - scale * (R @ model_pts.mean(axis=0))
        cand = scale * (model_fit @ R.T) + t
        cost = trimmed_chamfer(cand, scene_tree_pts, trim_frac,
                               symmetric=True, _scene_tree=tree_scene)
        if cost < best[0]:
            best = (cost, yaw, R)

    cost_yaw, yaw, R = best

    # ── Stage 4: ICP with trimmed Umeyama ──
    mu_model = model_pts.mean(axis=0)
    t = scene.mean(axis=0) - scale * (R @ mu_model)
    prev = cost_yaw
    n_done = 0

    for it in range(icp_iters):
        transformed = scale * (model_fit @ R.T) + t
        dist, idx = tree_scene.query(transformed, k=1)

        k_keep = max(6, int(round(trim_frac * len(dist))))
        keep = np.argsort(dist)[:k_keep]

        s_new, R_new, t_new = umeyama(
            model_fit[keep], scene_tree_pts[idx[keep]], with_scale=refine_scale)

        # Guard against a degenerate step (too few distinct correspondences).
        if not np.isfinite(s_new) or s_new <= 0:
            break
        if refine_scale:
            scale = s_new
        R, t = R_new, t_new
        n_done = it + 1

        cost = trimmed_chamfer(scale * (model_fit @ R.T) + t, scene_tree_pts,
                               trim_frac, symmetric=True, _scene_tree=tree_scene)
        improved = cost < prev - 1e-7
        prev = cost
        if not improved and it > 3:
            break

    # ── Report ──
    all_model = _subsample(model_pts, 20000)
    final_model = scale * (all_model @ R.T) + t
    c_m2s = trimmed_chamfer(_subsample(final_model, 8000), scene_full, trim_frac)
    c_s2m, _ = cKDTree(_subsample(final_model, 20000)).query(
        _subsample(scene_full, 8000), k=1)
    c_s2m = float(np.sort(c_s2m)[:max(1, int(0.8 * len(c_s2m)))].mean())

    diag = {
        "up_axis": up_diag,
        "scale": {
            "model_height": model_height,
            "scene_height": scene_height,
            "initial": scene_height / model_height,
            "final": float(scale),
            "refined": bool(refine_scale),
            "fit_pose": pose_note,
        },
        "yaw": {
            "best_deg": math.degrees(yaw),
            "n_hypotheses": int(max(1, n_yaw)),
            "chamfer_after_yaw": cost_yaw,
        },
        "icp": {
            "iterations": n_done,
            "trim_frac": trim_frac,
            "chamfer_model_to_scene": c_m2s,
            "chamfer_scene_to_model": c_s2m,
            "relative_to_height": c_m2s / max(scene_height, 1e-9),
        },
        "n_gaussians_used": int(len(scene)),
    }

    if verbose:
        print(f"  Fit body to Gaussians: {len(scene_full)} points, "
              f"scale={scale:.4f}, yaw={math.degrees(yaw):.1f}°")
        print(f"  Chamfer model→scene {c_m2s:.5f}, scene→model {c_s2m:.5f} "
              f"({100 * c_m2s / max(scene_height, 1e-9):.2f}% of height)")
        if up_diag.get("warning"):
            print(f"  WARNING: {up_diag['warning']}")

    return BodyFitResult(
        scale=float(scale),
        R=torch.from_numpy(R).float(),
        t=torch.from_numpy(t).float(),
        up_axis=up,
        chamfer_model_to_scene=c_m2s,
        chamfer_scene_to_model=c_s2m,
        diagnostics=diag,
    )


def align_body_model_to_gaussians(
    body_model,
    fit: BodyFitResult,
) -> None:
    """Apply a :class:`BodyFitResult` to *body_model* in place.

    Thin wrapper over ``align_to_scene`` that keeps the (scale, R, t) ordering
    identical to :meth:`BodyFitResult.apply`, so the fitted transform and the
    model's own deformation cannot drift apart.
    """
    body_model.align_to_scene(
        scale=fit.scale, rotation=fit.R.to(body_model.device),
        transl=fit.t.to(body_model.device))


def foot_anchor_offset(body_model, betas=None) -> torch.Tensor:
    """World-space position of the lowest foot joint in the rest pose.

    Useful for placing the body so that the feet sit on a ground plane, rather
    than centring the pelvis (which leaves the figure floating).
    """
    joints = body_model.get_rest_joints(betas)  # (J, 3)
    n = SMPL_NUM_JOINTS if joints.shape[0] >= SMPL_NUM_JOINTS else joints.shape[0]
    feet = joints[[L_FOOT, R_FOOT]] if n > R_FOOT else joints[-2:]
    return feet[torch.argmin(feet[:, 1])]


def head_height(body_model, betas=None) -> float:
    """Rest-pose height of the head joint above the lowest foot joint."""
    joints = body_model.get_rest_joints(betas)
    feet = foot_anchor_offset(body_model, betas)
    return float((joints[HEAD, 1] - feet[1]).abs())
