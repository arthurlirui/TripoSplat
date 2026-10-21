"""Closed-form registration of generated Gaussians into a photographed scene.

Problem
-------
TripoSplat emits Gaussians in a normalised model frame (AABB ``[-0.5, 0.5]³``,
Z-up).  The scene is given as calibrated cameras (OpenCV ``w2c``, Y-up SfM
world).  We need the 7-DoF similarity transform

    xyz_scene = s * (R @ xyz_model) + t

that places the generated subject where it actually was when photographed.

Why not silhouette-IoU grid search
----------------------------------
:mod:`~.alignment` searches ``(yaw, pitch, roll, scale)`` and scores each
candidate by mean silhouette IoU over **all** views.  On real captures that
objective is badly conditioned:

* The IoU surface is flat along the scale axis and its maximum sits at an
  inflated scale (measured here: a 2.6 m "person" beat the true 1.76 m).
  Larger silhouettes swallow the mask, so overlap keeps rising.
* Half the views in a handheld capture are **cropped by the frame border**.
  A cropped view has no bottom silhouette, so it cannot constrain scale at
  all, yet it votes with full weight.  On ``static/000001`` 15 of 30 views
  are border-touching.
* Front and back of a standing figure are near-mirror silhouettes, so a full
  6-DoF search can settle on a 180° flip and still report a good IoU.

What this module does instead
-----------------------------
Every parameter is pinned by an independent geometric constraint, so nothing
has to be searched except a single yaw angle:

1. **Up axis** ``u`` (2 DoF).  Triangulate the silhouette's topmost and
   bottommost pixels across views; the direction between the two
   triangulated 3D points is the subject's up axis in the world.  No prior on
   the SfM world frame is used, and the sign is fixed by consensus with the
   cameras' own up vectors.
2. **Subject centre** ``c`` (3 DoF).  Triangulate the silhouette centroid
   rays (robust IRLS).
3. **Scale** ``s`` (1 DoF).  The model has a known metric height along its own
   up axis ``h_model``; the scene's metric height ``H`` comes from matching
   each view's silhouette extent *along the projected up axis* to the
   prediction for a hypothetical ``H``.  Because the extent is measured along
   the projected up direction rather than the image's vertical, this is
   correct under camera roll and elevation, and cropped views are excluded
   from the fit.
4. **Yaw** (1 DoF).  With up/centre/scale fixed, only yaw remains; a 1-D
   sweep covers it.  A 180° flip is a separate point on this 1-D curve, so it
   has to win on its own merits rather than hiding in a 6-DoF search.
5. **Pose-only refinement.**  Optional short differentiable pass over the 7
   DoF (:func:`~.differentiable_refine.refine_alignment`), seeded from the
   closed-form solution.  Because the seed is already metrically correct the
   optimiser only has to polish, and the uprightness gate in
   :mod:`~.orientation` guards against it drifting into a flip.

The pipeline is deterministic apart from the refinement, needs no initial-view
pose, and degrades gracefully: with fewer than ``min_valid_views`` uncropped
views it relaxes to the least-cropped subset and reports reduced confidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from .cameras import Camera
from .coord_registry import MODEL2SCENE
from .gsplat_wrapper import render_alpha
from .quaternion import quaternion_compose, rotation_matrix_to_quaternion


# ═══════════════════════════════════════════════════════════════════════════
# Small geometry primitives (NumPy, float64 — independent of torch device)
# ═══════════════════════════════════════════════════════════════════════════

def camera_ray(cam: Camera, uv: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a pixel to a world-space ray.

    Returns ``(origin, direction)`` with ``direction`` unit length.  Uses the
    camera's per-frame intrinsics, which for this dataset differ between
    frames (``fl_x`` ranges 1780…5244) so a shared focal length would be
    wrong.
    """
    u, v = np.asarray(uv, dtype=np.float64).reshape(-1)[:2]
    u, v = float(u), float(v)
    d_cam = np.array([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.0])
    d_cam /= np.linalg.norm(d_cam)
    R = cam.c2w[:3, :3].detach().cpu().double().numpy()
    C = cam.c2w[:3, 3].detach().cpu().double().numpy()
    return C, R @ d_cam


def triangulate(rays: Sequence[tuple[np.ndarray, np.ndarray]],
                iters: int = 4,
                keep_frac: float = 0.7,
                min_keep: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Robust least-squares triangulation of a world point from several rays.

    Minimises ``Σ ||(I - ddᵀ)(X - C)||²`` — the squared distance from the
    point to each ray — but only over a *consensus set*: each iteration keeps
    the rays with the smallest residuals and refits (least trimmed squares).

    Being robust matters because the rays are not all aimed at the same
    physical point.  A silhouette's topmost pixel is the head in most views but
    a raised hand or a hat brim in others, and a single such view would
    otherwise pull the estimate by centimetres — enough to skew the metric
    scale, which is derived from that estimate.

    Trimming by rank (rather than by a residual threshold) avoids the usual
    failure of robust weighting: after an initial fit is dragged off by
    outliers, every residual looks moderate and a threshold-based scheme stops
    discriminating.

    Args:
        rays: sequence of ``(origin, direction)``; directions need not be unit.
        iters: refit iterations.
        keep_frac: fraction of rays kept per iteration.
        min_keep: never trim below this many rays.

    Returns:
        ``(point (3,), residuals (K,))`` with residuals measured against *all*
        input rays, so callers can see which ones were rejected.
    """
    C = np.stack([r[0] for r in rays])                      # (K, 3)
    D = np.stack([r[1] for r in rays])
    D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)

    # Projectors P_k = I - d dᵀ for every ray.
    P = np.eye(3)[None, :, :] - D[:, :, None] * D[:, None, :]  # (K, 3, 3)
    K = len(C)
    n_keep = max(min_keep, int(round(keep_frac * K)))
    keep = np.arange(K)                     # indices currently in the consensus

    X = np.zeros(3)
    for _ in range(max(1, iters)):
        Pk, Ck = P[keep], C[keep]
        A = Pk.sum(axis=0)
        b = np.einsum("kij,kj->i", Pk, Ck)
        X, *_ = np.linalg.lstsq(A, b, rcond=None)

        res = np.linalg.norm(np.einsum("kij,kj->ki", P, X - C), axis=1)
        if n_keep >= K:
            break
        keep = np.argsort(res)[:n_keep]

    res = np.linalg.norm(np.einsum("kij,kj->ki", P, X - C), axis=1)
    return X, res


def rotation_aligning(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Minimal rotation taking unit vector *src* onto unit vector *dst*.

    Used to tilt the model's own up axis onto the triangulated scene up axis.
    Falls back to a 180° turn about an arbitrary perpendicular when the
    vectors are antiparallel (Rodrigues is singular there).
    """
    a = np.asarray(src, dtype=np.float64)
    b = np.asarray(dst, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # Antiparallel: rotate 180° about any axis perpendicular to a.
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 0.0, 1.0]))
        axis /= np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    K = np.array([[0.0, -v[2], v[1]],
                  [v[2], 0.0, -v[0]],
                  [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def rotation_y(angle: float) -> np.ndarray:
    """Rotation about the scene +Y axis (the canonical up axis)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


def project_points(cam: Camera, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points (N, 3) with the pinhole model.

    Returns ``(u, v, z)``; ``z <= 0`` marks points behind the camera.
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    R = cam.w2c[:3, :3].detach().cpu().double().numpy()
    t = cam.w2c[:3, 3].detach().cpu().double().numpy()
    Xc = pts @ R.T + t
    z = Xc[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = cam.fx * Xc[:, 0] / zs + cam.cx
    v = cam.fy * Xc[:, 1] / zs + cam.cy
    return u, v, z


# ═══════════════════════════════════════════════════════════════════════════
# Per-view observations
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ViewObservation:
    """Silhouette measurements for one camera."""

    view_idx: int
    cam: Camera
    mask: np.ndarray            # (H, W) bool
    centroid_uv: np.ndarray     # (2,) silhouette centroid in pixels
    top_uv: np.ndarray          # (2,) topmost silhouette point (px)
    bottom_uv: np.ndarray       # (2,) bottommost silhouette point (px)
    area: int
    border_contact: float       # fraction of the 4 borders that is foreground
    clipped: bool
    weight: float = 1.0


def _mask_for_camera(cam: Camera, rmbg_fn: Callable | None,
                     alpha_threshold: float) -> np.ndarray:
    """Binary foreground mask, preferring the image's own alpha channel."""
    if cam.alpha is not None:
        return (cam.alpha.squeeze(0) > alpha_threshold).detach().cpu().numpy()
    if rmbg_fn is not None:
        from PIL import Image as PILImage

        img = (cam.image.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        rgba = rmbg_fn(PILImage.fromarray(img))
        arr = np.asarray(rgba, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 4:
            a = arr[..., 3] / 255.0
        else:
            a = np.ones(arr.shape[:2], dtype=np.float32)
        if a.shape[:2] != (cam.height, cam.width):
            a_pil = PILImage.fromarray((a * 255).astype(np.uint8), mode="L")
            a = np.asarray(a_pil.resize((cam.width, cam.height), PILImage.BILINEAR),
                           dtype=np.float32) / 255.0
        return a > alpha_threshold
    return (cam.image.mean(dim=0) > 0.05).detach().cpu().numpy()


def extract_observations(
    cameras: Sequence[Camera],
    rmbg_fn: Callable | None = None,
    alpha_threshold: float = 0.5,
    min_area_frac: float = 0.002,
) -> list[ViewObservation]:
    """Measure each view's silhouette and flag border-cropped ones.

    A view is *clipped* when any row/column of the foreground touches the image
    border: its silhouette then has no true top or bottom, so it carries no
    scale information and must be excluded from the height fit.
    """
    observations: list[ViewObservation] = []
    for i, cam in enumerate(cameras):
        mask = _mask_for_camera(cam, rmbg_fn, alpha_threshold)
        ys, xs = np.nonzero(mask)
        if len(ys) < 20 or len(ys) < min_area_frac * mask.size:
            continue

        H, W = mask.shape
        edges = [mask[0].mean(), mask[-1].mean(), mask[:, 0].mean(), mask[:, -1].mean()]
        contact = float(np.mean([e > 0.01 for e in edges]))
        clipped = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())

        ymin, ymax = int(ys.min()), int(ys.max())
        # Median column over the rows within 1 px of the extreme: robust to a
        # single stray pixel at the tip (hair, a fingertip).
        top_x = float(np.median(xs[ys <= ymin + 1]))
        bot_x = float(np.median(xs[ys >= ymax - 1]))
        observations.append(ViewObservation(
            view_idx=i,
            cam=cam,
            mask=mask,
            centroid_uv=np.array([xs.mean(), ys.mean()]),
            top_uv=np.array([top_x, float(ymin)]),
            bottom_uv=np.array([bot_x, float(ymax)]),
            area=int(len(ys)),
            border_contact=contact,
            clipped=clipped,
        ))
    return observations


def select_views(
    observations: Sequence[ViewObservation],
    min_valid_views: int = 6,
) -> tuple[list[ViewObservation], dict]:
    """Prefer uncropped views; relax to the least-cropped subset if needed.

    Cropped views are the single biggest source of scale error on handheld
    captures, so they are dropped whenever enough clean views remain.
    """
    clean = [o for o in observations if not o.clipped]
    info = {
        "n_observed": len(observations),
        "n_uncropped": len(clean),
        "relaxed": False,
    }

    if len(clean) >= min_valid_views:
        selected = clean
    else:
        # Keep the least border-contaminated half of the cropped views.
        ranked = sorted(observations, key=lambda o: (o.border_contact, -o.area))
        selected = ranked[: max(min_valid_views, len(ranked) // 2)]
        info["relaxed"] = True

    if not selected:
        return [], info

    # Down-weight views that touch the border a lot or are very small.
    max_area = max(o.area for o in selected)
    for o in selected:
        o.weight = (1.0 - 0.75 * o.border_contact) * math.sqrt(o.area / max_area)
    info["n_selected"] = len(selected)
    return selected, info


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — up axis
# ═══════════════════════════════════════════════════════════════════════════

def estimate_up_axis(obs: Sequence[ViewObservation],
                     trim_frac: float = 0.8) -> tuple[np.ndarray, dict]:
    """Recover the subject's vertical axis in world coordinates.

    Uses a **direction** constraint rather than triangulating extreme pixels.
    A 3-D direction ``u`` projects into view ``k`` as ``M_k u``, where
    ``M_k = J_k R_k`` is the projection Jacobian (``J_k`` the perspective
    scaling at the subject's depth, ``R_k`` the world-to-camera rotation).  If
    the silhouette's principal (long) axis in the image has unit direction
    ``d_k``, then ``M_k u ∥ d_k``, i.e. the single linear equation

        ``(d_k.y · M_k[0] − d_k.x · M_k[1]) · u = 0``.

    Stacking these over all views leaves ``u`` as the null vector.  This is
    robust where triangulating top/bottom pixels is not: the extreme pixel of
    a silhouette is a *different physical point* from a different viewpoint
    (a corner, a hand, a hat brim), whereas the principal axis is an aggregate
    property that every view agrees on.  Measured on a synthetic box-shaped
    subject, extremum triangulation erred by 56°, this method by a few degrees.

    A weak pull toward the cameras' own up vectors regularises degenerate
    configurations (e.g. all cameras on one horizontal line) and fixes the
    sign.  Views whose mask is nearly isotropic carry little directional
    information; they are dropped, as are the worst-fitting ``1 − trim_frac``.
    """
    entries = []
    for o in obs:
        ys, xs = np.nonzero(o.mask)
        if len(ys) < 20:
            continue
        X = np.stack([xs - xs.mean(), ys - ys.mean()]).astype(np.float64)
        C = (X @ X.T) / len(xs)
        evals, evecs = np.linalg.eigh(C)
        if evals.max() <= 1e-9:
            continue
        d = evecs[:, int(np.argmax(evals))]
        d = d / np.linalg.norm(d)
        elongation = math.sqrt(max(evals.max(), 0.0) / max(evals.min(), 1e-12))
        if elongation < 1.3:
            # Near-isotropic mask — no reliable long axis (e.g. looking
            # straight down the body, or a crouched pose).
            continue

        M = _projection_jacobian(o.cam)
        row = d[1] * M[0] - d[0] * M[1]
        n = float(np.linalg.norm(row))
        if n < 1e-12:
            continue
        entries.append((row / n, elongation))

    if len(entries) < 2:
        raise ValueError(
            "Cannot determine the up axis: need at least 2 views with an "
            f"elongated silhouette, got {len(entries)}."
        )

    cam_up = np.mean([-o.cam.c2w[:3, 1].detach().cpu().numpy() for o in obs], axis=0)
    cam_up_n = cam_up / max(np.linalg.norm(cam_up), 1e-9)
    lam = 0.5  # regularisation weight, in units of one average view

    def solve(use):
        A = np.stack([row * math.sqrt(elong) for row, elong in use])
        A = np.vstack([A, lam * np.eye(3)])
        b = np.concatenate([np.zeros(len(use)), lam * cam_up_n])
        _, _, vt = np.linalg.svd(A)
        return vt[-1]

    u = solve(entries)
    # Least-trimmed refit: discard views whose principal axis disagrees most.
    for _ in range(3):
        ang = np.array([float(np.linalg.norm(np.cross(row, u))) for row, _ in entries])
        keep_n = max(2, int(round(trim_frac * len(entries))))
        u_new = solve([entries[i] for i in np.argsort(ang)[:keep_n]])
        if float(np.dot(u_new, u)) < 0:
            u_new = -u_new
        converged = np.linalg.norm(u_new - u) < 1e-6
        u = u_new
        if converged:
            break

    u = u / np.linalg.norm(u)
    if float(np.dot(u, cam_up_n)) < 0:
        u = -u

    agreement = float(np.dot(u, cam_up_n))
    diag = {
        "camera_up_agreement": agreement,
        "n_views": len(entries),
        "elongation_median": float(np.median([e[1] for e in entries])),
    }
    if agreement < 0.5:
        diag["warning"] = (
            "Up axis disagrees with the cameras' up vectors "
            f"(dot={agreement:.2f}); the capture may be inverted, or the "
            "silhouettes are not dominated by the body's long axis."
        )
    return u, diag


def _projection_jacobian(cam: Camera) -> np.ndarray:
    """``M = diag(fx, fy) R_w2c`` (2×3): maps a world *direction* to an image
    direction.

    The perspective ``1/z`` factor is a positive scalar common to both rows;
    since the constraint rows below are normalised, it cancels, so the
    subject's depth need not be known here.
    """
    R_w2c = cam.w2c[:3, :3].detach().cpu().double().numpy()
    return np.array([[cam.fx, 0.0, 0.0], [0.0, cam.fy, 0.0]]) @ R_w2c


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 — subject centre
# ═══════════════════════════════════════════════════════════════════════════

def estimate_subject_centre(obs: Sequence[ViewObservation]) -> tuple[np.ndarray, dict]:
    """Triangulate the silhouette centroid to locate the subject in the scene."""
    rays = [camera_ray(o.cam, o.centroid_uv) for o in obs]
    centre, res = triangulate(rays)
    return centre, {"centroid_residual": float(np.mean(res)),
                    "centroid_residual_max": float(np.max(res))}


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 — metric scale
# ═══════════════════════════════════════════════════════════════════════════

def _observed_extent_along(
    obs: ViewObservation,
    up_dir_img: np.ndarray,
) -> float:
    """Silhouette extent in pixels measured along a given image direction."""
    ys, xs = np.nonzero(obs.mask)
    proj = xs * up_dir_img[0] + ys * up_dir_img[1]
    return float(proj.max() - proj.min())


def _image_up_direction(cam: Camera, centre: np.ndarray, up: np.ndarray) -> Optional[np.ndarray]:
    """Unit image-space direction of the world up axis, seen from *cam*.

    Returns ``None`` when the up axis is (nearly) along the viewing ray, i.e.
    the view looks straight down the subject's body and cannot measure height.
    """
    u1, v1, z1 = project_points(cam, centre[None, :])
    span = 0.1 * max(float(np.linalg.norm(up)), 1e-9)
    u2, v2, z2 = project_points(cam, (centre + span * up)[None, :])
    if z1[0] <= 1e-6 or z2[0] <= 1e-6:
        return None
    d = np.array([u2[0] - u1[0], v2[0] - v1[0]])
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return None
    d /= n
    # Long axis of the body must be measurable, not foreshortened to nothing.
    if n / span < 0.05:
        return None
    return d


def estimate_metric_height(
    obs: Sequence[ViewObservation],
    centre: np.ndarray,
    up: np.ndarray,
    height_range: tuple[float, float] = (0.2, 3.0),
) -> tuple[float, dict]:
    """Solve for the subject's height in metres from silhouette extents.

    For a hypothesised height ``H``, the subject is modelled as the segment
    ``centre ± (H/2)·up``.  Projecting its endpoints into each view predicts
    the silhouette's extent along the projected up direction; ``H`` is the
    value that matches the measured extents.  This is the physically correct
    relation for any camera roll or elevation — unlike matching the raw
    bounding-box height, which couples in foreshortening and lateral extent.

    The match is weighted by each view's reliability, so uncropped, large
    views dominate.
    """
    entries = []
    for o in obs:
        d = _image_up_direction(o.cam, centre, up)
        if d is None:
            continue
        extent_obs = _observed_extent_along(o, d)
        if extent_obs < 5:
            continue
        entries.append((o, d, extent_obs))

    if not entries:
        raise ValueError(
            "No view can measure the subject's height: every selected view "
            "either looks along the up axis or has a degenerate silhouette."
        )

    def cost(H: float) -> float:
        half = 0.5 * H
        total = 0.0
        for o, d, extent_obs in entries:
            pts = np.stack([centre + half * up, centre - half * up])
            u, v, z = project_points(o.cam, pts)
            if np.any(z <= 1e-6):
                return 1e12
            pred = math.hypot(u[0] - u[1], v[0] - v[1])
            total += o.weight * (pred - extent_obs) ** 2
        return total

    # Coarse log-grid then golden-section refinement; the cost is unimodal in H
    # because the predicted extent grows monotonically with H.
    lo, hi = math.log(height_range[0]), math.log(height_range[1])
    grid = np.linspace(lo, hi, 120)
    costs = [cost(math.exp(g)) for g in grid]
    best = int(np.argmin(costs))
    a = grid[max(best - 1, 0)]
    b = grid[min(best + 1, len(grid) - 1)]

    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c, d_ = b - gr * (b - a), a + gr * (b - a)
    fc, fd = cost(math.exp(c)), cost(math.exp(d_))
    for _ in range(60):
        if fc < fd:
            b, d_, fd = d_, c, fc
            c = b - gr * (b - a)
            fc = cost(math.exp(c))
        else:
            a, c, fc = c, d_, fd
            d_ = a + gr * (b - a)
            fd = cost(math.exp(d_))
    H = math.exp(0.5 * (a + b))

    # Per-view agreement at the solution, for diagnostics.
    half = 0.5 * H
    ratios = []
    for o, d, extent_obs in entries:
        u, v, z = project_points(o.cam, np.stack([centre + half * up, centre - half * up]))
        pred = math.hypot(u[0] - u[1], v[0] - v[1])
        ratios.append(pred / max(extent_obs, 1e-6))

    return H, {
        "n_views_fit": len(entries),
        "extent_ratio_median": float(np.median(ratios)),
        "extent_ratio_std": float(np.std(ratios)),
    }


def model_height_along_up(means: torch.Tensor, R_up: torch.Tensor) -> float:
    """Height of the Gaussian set along the direction *R_up* maps model-up to.

    ``R_up`` already contains :data:`MODEL2SCENE`, so the returned extent is
    the model's metric height scaled by 1 (i.e. in model units).
    """
    rotated = means @ R_up.T
    return float(rotated[:, 1].max() - rotated[:, 1].min())


def calibrate_scale_by_rendering(
    means, scales, quats, opacities,
    obs: Sequence[ViewObservation],
    centre: np.ndarray, up: np.ndarray,
    scale_init: float,
    model_centroid: np.ndarray,
    yaw: float,
    R_up: torch.Tensor,
    n_iter: int = 10,
    bracket: float = 1.6,
    alpha_threshold: float = 0.1,
) -> tuple[float, dict]:
    """Correct the scale so rendered and observed silhouettes agree.

    :func:`estimate_metric_height` measures the subject from the *observed*
    silhouettes but compares against the Gaussian *centres*' extent, which is
    smaller than the rendered silhouette by roughly one Gaussian radius at
    each end.  That makes the analytic scale biased high by a few percent.

    Here both sides are measured the same way — silhouette extent along the
    projected up axis — so the bias cancels.  The predicted extent grows
    monotonically with scale, so a bisection on ``log s`` is well posed and
    needs ~10 renders per view.

    Returns ``(scale, info)``.
    """
    device = means.device
    centre_t = torch.from_numpy(centre).float().to(device)
    mc = torch.from_numpy(model_centroid).float().to(device)
    R = torch.from_numpy(rotation_y(yaw)).float().to(device) @ R_up

    # Views able to measure height, with their observed extents.
    entries = []
    for o in obs:
        d = _image_up_direction(o.cam, centre, up)
        if d is None:
            continue
        extent_obs = _observed_extent_along(o, d)
        if extent_obs >= 5:
            entries.append((o, d, extent_obs))
    if not entries:
        return scale_init, {"calibrated": False, "reason": "no measurable view"}

    def ratio(s: float) -> float:
        """median(observed extent / rendered extent) over the measurable views."""
        t = centre_t - s * (mc @ R.T)
        mt, st, qt = _apply(means, scales, quats, s, R, t)
        vals = []
        for o, d, extent_obs in entries:
            alpha = render_alpha(mt, st, qt, opacities, o.cam)
            sil = (alpha.squeeze(0) > alpha_threshold).detach().cpu().numpy()
            ys, xs = np.nonzero(sil)
            if len(ys) < 5:
                continue
            proj = xs * d[0] + ys * d[1]
            extent_pred = float(proj.max() - proj.min())
            if extent_pred > 1e-6:
                vals.append(extent_obs / extent_pred)
        if not vals:
            return float("nan")
        return float(np.median(vals))

    # Bisect log(scale) so that ratio(s) == 1.  ratio is decreasing in s:
    # a bigger model renders a bigger silhouette, so observed/predicted falls.
    lo, hi = scale_init / bracket, scale_init * bracket
    r_lo, r_hi = ratio(lo), ratio(hi)
    if not np.isfinite(r_lo) or not np.isfinite(r_hi) or r_lo < 1.0 or r_hi > 1.0:
        # No sign change (degenerate): keep the analytic estimate.
        return scale_init, {"calibrated": False, "reason": "no bracket",
                            "ratio_lo": r_lo, "ratio_hi": r_hi}

    s = scale_init
    for _ in range(n_iter):
        mid = math.sqrt(lo * hi)  # geometric midpoint == bisection in log space
        r = ratio(mid)
        if not np.isfinite(r):
            break
        s = mid
        if r > 1.0:
            lo = mid          # model too small -> grow
        else:
            hi = mid          # model too big -> shrink

    return s, {
        "calibrated": True,
        "scale_before": scale_init,
        "scale_after": s,
        "rel_change": (s - scale_init) / scale_init,
        "final_ratio": ratio(s),
        "n_views": len(entries),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4 — yaw
# ═══════════════════════════════════════════════════════════════════════════

def _apply(means, scales, quats, s, R, t):
    from .alignment import _apply_transform

    return _apply_transform(means, scales, quats, s, R, t)


@torch.no_grad()
def _iou_against_masks(
    means, scales, quats, opacities, s, R, t,
    obs: Sequence[ViewObservation], alpha_threshold: float = 0.1,
) -> float:
    mt, st, qt = _apply(means, scales, quats, s, R, t)
    ious, weights = [], []
    for o in obs:
        alpha = render_alpha(mt, st, qt, opacities, o.cam)
        sil = (alpha.squeeze(0) > alpha_threshold).detach().cpu().numpy()
        inter = np.logical_and(sil, o.mask).sum()
        union = np.logical_or(sil, o.mask).sum()
        if union == 0:
            continue
        ious.append(inter / union)
        weights.append(o.weight)
    if not ious:
        return 0.0
    return float(np.average(ious, weights=weights))


def search_yaw(
    means, scales, quats, opacities,
    obs: Sequence[ViewObservation],
    centre: np.ndarray, up: np.ndarray, scale: float,
    model_centroid: np.ndarray,
    n_coarse: int = 36,
    n_fine: int = 24,
    alpha_threshold: float = 0.1,
) -> tuple[float, float, dict]:
    """Find the yaw that best explains all silhouettes.

    With up axis, centre and scale already fixed, the only remaining
    orientation freedom is a spin about the up axis.  A 1-D sweep therefore
    covers the whole space — including the 180° flip, which is just another
    sample on the same curve rather than a separate basin.

    Returns ``(yaw_radians, best_iou, info)``.
    """
    device = means.device
    R_up = torch.from_numpy(
        rotation_aligning(np.array([0.0, 1.0, 0.0]), up) @ MODEL2SCENE.double().numpy()
    ).float().to(device)
    centre_t = torch.from_numpy(centre).float().to(device)
    mc = torch.from_numpy(model_centroid).float().to(device)

    def evaluate(yaw: float):
        R = torch.from_numpy(rotation_y(yaw)).float().to(device) @ R_up
        # Keep the model centroid on the measured subject centre.
        t = centre_t - scale * (mc @ R.T)
        return _iou_against_masks(means, scales, quats, opacities,
                                  scale, R, t, obs, alpha_threshold), R, t

    coarse = [(y, *evaluate(y)[0:1]) for y in np.linspace(-math.pi, math.pi, n_coarse,
                                                          endpoint=False)]
    best_yaw = max(coarse, key=lambda e: e[1])[0]

    step = 2 * math.pi / n_coarse
    fine = [(y, evaluate(y)[0]) for y in np.linspace(best_yaw - step, best_yaw + step, n_fine)]
    best_yaw = max(fine, key=lambda e: e[1])[0]
    best_iou, R, t = evaluate(best_yaw)

    return best_yaw, best_iou, {"R": R, "t": t, "R_up": R_up,
                                "coarse_best": float(max(e[1] for e in coarse))}


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RegistrationResult:
    """Result of :func:`register_gaussians_to_scene`.

    The transform maps model-frame points to scene (world) coordinates::

        xyz_scene = scale * (R @ xyz_model) + t
    """

    scale: float
    R: torch.Tensor
    t: torch.Tensor
    up_axis: np.ndarray
    subject_centre: np.ndarray
    metric_height: float
    model_height: float
    yaw: float
    diagnostics: dict = field(default_factory=dict)

    def matrix(self) -> torch.Tensor:
        """4×4 homogeneous transform in scene coordinates."""
        M = torch.eye(4, dtype=self.R.dtype, device=self.R.device)
        M[:3, :3] = self.scale * self.R
        M[:3, 3] = self.t
        return M

    def apply(self, means, scales, quats):
        """Apply the transform to Gaussian attributes."""
        return _apply(means, scales, quats, self.scale, self.R, self.t)

    def to_dict(self) -> dict:
        return {
            "scale": float(self.scale),
            "R": self.R.detach().cpu().tolist(),
            "t": self.t.detach().cpu().tolist(),
            "matrix": self.matrix().detach().cpu().tolist(),
            "up_axis": np.asarray(self.up_axis).tolist(),
            "subject_centre": np.asarray(self.subject_centre).tolist(),
            "metric_height_m": float(self.metric_height),
            "model_height_model_units": float(self.model_height),
            "yaw_deg": math.degrees(self.yaw),
            "diagnostics": self.diagnostics,
        }


def register_gaussians_to_scene(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    cameras: Sequence[Camera],
    colors: torch.Tensor | None = None,
    rmbg_fn: Callable | None = None,
    alpha_threshold: float = 0.5,
    min_valid_views: int = 6,
    n_yaw_coarse: int = 36,
    n_yaw_fine: int = 24,
    calibrate_scale: bool = True,
    refine_iters: int = 0,
    refine_kwargs: dict | None = None,
    verbose: bool = True,
) -> RegistrationResult:
    """Register generated Gaussians to a photographed scene, closed form.

    Args:
        means: ``(N, 3)`` Gaussian centres in the **raw model frame** (the
            TripoSplat AABB frame; :data:`MODEL2SCENE` is folded into ``R``).
        scales: ``(N, 3)`` activated scales.
        quats:  ``(N, 4)`` quaternions.
        opacities: ``(N,)`` activated opacities.
        cameras: calibrated views.  Their images/masks must be loaded.
        colors: ``(N, K, 3)`` SH coefficients; required only when
            ``refine_iters > 0``.
        rmbg_fn: background remover used when a camera has no alpha channel.
        alpha_threshold: foreground threshold for the real masks.
        min_valid_views: how many uncropped views are needed before falling
            back to the least-cropped subset.
        n_yaw_coarse, n_yaw_fine: yaw sweep resolution.
        calibrate_scale: re-derive the scale from rendered silhouettes so the
            observed and predicted extents are measured the same way
            (:func:`calibrate_scale_by_rendering`).  Recommended.
        refine_iters: if > 0, run a short differentiable pose refinement seeded
            from the closed-form result.
        refine_kwargs: extra keyword arguments for
            :func:`~.differentiable_refine.refine_alignment`.
        verbose: print each stage's estimate.

    Returns:
        :class:`RegistrationResult` with the transform and diagnostics.
    """
    device = means.device
    means = means.detach().float()
    scales = scales.detach().float()
    quats = quats.detach().float()
    opacities = opacities.detach().float()

    def log(msg: str):
        if verbose:
            print(msg)

    # ── Observations ──
    observations = extract_observations(cameras, rmbg_fn=rmbg_fn,
                                        alpha_threshold=alpha_threshold)
    obs, sel_info = select_views(observations, min_valid_views=min_valid_views)
    if len(obs) < 3:
        raise ValueError(
            f"Registration needs at least 3 usable silhouettes, got {len(obs)} "
            f"(of {len(observations)} observed). Check masks / camera poses."
        )
    log(f"  Views: {sel_info['n_observed']} observed, {sel_info['n_uncropped']} uncropped, "
        f"{len(obs)} used" + ("  [relaxed: too few uncropped]" if sel_info["relaxed"] else ""))

    # ── Stage 1: up axis ──
    up, up_info = estimate_up_axis(obs)
    log(f"  Up axis (world): {np.round(up, 4).tolist()}, "
        f"camera agreement {up_info['camera_up_agreement']:.3f}")

    # ── Stage 2: subject centre ──
    centre, centre_info = estimate_subject_centre(obs)
    log(f"  Subject centre: {np.round(centre, 4).tolist()} "
        f"(median ray residual {centre_info['centroid_residual']:.4f} m)")

    # ── Stage 3: metric scale ──
    height, height_info = estimate_metric_height(obs, centre, up)
    R_up_np = rotation_aligning(np.array([0.0, 1.0, 0.0]), up) @ MODEL2SCENE.double().numpy()
    R_up = torch.from_numpy(R_up_np).float().to(device)
    h_model = model_height_along_up(means, R_up)
    scale = height / max(h_model, 1e-9)
    log(f"  Metric height: {height:.3f} m over {height_info['n_views_fit']} views "
        f"(model height {h_model:.4f} units) → scale {scale:.4f}")

    # ── Stage 4: yaw ──
    # Rotate the model centroid along with the model, so the rigid placement is
    # independent of the yaw that is about to be searched.
    model_centroid = means.mean(dim=0).detach().cpu().numpy()
    yaw, iou, yaw_info = search_yaw(
        means, scales, quats, opacities, obs, centre, up, scale,
        model_centroid, n_coarse=n_yaw_coarse, n_fine=n_yaw_fine,
        alpha_threshold=0.1,
    )

    # ── Stage 3b: rendering-consistent scale calibration ──
    # The analytic height compares observed silhouettes against Gaussian
    # *centres*; the rendered silhouette sticks out about one Gaussian radius
    # beyond them, which biases the scale high.  Re-derive it from rendered
    # silhouettes so both sides are measured identically.
    if calibrate_scale:
        scale, calib_info = calibrate_scale_by_rendering(
            means, scales, quats, opacities, obs, centre, up, scale,
            model_centroid, yaw, yaw_info["R_up"],
        )
        log(f"  Scale calibration: {calib_info.get('scale_before', scale):.4f} → "
            f"{scale:.4f} "
            f"({100 * calib_info.get('rel_change', 0.0):+.1f}%)")
    else:
        calib_info = None

    R = yaw_info["R"]
    t = yaw_info["t"]
    # Re-anchor the translation for the final scale.
    centre_t = torch.from_numpy(centre).float().to(device)
    mc_t = torch.from_numpy(model_centroid).float().to(device)
    t = centre_t - scale * (mc_t @ R.T)
    iou = _iou_against_masks(means, scales, quats, opacities, scale, R, t, obs, 0.1)
    log(f"  Yaw: {math.degrees(yaw):.1f}° → mean silhouette IoU {iou:.4f}")

    diagnostics = {
        "views": sel_info,
        "up_axis": up_info,
        "centre": centre_info,
        "height": height_info,
        "scale_calibration": calib_info,
        "yaw": {"coarse_best_iou": yaw_info["coarse_best"], "final_iou": iou},
        "method": "closed_form_axis_scale_yaw",
    }

    # ── Optional pose-only refinement ──
    if refine_iters > 0:
        if colors is None:
            raise ValueError("colors are required when refine_iters > 0")
        from .differentiable_refine import refine_alignment

        kwargs = dict(num_iterations=refine_iters, verbose=verbose)
        kwargs.update(refine_kwargs or {})
        log(f"  Refining pose differentiably ({refine_iters} iters, "
            f"{len(obs)} views)...")
        s_r, R_r, t_r = refine_alignment(
            means=means, scales=scales, quats=quats, opacities=opacities,
            colors=colors,
            cameras=[o.cam for o in obs],
            coarse_transform=(scale, R, t),
            rmbg_fn=rmbg_fn,
            **kwargs,
        )
        diagnostics["refine"] = {
            "scale_before": float(scale), "scale_after": float(s_r),
            "t_delta_norm": float((t_r - t).norm()),
        }
        scale, R, t = float(s_r), R_r, t_r
        iou_after = _iou_against_masks(means, scales, quats, opacities,
                                       scale, R, t, obs, 0.1)
        diagnostics["refine"]["iou_after"] = iou_after
        log(f"  After refinement: scale={scale:.4f}, IoU={iou_after:.4f}")

    return RegistrationResult(
        scale=scale, R=R, t=t,
        up_axis=up, subject_centre=centre,
        metric_height=height, model_height=h_model,
        yaw=yaw, diagnostics=diagnostics,
    )


def verify_registration(
    result: RegistrationResult,
    means, scales, quats, opacities,
    cameras: Sequence[Camera],
    rmbg_fn: Callable | None = None,
    alpha_threshold: float = 0.5,
    alpha_render_threshold: float = 0.1,
) -> dict:
    """Score a registration and flag the failure modes the closed form avoids.

    Returns per-view silhouette IoU plus three aggregate checks:

    * ``mean_iou`` — overall fit.
    * ``height_consistency`` — spread of the per-view implied metric height.
      A large spread means the model and the scene disagree about the
      subject's proportions/scale.
    * ``upright`` — fraction of rendered silhouettes that are head-up.  IoU
      alone cannot reject a 180° flip, so it is checked separately.
    """
    from .orientation import uprightness_report

    observations = extract_observations(cameras, rmbg_fn=rmbg_fn,
                                        alpha_threshold=alpha_threshold)
    clean = [o for o in observations if not o.clipped] or observations

    mt, st, qt = result.apply(means, scales, quats)
    ious, pred_masks, real_masks, implied = [], [], [], []

    with torch.no_grad():
        for o in clean:
            alpha = render_alpha(mt, st, qt, opacities, o.cam)
            sil = (alpha.squeeze(0) > alpha_render_threshold).detach().cpu().numpy()
            inter = np.logical_and(sil, o.mask).sum()
            union = np.logical_or(sil, o.mask).sum()
            if union:
                ious.append(float(inter / union))
            pred_masks.append(sil)
            real_masks.append(o.mask)

            ys, xs = np.nonzero(sil)
            if len(ys) > 20:
                cam = o.cam
                _, _, z = project_points(cam, result.subject_centre[None, :])
                if z[0] > 1e-6:
                    implied.append(float((ys.max() - ys.min() + 1) * z[0] / cam.fy))

    upright = uprightness_report(pred_masks, real_masks)
    return {
        "n_views": len(clean),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "min_iou": float(np.min(ious)) if ious else 0.0,
        "per_view_iou": ious,
        "implied_height_median": float(np.median(implied)) if implied else float("nan"),
        "height_consistency_std": float(np.std(implied)) if implied else float("nan"),
        "frac_upright": upright["frac_upright"],
        "flip_suspected": upright["flip_suspected"],
    }
