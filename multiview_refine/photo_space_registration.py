"""Photo-space registration: align generated Gaussians to the real world.

This module implements **camera-photo-space registration** — computing the
3D world coordinates of a generated Gaussian-Splatting model by comparing
what the model *renders* against what the camera actually *photographs*,
using image content rather than silhouettes alone.

Motivation
----------
The existing aligners in :mod:`alignment` rely on silhouette IoU and a
known canonical orientation.  That works for an initial placement but has
two limits the continual-learning / real-time-update loop must overcome:

1. **Silhouettes ignore texture.** Two poses with identical silhouettes but
   rotated 180° (front vs. back) are indistinguishable to IoU, and the
   uprightness heuristic in :mod:`orientation` only catches the vertical
   flip, not an in-plane yaw ambiguity.
2. **They are global and offline.** A full grid search cannot run every
   time a new frame arrives.

Photo-space registration instead works directly in the pixel grid of the
real photographs:

* Detect 2D feature points in each **real photo** (SIFT/ORB).
* Render the model from the same pose, detect features in the **render**,
  and — because we own the 3D scene — read back the **3D Gaussian centre**
  that projects to each rendered feature by rendering a depth/identity map.
* Match real-photo features ↔ render features across views.  Each match
  gives a real-photo pixel plus the 3D model point that should land there.
* Triangulate consistent real-photo pixels across ≥2 views into **scene
  3D points**, then solve a similarity transform (s, R, t) that maps the
  matched model points onto the triangulated scene points — a Kabsch /
  Umeyama problem.

The result is the model's world coordinates, recovered from photo content,
and because each new frame only adds a few correspondences the estimate
can be **incrementally refined** in real time.

Pipeline
--------
::

    real photo_i  ─┐                        ┌─ matched (pixel_i, model_pt)
    render_i      ─┤─ feature match ────────┤
    real photo_j  ─┤                        └─ matched (pixel_j, model_pt)
    render_j      ─┘

    (pixel_i, pixel_j) ──triangulate──▶ scene_pt   (uses real camera poses)
    (model_pt, scene_pt) pairs ──Umeyama──▶ (scale, R, t)

Coordinate conventions
-----------------------
Model points live in the TripoSplat AABB frame (same as
:func:`alignment.estimate_similarity_transform_multiview`).  Scene points
are in the Y-up world frame (:mod:`coord_registry`).  Cameras use the
OpenCV convention from :mod:`cameras`.  The returned ``(scale, R, t)`` has
the same meaning as everywhere else in the package::

    xyz_scene = scale * (R @ xyz_model) + t
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .cameras import Camera, invert_se3
from .gsplat_wrapper import render_gaussians, render_depth, render_alpha


# ═══════════════════════════════════════════════════════════════════════════
# 1. Feature extraction & matching (photo space)
# ═══════════════════════════════════════════════════════════════════════════

def _to_uint8(img: torch.Tensor) -> np.ndarray:
    """(3, H, W) float [0,1] → (H, W, 3) uint8 BGR-for-cv2 order is caller's
    responsibility; this returns RGB uint8."""
    arr = (img.detach().permute(1, 2, 0).cpu().numpy() * 255.0)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _make_detector(descriptor: str):
    """Build an OpenCV feature detector + descriptor.

    Returns a function ``detect(gray) -> (keypoints, descriptors)``.
    SIFT is preferred for accuracy; ORB is the fast fallback.
    """
    import cv2

    if descriptor == "sift":
        det = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.02,
                              edgeThreshold=20)
    elif descriptor == "orb":
        det = cv2.ORB_create(nfeatures=2000)
    else:
        raise ValueError(f"Unknown descriptor {descriptor!r}; use 'sift' or 'orb'")

    def detect(gray: np.ndarray):
        kp, des = det.detectAndCompute(gray, None)
        return kp, des

    return detect, descriptor


def _match_descriptors(des_a, des_b, descriptor: str, ratio: float = 0.75):
    """Lowe-ratio test matching. Returns array of (idx_a, idx_b) pairs."""
    import cv2

    if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
        return np.empty((0, 2), dtype=np.int64)

    if descriptor == "sift":
        # FLANN for SIFT (float32 descriptors).
        index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        # ORB: Hamming + brute force.
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = matcher.knnMatch(des_a, des_b, k=2)
    good = []
    for mm in matches:
        if len(mm) < 2:
            continue
        m, n = mm[0], mm[1]
        if m.distance < ratio * n.distance:
            good.append((m.queryIdx, m.trainIdx))
    return np.array(good, dtype=np.int64).reshape(-1, 2)


@dataclass
class ViewFeatures:
    """Features detected in one image (real or rendered)."""
    keypoints_xy: np.ndarray        # (K, 2) pixel coords
    descriptors: np.ndarray         # (K, D)
    source: str                     # "real" or "render"


def extract_features(image: torch.Tensor, descriptor: str = "sift") -> ViewFeatures:
    """Detect feature points in a (3, H, W) float image.

    Returns :class:`ViewFeatures` with pixel coordinates and descriptors.
    """
    import cv2

    detect, name = _make_detector(descriptor)
    rgb = _to_uint8(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kp, des = detect(gray)
    if des is None:
        pts = np.empty((0, 2), dtype=np.float32)
        des = np.empty((0, 1), dtype=np.float32)
    else:
        pts = np.array([k.pt for k in kp], dtype=np.float32)
    return ViewFeatures(keypoints_xy=pts, descriptors=des, source="?")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Render → 3D point lookup (the key step that makes it photo-space)
# ═══════════════════════════════════════════════════════════════════════════

def render_index_map(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    camera: Camera,
    alpha_threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Render a per-pixel **Gaussian index map**.

    For each pixel we record which Gaussian (by row index) contributed most
    to the alpha, plus the rendered alpha.  This is what lets us lift a 2D
    feature pixel in the render back to the 3D ``means`` row that produced
    it — the 2D→3D link that pure silhouette matching lacks.

    Implementation: gsplat does not expose a hit-id buffer directly in the
    high-level API, so we approximate it by rendering each Gaussian's
    normalised index as its "colour" and reading back the blended value.
    A blended index is a weighted average of contributor indices, which is
    accurate when one Gaussian dominates a pixel (the common case at
    feature keypoints on a surface).  We round to the nearest integer and
    mask by alpha.

    Returns:
        (index_map (H, W) int64, alpha (H, W) float32)
    """
    device = means.device
    N = means.shape[0]
    H, W = camera.height, camera.width

    # Encode index as colour: store i/N in the R channel, recover by
    # round(R * N).  Using a single channel keeps blending linear.
    idx_vals = (torch.arange(N, device=device, dtype=torch.float32) / max(N - 1, 1))
    colors = idx_vals.unsqueeze(1).expand(N, 3).contiguous()  # (N, 3)

    rgb, alpha, _ = render_gaussians(
        means, scales, quats, colors, opacities, camera,
        background=torch.zeros(3, device=device), sh_degree=None,
    )
    alpha_2d = alpha.squeeze(0)
    idx_r = rgb[0]
    idx_map = torch.round(idx_r * max(N - 1, 1)).long().clamp(0, N - 1)
    # Only trust the index where the surface is actually visible.
    idx_map = torch.where(alpha_2d > alpha_threshold, idx_map,
                          torch.full_like(idx_map, -1))
    return idx_map.cpu(), alpha_2d.cpu()


def features_to_3d(
    feats: ViewFeatures,
    idx_map: torch.Tensor,
    alpha: torch.Tensor,
    means: torch.Tensor,
    alpha_threshold: float = 0.1,
    border: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift 2D render features to 3D model points via the index map.

    Args:
        feats: Features detected in the rendered image.
        idx_map: (H, W) Gaussian index map from :func:`render_index_map`.
        alpha: (H, W) rendered alpha.
        means: (N, 3) Gaussian centres (model frame).
        alpha_threshold: discard features on near-transparent pixels.
        border: ignore features within *border* px of the image edge, where
            blending artefacts and false keypoints are most common.

    Returns:
        ``(pts2d (K,2), pts3d (K,3))`` — the render feature pixels that have
        a valid 3D back-link, and the corresponding model-frame 3D points.
    """
    H, W = idx_map.shape
    pts2d = feats.keypoints_xy
    if len(pts2d) == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    xs = np.clip(pts2d[:, 0].astype(np.int64), 0, W - 1)
    ys = np.clip(pts2d[:, 1].astype(np.int64), 0, H - 1)

    idx = idx_map.numpy()[ys, xs]
    al = alpha.numpy()[ys, xs]

    valid = (idx >= 0) & (al > alpha_threshold)
    # Drop border keypoints.
    valid &= (pts2d[:, 0] >= border) & (pts2d[:, 0] < W - border)
    valid &= (pts2d[:, 1] >= border) & (pts2d[:, 1] < H - border)

    idx = idx[valid]
    pts2d_v = pts2d[valid]
    means_np = means.detach().cpu().numpy()
    pts3d = means_np[idx]
    return pts2d_v.astype(np.float32), pts3d.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Cross-view correspondences & triangulation
# ═══════════════════════════════════════════════════════════════════════════

def _match_real_render(real_feat: ViewFeatures, rend_feat: ViewFeatures,
                       descriptor: str, ratio: float = 0.75):
    """Match real-photo features to render features.

    Returns ``(real_idx, rend_idx)`` arrays.
    """
    pairs = _match_descriptors(real_feat.descriptors, rend_feat.descriptors,
                               descriptor, ratio)
    if len(pairs) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return pairs[:, 0], pairs[:, 1]


def _fundamental_check(pts_a: np.ndarray, pts_b: np.ndarray, K_a, K_b,
                       w2c_a, w2c_b, thresh: float = 3.0) -> np.ndarray:
    """Epipolar consistency check between two views' matched pixels.

    Projects the matches through the *real* cameras' known poses and keeps
    those whose epipolar distance is below *thresh* pixels.  This rejects
    descriptor mismatches before triangulation.

    Returns a boolean mask over the input points.
    """
    # Essential matrix E = K_b^T [t]_x R K_a, with R,t from w2c_b @ c2w_a.
    c2w_a = invert_se3(w2c_a)
    rel = w2c_b @ c2w_a  # transforms a world point as seen by a into frame b
    R = rel[:3, :3].cpu().numpy()
    t = rel[:3, 3].cpu().numpy()

    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = K_b.T @ tx @ R @ K_a

    F = np.linalg.inv(K_b).T @ E @ np.linalg.inv(K_a)

    pts_a_h = np.hstack([pts_a, np.ones((len(pts_a), 1))])
    pts_b_h = np.hstack([pts_b, np.ones((len(pts_b), 1))])
    # Epipolar line in b: l_b = F @ p_a.  Distance = |p_b . l_b| / ||l_b[:2]||.
    lines_b = (F @ pts_a_h.T).T
    denom = np.linalg.norm(lines_b[:, :2], axis=1) + 1e-12
    dist_b = np.abs((pts_b_h * lines_b).sum(axis=1)) / denom
    # Symmetric: line in a from p_b.
    lines_a = (F.T @ pts_b_h.T).T
    denom_a = np.linalg.norm(lines_a[:, :2], axis=1) + 1e-12
    dist_a = np.abs((pts_a_h * lines_a).sum(axis=1)) / denom_a
    dist = np.maximum(dist_a, dist_b)
    return dist < thresh


def triangulate_points(cam_a: Camera, pts_a: np.ndarray,
                       cam_b: Camera, pts_b: np.ndarray) -> np.ndarray:
    """Linear DLT triangulation of matched pixel pairs from two views.

    Returns (M, 3) scene-frame 3D points.
    """
    P_a = (cam_a.w2c[:3, :].cpu().numpy())  # 3x4 projection (OpenCV cam)
    P_b = (cam_b.w2c[:3, :].cpu().numpy())

    # Intrinsics matrices.
    K_a = np.array([[cam_a.fx, 0, cam_a.cx], [0, cam_a.fy, cam_a.cy], [0, 0, 1]],
                   dtype=np.float64)
    K_b = np.array([[cam_b.fx, 0, cam_b.cx], [0, cam_b.fy, cam_b.cy], [0, 0, 1]],
                   dtype=np.float64)
    P_a = K_a @ P_a
    P_b = K_b @ P_b

    pts = []
    for (xa, ya), (xb, yb) in zip(pts_a, pts_b):
        A = np.stack([
            xa * P_a[2] - P_a[0],
            ya * P_a[2] - P_a[1],
            xb * P_b[2] - P_b[0],
            yb * P_b[2] - P_b[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        if abs(X[3]) < 1e-12:
            pts.append(np.array([np.nan, np.nan, np.nan]))
        else:
            pts.append(X[:3] / X[3])
    return np.array(pts, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Similarity transform: Umeyama (model points → scene points)
# ═══════════════════════════════════════════════════════════════════════════

def umeyama(model_pts: np.ndarray, scene_pts: np.ndarray,
            with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute the similarity transform mapping *model_pts* → *scene_pts*.

    Solves, in the least-squares sense::

        scene = scale * (R @ model) + t

    using the closed-form Umeyama method (rigid + uniform scale).  This is
    the heart of photo-space registration: once we have matched 3D model
    points and triangulated 3D scene points, the world coordinates of the
    *entire* Gaussian cloud follow from this single transform.

    Args:
        model_pts: (M, 3) source points (Gaussian centres, model frame).
        scene_pts: (M, 3) target points (triangulated from real photos).
        with_scale: estimate a uniform scale; False ⇒ R, t only (scale=1).

    Returns:
        ``(scale, R (3,3), t (3,))`` with the same convention as the rest
        of the package.
    """
    model_pts = np.asarray(model_pts, dtype=np.float64)
    scene_pts = np.asarray(scene_pts, dtype=np.float64)
    assert model_pts.shape == scene_pts.shape and model_pts.shape[1] == 3
    M = model_pts.shape[0]
    assert M >= 3, f"Need ≥3 correspondences for Umeyama, got {M}"

    mu_m = model_pts.mean(axis=0)
    mu_s = scene_pts.mean(axis=0)
    Mc = model_pts - mu_m
    Sc = scene_pts - mu_s

    # Cross-covariance.
    H = Mc.T @ Sc / M
    U, D, Vt = np.linalg.svd(H)

    # Correct for reflection.
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = Vt.T @ S @ U.T

    if with_scale:
        var_m = (Mc ** 2).sum() / M
        scale = (D * np.diag(S)).sum() / max(var_m, 1e-12)
    else:
        scale = 1.0

    t = mu_s - scale * (R @ mu_m)
    return float(scale), R.astype(np.float32), t.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Full registration driver
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Correspondence:
    """One (model 3D, scene 3D) correspondence plus provenance."""
    model_pt: np.ndarray       # (3,)
    scene_pt: np.ndarray       # (3,)
    views: list[int]           # camera indices that supported it
    reproj_err: float = 0.0    # mean reprojection error (px) over views


@dataclass
class RegistrationResult:
    """Outcome of :func:`register_photo_space`."""
    scale: float
    R: np.ndarray              # (3, 3)
    t: np.ndarray              # (3,)
    n_correspondences: int
    mean_reproj_err: float     # px
    inlier_ratio: float
    correspondences: list[Correspondence] = field(default_factory=list)


def _project(camera: Camera, pts3d: np.ndarray) -> np.ndarray:
    """Project (N, 3) world points into (N, 2) pixels via a Camera."""
    w2c = camera.w2c[:3, :].cpu().numpy()
    K = np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1]],
                 dtype=np.float64)
    p_cam = (w2c @ np.hstack([pts3d, np.ones((len(pts3d), 1))]).T).T
    z = p_cam[:, 2:3]
    z = np.where(np.abs(z) < 1e-12, 1e-12, z)
    uv = (K @ p_cam.T).T / z
    return uv[:, :2]


def _kabsch_ransac(model_pts: np.ndarray, scene_pts: np.ndarray,
                   max_iters: int = 1000, err_thresh: float = 0.02,
                   min_inliers: int = 8, seed: int = 0) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC over Umeyama to reject outlier correspondences.

    *err_thresh* is in scene-world units (metres).  Returns
    ``(scale, R, t, inlier_mask)``.
    """
    rng = np.random.default_rng(seed)
    M = len(model_pts)
    best_inliers = np.zeros(M, dtype=bool)
    best_score = 0
    if M < 3:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), best_inliers

    sample_size = min(3, M)
    for _ in range(max_iters):
        idx = rng.choice(M, sample_size, replace=False)
        try:
            s, R, t = umeyama(model_pts[idx], scene_pts[idx])
        except np.linalg.LinAlgError:
            continue
        pred = s * (model_pts @ R.T) + t
        err = np.linalg.norm(pred - scene_pts, axis=1)
        inliers = err < err_thresh
        score = int(inliers.sum())
        if score > best_score:
            best_score = score
            best_inliers = inliers
            if score >= M - 1:
                break

    if best_score >= 3:
        s, R, t = umeyama(model_pts[best_inliers], scene_pts[best_inliers])
    else:
        s, R, t = umeyama(model_pts, scene_pts)
        best_inliers = np.ones(M, dtype=bool)
    return s, R.astype(np.float32), t.astype(np.float32), best_inliers


def register_photo_space(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    cameras: list[Camera],
    descriptor: str = "sift",
    match_ratio: float = 0.75,
    epipolar_thresh: float = 3.0,
    ransac_err_thresh: float = 0.02,
    min_correspondences: int = 8,
    alpha_threshold: float = 0.1,
    max_views: int = 0,
    verbose: bool = True,
) -> Optional[RegistrationResult]:
    """Register a Gaussian model to world coordinates in photo space.

    This is the main entry point.  For each camera it (1) renders the model,
    (2) extracts features in both render and real photo, (3) matches them and
    lifts render features to 3D model points, then (4) triangulates the
    matched real-photo pixels across view pairs into scene 3D points and
    (5) solves the model→scene similarity transform with RANSAC-Umeyama.

    The transform recovers the model's **world coordinates** — exactly what
    is needed so that subsequent per-frame Gaussian updates can be applied
    in a consistent scene frame.

    Args:
        means, scales, quats, opacities: Gaussian attributes (model frame).
        cameras: Real cameras with loaded images and known poses.
        descriptor: ``"sift"`` (accurate) or ``"orb"`` (fast).
        match_ratio: Lowe ratio threshold.
        epipolar_thresh: Max epipolar distance (px) for a cross-view match.
        ransac_err_thresh: RANSAC inlier threshold in scene-world units (m).
        min_correspondences: Abort if fewer correspondences survive.
        alpha_threshold: Rendered-alpha cutoff for trusting a 3D back-link.
        max_views: Use at most this many cameras (0 = all).  Useful for
            real-time streaming where only the latest few frames matter.
        verbose: Print progress.

    Returns:
        :class:`RegistrationResult` or ``None`` if too few correspondences.
    """
    device = cameras[0].device
    means = means.detach().to(device).float()
    scales = scales.detach().to(device).float()
    quats = quats.detach().to(device).float()
    opacities = opacities.detach().to(device).float()

    cams = cameras if max_views <= 0 else cameras[:max_views]
    n = len(cams)
    if n < 2:
        if verbose:
            print("  register_photo_space: need ≥2 cameras, got", n)
        return None

    # ── Per-view: features + 3D back-links ──
    real_feats: list[ViewFeatures] = []
    rend_feats: list[ViewFeatures] = []
    # For each view, map render-feature index → (pixel2d, model_pt3d).
    rend_pts2d: list[np.ndarray] = []
    rend_pts3d: list[np.ndarray] = []

    if verbose:
        print(f"  register_photo_space: processing {n} views with {descriptor}...")
    for ci, cam in enumerate(cams):
        # Real photo features.
        if cam.image is None:
            real_feats.append(ViewFeatures(np.empty((0, 2)), np.empty((0, 1)), "real"))
            rend_feats.append(ViewFeatures(np.empty((0, 2)), np.empty((0, 1)), "render"))
            rend_pts2d.append(np.empty((0, 2)))
            rend_pts3d.append(np.empty((0, 3)))
            continue
        rf = extract_features(cam.image, descriptor)
        rf.source = "real"
        real_feats.append(rf)

        # Render the model from this pose and extract features + index map.
        rgb, alpha, _ = render_gaussians(
            means, scales, quats,
            torch.zeros((means.shape[0], 1, 3), device=device),
            opacities, cam,
            background=torch.zeros(3, device=device), sh_degree=0,
        )
        ef = extract_features(rgb, descriptor)
        ef.source = "render"
        rend_feats.append(ef)

        idx_map, alpha_2d = render_index_map(means, scales, quats, opacities, cam,
                                              alpha_threshold=alpha_threshold)
        p2, p3 = features_to_3d(ef, idx_map, alpha_2d, means,
                                alpha_threshold=alpha_threshold)
        rend_pts2d.append(p2)
        rend_pts3d.append(p3)

    # ── Build cross-view correspondences ──
    # For each view, match real↔render.  Then for each pair of views, find
    # matches that share the *same* model point (track across views) and
    # triangulate the real pixels.
    # We build, per view, an array real_idx → model_pt (via render match).
    model_pts_per_view: list[dict[int, np.ndarray]] = []
    real_pix_per_view: list[dict[int, np.ndarray]] = []
    for ci in range(n):
        mp = {}
        rp = {}
        if len(real_feats[ci].keypoints_xy) and len(rend_feats[ci].keypoints_xy):
            r_idx, e_idx = _match_real_render(real_feats[ci], rend_feats[ci],
                                              descriptor, match_ratio)
            for ri, ei in zip(r_idx, e_idx):
                if ei < len(rend_pts3d[ci]):
                    mp[int(ri)] = rend_pts3d[ci][ei]
                    rp[int(ri)] = rend_pts2d[ci][ei]
        model_pts_per_view.append(mp)
        real_pix_per_view.append(rp)

    # Group by model point identity (round to identify identical 3D points).
    # model_key = rounded model_pt; collect (view, real_pixel) per key.
    tracks: dict[tuple, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for ci in range(n):
        for ri, mpt in model_pts_per_view[ci].items():
            key = tuple(np.round(mpt, 4))
            tracks.setdefault(key, []).append((ci, real_pix_per_view[ci][ri], mpt))

    if verbose:
        print(f"  tracks: {len(tracks)} unique model points, "
              f"{sum(len(v) for v in tracks.values())} total view-observations")

    # Triangulate tracks seen in ≥2 views.
    corr_model: list[np.ndarray] = []
    corr_scene: list[np.ndarray] = []
    corr_views: list[list[int]] = []
    corr_reproj: list[float] = []
    for key, obs in tracks.items():
        if len(obs) < 2:
            continue
        # Use the first two observing views for triangulation.
        (ca, pa, ma), (cb, pb, mb) = obs[0], obs[1]
        scene_pt = triangulate_points(cams[ca], pa.reshape(1, 2),
                                      cams[cb], pb.reshape(1, 2))[0]
        if not np.all(np.isfinite(scene_pt)):
            continue
        # Reject points behind any camera.
        ok = True
        for ci, _, _ in obs:
            w2c = cams[ci].w2c[:3, :].cpu().numpy()
            p_cam = w2c @ np.append(scene_pt, 1.0)
            if p_cam[2] <= 0:
                ok = False
                break
        if not ok:
            continue

        # Reprojection error across all observing views.
        errs = []
        for ci, pxy, _ in obs:
            proj = _project(cams[ci], scene_pt.reshape(1, 3))[0]
            errs.append(float(np.linalg.norm(proj - pxy)))
        mean_err = float(np.mean(errs))

        corr_model.append(ma)
        corr_scene.append(scene_pt)
        corr_views.append([o[0] for o in obs])
        corr_reproj.append(mean_err)

    if len(corr_model) < min_correspondences:
        if verbose:
            print(f"  Too few correspondences: {len(corr_model)} < {min_correspondences}")
        return None

    model_arr = np.array(corr_model, dtype=np.float64)
    scene_arr = np.array(corr_scene, dtype=np.float64)

    # ── RANSAC Umeyama ──
    scale, R, t, inliers = _kabsch_ransac(
        model_arr, scene_arr,
        err_thresh=ransac_err_thresh, seed=0,
    )
    inlier_ratio = float(inliers.mean()) if len(inliers) else 0.0
    mean_reproj = float(np.mean([corr_reproj[i] for i in range(len(corr_reproj)) if inliers[i]])
                        ) if inliers.any() else float("nan")

    if verbose:
        print(f"  Umeyama: scale={scale:.4f}, inliers={int(inliers.sum())}/{len(inliers)} "
              f"({inlier_ratio:.0%}), mean reproj err={mean_reproj:.2f}px")

    correspondences = [
        Correspondence(model_pt=corr_model[i], scene_pt=corr_scene[i],
                       views=corr_views[i], reproj_err=corr_reproj[i])
        for i in range(len(corr_model)) if inliers[i]
    ]

    return RegistrationResult(
        scale=scale, R=R, t=t,
        n_correspondences=int(inliers.sum()),
        mean_reproj_err=mean_reproj,
        inlier_ratio=inlier_ratio,
        correspondences=correspondences,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6. Real-time incremental registration
# ═══════════════════════════════════════════════════════════════════════════

class IncrementalPhotoRegistrar:
    """Streaming photo-space registrar for real-time multi-view updates.

    As new camera frames arrive (e.g. from a live capture), this maintains
    a growing set of feature tracks and refines the model→scene similarity
    transform incrementally.  It is the entry point for the "real-time
    update of Gaussian content" the package targets: once the transform is
    stable, downstream code can splat new frames' Gaussians directly into
    the scene frame.

    Design:
      * The first frame establishes a set of model-point tracks.
      * Each subsequent frame is matched against the existing tracks (via
        descriptor matching to the nearest prior view) and new tracks are
        added.
      * After every *refine_every* frames (or when enough new tracks
        accumulate) the Umeyama transform is recomputed from all tracks.
      * The transform is **low-pass filtered** across updates to avoid
        jitter from noisy per-frame correspondences.
    """

    def __init__(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        opacities: torch.Tensor,
        descriptor: str = "orb",
        match_ratio: float = 0.8,
        refine_every: int = 3,
        min_tracks: int = 8,
        smoothing: float = 0.5,
        ransac_err_thresh: float = 0.02,
        min_scale: float = 0.01,
        max_scale: float = 100.0,
        verbose: bool = False,
    ):
        self.means = means.detach()
        self.scales = scales.detach()
        self.quats = quats.detach()
        self.opacities = opacities.detach()
        self.descriptor = descriptor
        self.match_ratio = match_ratio
        self.refine_every = refine_every
        self.min_tracks = min_tracks
        self.smoothing = smoothing
        self.ransac_err_thresh = ransac_err_thresh
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.verbose = verbose

        # Track store: model_key → list of (cam_idx, pixel2d, model_pt, descriptor).
        self._tracks: dict[tuple, list] = {}
        self._cams: list[Camera] = []
        self._frames_since_refine = 0
        # Current transform estimate.
        self._scale = 1.0
        self._R = np.eye(3, dtype=np.float32)
        self._t = np.zeros(3, dtype=np.float32)
        self._has_estimate = False

    @property
    def transform(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Current ``(scale, R, t)`` estimate (identity until enough tracks)."""
        return self._scale, self._R, self._t

    @property
    def num_tracks(self) -> int:
        return len(self._tracks)

    def add_frame(self, camera: Camera) -> Optional[RegistrationResult]:
        """Ingest one new camera frame and possibly refine the transform.

        Returns the :class:`RegistrationResult` if a refinement ran on this
        frame, else ``None``.
        """
        device = self.means.device
        cam = camera
        self._cams.append(cam)

        # Real features.
        if cam.image is None:
            return None
        rf = extract_features(cam.image, self.descriptor)
        rf.source = "real"

        # Render features + 3D back-links.
        rgb, alpha, _ = render_gaussians(
            self.means, self.scales, self.quats,
            torch.zeros((self.means.shape[0], 1, 3), device=device),
            self.opacities, cam,
            background=torch.zeros(3, device=device), sh_degree=0,
        )
        ef = extract_features(rgb, self.descriptor)
        ef.source = "render"
        idx_map, alpha_2d = render_index_map(
            self.means, self.scales, self.quats, self.opacities, cam,
            alpha_threshold=0.1,
        )
        p2, p3 = features_to_3d(ef, idx_map, alpha_2d, self.means, alpha_threshold=0.1)

        if len(rf.keypoints_xy) == 0 or len(ef.keypoints_xy) == 0:
            self._frames_since_refine += 1
            return None

        r_idx, e_idx = _match_real_render(rf, ef, self.descriptor, self.match_ratio)
        for ri, ei in zip(r_idx, e_idx):
            if ei < len(p3):
                key = tuple(np.round(p3[ei], 4))
                self._tracks.setdefault(key, []).append(
                    (len(self._cams) - 1, rf.keypoints_xy[ri], p3[ei]))

        self._frames_since_refine += 1
        if self._frames_since_refine < self.refine_every:
            return None
        self._frames_since_refine = 0
        return self.refine()

    def refine(self) -> Optional[RegistrationResult]:
        """Recompute the similarity transform from all accumulated tracks."""
        corr_model = []
        corr_scene = []
        corr_views = []
        corr_reproj = []
        for key, obs in self._tracks.items():
            if len(obs) < 2:
                continue
            (ca, pa, ma), (cb, pb, mb) = obs[0], obs[1]
            scene_pt = triangulate_points(self._cams[ca], pa.reshape(1, 2),
                                          self._cams[cb], pb.reshape(1, 2))[0]
            if not np.all(np.isfinite(scene_pt)):
                continue
            ok = True
            for ci, _, _ in obs:
                w2c = self._cams[ci].w2c[:3, :].cpu().numpy()
                if (w2c @ np.append(scene_pt, 1.0))[2] <= 0:
                    ok = False
                    break
            if not ok:
                continue
            errs = []
            for ci, pxy, _ in obs:
                proj = _project(self._cams[ci], scene_pt.reshape(1, 3))[0]
                errs.append(float(np.linalg.norm(proj - pxy)))
            corr_model.append(ma)
            corr_scene.append(scene_pt)
            corr_views.append([o[0] for o in obs])
            corr_reproj.append(float(np.mean(errs)))

        if len(corr_model) < self.min_tracks:
            if self.verbose:
                print(f"  incremental: {len(corr_model)} tracks < {self.min_tracks}, waiting")
            return None

        model_arr = np.array(corr_model, dtype=np.float64)
        scene_arr = np.array(corr_scene, dtype=np.float64)
        scale, R, t, inliers = _kabsch_ransac(
            model_arr, scene_arr, err_thresh=self.ransac_err_thresh, seed=0)

        # Degeneracy guard: a near-zero or wildly large scale means the
        # correspondences collapsed (e.g. all model points mapped to one
        # scene point).  Reject the update and keep the previous estimate
        # rather than poisoning the state.
        if not (self.min_scale < scale < self.max_scale):
            if self.verbose:
                print(f"  incremental: degenerate scale {scale:.4f} rejected, "
                      f"keeping previous estimate")
            return None

        # Low-pass smoothing against the previous estimate.
        if self._has_estimate and self.smoothing > 0:
            a = self.smoothing
            scale = a * self._scale + (1 - a) * scale
            # SLERP between rotation matrices via quaternions.
            from .quaternion import matrix_to_quat, quat_to_matrix
            q0 = matrix_to_quat(self._R[None])[0]
            q1 = matrix_to_quat(R[None])[0]
            qs = _slerp_quat(q0, q1, 1 - a)
            R = quat_to_matrix(qs[None])[0].astype(np.float32)
            t = (a * self._t + (1 - a) * t).astype(np.float32)

        self._scale = float(scale)
        self._R = np.asarray(R, dtype=np.float32)
        self._t = np.asarray(t, dtype=np.float32)
        self._has_estimate = True

        mean_reproj = float(np.mean([corr_reproj[i] for i in range(len(corr_reproj)) if inliers[i]])
                            ) if inliers.any() else float("nan")
        if self.verbose:
            print(f"  incremental refine: scale={self._scale:.4f}, "
                  f"inliers={int(inliers.sum())}/{len(inliers)}, "
                  f"reproj={mean_reproj:.2f}px, tracks={len(self._tracks)}")

        return RegistrationResult(
            scale=self._scale, R=self._R, t=self._t,
            n_correspondences=int(inliers.sum()),
            mean_reproj_err=mean_reproj,
            inlier_ratio=float(inliers.mean()) if len(inliers) else 0.0,
        )


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two (w,x,y,z) quaternions."""
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * t
    sin_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * math.sin(theta) / sin_0
    s1 = math.sin(theta) / sin_0
    return s0 * q0 + s1 * q1
