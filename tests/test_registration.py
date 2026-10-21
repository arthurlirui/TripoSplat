"""Unit tests for closed-form multi-view registration.

Covers the geometry primitives (triangulation, up-vector alignment, projection,
view selection) without a GPU, plus an end-to-end synthetic recovery test that
is skipped when CUDA is unavailable.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiview_refine.cameras import Camera, invert_se3
from multiview_refine.registration import (
    ViewObservation,
    camera_ray,
    extract_observations,
    project_points,
    rotation_aligning,
    rotation_y,
    select_views,
    triangulate,
)


# ── helpers ─────────────────────────────────────────────────────────────────

def make_camera(position, look_at, fx=500.0, fy=500.0, W=640, H=480, up=(0, 1, 0)):
    """OpenCV-convention camera (camera +Y points down in the image)."""
    position = torch.tensor(position, dtype=torch.float64)
    look_at = torch.tensor(look_at, dtype=torch.float64)
    up = torch.tensor(up, dtype=torch.float64)

    forward = look_at - position
    forward = forward / forward.norm()
    right = torch.cross(forward, up, dim=0)
    right = right / right.norm()
    # OpenCV is right-handed with z = x × y, so y (image down) = z × x.
    # Using ``cross(right, forward)`` here would produce det = -1, i.e. a
    # mirrored camera whose "up" points into the ground.
    down = torch.cross(forward, right, dim=0)
    down = down / down.norm()

    c2w = torch.eye(4, dtype=torch.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = position
    c2w = c2w.float()

    return Camera(
        w2c=invert_se3(c2w), c2w=c2w, position=c2w[:3, 3].clone(),
        fx=fx, fy=fy, cx=W / 2.0, cy=H / 2.0, width=W, height=H,
    )


def orbit_cameras(n=8, radius=4.0, height=0.4, target=(0.0, 0.0, 0.0)):
    cams = []
    tgt = np.array(target, dtype=np.float64)
    for i in range(n):
        a = 2 * math.pi * i / n
        pos = tgt + np.array([radius * math.sin(a), height, radius * math.cos(a)])
        cams.append(make_camera(pos, tgt))
    return cams


# ── triangulation ───────────────────────────────────────────────────────────

def test_triangulate_recovers_point():
    """Rays from several cameras through a known point meet at that point."""
    X = np.array([0.3, -0.4, 1.2])
    cams = orbit_cameras(8)
    rays = []
    for cam in cams:
        u, v, z = project_points(cam, X[None, :])
        assert z[0] > 0
        rays.append(camera_ray(cam, (u[0], v[0])))

    est, res = triangulate(rays)
    np.testing.assert_allclose(est, X, atol=1e-4)
    assert res.mean() < 1e-4


def test_triangulate_rejects_outliers():
    """IRLS keeps a few corrupted rays from dragging the estimate."""
    X = np.array([0.0, 0.5, 2.0])
    cams = orbit_cameras(12)
    rays = []
    for i, cam in enumerate(cams):
        u, v, z = project_points(cam, X[None, :])
        if i < 3:
            # Steer three rays far off target (simulates a hand/hat top).
            u, v = u[0] + 0.35 * cam.width, v[0] + 0.3 * cam.height
        rays.append(camera_ray(cam, (u, v)))

    est, _ = triangulate(rays, iters=6)
    # Plain least squares would be pulled metres away; IRLS must stay close.
    assert np.linalg.norm(est - X) < 0.05, f"IRLS failed: {est}"


def test_triangulate_needs_two_rays():
    X = np.array([1.0, 0.0, 0.0])
    cams = orbit_cameras(4)[:2]
    rays = []
    for cam in cams:
        u, v, _ = project_points(cam, X[None, :])
        rays.append(camera_ray(cam, (u[0], v[0])))
    est, _ = triangulate(rays)
    # Two rays determine the point up to the depth along their plane.
    assert np.linalg.norm(est - X) < 0.05


# ── up-axis alignment ───────────────────────────────────────────────────────

def test_rotation_aligning_maps_vector_exactly():
    for src, dst in [
        (np.array([0.0, 1.0, 0.0]), np.array([0.1, 0.98, 0.05])),
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
    ]:
        R = rotation_aligning(src, dst)
        got = R @ (src / np.linalg.norm(src))
        exp = dst / np.linalg.norm(dst)
        np.testing.assert_allclose(got, exp, atol=1e-9)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0)


def test_rotation_aligning_handles_antiparallel():
    """Rodrigues is singular for opposite vectors; the fallback must still work."""
    src = np.array([0.0, 1.0, 0.0])
    R = rotation_aligning(src, -src)
    np.testing.assert_allclose(R @ src, -src, atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_rotation_y_preserves_y_and_rotates_xz():
    for deg in (0.0, 30.0, -75.0, 180.0):
        R = rotation_y(math.radians(deg))
        np.testing.assert_allclose(R @ np.array([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
        # R_y(θ) applied to +X gives (cosθ, 0, -sinθ).
        np.testing.assert_allclose(
            R @ np.array([1.0, 0.0, 0.0]),
            [math.cos(math.radians(deg)), 0.0, -math.sin(math.radians(deg))],
            atol=1e-12,
        )


# ── projection ──────────────────────────────────────────────────────────────

def test_project_points_matches_pinhole():
    cam = make_camera((0, 0, 5), (0, 0, 0))
    # +Y world is *up*, and image v grows downward, so a point above the
    # optical axis must land at v < cy.
    X = np.array([[0.0, 0.0, 0.0], [0.5, 0.25, 0.0]])
    u, v, z = project_points(cam, X)
    np.testing.assert_allclose(u, [cam.cx, cam.cx + cam.fx * 0.5 / 5.0], rtol=1e-6)
    np.testing.assert_allclose(v, [cam.cy, cam.cy - cam.fy * 0.25 / 5.0], rtol=1e-6)
    np.testing.assert_allclose(z, [5.0, 5.0], rtol=1e-6)


def test_project_points_flags_points_behind_camera():
    cam = make_camera((0, 0, 5), (0, 0, 0))
    _, _, z = project_points(cam, np.array([[0.0, 0.0, 10.0]]))
    assert z[0] < 0


# ── view selection ──────────────────────────────────────────────────────────

def _fake_obs(idx, clipped, contact=0.0, area=1000):
    return ViewObservation(
        view_idx=idx, cam=None, mask=np.zeros((4, 4), bool),
        centroid_uv=np.zeros(2), top_uv=np.zeros(2), bottom_uv=np.zeros(2),
        area=area, border_contact=contact, clipped=clipped,
    )


def test_select_views_prefers_uncropped():
    obs = [_fake_obs(i, clipped=(i < 4), contact=0.5) for i in range(12)]
    selected, info = select_views(obs, min_valid_views=6)
    assert info["relaxed"] is False
    assert len(selected) == 8
    assert all(not o.clipped for o in selected)


def test_select_views_relaxes_when_starved():
    """With too few clean views, fall back to the least-contaminated ones."""
    obs = [_fake_obs(0, False, area=500),
           _fake_obs(1, True, contact=1.0, area=900),
           _fake_obs(2, True, contact=0.25, area=800),
           _fake_obs(3, True, contact=0.75, area=700)]
    selected, info = select_views(obs, min_valid_views=6)
    assert info["relaxed"] is True
    assert len(selected) == 4
    # Lowest border contact first.
    assert selected[0].view_idx == 0
    assert selected[1].view_idx == 2


def test_select_views_weights_penalise_clipping():
    obs = [_fake_obs(0, False, contact=0.0, area=1000),
           _fake_obs(1, False, contact=0.5, area=1000)]
    selected, _ = select_views(obs, min_valid_views=1)
    assert selected[0].weight > selected[1].weight


# ── observations from real masks ────────────────────────────────────────────

def test_extract_observations_flags_border_contact():
    """A silhouette touching an edge must be marked clipped."""
    cam = make_camera((0, 0, 5), (0, 0, 0), W=64, H=64)
    alpha_full = torch.zeros(1, 64, 64)
    alpha_full[0, :, 10:40] = 1.0          # spans top to bottom -> clipped
    cam.alpha = alpha_full
    obs = extract_observations([cam])
    assert len(obs) == 1 and obs[0].clipped

    alpha_in = torch.zeros(1, 64, 64)
    alpha_in[0, 15:50, 20:40] = 1.0        # interior only
    cam.alpha = alpha_in
    obs = extract_observations([cam])
    assert len(obs) == 1 and not obs[0].clipped
    assert obs[0].top_uv[1] == pytest.approx(15.0)
    assert obs[0].bottom_uv[1] == pytest.approx(49.0)


def test_extract_observations_ignores_empty_masks():
    cam = make_camera((0, 0, 5), (0, 0, 0), W=64, H=64)
    cam.alpha = torch.zeros(1, 64, 64)
    assert extract_observations([cam]) == []


# ── end-to-end synthetic recovery ───────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for rendering")
def test_registration_recovers_synthetic_transform():
    """Render a known transform's silhouettes, then recover that transform."""
    from pointsplat_bridge._gsplat_preload import preload_gsplat_cuda
    preload_gsplat_cuda()

    from multiview_refine.coord_registry import MODEL2SCENE
    from multiview_refine.gsplat_wrapper import render_alpha
    from multiview_refine.quaternion import (
        quaternion_compose, rotation_matrix_to_quaternion,
    )
    from multiview_refine.registration import (
        register_gaussians_to_scene, verify_registration,
    )

    dev = "cuda"
    torch.manual_seed(0)

    # A crude "person" in the TripoSplat **model** frame: elongated along the
    # model's up axis, which is +Z (see coord_registry.MODEL2SCENE).
    N = 3000
    pts = torch.randn(N, 3) * torch.tensor([0.25, 0.25, 1.0])
    pts = torch.clamp(pts, -0.5, 0.5)
    means = pts
    scales = torch.full((N, 3), 0.015)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(N, 1)
    opac = torch.full((N,), 0.95)

    s_true, yaw_true = 1.4, math.radians(-70.0)
    R_true = (torch.from_numpy(rotation_y(yaw_true)).float().to(dev)
              @ MODEL2SCENE.to(dev))
    t_true = torch.tensor([0.1, 0.0, -0.15], device=dev)

    means_w = s_true * (means.to(dev) @ R_true.T) + t_true
    q_w = quaternion_compose(rotation_matrix_to_quaternion(R_true), quats.to(dev))

    cams = [c.to(dev) for c in orbit_cameras(10, radius=4.0, height=0.5)]
    for cam in cams:
        with torch.no_grad():
            a = render_alpha(means_w, scales.to(dev) * s_true, q_w, opac.to(dev), cam)
        cam.alpha = a.detach()

    res = register_gaussians_to_scene(
        means.to(dev), scales.to(dev), quats.to(dev), opac.to(dev), cams,
        verbose=False,
    )

    # Scale within 8 %, yaw within 10°, centre within 10 cm.
    assert res.scale == pytest.approx(s_true, rel=0.08), f"scale {res.scale} vs {s_true}"
    yaw_err = abs(((math.degrees(res.yaw) - math.degrees(yaw_true) + 180) % 360) - 180)
    assert yaw_err < 10.0, f"yaw error {yaw_err:.1f} deg"

    metrics = verify_registration(
        res, means.to(dev), scales.to(dev), quats.to(dev), opac.to(dev), cams)
    assert metrics["mean_iou"] > 0.8, metrics
    assert not metrics["flip_suspected"], metrics
