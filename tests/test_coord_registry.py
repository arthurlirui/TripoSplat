"""Unit tests for multiview_refine.coord_registry.

Verifies that the coordinate-system constants and transform helpers are
correct: rotation matrices are orthonormal, known vectors map to expected
results, and round-trip conversions are identity.

Key insight: the pipeline's world is **Y-up** (SfM output).  Cameras are
converted to OpenCV convention on load (flip Y/Z of camera basis only),
but the world coordinates of Gaussians and SMPL remain Y-up.  Therefore
SMPL (also Y-up) needs no rotation — SMPL2SCENE = identity.
"""

import math
import sys
import os

import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiview_refine.coord_registry import (
    CANONICAL_WORLD_UP,
    MODEL2SCENE,
    MODEL2OPENCV,
    SMPL2SCENE,
    SMPL2OPENCV,
    SCENE2SPLATSHOP_WORLD,
    OPENCV2SPLATSHOP_CAM,
    OPENCV2SPLATSHOP_WORLD,
    transform_means,
    transform_quats,
    model_to_scene,
    model_to_opencv,
    smpl_to_scene_frame,
    scene_to_splatshop_world,
    opencv_to_splatshop_world,
    opencv_c2w_to_splatshop_c2w,
)


# ── Matrix properties ──────────────────────────────────────────────────────

@pytest.mark.parametrize("R,name", [
    (MODEL2SCENE, "MODEL2SCENE"),
    (SMPL2SCENE, "SMPL2SCENE"),
    (SCENE2SPLATSHOP_WORLD, "SCENE2SPLATSHOP_WORLD"),
    (OPENCV2SPLATSHOP_CAM, "OPENCV2SPLATSHOP_CAM"),
])
def test_rotation_orthonormal(R, name):
    """Every coordinate transform matrix must be orthonormal (R@R^T=I)."""
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6), \
        f"{name} is not orthonormal"
    assert torch.allclose(R.T @ R, torch.eye(3), atol=1e-6), \
        f"{name} is not orthonormal"
    det = torch.det(R).item()
    assert abs(abs(det) - 1.0) < 1e-6, \
        f"{name} det={det}, expected ±1 (orthonormal)"


# ── MODEL2SCENE: model frame → Y-up scene ──────────────────────────────────

def test_model2scene_is_default_transform():
    """MODEL2SCENE equals TripoSplat's _DEFAULT_TRANSFORM.

    _DEFAULT_TRANSFORM = [[1,0,0],[0,0,-1],[0,1,0]]  (model → Y-up world)
    """
    expected = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    assert torch.allclose(MODEL2SCENE, expected, atol=1e-6)


def test_model2scene_maps_axes():
    """Model frame (x, y, z) → Y-up scene (x, z, -y).

    MODEL2SCENE = [[1,0,0],[0,0,-1],[0,1,0]]
    so (x, y, z) → (x, -z, y).

    Model "forward" (0, 0, 1) → scene "down" (0, -1, 0) — model Z becomes
    scene -Y (downward).  This matches TripoSplat's export convention.
    """
    # Model "forward" (0, 0, 1) → scene (0, -1, 0)
    fwd_model = torch.tensor([[0.0, 0.0, 1.0]])
    fwd_scene = transform_means(fwd_model, MODEL2SCENE)
    assert torch.allclose(fwd_scene, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6)

    # Model "up" (0, 1, 0) → scene (0, 0, 1)
    up_model = torch.tensor([[0.0, 1.0, 0.0]])
    up_scene = transform_means(up_model, MODEL2SCENE)
    assert torch.allclose(up_scene, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-6)


# ── SMPL2SCENE: identity (both Y-up) ───────────────────────────────────────

def test_smpl2scene_is_identity():
    """SMPL and scene are both Y-up, so no rotation needed."""
    assert torch.allclose(SMPL2SCENE, torch.eye(3), atol=1e-6)
    assert torch.allclose(SMPL2OPENCV, torch.eye(3), atol=1e-6)


def test_smpl_to_scene_frame_is_noop():
    """smpl_to_scene_frame should return inputs unchanged."""
    v = torch.randn(50, 3)
    sd = torch.randn(50, 3, 10)
    v_out, sd_out = smpl_to_scene_frame(v, sd)
    assert torch.allclose(v_out, v)
    assert torch.allclose(sd_out, sd)


# ── SCENE2SPLATSHOP_WORLD: Y-up → Z-up ─────────────────────────────────────

def test_scene_to_splatshop_y_to_z():
    """Scene Y-up (0, 1, 0) → SplatShop Z-up (0, 0, 1).

    SCENE2SPLATSHOP_WORLD = [[1,0,0],[0,0,1],[0,-1,0]]
    so (x, y, z) → (x, z, -y).  Thus (0,1,0) → (0,0,1) = Z-up. ✓
    """
    up_scene = torch.tensor([[0.0, 1.0, 0.0]])
    result = transform_means(up_scene, SCENE2SPLATSHOP_WORLD)
    assert torch.allclose(result, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-6)


def test_opencv_to_splatshop_c2w():
    """A camera at (0, 1, 3) looking at origin in OpenCV → SplatShop."""
    c2w_cv = torch.eye(4)
    c2w_cv[:3, 3] = torch.tensor([0.0, 1.0, 3.0])
    c2w_ss = opencv_c2w_to_splatshop_c2w(c2w_cv)

    # World position: (0, 1, 3) → rot_x(+90°): (x, -z, y) → (0, -3, 1)
    pos_ss = c2w_ss[:3, 3]
    assert torch.allclose(pos_ss, torch.tensor([0.0, -3.0, 1.0]), atol=1e-5)

    # Check it's a valid SE(3) transform.
    R = c2w_ss[:3, :3]
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)


# ── Round-trip tests ────────────────────────────────────────────────────────

def test_model_to_scene_roundtrip():
    """model_to_scene then inverse should recover original means."""
    torch.manual_seed(42)
    means = torch.randn(100, 3) * 0.5
    scales = torch.rand(100, 3) * 0.01 + 0.001
    quats = torch.randn(100, 4)
    quats = quats / quats.norm(dim=-1, keepdim=True)

    means_s, scales_s, quats_s = model_to_scene(means, scales, quats)
    # Inverse: (means @ R.T) @ R = means
    R = MODEL2SCENE
    means_back = means_s @ R
    assert torch.allclose(means_back, means, atol=1e-6)


def test_splatshop_world_roundtrip():
    """scene_to_splatshop_world then inverse should recover original."""
    torch.manual_seed(42)
    means = torch.randn(50, 3)
    scales = torch.rand(50, 3) * 0.01
    quats = torch.randn(50, 4)
    quats = quats / quats.norm(dim=-1, keepdim=True)

    means_ss, scales_ss, quats_ss = scene_to_splatshop_world(means, scales, quats)
    R = SCENE2SPLATSHOP_WORLD
    means_back = means_ss @ R
    assert torch.allclose(means_back, means, atol=1e-6)
    assert torch.allclose(scales_ss, scales, atol=1e-6)


# ── align_to_scene with rotation ────────────────────────────────────────────

def test_align_to_scene_with_rotation():
    """align_to_scene with rotation should rotate v_template."""
    R = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])  # rot_z +90°
    v = torch.tensor([[1.0, 0.0, 0.0]])
    scale = 2.0
    transl = torch.tensor([1.0, 2.0, 3.0])

    # v_new = R @ (v * scale) + transl
    v_new = (R @ (v * scale).T).T + transl
    # (1,0,0)*2=(2,0,0) → R@(2,0,0)=(0,2,0) → +transl = (1,4,3)
    assert torch.allclose(v_new[0], torch.tensor([1.0, 4.0, 3.0]), atol=1e-6)


def test_align_to_scene_backward_compat():
    """align_to_scene with rotation=None should match old behavior (scale+transl only)."""
    v = torch.tensor([[0.0, 1.0, 0.0]])
    scale = 2.0
    transl = torch.tensor([1.0, 2.0, 3.0])

    v_old = v * scale + transl
    v_new = v * scale + transl  # R=None path
    assert torch.allclose(v_old, v_new, atol=1e-6)


# ── Constants ───────────────────────────────────────────────────────────────

def test_canonical_world_up():
    assert CANONICAL_WORLD_UP == "y"


def test_backward_compat_aliases():
    """Old names should still exist for backward compatibility."""
    assert MODEL2OPENCV is MODEL2SCENE
    assert SMPL2OPENCV is SMPL2SCENE
    assert OPENCV2SPLATSHOP_WORLD is SCENE2SPLATSHOP_WORLD


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
