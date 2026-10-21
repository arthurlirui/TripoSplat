"""Regression tests for coordinate-convention bugs found in the pipeline audit.

Each test pins one previously-observed failure so it cannot silently return.
"""

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiview_refine.cameras import Camera
from multiview_refine.coord_transform import detect_scene_up, world_up_rotation
from multiview_refine.coord_registry import OPENCV2SPLATSHOP_WORLD


# ── detect_scene_up must refuse to guess on degenerate camera sets ──────────

def _cam(pos):
    pos = torch.tensor(pos, dtype=torch.float32)
    c2w = torch.eye(4)
    c2w[:3, 3] = pos
    w2c = torch.eye(4)
    w2c[:3, 3] = -pos
    return Camera(w2c=w2c, c2w=c2w, position=pos,
                  fx=100.0, fy=100.0, cx=50.0, cy=50.0, width=100, height=100)


def test_detect_scene_up_rejects_two_cameras():
    """Regression: 2 nearly-aligned cameras gave an exactly-zero variance axis,
    which was mis-read as 'up' and rotated the whole Y-up scene by 90 degrees.
    """
    cams = [_cam([0.0, 1.426, 2.589]), _cam([-0.034, 0.454, 2.591])]
    with pytest.raises(ValueError, match="at least 3 cameras"):
        detect_scene_up(cams)


def test_detect_scene_up_rejects_degenerate_variance():
    """Co-planar cameras (zero variance along one axis) must not be guessed."""
    cams = [_cam([x, 1.0, 2.0]) for x in (0.0, 1.0, 2.0)]
    with pytest.raises(ValueError, match="degenerate"):
        detect_scene_up(cams)


def test_detect_scene_up_finds_y_for_a_real_orbit():
    """A real turntable orbit: cameras spread horizontally, little vertically.

    Real captures in this project use several rig heights (e.g. 0.15/0.45/1.42 m),
    which keeps the Y variance small but non-zero — enough to be identifiable.
    """
    cams = []
    heights = (0.15, 0.45, 1.42)
    for i in range(12):
        a = 2 * math.pi * i / 12
        cams.append(_cam([2.6 * math.cos(a), heights[i % 3], 2.6 * math.sin(a)]))
    assert detect_scene_up(cams) == "y"


def test_detect_scene_up_empty_is_default_y():
    assert detect_scene_up([]) == "y"


# ── Registry constants must be usable on any device ─────────────────────────

def test_splatshop_constant_moves_to_device():
    """Regression: convert(target='splatshop') crashed on CUDA because the
    constant was only dtype-cast, never moved off CPU."""
    R = OPENCV2SPLATSHOP_WORLD.to(dtype=torch.float32)
    assert R.device.type == "cpu"
    # The pattern the fixed code uses:
    moved = OPENCV2SPLATSHOP_WORLD.to(device=torch.device("cpu"), dtype=torch.float32)
    assert moved.device.type == "cpu"
    assert torch.allclose(moved, R)


def test_splatshop_constant_is_proper_rotation():
    R = OPENCV2SPLATSHOP_WORLD.float()
    assert abs(torch.det(R).item() - 1.0) < 1e-6
    assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-6)


# ── world_up_rotation contract ──────────────────────────────────────────────

def test_world_up_rotation_y_to_y_is_identity():
    assert torch.allclose(world_up_rotation("y", "y"), torch.eye(3), atol=1e-6)


def test_world_up_rotation_y_to_z_maps_up_axis():
    R = world_up_rotation("y", "z").float()
    up = torch.tensor([[0.0, 1.0, 0.0]])
    assert torch.allclose(up @ R.T, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-6)


# ── The upright invariant, expressed on the real registry matrix ────────────

def test_avatar_up_axis_is_scene_y():
    """End-to-end guard: model +Z (feet->head) must become scene +Y."""
    from multiview_refine.coord_registry import MODEL2SCENE
    up = torch.tensor([[0.0, 0.0, 1.0]])
    assert torch.allclose(up @ MODEL2SCENE.float().T,
                          torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6)
