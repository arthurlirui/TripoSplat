"""Coordinate-system registry — single source of truth for frame conventions.

Coordinate frames in play
-------------------------
The pipeline distinguishes **camera convention** from **world convention**:

* **Camera convention** — how the camera's local axes are oriented:
    - OpenGL/Nerfstudio: +X right, +Y up, -Z forward
    - OpenCV:            +X right, +Y down, +Z forward
  ``cameras.py`` converts OpenGL c2w → OpenCV c2w on load (``diag(1,-1,-1,1)``).
  This flips only the camera's basis vectors, **not** the world.

* **World convention** — how the scene's world axes are oriented.  This is
  determined by the SfM reconstruction and is **Y-up** for the turntable body
  scan datasets used here (camera positions have the smallest spread along Y).

  Gaussians live in the **Y-up world frame**.  gsplat renders them correctly
  using OpenCV w2c matrices — the camera convention flip is handled by the
  w2c, and the world stays Y-up.

* **SMPL frame** — Y-up, Z-forward, origin at pelvis (~1.7 m tall).  Since the
  world is also Y-up, SMPL requires **no rotation** to align — only scale +
  translation.

* **SplatShop frame** — OpenGL camera (+X R, +Y U, -Z F) + Z-up world.  Export
  rotates the Y-up world to Z-up and flips cameras back to OpenGL.

* **TripoSplat model frame** — normalized AABB ``[-0.5, 0.5]³``, **Z-up**: the
  avatar's head→feet axis is model **+Z**.  ``MODEL2SCENE`` below rotates it
  into the Y-up scene frame.

Conversions
-----------
Model → Scene (Y-up) : ``MODEL2SCENE`` (3×3 rotation)
SMPL  → Scene (Y-up) : identity (both Y-up) — ``SMPL2SCENE = I``
Scene (Y-up) → SplatShop world : ``SCENE2SPLATSHOP_WORLD`` (rot_x −90°: Y-up→Z-up)
OpenCV cam → SplatShop cam     : ``OPENCV2SPLATSHOP_CAM`` (diag(1,-1,-1))
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from .quaternion import quaternion_compose, rotation_matrix_to_quaternion

# ── Canonical world frame ───────────────────────────────────────────────────
#
# The system's world is **Y-up** (determined by SfM).  Cameras are stored as
# OpenCV w2c for gsplat, but the world coordinates of Gaussians and SMPL are
# Y-up.  This is sometimes called "OpenGL world + OpenCV camera".

CANONICAL_WORLD_UP: Literal["y"] = "y"

# ── TripoSplat model frame → Y-up scene ─────────────────────────────────────
#
# TripoSplat stores Gaussians in a Z-up AABB [-0.5, 0.5]³ whose +Z axis runs
# from the avatar's feet toward its head.  The scene world is Y-up, so the
# model must be rotated -90° about X.
#
# Convention note: every consumer applies this matrix as ``means @ R.T``
# (row-vector form), which is equivalent to the column-vector product
# ``R @ means``.  So ``MODEL2SCENE`` below must be written in **column-vector**
# form to yield the documented mapping (x, y, z) → (x, z, -y):
#
#     R = [[1, 0, 0],      R @ (0, 1, 0) = (0, 0, -1)   # model Y → scene -Z
#          [0, 0, 1],      R @ (0, 0, 1) = (0, 1, 0)    # model Z → scene +Y (up)
#          [0, -1, 0]]
#
# An earlier version stored the matrix in Tripo's *row-vector* export form
# ([[1,0,0],[0,0,-1],[0,1,0]], i.e. the transpose of the above).  Because the
# code applies it with ``means @ R.T``, that transposition mapped model +Z to
# scene -Y and left every render 180° upside down.  See tests
# ``test_model2scene_upright`` which pins the avatar's up axis.
MODEL2SCENE = torch.tensor(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, -1.0, 0.0]],
    dtype=torch.float32,
)

# Backward-compatible alias (the old name was MODEL2OPENCV, but the transform
# target is the Y-up scene, not an OpenCV world).
MODEL2OPENCV = MODEL2SCENE  # deprecated alias, kept for run_pipeline.py

# ── Which way does the avatar face, after MODEL2SCENE? ──────────────────────
#
# Verified empirically by rendering the converted model from the four scene
# axes (see output/pipeline_fixed/model_facing.png): a camera on **+X** sees
# the face, one on -X sees the back.  TripoSplat always generates the subject
# facing its canonical camera, so once MODEL2SCENE is applied the facing
# direction is a fixed scene axis — it is a constant, not something to search
# for.
#
# Because the model is generated *from* a specific view, that view's camera
# must lie on this axis: ``align_from_init_view`` uses the constant to solve
# the yaw analytically instead of guessing it from silhouettes.
MODEL_FRONT_SCENE = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

# ── SMPL Y-up → Y-up scene: identity ────────────────────────────────────────
#
# SMPL native frame is Y-up, Z-forward — exactly the same as the Y-up scene
# world.  No rotation is needed; only scale + translation via align_to_scene.
#
# The previous version used diag(1,-1,1) which incorrectly flipped SMPL
# upside-down.  That was based on the mistaken assumption that the scene
# world was OpenCV Y-down.  In reality, cameras.py only flips the camera
# basis (not the world), so the world remains Y-up.
SMPL2SCENE = torch.eye(3, dtype=torch.float32)

# Backward-compatible alias.
SMPL2OPENCV = SMPL2SCENE  # identity — no rotation needed

# ── Scene (Y-up) → SplatShop (Z-up world) ───────────────────────────────────
#
# SplatShop uses OpenGL cameras (+X R, +Y U, -Z F) with a Z-up world.
# World: Y-up → Z-up = rot_x(+90°): (x, y, z) → (x, -z, y)
#   [[1,0,0],[0,cos(90),-sin(90)],[0,sin(90),cos(90)]]
#   = [[1,0,0],[0,0,-1],[0,1,0]]
# This maps (0,1,0) → (0,0,1) = Z-up. ✓
SCENE2SPLATSHOP_WORLD = torch.tensor(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0],
     [0.0, 1.0, 0.0]],
    dtype=torch.float32,
)

# Camera: OpenCV → OpenGL = diag(1, -1, -1).
OPENCV2SPLATSHOP_CAM = torch.tensor(
    [[1.0, 0.0, 0.0],
     [0.0, -1.0, 0.0],
     [0.0, 0.0, -1.0]],
    dtype=torch.float32,
)

# Backward-compatible aliases for the old names.
OPENCV2SPLATSHOP_WORLD = SCENE2SPLATSHOP_WORLD


# ── Transform helpers ───────────────────────────────────────────────────────

def transform_means(means: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate point positions: ``means_new = (R @ means.T).T`` = ``means @ R.T``."""
    return means @ R.T.to(means)


def transform_quats(quats: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate Gaussian quaternions by rotation matrix R (left-composition).

    ``q_new = q(R) ⊗ q_old``
    """
    q_R = rotation_matrix_to_quaternion(R.to(quats))
    return quaternion_compose(q_R, quats)


def transform_scales(scales: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Scales are invariant under pure rotation (anisotropic axes rotate with
    the quaternion, not the scale vector).

    For a similarity transform with uniform scale ``s``, multiply scales by
    ``s``.  Under pure rotation this is a no-op.
    """
    return scales


def model_to_scene(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert TripoSplat model-frame Gaussians to the Y-up scene frame.

    Args:
        means:  (N, 3) in model/AABB frame.
        scales: (N, 3) activated or log-space (unchanged).
        quats:  (N, 4) (w, x, y, z).

    Returns:
        (means_scene, scales_scene, quats_scene) in the Y-up scene frame.
    """
    R = MODEL2SCENE.to(means)
    means_scene = transform_means(means, R)
    quats_scene = transform_quats(quats, R)
    return means_scene, scales, quats_scene


# Backward-compatible alias.
def model_to_opencv(means, scales, quats):
    """Deprecated alias for :func:`model_to_scene`."""
    return model_to_scene(means, scales, quats)


def smpl_to_scene_frame(
    v_template: torch.Tensor,
    shapedirs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate SMPL mesh data to the scene frame.

    Since both SMPL and the scene are Y-up, this is identity (no-op).
    Kept for API symmetry with :func:`model_to_scene`.

    Args:
        v_template: (V, 3) SMPL template vertices.
        shapedirs:  (V, 3, D) blendshape directions.

    Returns:
        (v_template, shapedirs) unchanged.
    """
    return v_template, shapedirs


# Backward-compatible alias.
def smpl_to_opencv_frame(v_template, shapedirs):
    """Deprecated alias for :func:`smpl_to_scene_frame`."""
    return smpl_to_scene_frame(v_template, shapedirs)


def scene_to_splatshop_world(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert splat from Y-up scene to SplatShop Z-up world.

    Scales unchanged (pure rotation).  Cameras handled separately via
    ``OPENCV2SPLATSHOP_CAM`` on c2w matrices.
    """
    R = SCENE2SPLATSHOP_WORLD.to(means)
    means_ss = transform_means(means, R)
    quats_ss = transform_quats(quats, R)
    return means_ss, scales, quats_ss


# Backward-compatible alias.
def opencv_to_splatshop_world(means, scales, quats):
    """Deprecated alias for :func:`scene_to_splatshop_world`."""
    return scene_to_splatshop_world(means, scales, quats)


def opencv_c2w_to_splatshop_c2w(c2w_cv: torch.Tensor) -> torch.Tensor:
    """Convert a 4×4 OpenCV c2w to SplatShop (OpenGL camera + Z-up world).

    First rotate the world (position + basis) by ``SCENE2SPLATSHOP_WORLD``,
    then flip the camera basis to OpenGL via ``diag(1,-1,-1,1)``.

    The world rotation is applied as ``[[R_w, 0],[0,1]] @ c2w_cv``.
    The camera flip is ``c2w @ diag(1,-1,-1,1)`` (post-multiply, flips Y/Z
    columns = camera basis).
    """
    R_w = SCENE2SPLATSHOP_WORLD.to(c2w_cv)
    T_world = torch.eye(4, dtype=c2w_cv.dtype, device=c2w_cv.device)
    T_world[:3, :3] = R_w
    c2w_rotated = T_world @ c2w_cv

    flip = torch.diag(torch.tensor(
        [1.0, -1.0, -1.0, 1.0], dtype=c2w_cv.dtype, device=c2w_cv.device))
    return c2w_rotated @ flip
