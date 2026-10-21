"""Coordinate-system conversion for Gaussian splats and cameras.

TripoSplat generates splats in a model/scene frame, while the Nerfstudio-style
``transforms_*.json`` stores camera-to-world matrices in the OpenGL / Nerfstudio
convention (+X right, +Y up, -Z forward).  The downstream gsplat pipeline works
in OpenCV (+X right, +Y down, +Z forward) — :mod:`multiview_refine.cameras`
already performs that flip on load.

This module provides a single ``convert`` entry point that re-expresses a splat
plus its cameras in one of three unified target conventions:

* ``opencv``   — camera c2w in OpenCV, world unchanged (the existing pipeline).
* ``opengl``   — camera c2w in OpenGL (+X right, +Y up, -Z forward), world
                 unchanged.
* ``splatshop`` — camera c2w in OpenGL **and** world up rotated to +Z, matching
                 the SplatShop editor (m-schuetz/Splatshop): OpenGL camera
                 convention with a Z-up world.

The OpenGL<->OpenCV flip is the self-inverse matrix ``diag(1, -1, -1, 1)``;
the same op converts in both directions.  World-up rotation is a proper
rotation ``R_w`` (e.g. Y-up -> Z-up = rot_x(+90deg)) applied to both splat
centres/quaternions and camera positions/rotations, so splat and cameras stay
registered.

Reuses :func:`multiview_refine.alignment._apply_transform` for the splat
similarity transform and :mod:`multiview_refine.quaternion` for quaternion
composition, keeping a single source of truth for those operations.
"""

from __future__ import annotations

import json
import os
from typing import Literal, Sequence

import numpy as np
import torch

from .cameras import Camera
from .ply_io import save_gaussians_ply
from .coord_registry import OPENCV2SPLATSHOP_WORLD, OPENCV2SPLATSHOP_CAM

Target = Literal["opencv", "opengl", "splatshop"]
WorldUp = Literal["x", "y", "z"]
SrcFrame = Literal["auto", "opencv", "opengl"]

# OpenGL (+X right, +Y up, -Z forward)  <->  OpenCV (+X right, +Y down, +Z forward).
# Self-inverse: flipping Y and Z columns of the c2w works both directions.
CV2GL = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32))


def opencv_c2w_to_opengl_c2w(c2w_cv: torch.Tensor) -> torch.Tensor:
    """Convert a 4x4 OpenCV camera-to-world to OpenGL/Nerfstudio convention.

    ``c2w_gl = c2w_cv @ diag(1, -1, -1, 1)``.  Self-inverse with
    :func:`multiview_refine.cameras.opengl_c2w_to_opencv_c2w`.
    """
    return c2w_cv @ CV2GL.to(c2w_cv)


def world_up_rotation(src_up: WorldUp, dst_up: WorldUp) -> torch.Tensor:
    """Return a (3, 3) rotation mapping a source world-up axis to a target one.

    ``R_w @ e_src == e_dst`` where ``e_axis`` is the unit vector along that
    axis.  Identity when ``src_up == dst_up``.  Only the six axis-aligned
    rotations needed for up-axis swaps are supported; arbitrary rotations are
    out of scope (use ``model_to_world`` for those).
    """
    axes = {"x": torch.tensor([1.0, 0.0, 0.0]),
            "y": torch.tensor([0.0, 1.0, 0.0]),
            "z": torch.tensor([0.0, 0.0, 1.0])}
    if src_up not in axes or dst_up not in axes:
        raise ValueError(f"up axes must be one of {set(axes)}, got {src_up!r}, {dst_up!r}")
    a, b = axes[src_up], axes[dst_up]
    if src_up == dst_up:
        return torch.eye(3)
    # Rotation taking a -> b via Rodrigues around a x b.
    v = torch.cross(a, b, dim=0)
    s = v.norm()
    c = float(a @ b)
    if s < 1e-8:
        # 180-degree flip: a == -b. Rotate 180deg around any axis orthogonal to a.
        # Pick the world axis least aligned with a.
        candidates = [axes[k] for k in axes if k != src_up]
        cand = max(candidates, key=lambda w: 1.0 - abs(float(a @ w)))
        axis = torch.cross(a, cand, dim=0)
        axis = axis / axis.norm()
        K = _skew(axis)
        R = 2.0 * (axis.unsqueeze(1) @ axis.unsqueeze(0)) - torch.eye(3)
        return R
    K = _skew(v / s)
    R = torch.eye(3) + K + K @ K * ((1.0 - c) / (s * s))
    return R


def _skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v
    return torch.tensor([[0, -z, y],
                         [z, 0, -x],
                         [-y, x, 0]], dtype=v.dtype)


def detect_scene_up(cameras: Sequence[Camera], min_var: float = 1e-6) -> WorldUp:
    """Heuristically guess the scene's world-up axis from camera positions.

    For a turntable-style body scan the cameras orbit roughly in a horizontal
    plane around the subject, so their positions have the smallest spread
    along the world-up axis.  We report the axis of least position variance.

    **This is unreliable on degenerate camera sets.**  With only two cameras
    nearly aligned along one axis, that axis's variance can be exactly zero,
    which makes it look like "up".  Measured on ``static/000001/mvs``:

        transforms_train (30 cams) -> "y"   (var: x=3.48, y=0.57, z=3.51)
        transforms_val   ( 2 cams) -> "z"   (var: x=5.9e-4, y=0.47, z=0.0)  ← wrong

    Trusting that "z" rotates the whole Y-up scene by 90° and silently lays the
    subject on its side.  So when the smallest variance is degenerate we raise
    instead of guessing, and callers that know their world (this project is
    always Y-up: SfM ``points3d.ply`` is centred at the origin and the SMPL-X
    template is Y-up) should pass ``src_world_up="y"`` explicitly rather than
    relying on detection.

    Args:
        cameras: cameras to inspect.
        min_var: if the smallest per-axis variance is below this, the set is
            too degenerate to judge and ``ValueError`` is raised.

    Returns:
        ``"x"``, ``"y"`` or ``"z"``.

    Raises:
        ValueError: fewer than 3 cameras, or a degenerate (near-zero) variance
            that makes the guess meaningless.
    """
    if not cameras:
        return "y"
    if len(cameras) < 3:
        raise ValueError(
            f"detect_scene_up needs at least 3 cameras to judge the up axis "
            f"(got {len(cameras)}). Pass src_world_up explicitly instead."
        )
    pos = torch.stack([c.position.float().cpu() for c in cameras])  # (N, 3)
    var = pos.var(dim=0)
    vmin = float(var.min().item())
    if vmin < min_var:
        raise ValueError(
            "detect_scene_up: degenerate camera set — per-axis variance "
            f"{[round(float(v), 6) for v in var]} has a near-zero axis "
            f"({vmin:.2e}), so the 'up' axis cannot be inferred. "
            "Pass src_world_up explicitly (this project's scenes are Y-up)."
        )
    idx = int(torch.argmin(var).item())
    return ["x", "y", "z"][idx]


def _apply_world_rotation_to_cameras(
    cameras_cv: Sequence[Camera], R_w: torch.Tensor
) -> list[torch.Tensor]:
    """Return per-camera 4x4 OpenCV c2w with world rotation R_w applied.

    A point p_world -> R_w @ p_world; the camera basis rotates the same way,
    so c2w_new = [[R_w, 0],[0,1]] @ c2w.
    """
    R_w = R_w.to(torch.float32)
    out = []
    for cam in cameras_cv:
        c2w = cam.c2w.float().cpu()
        T = torch.eye(4, dtype=torch.float32)
        T[:3, :3] = R_w
        out.append(T @ c2w)
    return out


def convert(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    cameras_cv: Sequence[Camera],
    *,
    target: Target,
    src_world_up: WorldUp = "y",
    src_frame: SrcFrame = "auto",
    model_to_world: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
):
    """Convert splat + cameras to a unified target convention.

    Args:
        means:   (N, 3) splat centres in the source frame.
        scales:  (N, 3) log-space scales.
        quats:   (N, 4) quaternions (w, x, y, z).
        cameras_cv: sequence of :class:`Camera` as produced by
            :func:`multiview_refine.cameras.load_cameras` (OpenCV c2w).
        target:  one of ``opencv`` / ``opengl`` / ``splatshop``.
        src_world_up: which world axis is "up" in the source scene.
            Used when ``src_frame="auto"`` or ``src_frame="opengl"``.
            Ignored when ``src_frame="opencv"`` (world is always Y-down).
        src_frame: explicit source world frame.  ``"opencv"`` means the
            splat is already in the canonical OpenCV scene frame (Y-down,
            +Z forward) — this avoids the ambiguity of ``detect_scene_up``
            which cannot distinguish Y-up from Y-down.  ``"auto"`` uses
            ``src_world_up`` as before.  Default ``"auto"`` (backward compat).
        model_to_world: optional (3, 4) [R|t] applied to the splat *before* the
            world-up rotation, to bring a raw model-frame splat into the scene
            world frame.  Default identity.
        device:  device for returned tensors.

    Returns:
        dict with keys:
            ``means``, ``scales``, ``quats``  — converted splat tensors.
            ``cameras``      — list of 4x4 c2w matrices in the target camera
                               convention (numpy float32).
            ``intrinsics``   — list of dicts {fx, fy, cx, cy, w, h} per camera.
            ``R_w``          — (3, 3) world-up rotation actually applied.
            ``axis_map``     — human-readable description string.
    """
    from .alignment import _apply_transform

    if target not in ("opencv", "opengl", "splatshop"):
        raise ValueError(f"target must be opencv/opengl/splatshop, got {target!r}")

    # 1. Optional model -> scene-world transform on the splat only.
    if model_to_world is not None:
        m2w = torch.as_tensor(model_to_world, dtype=torch.float32)
        if m2w.shape != (3, 4):
            raise ValueError(f"model_to_world must be (3,4), got {m2w.shape}")
        R_m, t_m = m2w[:, :3], m2w[:, 3]
        means, scales, quats = _apply_transform(means, scales, quats, 1.0, R_m, t_m)

    # 2. World-up rotation (only splatshop changes the world up; opencv/opengl
    #    keep the source world unchanged).
    dst_up: WorldUp = "z" if target == "splatshop" else src_world_up
    # The registry constants are CPU tensors; the splat may be on CUDA.  Move
    # them onto the splat's device before the matmul (a dtype-only .to() here
    # used to raise "Expected all tensors to be on the same device").
    dev_w = means.device
    if target == "splatshop" and src_frame == "opencv":
        # OpenCV world is Y-down; SplatShop wants Z-up.
        # Use the centralised OPENCV2SPLATSHOP_WORLD rotation (rot_x +90°).
        R_w = OPENCV2SPLATSHOP_WORLD.to(device=dev_w, dtype=torch.float32)
        means, scales, quats = _apply_transform(
            means, scales, quats, 1.0, R_w, torch.zeros(3, device=dev_w))
    else:
        R_w = world_up_rotation(src_world_up, dst_up).to(device=dev_w, dtype=torch.float32)
        if not torch.allclose(R_w, torch.eye(3, device=dev_w)):
            means, scales, quats = _apply_transform(
                means, scales, quats, 1.0, R_w, torch.zeros(3, device=dev_w))

    # 3. Camera c2w: apply the same world rotation, then flip to target cam conv.
    cams_world = _apply_world_rotation_to_cameras(cameras_cv, R_w)  # OpenCV c2w, rotated world
    cameras_out: list[np.ndarray] = []
    for c2w_cv in cams_world:
        if target == "opencv":
            c2w = c2w_cv
        else:  # opengl or splatshop -> OpenGL camera
            c2w = opencv_c2w_to_opengl_c2w(c2w_cv)
        cameras_out.append(c2w.numpy().astype(np.float32))

    intrinsics = [{"fx": float(c.fx), "fy": float(c.fy),
                   "cx": float(c.cx), "cy": float(c.cy),
                   "w": int(c.width), "h": int(c.height)} for c in cameras_cv]

    axis_map = {
        "opencv":   "OpenCV cam (+X R, +Y D, +Z F) · world up = " + dst_up,
        "opengl":   "OpenGL cam (+X R, +Y U, -Z F) · world up = " + dst_up,
        "splatshop": "SplatShop: OpenGL cam (+X R, +Y U, -Z F) · world Z-up",
    }[target]

    dev = torch.device(device)
    return {
        "means": means.to(dev),
        "scales": scales.to(dev),
        "quats": quats.to(dev),
        "cameras": cameras_out,
        "intrinsics": intrinsics,
        "R_w": R_w,
        "axis_map": axis_map,
    }


def export_ply(
    path: str,
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor | None = None,
):
    """Write a converted splat to a 3DGS-compatible PLY (wraps ply_io)."""
    save_gaussians_ply(path, means, scales, quats, opacities, sh0, shN)


def export_cameras_json(
    path: str,
    cameras: Sequence[np.ndarray],
    intrinsics: Sequence[dict],
    *,
    convention: Target,
):
    """Write cameras as a Nerfstudio-schema JSON in the target convention.

    Each frame carries ``transform_matrix`` (4x4 row-major c2w in *convention*),
    ``fl_x/fl_y/cx/cy``, ``w/h`` and ``cam_idx``.  Top-level intrinsics mirror
    the first frame for compatibility with loaders expecting globals.
    """
    if not cameras:
        raise ValueError("no cameras to export")
    frames = []
    for i, (c2w, intr) in enumerate(zip(cameras, intrinsics)):
        frames.append({
            "cam_idx": i,
            "file_path": f"images/{i:02d}",
            "fl_x": intr["fx"], "fl_y": intr["fy"],
            "cx": intr["cx"], "cy": intr["cy"],
            "w": intr["w"], "h": intr["h"],
            "transform_matrix": np.asarray(c2w, dtype=np.float32).tolist(),
        })
    g = intrinsics[0]
    data = {
        "convention": convention,
        "h": g["h"], "w": g["w"],
        "cx": g["cx"], "cy": g["cy"],
        "fl_x": g["fx"], "fl_y": g["fy"],
        "frames": frames,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
