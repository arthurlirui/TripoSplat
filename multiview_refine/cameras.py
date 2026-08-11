"""Camera I/O and coordinate-system conversion for multi-view refinement.

Reads Nerfstudio-style ``transforms_{train,val,test}.json`` and produces
:class:`Camera` objects in the OpenCV convention (+X right, +Y down, +Z forward)
expected by gsplat.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from PIL import Image


# ── Convention helpers ──────────────────────────────────────────────────────

# Nerfstudio / OpenGL cameras: +X right, +Y up, -Z forward.
# OpenCV cameras:              +X right, +Y down, +Z forward.
# Conversion: flip Y and Z axes of the camera-to-world rotation.
_GL2CV = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32))


def opengl_c2w_to_opencv_c2w(c2w_gl: torch.Tensor) -> torch.Tensor:
    """Convert a 4x4 OpenGL/Nerfstudio camera-to-world to OpenCV convention.

    ``c2w_cv = c2w_gl @ diag(1, -1, -1, 1)``.
    """
    return c2w_gl @ _GL2CV.to(c2w_gl)


def invert_se3(c2w: torch.Tensor) -> torch.Tensor:
    """Invert a 4x4 rigid SE(3) transform (rotation + translation)."""
    w2c = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ t
    return w2c


# ── Camera dataclass ────────────────────────────────────────────────────────

@dataclass
class Camera:
    """A single calibrated perspective camera in OpenCV convention.

    Attributes:
        w2c:       (4, 4) world-to-camera transform (OpenCV, row-major).
        c2w:       (4, 4) camera-to-world transform (OpenCV).
        position:  (3,)   camera centre in world coordinates.
        fx, fy:    focal lengths in pixels.
        cx, cy:    principal point in pixels (0-based).
        width, height: image resolution.
        image:     (3, H, W) float32 in [0, 1], or ``None`` if not loaded.
        alpha:     (1, H, W) float32 foreground mask in [0, 1], or ``None``.
        cam_idx:   integer index from the transforms file.
        file_path: extension-less path as written in the JSON (e.g. ``"images/02"``).
    """

    w2c: torch.Tensor
    c2w: torch.Tensor
    position: torch.Tensor
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    image: torch.Tensor | None = None
    alpha: torch.Tensor | None = None
    cam_idx: int = -1
    file_path: str = ""

    @property
    def device(self) -> torch.device:
        return self.w2c.device

    def to(self, device: torch.device | str) -> "Camera":
        """Move all tensors to *device* and return a new Camera."""
        return Camera(
            w2c=self.w2c.to(device),
            c2w=self.c2w.to(device),
            position=self.position.to(device),
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            width=self.width, height=self.height,
            image=self.image.to(device) if self.image is not None else None,
            alpha=self.alpha.to(device) if self.alpha is not None else None,
            cam_idx=self.cam_idx,
            file_path=self.file_path,
        )


# ── Image loading ───────────────────────────────────────────────────────────

def _load_image_tensor(path: str, image_scale: float = 1.0) -> torch.Tensor:
    """Load an image as (3, H, W) float32 in [0, 1].

    If *image_scale* < 1 the image is bilinearly downscaled.
    """
    img = Image.open(path).convert("RGB")
    if image_scale != 1.0:
        new_w = max(1, int(round(img.width * image_scale)))
        new_h = max(1, int(round(img.height * image_scale)))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)


def _load_alpha_tensor(path: str, image_scale: float = 1.0) -> torch.Tensor | None:
    """Load the alpha channel as (1, H, W) float32 in [0, 1] if present."""
    img = Image.open(path)
    if img.mode != "RGBA":
        return None
    if image_scale != 1.0:
        new_w = max(1, int(round(img.width * image_scale)))
        new_h = max(1, int(round(img.height * image_scale)))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    alpha = np.asarray(img, dtype=np.float32)[..., 3:4] / 255.0  # (H, W, 1)
    return torch.from_numpy(alpha).permute(2, 0, 1).contiguous()  # (1, H, W)


def _scale_intrinsic(value: float, old_size: int, new_size: int) -> float:
    """Scale a focal length / principal-point value proportionally to image size."""
    return value * (new_size / old_size)


# ── Transforms JSON parsing ─────────────────────────────────────────────────

def _resolve_image_path(base_dir: str, file_path: str) -> str:
    """Resolve an extension-less ``file_path`` from transforms.json to a real file.

    Tries the path as-is first, then appends common image extensions.
    """
    candidates = [file_path]
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"):
        candidates.append(file_path + ext)
    for c in candidates:
        full = os.path.join(base_dir, c)
        if os.path.isfile(full):
            return full
    # Return the .png fallback path so the error message is sensible.
    return os.path.join(base_dir, file_path + ".png")


def _build_camera_from_frame(
    frame: dict,
    global_intrinsics: dict,
    base_dir: str,
    device: torch.device,
    image_scale: float,
    load_image: bool,
) -> Camera:
    """Construct a :class:`Camera` from a single transforms.json frame dict."""
    # Per-frame intrinsics override the global ones when present.
    h = frame.get("h", global_intrinsics["h"])
    w = frame.get("w", global_intrinsics["w"])
    fl_x = frame.get("fl_x", global_intrinsics.get("fl_x"))
    fl_y = frame.get("fl_y", global_intrinsics.get("fl_y"))
    cx_f = frame.get("cx", global_intrinsics.get("cx", w / 2.0))
    cy_f = frame.get("cy", global_intrinsics.get("cy", h / 2.0))

    if fl_x is None or fl_y is None:
        # Fall back to camera_angle_x / camera_angle_y.
        angle_x = frame.get("camera_angle_x", global_intrinsics.get("camera_angle_x"))
        angle_y = frame.get("camera_angle_y", global_intrinsics.get("camera_angle_y"))
        if angle_x is not None:
            fl_x = (w / 2.0) / np.tan(angle_x / 2.0)
        if angle_y is not None:
            fl_y = (h / 2.0) / np.tan(angle_y / 2.0)
    if fl_x is None or fl_y is None:
        raise ValueError(
            f"Cannot determine focal length for frame {frame.get('file_path')!r}; "
            "neither fl_x/fl_y nor camera_angle_x/y is present."
        )

    # Camera-to-world in OpenGL / Nerfstudio convention (row-major 4x4).
    tm = np.array(frame["transform_matrix"], dtype=np.float32)
    c2w_gl = torch.from_numpy(tm)

    # Convert to OpenCV convention and invert to get world-to-camera.
    c2w_cv = opengl_c2w_to_opencv_c2w(c2w_gl)
    w2c_cv = invert_se3(c2w_cv)
    position = c2w_cv[:3, 3]

    # Load image (scaled) and adjust intrinsics accordingly.
    image_tensor = None
    alpha_tensor = None
    actual_h, actual_w = int(h), int(w)
    if load_image:
        img_path = _resolve_image_path(base_dir, frame["file_path"])
        image_tensor = _load_image_tensor(img_path, image_scale).to(device)
        alpha_tensor = _load_alpha_tensor(img_path, image_scale)
        if alpha_tensor is not None:
            alpha_tensor = alpha_tensor.to(device)
        actual_h, actual_w = image_tensor.shape[1], image_tensor.shape[2]
        fx_scaled = _scale_intrinsic(fl_x, w, actual_w)
        fy_scaled = _scale_intrinsic(fl_y, h, actual_h)
        cx_scaled = _scale_intrinsic(cx_f, w, actual_w)
        cy_scaled = _scale_intrinsic(cy_f, h, actual_h)
    else:
        fx_scaled, fy_scaled = float(fl_x), float(fl_y)
        cx_scaled, cy_scaled = float(cx_f), float(cy_f)

    return Camera(
        w2c=w2c_cv.to(device),
        c2w=c2w_cv.to(device),
        position=position.to(device),
        fx=fx_scaled,
        fy=fy_scaled,
        cx=cx_scaled,
        cy=cy_scaled,
        width=actual_w,
        height=actual_h,
        image=image_tensor,
        alpha=alpha_tensor,
        cam_idx=int(frame.get("cam_idx", -1)),
        file_path=frame["file_path"],
    )


def load_cameras(
    transforms_json: str,
    base_dir: str | None = None,
    device: torch.device | str = "cuda",
    image_scale: float = 1.0,
    load_image: bool = True,
) -> list[Camera]:
    """Load cameras from a Nerfstudio transforms JSON file.

    Args:
        transforms_json: path to ``transforms_{train,val,test}.json``.
        base_dir: directory containing ``images/`` and the JSON. Defaults to the
            JSON file's parent directory.
        device: target device for camera tensors.
        image_scale: downscale factor for loaded images (and matching intrinsics).
        load_image: if False, only camera parameters are loaded (images set to None).
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(transforms_json))
    device = torch.device(device)

    with open(transforms_json, "r") as f:
        data = json.load(f)

    global_intrinsics = {
        k: data[k]
        for k in ("h", "w", "cx", "cy", "fl_x", "fl_y",
                  "camera_angle_x", "camera_angle_y")
        if k in data
    }

    cameras: list[Camera] = []
    for frame in data["frames"]:
        cam = _build_camera_from_frame(
            frame, global_intrinsics, base_dir, device, image_scale, load_image
        )
        cameras.append(cam)
    return cameras


# ── Utilities ───────────────────────────────────────────────────────────────

def find_camera_by_filename(cameras: Sequence[Camera], filename: str) -> Camera:
    """Find the camera whose ``file_path`` or basename matches *filename*.

    ``filename`` may be given as ``"00.png"``, ``"00"``, ``"images/00"``, etc.
    """
    target = os.path.basename(filename)
    # Strip extension.
    stem = os.path.splitext(target)[0]

    for cam in cameras:
        fp_base = os.path.basename(cam.file_path)
        fp_stem = os.path.splitext(fp_base)[0]
        if fp_stem == stem or fp_base == target or cam.file_path == filename:
            return cam
    raise ValueError(
        f"No camera matches {filename!r}. Available file_paths: "
        f"{[c.file_path for c in cameras]}"
    )


def compute_cameras_extent(cameras: Sequence[Camera]) -> float:
    """Compute the scene ``cameras_extent`` used for densification thresholds.

    This is the maximum distance between any two camera centres, scaled by 1.1
    (following the original 3DGS convention).
    """
    if len(cameras) < 2:
        return 1.0
    positions = torch.stack([c.position for c in cameras])  # (N, 3)
    # Pairwise distances via cdist.
    dists = torch.cdist(positions, positions)
    max_dist = dists.max().item()
    return max_dist * 1.1


def split_train_test(
    cameras: Sequence[Camera], init_view: str
) -> tuple[list[Camera], list[Camera]]:
    """Split cameras into train / test sets.

    The init-view camera is always in the training set. If there are enough
    cameras, every 8th camera (excluding the init view) is held out for testing.
    """
    init_cam = find_camera_by_filename(cameras, init_view)
    test: list[Camera] = []
    train: list[Camera] = []
    for i, cam in enumerate(cameras):
        if cam is init_cam:
            train.append(cam)
        elif i % 8 == 0:
            test.append(cam)
        else:
            train.append(cam)
    if len(test) == 0 and len(cameras) > 1:
        # Fallback: use the second camera as test.
        for cam in cameras:
            if cam is not init_cam:
                test.append(cam)
                break
    return train, test
