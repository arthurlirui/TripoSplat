"""Lightweight monocular-frame container and SMPL estimation.

A ``MonocularFrame`` bundles everything the continual refiner needs for one
observed frame: the RGB image, the camera (intrinsics + extrinsics), and the
estimated SMPL parameters (betas, body_pose, global_orient, transl).
SMPL estimation here is a thin wrapper around an external regressor; the
actual model (e.g. SMPLer-X / YOLO11-pose) is injected so this module stays
backend-agnostic and unit-testable without heavyweight dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────
#  SMPL regressor protocol
# ──────────────────────────────────────────────────────────────────────
class SMPLRegressor(Protocol):
    """Interface for a monocular SMPL regressor.

    A concrete regressor maps an RGB image (H, W, 3) in [0, 1] to SMPL
    parameters. Implementations may wrap SMPLer-X, PIXIE, ExPose, etc.
    """

    def __call__(self, image: np.ndarray) -> dict:
        """Return dict with keys: betas (10,), body_pose (69,),
        global_orient (3,), transl (3,), all as torch tensors on `device`."""
        ...


@dataclass
class MonocularFrame:
    """One observed monocular frame.

    Attributes:
        image:   (3, H, W) float32 tensor in [0, 1], CHW order for rendering.
        camera:  a ``multiview_refine.cameras.Camera`` with w2c / intrinsics.
        betas:       (10,) SMPL shape parameters.
        body_pose:   (69,) SMPL body pose (23 joints axis-angle, no root).
        global_orient:(3,) global/root orientation.
        transl:      (3,) global translation.
        index:   frame index in the original stream (for bookkeeping).
    """

    image: torch.Tensor
    camera: object  # multiview_refine.cameras.Camera
    betas: torch.Tensor
    body_pose: torch.Tensor
    global_orient: torch.Tensor
    transl: torch.Tensor
    index: int = -1

    def to(self, device: str | torch.device) -> "MonocularFrame":
        return MonocularFrame(
            image=self.image.to(device),
            camera=self.camera,
            betas=self.betas.to(device),
            body_pose=self.body_pose.to(device),
            global_orient=self.global_orient.to(device),
            transl=self.transl.to(device),
            index=self.index,
        )

    def joint_transforms(self, smpl) -> torch.Tensor:
        """Per-joint relative transforms for this frame's pose."""
        return smpl.get_joint_transforms(
            betas=self.betas,
            body_pose=self.body_pose,
            global_orient=self.global_orient,
            transl=self.transl,
        )


# ──────────────────────────────────────────────────────────────────────
#  Estimation helpers
# ──────────────────────────────────────────────────────────────────────
def estimate_frame_smpl(
    regressor: SMPLRegressor,
    image_hwc: np.ndarray,
    camera,
    device: str = "cuda",
    index: int = -1,
) -> MonocularFrame:
    """Estimate SMPL params for a single frame and build a ``MonocularFrame``.

    Args:
        regressor:  a callable returning SMPL params (see :class:`SMPLRegressor`).
        image_hwc:  (H, W, 3) uint8 or float32 image.
        camera:     the observing camera (intrinsics/extrinsics for this frame).
        device:     torch device.
        index:      frame index in the stream.
    """
    params = regressor(image_hwc)
    img = torch.as_tensor(np.asarray(image_hwc), dtype=torch.float32, device=device)
    if img.max() > 1.5:  # uint8 -> [0,1]
        img = img / 255.0
    img = img.permute(2, 0, 1).contiguous()  # HWC -> CHW

    return MonocularFrame(
        image=img,
        camera=camera,
        betas=torch.as_tensor(params["betas"], dtype=torch.float32, device=device),
        body_pose=torch.as_tensor(params["body_pose"], dtype=torch.float32, device=device),
        global_orient=torch.as_tensor(params["global_orient"], dtype=torch.float32, device=device),
        transl=torch.as_tensor(params["transl"], dtype=torch.float32, device=device),
        index=index,
    )
