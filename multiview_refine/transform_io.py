"""Persistent similarity/rigid transforms between Gaussian frames and scenes.

The pipeline produces several transforms that must survive between stages and
between runs:

* ``MODEL2SCENE`` (a fixed frame convention),
* the alignment ``(scale, R, t)`` that maps the raw Tripo AABB frame onto the
  SfM scene,
* optional per-run refinements of that alignment.

Before this module each script wrote its own ad-hoc dict — ``run_pipeline.py``
used ``rotation``/``translation`` keys while ``run_pipeline_verify.py`` used
``R``/``t``, neither recorded the frame conventions, and nothing could be
validated on load.  That is exactly how a transposed matrix went unnoticed.

This module defines **one** self-describing schema, and a loader that verifies
the invariant that matters: applying the transform must leave an upright
subject upright.

Schema (version 1)::

    {
      "schema": "triposplat.gaussian_transform",
      "version": 1,
      "created": "2026-10-21T12:00:00",
      "name": "analytic_init_view",
      "source_frame": "triposplat_model_aabb",   # Z-up, [-0.5, 0.5]^3
      "target_frame": "scene_y_up",              # SfM world
      "scale": 1.7659,
      "R": [[...], [...], [...]],                # column-vector, row-major
      "t": [x, y, z],
      "convention": "xyz = scale * (R @ xyz_model) + t",
      "up": {"model": [0,0,1], "scene": [0,1,0]},
      "init_view": "00.png",
      "init_camera_position": [...],
      "metrics": {"iou": 0.7348, "upright": 0.80},
      "notes": "..."
    }

Usage::

    from multiview_refine.transform_io import (
        GaussianTransform, save_transform, load_transform, apply_transform,
    )

    tfm = GaussianTransform.from_align(scale, R, t, init_view="00.png")
    save_transform("output/alignment.json", tfm)

    tfm = load_transform("output/alignment.json")     # validated
    means_a, scales_a, quats_a = tfm.apply(means, scales, quats)

    # Or apply straight to a PLY without touching tensors by hand:
    convert_ply(src_ply, dst_ply, tfm)
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np
import torch

SCHEMA_NAME = "triposplat.gaussian_transform"
SCHEMA_VERSION = 1

SOURCE_MODEL = "triposplat_model_aabb"
TARGET_SCENE = "scene_y_up"

# Model's feet->head axis, and the scene's up axis.  The pair is the single
# invariant a correct transform must preserve; see ``GaussianTransform.validate``.
MODEL_UP = (0.0, 0.0, 1.0)
SCENE_UP = (0.0, 1.0, 0.0)


def _to_list(x) -> Optional[list]:
    """Convert a torch/numpy/list payload to a plain list (CUDA-safe)."""
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1).tolist()


# ═══════════════════════════════════════════════════════════════════════════
# Dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GaussianTransform:
    """A similarity transform ``xyz_scene = scale * (R @ xyz_model) + t``.

    ``R`` is stored row-major in **column-vector** form, i.e. it is used as
    ``R @ v``.  When applying to row-vector tensors the equivalent is
    ``v @ R.T`` — ``apply()`` does this for you.
    """

    scale: float
    R: np.ndarray                 # (3, 3)
    t: np.ndarray                 # (3,)
    name: str = "alignment"
    source_frame: str = SOURCE_MODEL
    target_frame: str = TARGET_SCENE
    init_view: Optional[str] = None
    init_camera_position: Optional[np.ndarray] = None
    image_scale: Optional[float] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    created: Optional[str] = None

    # ── construction ────────────────────────────────────────────────────

    @classmethod
    def from_align(cls, scale, R, t, **kw) -> "GaussianTransform":
        """Build from torch or numpy ``(scale, R, t)`` returned by the aligners.

        Accepts CUDA tensors: everything is moved to host first, so a caller
        never has to remember ``.cpu()``.
        """
        def to_np(x, shape=None):
            if isinstance(x, torch.Tensor):
                x = x.detach().float().cpu().numpy()
            a = np.asarray(x, dtype=np.float64)
            return a.reshape(shape) if shape else a

        if kw.get("init_camera_position") is not None:
            kw["init_camera_position"] = to_np(kw["init_camera_position"], (3,))

        return cls(
            scale=float(to_np(scale)),
            R=to_np(R, (3, 3)),
            t=to_np(t, (3,)),
            **kw,
        )

    # ── validation ──────────────────────────────────────────────────────

    def validate(self, *, strict_upright: bool = True, atol: float = 1e-5) -> list[str]:
        """Check the transform is well-formed. Returns a list of warnings.

        Raises:
            ValueError: on a hard error (non-finite, non-orthonormal R,
                reflection, non-positive scale).
        """
        warnings: list[str] = []

        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(f"scale must be finite and > 0, got {self.scale}")
        for nm, arr, shape in (("R", self.R, (3, 3)), ("t", self.t, (3,))):
            if np.asarray(arr).shape != shape:
                raise ValueError(f"{nm} must be {shape}, got {np.asarray(arr).shape}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{nm} contains non-finite values")

        R = np.asarray(self.R, dtype=np.float64)
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-4):
            raise ValueError(
                "R is not orthonormal (R @ R.T != I). A transposed or scaled "
                "matrix here silently corrupts every render.\n" + str(R)
            )
        det = float(np.linalg.det(R))
        if det < 0:
            raise ValueError(f"R is a reflection (det={det:.6f}); it would mirror the model")

        if strict_upright and self.source_frame == SOURCE_MODEL and self.target_frame == TARGET_SCENE:
            up_scene = R @ np.array(MODEL_UP, dtype=np.float64)
            if not np.allclose(up_scene, np.array(SCENE_UP, dtype=np.float64), atol=1e-3):
                raise ValueError(
                    f"model up {MODEL_UP} maps to {np.round(up_scene, 4)}, expected "
                    f"scene up {SCENE_UP}. This transform would leave the subject "
                    "tipped over or upside down — check MODEL2SCENE and the yaw."
                )

        return warnings

    # ── application ─────────────────────────────────────────────────────

    def apply(self, means, scales=None, quats=None, *, device=None):
        """Apply to Gaussian attributes.

        Args:
            means: (N, 3) centres in the source frame.
            scales: (N, 3) activated scales, or None to skip.
            quats: (N, 4) quaternions (w, x, y, z), or None to skip.
            device: target device; defaults to ``means.device``.

        Returns:
            ``(means, scales, quats)`` with the same leading shape; ``scales``
            and ``quats`` are ``None`` when not supplied.
        """
        from .quaternion import quaternion_compose, rotation_matrix_to_quaternion

        dev = device or (means.device if isinstance(means, torch.Tensor) else "cpu")
        R = torch.as_tensor(self.R, dtype=torch.float32, device=dev)
        t = torch.as_tensor(self.t, dtype=torch.float32, device=dev)
        s = float(self.scale)

        means = means.to(dev).float()
        means_out = s * (means @ R.T) + t

        scales_out = None
        if scales is not None:
            scales_out = scales.to(dev).float() * s

        quats_out = None
        if quats is not None:
            q_R = rotation_matrix_to_quaternion(R)
            quats_out = quaternion_compose(q_R, quats.to(dev).float())

        return means_out, scales_out, quats_out

    def inverse(self) -> "GaussianTransform":
        """The inverse similarity transform (scene → model)."""
        Rinv = np.asarray(self.R, dtype=np.float64).T
        sinv = 1.0 / float(self.scale)
        tinv = -(sinv * (Rinv @ np.asarray(self.t, dtype=np.float64)))
        return GaussianTransform(
            scale=sinv, R=Rinv, t=tinv,
            name=f"{self.name}:inverse",
            source_frame=self.target_frame,
            target_frame=self.source_frame,
            notes="auto-generated inverse",
            created=self.created,
        )

    def compose(self, other: "GaussianTransform") -> "GaussianTransform":
        """``self ∘ other`` — apply *other* first, then *self*."""
        s = float(self.scale) * float(other.scale)
        R = np.asarray(self.R, dtype=np.float64) @ np.asarray(other.R, dtype=np.float64)
        t = (float(self.scale) *
             (np.asarray(self.R, dtype=np.float64) @ np.asarray(other.t, dtype=np.float64))
             + np.asarray(self.t, dtype=np.float64))
        return GaussianTransform(
            scale=s, R=R, t=t,
            name=f"{self.name}∘{other.name}",
            source_frame=other.source_frame,
            target_frame=self.target_frame,
            notes="composed transform",
        )

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_NAME,
            "version": SCHEMA_VERSION,
            "created": self.created or _dt.datetime.now().isoformat(timespec="seconds"),
            "name": self.name,
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "scale": float(self.scale),
            "R": np.asarray(self.R, dtype=np.float64).tolist(),
            "t": np.asarray(self.t, dtype=np.float64).tolist(),
            "convention": "xyz = scale * (R @ xyz_model) + t",
            "up": {"model": list(MODEL_UP), "scene": list(SCENE_UP)},
            "init_view": self.init_view,
            "init_camera_position": _to_list(self.init_camera_position),
            "image_scale": self.image_scale,
            "metrics": dict(self.metrics),
            "notes": self.notes,
        }

    def summary(self) -> str:
        """One-line human summary, with the implied subject height if known."""
        parts = [f"{self.name}: scale={self.scale:.4f}"]
        h = self.metrics.get("subject_height_m")
        if h is not None:
            parts.append(f"height={h:.2f}m")
        for k in ("iou", "psnr", "upright"):
            if k in self.metrics:
                v = self.metrics[k]
                parts.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
        return "  ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════════

def save_transform(path: str, tfm: GaussianTransform, *, validate: bool = True) -> str:
    """Write *tfm* to *path* as JSON. Returns the path written."""
    if validate:
        tfm.validate()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tfm.to_dict(), f, indent=2)
    return path


def load_transform(path: str, *, validate: bool = True,
                   strict_upright: bool = True) -> GaussianTransform:
    """Read a transform JSON and verify it before use.

    Also accepts two legacy layouts so existing files keep working:

    * ``{"scale", "R"/"rotation", "t"/"translation", ...}``
    * ``{"scale", "rotation", "translation", ...}`` where ``rotation`` is the
      product ``R @ MODEL2SCENE`` (as written by an older ``run_pipeline.py``).

    Raises:
        FileNotFoundError, ValueError (schema/shape/invariant errors).
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(d).__name__}")

    R = d.get("R", d.get("rotation"))
    t = d.get("t", d.get("translation"))
    if R is None or t is None:
        raise ValueError(f"{path}: missing rotation (R/rotation) or translation (t/translation)")
    if "scale" not in d:
        raise ValueError(f"{path}: missing 'scale'")

    tfm = GaussianTransform.from_align(
        d["scale"], R, t,
        name=d.get("name", d.get("method", os.path.basename(path))),
        source_frame=d.get("source_frame", SOURCE_MODEL),
        target_frame=d.get("target_frame", TARGET_SCENE),
        init_view=d.get("init_view"),
        init_camera_position=d.get("init_camera_position"),
        image_scale=d.get("image_scale"),
        metrics=d.get("metrics", {}),
        notes=d.get("notes", ""),
        created=d.get("created"),
    )

    if validate:
        tfm.validate(strict_upright=strict_upright)
    return tfm


# ═══════════════════════════════════════════════════════════════════════════
# PLY convenience
# ═══════════════════════════════════════════════════════════════════════════

def convert_ply(src_ply: str, dst_ply: str, tfm: GaussianTransform,
                device: str = "cpu") -> str:
    """Apply *tfm* to a Gaussian PLY and write the result.

    Scales are stored log-space in PLY, so a uniform scale ``s`` adds
    ``log(s)`` rather than multiplying.
    """
    from .ply_io import load_gaussians_ply, save_gaussians_ply

    means, scales_raw, quats, opac_raw, sh0, shN = load_gaussians_ply(src_ply, device=device)
    scales_act = torch.exp(scales_raw)

    means_o, scales_o, quats_o = tfm.apply(means, scales_act, quats, device=device)
    scales_raw_o = torch.log(scales_o.clamp_min(1e-12))

    save_gaussians_ply(dst_ply, means_o, scales_raw_o, quats_o, opac_raw, sh0,
                       shN if shN.numel() > 0 else None)
    return dst_ply
