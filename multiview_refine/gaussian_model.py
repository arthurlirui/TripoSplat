"""Refineable 3D Gaussian model for multi-view optimization.

Wraps Gaussian parameters (means, scales, quats, opacities, SH coefficients)
as ``torch.nn.Parameter`` objects and integrates with gsplat's
:class:`gsplat.DefaultStrategy` for adaptive density control (clone/split/prune
+ opacity reset).

Parameters are stored in **raw** (pre-activation) form:
  - ``means``:     (N, 3)  world-space — no activation.
  - ``scales``:    (N, 3)  log-space — activate with ``exp()``.
  - ``quats``:     (N, 4)  raw quaternion (w, x, y, z) — normalise on use.
  - ``opacities``: (N,)    logit — activate with ``sigmoid()``.
  - ``sh0``:       (N, 1, 3)  SH degree-0 DC coefficients.
  - ``shN``:       (N, K-1, 3)  higher-order SH coefficients (empty for degree 0).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gsplat
from gsplat.strategy import DefaultStrategy

from .quaternion import quaternion_compose, rotation_matrix_to_quaternion

# SH degree-0 constant for colour conversion.
SH_C0 = 0.28209479177387814


class RefineableGaussians(nn.Module):
    """A set of 3D Gaussians optimisable via gradient descent.

    Parameters are stored raw (log-scales, logit-opacity, unnormalised quats)
    and activated lazily via the :attr:`activated_*` properties.
    """

    def __init__(
        self,
        sh_degree: int = 0,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.device = torch.device(device)

        # Parameters (allocated in from_tripo_splat / from_point_cloud).
        self._means: nn.Parameter | None = None
        self._scales: nn.Parameter | None = None
        self._quats: nn.Parameter | None = None
        self._opacities: nn.Parameter | None = None
        self._sh0: nn.Parameter | None = None
        self._shN: nn.Parameter | None = None

        # Strategy + optimizer + state (set up in training_setup).
        self.strategy: DefaultStrategy | None = None
        self.optimizers: dict[str, torch.optim.Optimizer] | None = None
        self.state: dict | None = None
        self.scene_scale: float = 1.0

    # ── Properties (activated) ──────────────────────────────────────────

    @property
    def means(self) -> torch.Tensor:
        """World-space centres (N, 3)."""
        return self._means

    @property
    def scales(self) -> torch.Tensor:
        """Activated scales (N, 3) = exp(raw)."""
        return self._scales.exp()

    @property
    def raw_scales(self) -> torch.Tensor:
        """Raw (log-space) scales (N, 3)."""
        return self._scales

    @property
    def quats(self) -> torch.Tensor:
        """Normalised quaternions (N, 4)."""
        return F.normalize(self._quats, dim=-1)

    @property
    def raw_quats(self) -> torch.Tensor:
        """Raw (unnormalised) quaternions (N, 4)."""
        return self._quats

    @property
    def opacities(self) -> torch.Tensor:
        """Activated opacities (N,) = sigmoid(raw)."""
        return torch.sigmoid(self._opacities)

    @property
    def raw_opacities(self) -> torch.Tensor:
        """Raw (logit) opacities (N,)."""
        return self._opacities

    @property
    def sh0(self) -> torch.Tensor:
        """SH degree-0 coefficients (N, 1, 3)."""
        return self._sh0

    @property
    def shN(self) -> torch.Tensor:
        """Higher-order SH coefficients (N, K-1, 3) or empty."""
        return self._shN

    @property
    def n_gaussians(self) -> int:
        return self._means.shape[0] if self._means is not None else 0

    @property
    def n_sh_bases(self) -> int:
        """Total SH bases = (degree + 1)^2."""
        return (self.sh_degree + 1) ** 2

    # ── Initialization from TripoSplat Gaussian ─────────────────────────

    @classmethod
    def from_tripo_splat(
        cls,
        tripo_gaussian,
        similarity_transform: tuple[float, torch.Tensor, torch.Tensor],
        sh_degree: int = 0,
        device: str | torch.device = "cuda",
    ) -> "RefineableGaussians":
        """Create from a TripoSplat ``Gaussian`` object + 7-DoF similarity transform.

        Args:
            tripo_gaussian: The ``Gaussian`` object from ``TripoSplatPipeline.run()``.
            similarity_transform: ``(scale, R, t)`` where:
                - scale: float
                - R: (3, 3) rotation matrix
                - t: (3,) translation
                The transform maps: ``xyz_new = scale * (R @ xyz_old) + t``.
            sh_degree: SH degree (0 = DC only, matching TripoSplat output).
            device: target device.

        The TripoSplat Gaussian stores:
          - ``get_xyz`` (N, 3): activated means in AABB [-0.5, 0.5].
          - ``get_scaling`` (N, 3): activated scales.
          - ``_rotation`` (N, 4): raw quaternions (before rots_bias).
          - ``get_opacity`` (N, 1): activated opacities.
          - ``_features_dc`` (N, 1, 3): SH DC coefficients.
        """
        scale, R, t = similarity_transform
        R = R.to(device).float()
        t = t.to(device).float()
        g = tripo_gaussian

        # ── Means: apply similarity transform ──
        xyz = g.get_xyz.detach().to(device).float()  # (N, 3) in AABB
        means = (scale * (xyz @ R.T) + t).contiguous()

        # ── Scales: apply scale factor, enforce minimum, convert to log-space ──
        # The similarity transform scales all distances by `scale`, so
        # activated scales should be multiplied by `scale` too. We also
        # enforce a minimum scale relative to the scene extent so that
        # Gaussians are not sub-pixel at any training view.
        scales_activated = g.get_scaling.detach().to(device).float() * scale  # (N, 3)
        # Minimum scale = 0.1% of scene extent (ensures visibility at all views)
        min_scale = 0.001 * abs(scale) * 3.0  # rough scene size estimate
        scales_activated = scales_activated.clamp_min(min_scale)
        scales_raw = torch.log(scales_activated.clamp_min(1e-8)).contiguous()

        # ── Quaternions: rotate by R ──
        # TripoSplat stores raw quat in _rotation; the activated quat is
        # _rotation + rots_bias where rots_bias = [1, 0, 0, 0].
        # We need to compose: q_new = q_R * q_old * q_R^{-1}
        # where q_R is the quaternion of R.
        raw_quats = g._rotation.detach().to(device).float()  # (N, 4)
        rots_bias = g.rots_bias.to(device).float()  # (1, 4) = [1, 0, 0, 0]
        quats_activated = raw_quats + rots_bias[None, :]  # (N, 4)
        # Convert R to quaternion
        q_R = rotation_matrix_to_quaternion(R)  # (4,)
        quats_new = quaternion_compose(q_R, quats_activated)  # (N, 4)
        # Store raw (gsplat normalises internally, so we can store the
        # activated quaternion directly as "raw"; normalisation happens in
        # the quats property).
        quats_raw = quats_new.contiguous()

        # ── Opacities: convert activated → logit ──
        opac_activated = g.get_opacity.detach().to(device).float().squeeze(-1)  # (N,)
        opac_activated = opac_activated.clamp(1e-6, 1 - 1e-6)
        opacities_raw = torch.log(opac_activated / (1 - opac_activated)).contiguous()

        # ── SH DC coefficients: copy directly ──
        # gsplat 1.5.x's rasterization() adds the +0.5 bias internally
        # (rendering.py line 525: `colors = clamp_min(colors + 0.5, 0)`),
        # so we store the standard 3DGS SH DC coefficients as-is.
        sh0 = g._features_dc.detach().to(device).float().contiguous()  # (N, 1, 3)

        # ── Higher-order SH: zeros (TripoSplat is degree 0) ──
        n_rest = (sh_degree + 1) ** 2 - 1
        if n_rest > 0:
            shN = torch.zeros(
                (sh0.shape[0], n_rest, 3), dtype=torch.float32, device=device
            )
        else:
            shN = torch.zeros((sh0.shape[0], 0, 3), dtype=torch.float32, device=device)

        model = cls(sh_degree=sh_degree, device=device)
        model._means = nn.Parameter(means)
        model._scales = nn.Parameter(scales_raw)
        model._quats = nn.Parameter(quats_raw)
        model._opacities = nn.Parameter(opacities_raw)
        model._sh0 = nn.Parameter(sh0)
        # shN: if empty (degree 0), mark as non-trainable so the densification
        # strategy doesn't expect an optimizer for it.
        model._shN = nn.Parameter(shN, requires_grad=(shN.numel() > 0))
        return model

    # ── Training setup ──────────────────────────────────────────────────

    def training_setup(
        self,
        scene_scale: float,
        strategy: Optional[DefaultStrategy] = None,
    ):
        """Set up the optimizer, densification strategy, and running state.

        Args:
            scene_scale: Characteristic scene radius (e.g. cameras_extent).
                Used for densification thresholds.
            strategy: A gsplat ``DefaultStrategy`` instance. If None, creates
                one with default hyperparameters.
        """
        self.scene_scale = scene_scale
        if strategy is None:
            strategy = DefaultStrategy(verbose=True)
        self.strategy = strategy

        # Build the optimizer dict. gsplat's strategy expects per-parameter
        # optimizers that it can extend (add/remove state) during densification.
        # We use plain torch.optim.Adam for each parameter group.
        lr_means = 1.6e-4 * scene_scale
        lr_scales = 5e-3
        lr_quats = 1e-3
        lr_opacities = 5e-2
        lr_sh0 = 2.5e-3
        lr_shN = 2.5e-3 / 20.0

        self.optimizers = {
            "means": torch.optim.Adam(
                [{"params": [self._means], "lr": lr_means, "name": "means"}],
                eps=1e-15,
            ),
            "scales": torch.optim.Adam(
                [{"params": [self._scales], "lr": lr_scales, "name": "scales"}],
                eps=1e-15,
            ),
            "quats": torch.optim.Adam(
                [{"params": [self._quats], "lr": lr_quats, "name": "quats"}],
                eps=1e-15,
            ),
            "opacities": torch.optim.Adam(
                [{"params": [self._opacities], "lr": lr_opacities, "name": "opacities"}],
                eps=1e-15,
            ),
            "sh0": torch.optim.Adam(
                [{"params": [self._sh0], "lr": lr_sh0, "name": "sh0"}],
                eps=1e-15,
            ),
        }
        if self._shN.numel() > 0:
            self.optimizers["shN"] = torch.optim.Adam(
                [{"params": [self._shN], "lr": lr_shN, "name": "shN"}],
                eps=1e-15,
            )

        self.state = self.strategy.initialize_state(scene_scale=scene_scale)
        self.lr_means_init = lr_means
        self.lr_means_final = 1.6e-6 * scene_scale
        self.lr_means_max_steps = 30000
        # Initialize the persistent params dict so the strategy can modify it
        # in-place during densification.
        self._params_dict = None

    def get_params(self) -> dict[str, nn.Parameter]:
        """Return the params dict expected by gsplat's strategy.

        This dict is **persistent** — it is created once and updated in-place
        by the strategy during densification. The model's ``_means`` etc. are
        kept in sync via ``_sync_params_from_dict()`` after each densification
        step.
        """
        if not hasattr(self, "_params_dict") or self._params_dict is None:
            self._params_dict = {
                "means": self._means,
                "scales": self._scales,
                "quats": self._quats,
                "opacities": self._opacities,
                "sh0": self._sh0,
                "shN": self._shN,
            }
        return self._params_dict

    def get_colors(self) -> torch.Tensor:
        """Return SH coefficients for gsplat rasterization.

        For degree 0: returns (N, 1, 3) = sh0.
        For higher degrees: returns (N, K, 3) = cat(sh0, shN).
        """
        if self._shN.numel() > 0:
            return torch.cat([self._sh0, self._shN], dim=1)  # (N, K, 3)
        return self._sh0  # (N, 1, 3)

    def update_learning_rate(self, iteration: int):
        """Exponential decay of the means learning rate."""
        if self.lr_means_max_steps <= 0:
            return
        progress = min(iteration / self.lr_means_max_steps, 1.0)
        lr = self.lr_means_init * (self.lr_means_final / self.lr_means_init) ** progress
        self.optimizers["means"].param_groups[0]["lr"] = lr

    def step_pre_backward(self, info: dict):
        """Call before ``loss.backward()``. Retains grad on means2d."""
        self.strategy.step_pre_backward(
            self.get_params(), self.optimizers, self.state, step=info.get("step", 0), info=info
        )

    def step_post_backward(self, info: dict, step: int, packed: bool = True):
        """Call after ``loss.backward()`` and before ``optimizer.step()``.

        Performs densification (clone/split/prune) and opacity reset.
        Updates the model's parameters in-place to reflect any changes.
        """
        self.strategy.step_post_backward(
            self.get_params(), self.optimizers, self.state,
            step=step, info=info, packed=packed,
        )
        # The strategy may have replaced parameters in the dict; sync back.
        self._sync_params_from_dict()

    def optimizer_step(self):
        """Step all optimizers."""
        for opt in self.optimizers.values():
            opt.step()

    def optimizer_zero_grad(self):
        """Zero all optimizer gradients."""
        for opt in self.optimizers.values():
            opt.zero_grad()

    def _sync_params_from_dict(self):
        """Sync parameter tensors back from the params dict after densification.

        The strategy replaces entries in the params dict with new nn.Parameters
        during clone/split/prune. We sync them back to the model's attributes
        so that the next render call uses the updated parameters.
        """
        params = self.get_params()
        self._means = params["means"]
        self._scales = params["scales"]
        self._quats = params["quats"]
        self._opacities = params["opacities"]
        self._sh0 = params["sh0"]
        self._shN = params["shN"]
        # Also update the optimizer param_groups to point to the new parameters.
        for name, opt in self.optimizers.items():
            if name in params:
                opt.param_groups[0]["params"] = [params[name]]

    # ── Export ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def save_ply(self, path: str):
        """Save to a standard 3DGS-compatible PLY file.

        Vertex attributes: x, y, z, nx, ny, nz, f_dc_0..2,
        [f_rest_0..], opacity, scale_0..2, rot_0..3.
        All values are stored in raw (pre-activation) form.
        """
        means = self._means.detach().cpu().numpy()
        normals = np.zeros_like(means)
        sh0 = self._sh0.detach().cpu().numpy()  # (N, 1, 3)
        # Transpose to (N, 3, 1) then flatten to (N, 3) for f_dc
        f_dc = sh0.transpose(0, 2, 1).reshape(means.shape[0], -1)  # (N, 3)
        shN_np = (
            self._shN.detach().cpu().numpy() if self._shN.numel() > 0 else np.zeros((means.shape[0], 0, 3))
        )
        f_rest = shN_np.transpose(0, 2, 1).reshape(means.shape[0], -1)  # (N, (K-1)*3)
        opacities = self._opacities.detach().cpu().numpy()  # logit
        scales = self._scales.detach().cpu().numpy()  # log
        rotations = F.normalize(self._quats, dim=-1).detach().cpu().numpy()  # normalized quat

        # Build attribute names
        attr_names = ["x", "y", "z", "nx", "ny", "nz"]
        for i in range(f_dc.shape[1]):
            attr_names.append(f"f_dc_{i}")
        for i in range(f_rest.shape[1]):
            attr_names.append(f"f_rest_{i}")
        attr_names.append("opacity")
        for i in range(3):
            attr_names.append(f"scale_{i}")
        for i in range(4):
            attr_names.append(f"rot_{i}")

        # Build the structured array
        data = np.concatenate(
            [means, normals, f_dc, f_rest, opacities.reshape(-1, 1), scales, rotations],
            axis=1,
        )
        dtype_full = [(name, "f4") for name in attr_names]
        elements = np.empty(data.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, data))

        # Write PLY
        header = "ply\nformat binary_little_endian 1.0\n"
        header += f"element vertex {data.shape[0]}\n"
        for name in attr_names:
            header += f"property float {name}\n"
        header += "end_header\n"

        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(elements.tobytes())

    @torch.no_grad()
    def save_splat(self, path: str):
        """Save to a .splat file (32 bytes per splat) for web viewers."""
        means = self._means.detach().cpu().numpy()
        scales = self._scales.exp().detach().cpu().numpy()
        opacities = torch.sigmoid(self._opacities).detach().cpu().numpy()
        quats = F.normalize(self._quats, dim=-1).detach().cpu().numpy()
        # SH DC -> RGB: rgb = f_dc * C0 + 0.5 (standard 3DGS convention).
        sh0 = self._sh0.detach().cpu().numpy()  # (N, 1, 3)
        rgb = (sh0[:, 0, :] * SH_C0 + 0.5).clip(0, 1)

        # Sort by descending opacity * prod(scale) for alpha blending
        sort_key = opacities.flatten() * scales.prod(axis=-1)
        order = np.argsort(-sort_key)

        means = means[order]
        scales = scales[order]
        opacities = opacities[order]
        quats = quats[order]
        rgb = rgb[order]

        rgba = np.concatenate(
            [rgb, opacities.reshape(-1, 1)], axis=1
        ).clip(0, 1) * 255
        rgba = rgba.astype(np.uint8)
        scales = scales.astype(np.float32)
        quats_u8 = np.clip(quats * 128 + 128, 0, 255).astype(np.uint8)

        with open(path, "wb") as f:
            for i in range(means.shape[0]):
                f.write(means[i].astype(np.float32).tobytes())  # 12B xyz
                f.write(scales[i].astype(np.float32).tobytes())  # 12B scale
                f.write(rgba[i].tobytes())  # 4B rgba
                f.write(quats_u8[i].tobytes())  # 4B rot
