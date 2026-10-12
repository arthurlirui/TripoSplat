"""Elastic Weight Consolidation (EWC) prior for 3D Gaussian parameters.

Implements Sec. 3.3.2 of the paper. After each clip is learned we estimate
a per-parameter Fisher-information importance map as the running, squared
photometric-loss gradient accumulated over the clip's frames (plus a small
replay subset). When learning from a new clip, a quadratic penalty anchors
parameters to their post-previous-clip values, weighted by importance, so
that well-constrained Gaussians (visible in earlier viewpoints) are
protected from forgetting while under-observed ones remain plastic.

The importance map is stored as a dict of tensors mirroring the parameter
shapes of :class:`SkinnedGaussians` (means, scales, quats, opacities, sh0,
and optionally skin_weights).
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


# Parameter names whose EWC importance we track. Skin weights are included
# only when they are optimizable.
PARAM_NAMES = ("_means", "_scales", "_quats", "_opacities", "_sh0")


class EWCPrior:
    """Per-parameter Fisher importance + EWC penalty.

    Attributes:
        importance: dict[name -> Tensor] running Fisher map (same shape as params).
        star:       dict[name -> Tensor] parameter snapshot at the last update.
        gamma:      forgetting factor for the running importance (Eq. 6).
    """

    def __init__(self, gamma: float = 0.9, device: str | torch.device = "cuda"):
        self.gamma = gamma
        self.device = torch.device(device)
        self.importance: dict[str, torch.Tensor] = {}
        self.star: dict[str, torch.Tensor] = {}

    # ── snapshot + importance update ───────────────────────────────
    @torch.no_grad()
    def update(self, model: nn.Module, grads: Iterable[tuple[str, torch.Tensor]],
               scale: float = 1.0) -> None:
        """Update the running Fisher map and snapshot the parameters.

        Called once after a clip is finished. ``grads`` is an iterable of
        (name, grad) pairs where ``grad`` is the per-parameter squared
        gradient contribution accumulated over the clip. ``scale`` divides
        the contribution (e.g. by the number of frames).

        Args:
            model: the :class:`SkinnedGaussians` whose parameters we snapshot.
            grads: iterable of (param_name, squared_grad_tensor).
            scale: normalization factor (number of accumulated frames).
        """
        new_imp = {n: (g / max(scale, 1e-8)) for n, g in grads}

        if not self.importance:
            # first task: initialize.
            self.importance = {n: g.detach() for n, g in new_imp.items()}
        else:
            for n, g in new_imp.items():
                g = g.detach()
                if n in self.importance and self.importance[n].shape == g.shape:
                    self.importance[n] = (
                        self.gamma * self.importance[n] + (1 - self.gamma) * g
                    )
                else:
                    # shape changed (densification): reset this parameter's map.
                    self.importance[n] = g

        # snapshot parameters (detached copies) at their current values.
        self.star = {}
        for n in PARAM_NAMES:
            p = getattr(model, n, None)
            if p is not None:
                self.star[n] = p.detach().clone()
        # skin weights, if optimizable.
        if getattr(model, "_weights_optimizable", False) and getattr(model, "_skin_weights", None) is not None:
            self.star["_skin_weights"] = model._skin_weights.detach().clone()

    # ── penalty ────────────────────────────────────────────────────
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """EWC quadratic penalty (Eq. 7), summed over tracked parameters.

        Returns 0 if no snapshot exists yet (first task).
        """
        if not self.star:
            return torch.zeros((), device=self.device)

        total = torch.zeros((), device=self.device)
        for n, theta_star in self.star.items():
            p = getattr(model, n, None)
            if n == "_skin_weights":
                p = getattr(model, "_skin_weights", None)
            if p is None:
                continue
            F = self.importance.get(n)
            if F is None or F.shape != p.shape:
                # shape mismatch due to densification: skip this parameter
                # until the next :meth:`update` resets its map.
                continue
            total = total + 0.5 * (F * (p - theta_star) ** 2).sum()
        return total

    # ── utilities ──────────────────────────────────────────────────
    def state_dict(self) -> dict:
        return {
            "gamma": self.gamma,
            "importance": {k: v.cpu() for k, v in self.importance.items()},
            "star": {k: v.cpu() for k, v in self.star.items()},
        }

    def load_state_dict(self, sd: dict) -> None:
        self.gamma = sd.get("gamma", self.gamma)
        self.importance = {k: v.to(self.device) for k, v in sd.get("importance", {}).items()}
        self.star = {k: v.to(self.device) for k, v in sd.get("star", {}).items()}
