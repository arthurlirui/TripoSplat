"""Pose-conditioned selective reactivation (Sec. 3.3.4).

Maintains a per-Gaussian accumulated visibility count over observed frames
and modulates the 3DGS densification gradient threshold per Gaussian so
that frequently-seen Gaussians are pruned more aggressively and
under-observed ones are split/cloned more readily. This steers the
optimization capacity toward novel detail as the stream progresses.
"""

from __future__ import annotations

import torch


class VisibilityAccumulator:
    """Per-Gaussian visibility counts + densification-threshold modulation.

    Args:
        n_gaussians: initial number of Gaussians.
        rho:         reactivation strength (Eq. 8).
        base_grad:   base densification gradient threshold g0.
    """

    def __init__(self, n_gaussians: int, rho: float = 0.5,
                 base_grad: float = 0.0002, device: str | torch.device = "cuda"):
        self.rho = rho
        self.base_grad = base_grad
        self.device = torch.device(device)
        self.counts = torch.zeros(n_gaussians, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def update_from_alpha(self, alpha: torch.Tensor, meta: dict) -> None:
        """Accumulate visibility from a rendered alpha map.

        Args:
            alpha: (1, H, W) rendered alpha (not used directly here).
            meta:  gsplat rasterization meta dict. If it contains
                   ``"radii"`` or a per-Gaussian visibility mask, we use it;
                   otherwise we mark all rendered Gaussians as visible.
        """
        # gsplat meta may expose visibility via `visible_id` or `radii`.
        n = self.counts.shape[0]
        if meta is None:
            return
        # Heuristic: any Gaussian with a positive radius in this frame is visible.
        radii = meta.get("radii")
        if radii is not None:
            vis = (radii > 0).float()
            if vis.shape[0] == n:
                self.counts = self.counts + vis
                return
        # Fallback: mark all current Gaussians as visible once.
        self.counts = self.counts + 1.0

    @torch.no_grad()
    def resize(self, n_gaussians: int) -> None:
        """Resize the visibility tensor after densification (preserve old counts)."""
        if n_gaussians == self.counts.shape[0]:
            return
        if n_gaussians > self.counts.shape[0]:
            pad = torch.zeros(n_gaussians - self.counts.shape[0],
                              device=self.device, dtype=self.counts.dtype)
            self.counts = torch.cat([self.counts, pad], dim=0)
        else:
            self.counts = self.counts[:n_gaussians]

    @torch.no_grad()
    def per_gaussian_threshold(self) -> torch.Tensor:
        """Per-Gaussian densification gradient threshold (Eq. 8).

        Returns a tensor of shape (N,) with the threshold for each Gaussian.
        Gaussians with high historical visibility get a higher threshold
        (harder to densify); under-observed ones get a lower threshold.
        """
        if self.counts.numel() == 0 or self.counts.max() < 1e-6:
            return torch.full_like(self.counts, self.base_grad)
        v_bar = self.counts / self.counts.max().clamp_min(1.0)
        return self.base_grad * (1.0 + self.rho * v_bar)

    @torch.no_grad()
    def grad_scale_factors(self) -> torch.Tensor:
        """Per-Gaussian multiplicative scale for ``state["grad2d"]``.

        Returns ``base_grad / threshold_i`` so that, after scaling the
        accumulated gradient by this factor, the scalar comparison
        ``scaled_grads > base_grad`` inside gsplat's ``_grow_gs`` is
        equivalent to ``grads > threshold_i`` (the per-Gaussian threshold).
        """
        threshold = self.per_gaussian_threshold()
        return self.base_grad / threshold.clamp_min(1e-12)

    @torch.no_grad()
    def state_dict(self) -> dict:
        return {"rho": self.rho, "base_grad": self.base_grad,
                "counts": self.counts.cpu()}

    @torch.no_grad()
    def load_state_dict(self, sd: dict) -> None:
        self.rho = sd.get("rho", self.rho)
        self.base_grad = sd.get("base_grad", self.base_grad)
        self.counts = sd.get("counts", self.counts).to(self.device)


class SelectiveReactivationStrategy:
    """Mixin that wires :class:`VisibilityAccumulator` into a gsplat strategy.

    Wraps an existing ``DefaultStrategy`` (or compatible) instance and scales
    ``state["grad2d"]`` by per-Gaussian factors before the strategy's
    ``_grow_gs`` runs, so that under-observed Gaussians are densified more
    readily and frequently-seen ones are densified less (Eq. 8).

    Usage::

        strategy = SelectiveReactivationStrategy(
            DefaultStrategy(...), visibility_accumulator)
        strategy.step_post_backward(params, optimizers, state, step, info)
    """

    def __init__(self, base_strategy, vis: VisibilityAccumulator):
        self._base = base_strategy
        self._vis = vis

    def __getattr__(self, name):
        # Delegate all unknown attributes/methods to the wrapped strategy.
        return getattr(self._base, name)

    @torch.no_grad()
    def step_post_backward(self, params, optimizers, state, step, info, packed=False):
        # Run the normal post-backward, but intercept _grow_gs to apply
        # per-Gaussian gradient scaling before the threshold comparison.
        original_grow_gs = self._base._grow_gs

        @torch.no_grad()
        def scaled_grow_gs(p, opt, st, s):
            if "grad2d" in st and st["grad2d"].shape[0] == self._vis.counts.shape[0]:
                st["grad2d"].mul_(self._vis.grad_scale_factors())
            return original_grow_gs(p, opt, st, s)

        self._base._grow_gs = scaled_grow_gs
        try:
            self._base.step_post_backward(
                params, optimizers, state, step, info, packed=packed)
        finally:
            self._base._grow_gs = original_grow_gs
