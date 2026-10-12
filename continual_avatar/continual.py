"""ContinualRefiner — the online continual-learning refinement loop.

Implements Algorithm 1 of the paper. Given a warm-started drivable
:class:`SkinnedGaussians` avatar and a stream of monocular clips, it
optimizes the avatar on each clip while:

  * protecting previously learned parameters with an EWC prior
    (``continual_avatar.ewc.EWCPrior``),
  * replaying representative historical frames from a viewpoint-stratified
    buffer (``continual_avatar.replay.ReplayBuffer``),
  * steering densification toward under-observed body regions via
    pose-conditioned selective reactivation
    (``continual_avatar.reactivation.VisibilityAccumulator``).

The refiner wires :class:`SkinnedGaussians` (which is a plain
``nn.Module`` and does not itself own a gsplat densification strategy) into
gsplat's ``DefaultStrategy`` via a small training manager.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ewc import EWCPrior, PARAM_NAMES
from .frame import MonocularFrame
from .reactivation import VisibilityAccumulator, SelectiveReactivationStrategy
from .replay import ReplayBuffer


# ──────────────────────────────────────────────────────────────────────
#  Photometric loss (L1 + DSSIM), mirrors multiview_refine.losses but kept
#  local to avoid a hard dependency when used outside the repo.
# ──────────────────────────────────────────────────────────────────────
def _ssim(x: torch.Tensor, y: torch.Tensor, win_size: int = 11) -> torch.Tensor:
    """Gaussian-windowed SSIM for (3, H, W) tensors."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    coords = torch.arange(win_size, dtype=x.dtype, device=x.device) - win_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    win = g[:, None] * g[None, :]
    win = win[None, None].expand(x.shape[0], 1, -1, -1).contiguous()
    pad = win_size // 2

    mu_x = F.conv2d(x.unsqueeze(0), win, padding=pad, groups=x.shape[0])[0]
    mu_y = F.conv2d(y.unsqueeze(0), win, padding=pad, groups=y.shape[0])[0]
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x2 = F.conv2d(x.unsqueeze(0) * x.unsqueeze(0), win, padding=pad, groups=x.shape[0])[0] - mu_x2
    sig_y2 = F.conv2d(y.unsqueeze(0) * y.unsqueeze(0), win, padding=pad, groups=y.shape[0])[0] - mu_y2
    sig_xy = F.conv2d(x.unsqueeze(0) * y.unsqueeze(0), win, padding=pad, groups=x.shape[0])[0] - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2) + 1e-12)
    return ssim_map.mean()


def photometric_loss(rendered: torch.Tensor, gt: torch.Tensor,
                     lambda_l1: float = 0.8, lambda_dssim: float = 0.2) -> torch.Tensor:
    l1 = F.l1_loss(rendered, gt)
    dssim = 1.0 - _ssim(rendered, gt)
    return lambda_l1 * l1 + lambda_dssim * dssim


# ──────────────────────────────────────────────────────────────────────
#  Training manager: wires SkinnedGaussians into gsplat DefaultStrategy.
# ──────────────────────────────────────────────────────────────────────
class _SkinnedTrainingManager:
    """Owns per-parameter Adam optimizers + gsplat ``DefaultStrategy`` state.

    ``SkinnedGaussians`` stores parameters in canonical space and exposes
    LBS deformation + rendering via ``render_posed``. This manager adds the
    optimizer dict and densification hooks the strategy expects, mirroring
    ``multiview_refine.gaussian_model.RefineableGaussians`` but without
    requiring that base class.
    """

    def __init__(self, avatar, scene_scale: float, grow_grad2d: float = 2e-4,
                 refine_start: int = 500, refine_stop: int = 15000,
                 refine_every: int = 100, reset_every: int = 3000,
                 vis: Optional["VisibilityAccumulator"] = None):
        from gsplat.strategy import DefaultStrategy
        self.avatar = avatar
        self.scene_scale = scene_scale
        base_strategy = DefaultStrategy(
            prune_opa=0.005, grow_grad2d=grow_grad2d,
            grow_scale3d=0.01, grow_scale2d=0.05,
            prune_scale3d=0.1, prune_scale2d=0.15,
            refine_start_iter=refine_start, refine_stop_iter=refine_stop,
            refine_every=refine_every, reset_every=reset_every, verbose=False,
        )
        if vis is not None:
            self.strategy = SelectiveReactivationStrategy(base_strategy, vis)
        else:
            self.strategy = base_strategy
        lr_means = 1.6e-4 * scene_scale
        self.optimizers = {
            "means": torch.optim.Adam([{"params": [avatar._means], "lr": lr_means, "name": "means"}], eps=1e-15),
            "scales": torch.optim.Adam([{"params": [avatar._scales], "lr": 5e-3, "name": "scales"}], eps=1e-15),
            "quats": torch.optim.Adam([{"params": [avatar._quats], "lr": 1e-3, "name": "quats"}], eps=1e-15),
            "opacities": torch.optim.Adam([{"params": [avatar._opacities], "lr": 5e-2, "name": "opacities"}], eps=1e-15),
            "sh0": torch.optim.Adam([{"params": [avatar._sh0], "lr": 2.5e-3, "name": "sh0"}], eps=1e-15),
        }
        if getattr(avatar, "_weights_optimizable", False) and avatar._skin_weights is not None:
            self.optimizers["skin_weights"] = torch.optim.Adam(
                [{"params": [avatar._skin_weights], "lr": 1e-3, "name": "skin_weights"}], eps=1e-15)
        self.state = self.strategy.initialize_state(scene_scale=scene_scale)
        self._params_dict = None

    def get_params(self) -> dict:
        a = self.avatar
        if self._params_dict is None:
            self._params_dict = {
                "means": a._means, "scales": a._scales, "quats": a._quats,
                "opacities": a._opacities, "sh0": a._sh0,
            }
        return self._params_dict

    def step_pre_backward(self, info: dict):
        self.strategy.step_pre_backward(
            self.get_params(), self.optimizers, self.state,
            step=info.get("step", 0), info=info)

    def step_post_backward(self, info: dict, step: int, packed: bool = True):
        self.strategy.step_post_backward(
            self.get_params(), self.optimizers, self.state,
            step=step, info=info, packed=packed)
        # sync strategy-side parameter replacements back onto the avatar.
        d = self._params_dict
        self.avatar._means = nn.Parameter(d["means"])
        self.avatar._scales = nn.Parameter(d["scales"])
        self.avatar._quats = nn.Parameter(d["quats"])
        self.avatar._opacities = nn.Parameter(d["opacities"])
        self.avatar._sh0 = nn.Parameter(d["sh0"])

    def optimizer_step(self):
        for opt in self.optimizers.values():
            opt.step()

    def optimizer_zero_grad(self):
        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=True)


# ──────────────────────────────────────────────────────────────────────
#  Refiner config
# ──────────────────────────────────────────────────────────────────────
@dataclass
class RefinerConfig:
    steps_per_clip: int = 500
    scene_scale: float = 1.0
    grow_grad2d: float = 2e-4
    refine_start: int = 100
    refine_stop: int = 400
    refine_every: int = 100
    reset_every: int = 3000
    # loss weights
    lambda_l1: float = 0.8
    lambda_dssim: float = 0.2
    lambda_ewc: float = 400.0
    lambda_replay: float = 0.5
    lambda_weights: float = 0.01
    # continual-learning
    ewc_gamma: float = 0.9
    replay_capacity: int = 64
    replay_n_bins: int = 8
    reactivation_rho: float = 0.5
    replay_batch: int = 4
    seed: int = 42
    background: tuple = (0.0, 0.0, 0.0)
    sh_degree: int = 0
    device: str = "cuda"


# ──────────────────────────────────────────────────────────────────────
#  ContinualRefiner
# ──────────────────────────────────────────────────────────────────────
class ContinualRefiner:
    """Online continual refinement of a drivable avatar (Algorithm 1)."""

    def __init__(self, avatar, smpl, cfg: Optional[RefinerConfig] = None):
        self.avatar = avatar
        self.smpl = smpl
        self.cfg = cfg or RefinerConfig()
        random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        self.device = self.cfg.device

        self.tm: Optional[_SkinnedTrainingManager] = None
        self.ewc = EWCPrior(gamma=self.cfg.ewc_gamma, device=self.device)
        self.replay = ReplayBuffer(capacity=self.cfg.replay_capacity,
                                   n_bins=self.cfg.replay_n_bins)
        self.vis = VisibilityAccumulator(
            avatar.n_gaussians, rho=self.cfg.reactivation_rho,
            base_grad=self.cfg.grow_grad2d, device=self.device)
        self.bg = torch.tensor(self.cfg.background, dtype=torch.float32, device=self.device)
        self.global_step = 0
        self.clip_index = 0

    # ── setup (call once before the first clip) ──
    def setup(self) -> None:
        self.tm = _SkinnedTrainingManager(
            self.avatar, scene_scale=self.cfg.scene_scale,
            grow_grad2d=self.cfg.grow_grad2d,
            refine_start=self.cfg.refine_start, refine_stop=self.cfg.refine_stop,
            refine_every=self.cfg.refine_every, reset_every=self.cfg.reset_every,
            vis=self.vis)

    # ── render one frame for the current avatar pose ──
    def _render(self, frame: MonocularFrame):
        jt = frame.joint_transforms(self.smpl)
        return self.avatar.render_posed(
            jt, frame.camera, background=self.bg,
            sh_degree=self.cfg.sh_degree)

    # ── per-step loss ──
    def _step_loss(self, frames: Sequence[MonocularFrame],
                   replay_frames: Sequence[MonocularFrame]) -> torch.Tensor:
        frame = random.choice(frames)
        rgb, alpha, meta = self._render(frame)
        # crop/pad to match gt size if slight mismatch.
        gt = self._match_size(frame.image, rgb.shape[-2:])
        loss = photometric_loss(rgb, gt, self.cfg.lambda_l1, self.cfg.lambda_dssim)

        # update visibility from this render (no grad).
        with torch.no_grad():
            self.vis.update_from_alpha(alpha, meta)

        # replay loss over historical frames.
        if replay_frames:
            rf = random.choice(replay_frames)
            r_rgb, _, _ = self._render(rf)
            r_gt = self._match_size(rf.image, r_rgb.shape[-2:])
            loss = loss + self.cfg.lambda_replay * photometric_loss(
                r_rgb, r_gt, self.cfg.lambda_l1, self.cfg.lambda_dssim)

        # EWC penalty.
        if self.cfg.lambda_ewc > 0:
            loss = loss + self.cfg.lambda_ewc * self.ewc.penalty(self.avatar)

        # skin-weight regularization.
        if getattr(self.avatar, "_weights_optimizable", False):
            loss = loss + self.avatar.weight_regularization_loss(
                self.cfg.lambda_weights)
        return loss, meta

    @staticmethod
    def _match_size(img: torch.Tensor, target_hw) -> torch.Tensor:
        H, W = target_hw
        if img.shape[-2:] == (H, W):
            return img
        return F.interpolate(img.unsqueeze(0), size=(H, W), mode="bilinear",
                             align_corners=False).squeeze(0)

    # ── public: refine one clip ──
    def refine_clip(self, clip: Sequence[MonocularFrame]) -> dict:
        """Run ``steps_per_clip`` optimization steps on ``clip`` (Algorithm 1).

        Returns a dict of diagnostics (final loss, n_gaussians, etc.).
        """
        assert self.tm is not None, "call setup() first"
        clip = list(clip)
        last_loss = float("nan")
        for it in range(self.cfg.steps_per_clip):
            self.global_step += 1
            self.avatar.train()
            self.tm.optimizer_zero_grad()

            replay_frames = self.replay.sample(self.cfg.replay_batch) if len(self.replay) else []
            loss, meta = self._step_loss(clip, replay_frames)

            info = dict(meta); info["step"] = self.global_step
            self.tm.step_pre_backward(info)
            loss.backward()
            self.tm.step_post_backward(info, step=self.global_step)
            self.tm.optimizer_step()

            # keep visibility tensor sized to the (possibly densified) avatar.
            self.vis.resize(self.avatar.n_gaussians)
            last_loss = float(loss.item())

        # ── post-clip: update Fisher map + replay buffer + snapshot ──
        self._update_ewc(clip)
        self.replay.add_many(clip)
        self.clip_index += 1
        return {"clip": self.clip_index, "loss": last_loss,
                "n_gaussians": self.avatar.n_gaussians,
                "replay_size": len(self.replay)}

    def _update_ewc(self, clip: Sequence[MonocularFrame]) -> None:
        """Estimate Fisher importance as the mean squared photometric-loss
        gradient over the clip's frames (Eq. 6)."""
        accum = {n: None for n in PARAM_NAMES}
        n_frames = 0
        # Use a small subsample of the clip for efficiency.
        sample = clip if len(clip) <= 8 else random.sample(clip, 8)
        for frame in sample:
            self.avatar.train()
            self.tm.optimizer_zero_grad()
            rgb, _, meta = self._render(frame)
            gt = self._match_size(frame.image, rgb.shape[-2:])
            loss = photometric_loss(rgb, gt, self.cfg.lambda_l1, self.cfg.lambda_dssim)
            loss.backward()
            for n in PARAM_NAMES:
                p = getattr(self.avatar, n, None)
                if p is not None and p.grad is not None:
                    sq = p.grad.detach().pow(2)
                    accum[n] = sq if accum[n] is None else accum[n] + sq
            if getattr(self.avatar, "_weights_optimizable", False) and self.avatar._skin_weights is not None:
                sw = self.avatar._skin_weights
                if sw.grad is not None:
                    sq = sw.grad.detach().pow(2)
                    accum["_skin_weights"] = sq if accum.get("_skin_weights") is None else accum["_skin_weights"] + sq
            n_frames += 1

        grads = [(n, g) for n, g in accum.items() if g is not None]
        self.ewc.update(self.avatar, grads, scale=n_frames)

    # ── state for checkpointing ──
    def state_dict(self) -> dict:
        return {
            "global_step": self.global_step,
            "clip_index": self.clip_index,
            "ewc": self.ewc.state_dict(),
            "vis": self.vis.state_dict(),
            "replay_meta": self.replay.state(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.global_step = sd.get("global_step", 0)
        self.clip_index = sd.get("clip_index", 0)
        self.ewc.load_state_dict(sd.get("ewc", {}))
        self.vis.load_state_dict(sd.get("vis", {}))
