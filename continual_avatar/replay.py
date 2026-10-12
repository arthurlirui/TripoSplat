"""Viewpoint-stratified replay buffer (Sec. 3.3.3).

Stores a compact, pose-diverse set of historical frames and re-renders them
jointly with incoming data to provide episodic-memory replay. Frames are
stratified by the azimuth of the SMPL global orientation into a fixed
number of bins; within each bin a quota is enforced using a
coverage-greedy (herding) selection that maximizes the minimum feature
distance to already-buffered frames.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import List

import torch


class ReplayBuffer:
    """Viewpoint-stratified episodic-memory buffer.

    Args:
        capacity:    total number of frames to retain.
        n_bins:      number of azimuth bins (strata).
        feat_fn:     optional callable(frame) -> 1-D feature tensor for herding.
                     If None, the frame's SMPL body_pose flattened is used.
    """

    def __init__(self, capacity: int = 64, n_bins: int = 8, feat_fn=None):
        self.capacity = capacity
        self.n_bins = n_bins
        self.per_bin = max(1, capacity // n_bins)
        self.feat_fn = feat_fn
        # bin_idx -> list of (frame, feat)
        self._bins: dict[int, list] = defaultdict(list)

    # ── helpers ────────────────────────────────────────────────────
    @staticmethod
    def _azimuth_bin(global_orient: torch.Tensor, n_bins: int) -> int:
        """Map a global_orient axis-angle (3,) to an azimuth bin.

        Uses the yaw of the rotation (projected onto the ground plane).
        """
        # convert axis-angle to a rough yaw via the vector direction.
        v = global_orient.detach().cpu().numpy()
        # axis * angle magnitude -> a pseudo-yaw proxy; robust enough for binning.
        yaw = math.atan2(v[1], v[0]) if (v[0] ** 2 + v[1] ** 2) > 1e-8 else 0.0
        # map yaw in [-pi, pi] to [0, n_bins-1]
        b = int(((yaw + math.pi) / (2 * math.pi)) * n_bins)
        return min(max(b, 0), n_bins - 1)

    def _feat(self, frame) -> torch.Tensor:
        if self.feat_fn is not None:
            f = self.feat_fn(frame)
            return f.detach().cpu().reshape(-1)
        return frame.body_pose.detach().cpu().reshape(-1)

    def _coverage_score(self, feat: torch.Tensor, bin_frames: list) -> float:
        """Min distance to existing frames in the bin (higher = more novel)."""
        if not bin_frames:
            return float("inf")
        dists = [torch.norm(feat - f[1]).item() for f in bin_frames]
        return min(dists)

    # ── public API ─────────────────────────────────────────────────
    def add(self, frame) -> None:
        """Attempt to insert ``frame`` into the buffer with herding."""
        b = self._azimuth_bin(frame.global_orient, self.n_bins)
        feat = self._feat(frame)
        bin_frames = self._bins[b]

        if len(bin_frames) < self.per_bin:
            bin_frames.append((frame, feat))
            return

        # bin full: replace the frame with the smallest coverage score if
        # the new frame is more novel. For each existing slot i, its score is
        # the min distance to all *other* slots in the bin (lower = redundant).
        n = len(bin_frames)
        scores = [
            self._coverage_score(bin_frames[j][1],
                                 [bin_frames[k] for k in range(n) if k != j])
            for j in range(n)
        ]
        worst = min(range(n), key=lambda j: scores[j])
        new_score = self._coverage_score(feat, bin_frames)
        if new_score > scores[worst]:
            bin_frames[worst] = (frame, feat)

    def add_many(self, frames) -> None:
        for f in frames:
            self.add(f)

    def sample(self, batch_size: int) -> List:
        """Sample a random batch of frames across all bins (no replacement)."""
        all_frames = [f for b in self._bins.values() for f, _ in b]
        if not all_frames:
            return []
        bs = min(batch_size, len(all_frames))
        idx = torch.randperm(len(all_frames))[:bs].tolist()
        return [all_frames[i] for i in idx]

    def __len__(self) -> int:
        return sum(len(b) for b in self._bins.values())

    def state(self) -> dict:
        return {"capacity": self.capacity, "n_bins": self.n_bins,
                "per_bin": self.per_bin, "count": len(self)}
