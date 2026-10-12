"""ContinualGS — Continual-Learning Drivable 3D Gaussian Avatars.

This package implements the algorithm described in
``latex/continual_avatar`` (ICME 2027 submission):
  1. Warm-start a drivable 3DGS avatar from a single monocular frame via a
     feed-forward Gaussian generator + SMPL registration + skeleton rigging.
  2. Continually refine the avatar over a monocular video stream while
     mitigating catastrophic forgetting with:
       - an EWC prior on Gaussian parameters (Fisher-information importance),
       - a viewpoint-stratified replay buffer,
       - pose-conditioned selective reactivation of densification.

Modules:
  warm_start   : single-frame -> drivable avatar.
  ewc          : Fisher importance map + EWC penalty.
  replay       : viewpoint-stratified replay buffer.
  reactivation : per-Gaussian visibility counts + densification modulation.
  continual    : the online refinement loop tying everything together.
  frame        : a lightweight monocular-frame container (image + camera + SMPL).
"""

from .frame import MonocularFrame, estimate_frame_smpl
from .warm_start import warm_start_avatar
from .ewc import EWCPrior
from .replay import ReplayBuffer
from .reactivation import VisibilityAccumulator
from .continual import ContinualRefiner

__all__ = [
    "MonocularFrame",
    "estimate_frame_smpl",
    "warm_start_avatar",
    "EWCPrior",
    "ReplayBuffer",
    "VisibilityAccumulator",
    "ContinualRefiner",
]
