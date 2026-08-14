"""Skinned Gaussian Splatting for human body animation.

This package implements SMPL skeleton-driven 3D Gaussian Splatting:
  1. Gaussians are defined in canonical (rest-pose) space.
  2. Each Gaussian stores skinning weights (inherited from SMPL vertices).
  3. Given a new pose, LBS (Linear Blend Skinning) deforms Gaussians
     from canonical to posed space.
  4. gsplat renders the deformed Gaussians.
"""

from .smpl_model import SMPLWrapper
from .skinning import lbs_deform_gaussians
from .skinned_gaussians import SkinnedGaussians
from .rigging import rig_gaussians_to_smpl
from .pose_io import load_pose_sequence, generate_demo_poses
from .animation import render_animation

__all__ = [
    "SMPLWrapper",
    "lbs_deform_gaussians",
    "SkinnedGaussians",
    "rig_gaussians_to_smpl",
    "load_pose_sequence",
    "generate_demo_poses",
    "render_animation",
]
