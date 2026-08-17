"""Skinned Gaussian Splatting for human body animation.

This package implements SMPL/SMPL-X skeleton-driven 3D Gaussian Splatting:
  1. Gaussians are defined in canonical (rest-pose) space.
  2. Each Gaussian stores skinning weights (inherited from SMPL/SMPL-X vertices).
  3. Given a new pose, LBS (Linear Blend Skinning) deforms Gaussians
     from canonical to posed space.
  4. Expression blendshapes deform facial Gaussians.
  5. gsplat renders the deformed Gaussians.
"""

from .smpl_model import SMPLWrapper
from .smplx_model import SMPLXWrapper
from .skinning import lbs_deform_gaussians
from .skinned_gaussians import SkinnedGaussians
from .rigging import rig_gaussians_to_smpl, rig_gaussians_to_smplx
from .pose_io import load_pose_sequence, generate_demo_poses, save_pose_sequence
from .animation import render_animation, render_animation_comparison

__all__ = [
    "SMPLWrapper",
    "SMPLXWrapper",
    "lbs_deform_gaussians",
    "SkinnedGaussians",
    "rig_gaussians_to_smpl",
    "rig_gaussians_to_smplx",
    "load_pose_sequence",
    "generate_demo_poses",
    "save_pose_sequence",
    "render_animation",
    "render_animation_comparison",
]
