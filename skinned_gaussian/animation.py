"""Animation rendering — drive skinned Gaussians with a pose sequence."""

from __future__ import annotations

import os
import time
import numpy as np
import torch
import cv2

from .smpl_model import SMPLWrapper
from .skinned_gaussians import SkinnedGaussians


def render_animation(
    skinned: SkinnedGaussians,
    smpl: SMPLWrapper,
    pose_sequence: dict,
    camera,
    output_path: str,
    fps: int = 30,
    background: torch.Tensor | None = None,
    sh_degree: int = 0,
    verbose: bool = True,
):
    """Render an animation by driving skinned Gaussians with a pose sequence.

    For each frame:
      1. Compute SMPL joint transforms from pose parameters.
      2. Deform canonical Gaussians to posed space via LBS.
      3. Render with gsplat.
      4. Write frame to video.

    Args:
        skinned: SkinnedGaussians in canonical space.
        smpl: SMPLWrapper for joint transform computation.
        pose_sequence: Dict with 'betas', 'body_pose' (T,69),
                       'global_orient' (T,3), 'transl' (T,3).
        camera: Camera to render from (fixed or per-frame).
        output_path: Output .mp4 path.
        fps: Video FPS.
        background: (3,) background colour, default black.
        sh_degree: SH degree for rendering.
        verbose: Print progress.
    """
    device = skinned.device
    n_frames = pose_sequence['n_frames']
    betas = pose_sequence['betas']
    body_pose = pose_sequence['body_pose']
    global_orient = pose_sequence['global_orient']
    transl = pose_sequence['transl']

    # Defaults for missing pose parameters.
    if body_pose is None:
        body_pose = torch.zeros(n_frames, 69, device=device)
    if global_orient is None:
        global_orient = torch.zeros(n_frames, 3, device=device)

    if background is None:
        background = torch.zeros(3, device=device)

    # Determine video dimensions.
    W, H = camera.width, camera.height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    if verbose:
        print(f"\nRendering {n_frames} frames at {W}x{H} @ {fps} fps...")
        t0 = time.time()

    for frame_idx in range(n_frames):
        # Compute joint transforms for this frame.
        joint_transforms = smpl.get_joint_transforms(
            betas=betas,
            body_pose=body_pose[frame_idx],
            global_orient=global_orient[frame_idx],
            transl=transl[frame_idx] if transl is not None else None,
        )

        # Deform + render.
        with torch.no_grad():
            rgb, alpha, meta = skinned.render_posed(
                joint_transforms, camera,
                background=background, sh_degree=sh_degree,
            )

        # Convert to BGR for OpenCV.
        frame_np = (rgb.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        vw.write(frame_bgr)

        if verbose and (frame_idx % 20 == 0 or frame_idx == 0):
            elapsed = time.time() - t0
            print(f"  frame {frame_idx:4d}/{n_frames}: rgb_mean={rgb.mean():.4f}, "
                  f"alpha_mean={alpha.mean():.4f}, {elapsed:.1f}s")

    vw.release()
    if verbose:
        elapsed = time.time() - t0
        print(f"\n  Video saved: {output_path} ({n_frames} frames, {elapsed:.1f}s)")
        print(f"  File size: {os.path.getsize(output_path) // 1024} KB")


def render_animation_comparison(
    skinned: SkinnedGaussians,
    smpl: SMPLWrapper,
    pose_sequence: dict,
    camera,
    output_path: str,
    fps: int = 30,
    background: torch.Tensor | None = None,
    sh_degree: int = 0,
):
    """Render a side-by-side comparison: canonical (rest pose) vs animated.

    Left: Gaussians in canonical (rest) pose.
    Right: Gaussians deformed by the current frame's pose.
    """
    device = skinned.device
    n_frames = pose_sequence['n_frames']
    betas = pose_sequence['betas']
    body_pose = pose_sequence['body_pose']
    global_orient = pose_sequence['global_orient']
    transl = pose_sequence['transl']

    # Defaults for missing pose parameters.
    if body_pose is None:
        body_pose = torch.zeros(n_frames, 69, device=device)
    if global_orient is None:
        global_orient = torch.zeros(n_frames, 3, device=device)

    if background is None:
        background = torch.zeros(3, device=device)

    W, H = camera.width, camera.height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(output_path, fourcc, fps, (W * 2, H))

    # Rest-pose joint transforms (all zeros).
    rest_transforms = smpl.get_joint_transforms(betas=betas)

    print(f"\nRendering comparison: {n_frames} frames (canonical | animated)...")
    t0 = time.time()

    for frame_idx in range(n_frames):
        joint_transforms = smpl.get_joint_transforms(
            betas=betas,
            body_pose=body_pose[frame_idx],
            global_orient=global_orient[frame_idx],
            transl=transl[frame_idx] if transl is not None else None,
        )

        with torch.no_grad():
            # Canonical (rest pose).
            rgb_rest, _, _ = skinned.render_posed(
                rest_transforms, camera, background=background, sh_degree=sh_degree)
            # Animated.
            rgb_anim, _, _ = skinned.render_posed(
                joint_transforms, camera, background=background, sh_degree=sh_degree)

        rest_bgr = cv2.cvtColor((rgb_rest.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        anim_bgr = cv2.cvtColor((rgb_anim.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        frame = np.concatenate([rest_bgr, anim_bgr], axis=1)

        cv2.putText(frame, 'Canonical (rest)', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f'Animated frame {frame_idx}', (W + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        vw.write(frame)

        if frame_idx % 20 == 0:
            print(f"  frame {frame_idx:4d}/{n_frames}, {time.time()-t0:.1f}s")

    vw.release()
    print(f"\n  Comparison video: {output_path} ({os.path.getsize(output_path)//1024} KB)")
