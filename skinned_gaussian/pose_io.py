"""SMPL pose parameter I/O and demo pose generation."""

from __future__ import annotations

import json
import math
import numpy as np
import torch


def load_pose_sequence(path: str, device: str = 'cuda') -> dict:
    """Load a SMPL pose sequence from JSON or NPZ.

    Expected JSON format:
    {
        "betas": [10 floats],
        "body_pose": [[69 floats], ...],   # (T, 69) = 23 joints × 3
        "global_orient": [[3 floats], ...], # (T, 3)
        "transl": [[3 floats], ...]         # (T, 3)
    }

    Returns a dict with torch tensors on the specified device.
    """
    dev = torch.device(device)

    if path.endswith('.json'):
        with open(path, 'r') as f:
            data = json.load(f)
    elif path.endswith('.npz'):
        data = dict(np.load(path, allow_pickle=True))
    else:
        raise ValueError(f"Unsupported pose file format: {path}")

    result = {
        'betas': torch.tensor(data['betas'], dtype=torch.float32, device=dev) if 'betas' in data else torch.zeros(10, device=dev),
        'body_pose': torch.tensor(data['body_pose'], dtype=torch.float32, device=dev) if 'body_pose' in data else None,
        'global_orient': torch.tensor(data['global_orient'], dtype=torch.float32, device=dev) if 'global_orient' in data else None,
        'transl': torch.tensor(data['transl'], dtype=torch.float32, device=dev) if 'transl' in data else None,
    }

    # Determine number of frames.
    for key in ['body_pose', 'global_orient', 'transl']:
        if result[key] is not None:
            result['n_frames'] = result[key].shape[0]
            break
    else:
        result['n_frames'] = 0

    print(f"  Loaded pose sequence: {result['n_frames']} frames")
    return result


def generate_demo_poses(
    n_frames: int = 120,
    betas: torch.Tensor | None = None,
    device: str = 'cuda',
) -> dict:
    """Generate a simple demo pose sequence (arm wave + slight rotation).

    Creates a sequence where:
      - The right arm raises and lowers (shoulder rotation).
      - The body slightly rotates left/right (global orientation).
      - A subtle breathing motion (spine joint).

    Args:
        n_frames: Number of frames to generate.
        betas: (10,) shape parameters. If None, uses zeros (mean body).
        device: torch device.

    Returns:
        Dict with 'betas', 'body_pose' (n, 69), 'global_orient' (n, 3),
        'transl' (n, 3), 'n_frames'.
    """
    dev = torch.device(device)

    if betas is None:
        betas = torch.zeros(10, device=dev)

    # Time parameter.
    t = torch.linspace(0, 2 * math.pi, n_frames, device=dev)  # (n_frames,)

    # Body pose: 23 joints × 3 = 69 params, all zeros initially.
    # NOTE: body_pose excludes root (joint 0), so body_pose slot i ↔ SMPL joint i+1.
    # To target SMPL joint j, index (j-1)*3 in the 69-vector.
    # SMPL joints: 1=L_Hip, 2=R_Hip, 3=Spine1, 6=Spine2,
    #   13=L_Shoulder, 14=L_Elbow, 16=R_Shoulder, 17=R_Elbow
    body_pose = torch.zeros(n_frames, 69, device=dev)

    # R_Shoulder (SMPL joint 16) → body_pose slot 15
    body_pose[:, 15 * 3 + 0] = torch.sin(t) * 0.8   # R_Shoulder_X: arm raise
    body_pose[:, 15 * 3 + 2] = torch.cos(t) * 0.3   # R_Shoulder_Z: arm rotation

    # R_Elbow (SMPL joint 17) → body_pose slot 16
    body_pose[:, 16 * 3 + 0] = 0.5 + torch.sin(t) * 0.3  # R_Elbow_X: bend

    # L_Shoulder (SMPL joint 13) → body_pose slot 12
    body_pose[:, 12 * 3 + 0] = torch.sin(t + math.pi) * 0.3  # L_Shoulder_X: counter-motion

    # Spine1 (SMPL joint 3) → body_pose slot 2
    body_pose[:, 2 * 3 + 0] = torch.sin(t * 2) * 0.05  # Spine1_X: breathing

    # Spine2 (SMPL joint 6) → body_pose slot 5
    body_pose[:, 5 * 3 + 0] = torch.sin(t * 2) * 0.05  # Spine2_X: breathing

    # Global orientation: slight left-right rotation
    global_orient = torch.zeros(n_frames, 3, device=dev)
    global_orient[:, 1] = torch.sin(t * 0.5) * 0.2  # Y rotation (turn left/right)

    # Global translation: stay in place
    transl = torch.zeros(n_frames, 3, device=dev)

    print(f"  Generated {n_frames} demo frames (arm wave + rotation)")
    return {
        'betas': betas,
        'body_pose': body_pose,
        'global_orient': global_orient,
        'transl': transl,
        'n_frames': n_frames,
    }


def save_pose_sequence(path: str, poses: dict):
    """Save a pose sequence to JSON."""
    data = {}
    for key in ['betas', 'body_pose', 'global_orient', 'transl']:
        if key in poses and poses[key] is not None:
            data[key] = poses[key].detach().cpu().numpy().tolist()
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved pose sequence to {path}")
