"""SMPL pose parameter I/O and demo pose generation.

Poses are authored in the 63-dim shared body layout (joints 1..21, root
excluded) so one sequence drives either skeleton.  ``match_body_pose`` is
applied at consumption time to pad to SMPL's 69 dims.
"""

from __future__ import annotations

import json
import math
import numpy as np
import torch

from .joints import (
    BODY_POSE_DIM,
    L_ELBOW,
    L_SHOULDER,
    R_ELBOW,
    R_SHOULDER,
    SPINE1,
    SPINE2,
    body_pose_slot,
)


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

    # Body pose in the shared 63-dim layout (21 joints, root excluded).
    # Authoring it here rather than per-skeleton means SMPL gets padded to 69
    # at consumption time instead of SMPL-X receiving an over-long vector.
    body_pose = torch.zeros(n_frames, BODY_POSE_DIM, device=dev)

    body_pose[:, body_pose_slot(R_SHOULDER) + 0] = torch.sin(t) * 0.8   # arm raise
    body_pose[:, body_pose_slot(R_SHOULDER) + 2] = torch.cos(t) * 0.3   # arm twist
    body_pose[:, body_pose_slot(R_ELBOW) + 0] = 0.5 + torch.sin(t) * 0.3  # bend
    body_pose[:, body_pose_slot(L_SHOULDER) + 0] = torch.sin(t + math.pi) * 0.3
    body_pose[:, body_pose_slot(SPINE1) + 0] = torch.sin(t * 2) * 0.05  # breathing
    body_pose[:, body_pose_slot(SPINE2) + 0] = torch.sin(t * 2) * 0.05

    # Global orientation: slight left-right rotation
    global_orient = torch.zeros(n_frames, 3, device=dev)
    global_orient[:, 1] = torch.sin(t * 0.5) * 0.2  # Y rotation (turn left/right)

    # Global translation: stay in place
    transl = torch.zeros(n_frames, 3, device=dev)

    # Jaw pose: mouth open/close (SMPL-X joint 22).
    jaw_pose = torch.zeros(n_frames, 3, device=dev)
    jaw_pose[:, 0] = (torch.sin(t * 2) * 0.3 + 0.3).clamp(min=0)  # jaw opens periodically

    # Eye pose: subtle gaze movement (SMPL-X joints 23, 24).
    leye_pose = torch.zeros(n_frames, 3, device=dev)
    leye_pose[:, 1] = torch.sin(t * 0.3) * 0.2  # L eye look left/right
    reye_pose = torch.zeros(n_frames, 3, device=dev)
    reye_pose[:, 1] = torch.sin(t * 0.3) * 0.2  # R eye same direction

    # Expression coefficients: 100-dim, use first few for simple expressions.
    # Expression 0: mouth smile, Expression 1: brow raise, etc.
    expression = torch.zeros(n_frames, 100, device=dev)
    expression[:, 0] = torch.sin(t) * 0.5  # smile/frown cycle
    expression[:, 1] = torch.sin(t * 0.7) * 0.3  # brow movement

    print(f"  Generated {n_frames} demo frames (arm wave + rotation + jaw + expression)")
    return {
        'betas': betas,
        'body_pose': body_pose,
        'global_orient': global_orient,
        'transl': transl,
        'jaw_pose': jaw_pose,
        'leye_pose': leye_pose,
        'reye_pose': reye_pose,
        'expression': expression,
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
