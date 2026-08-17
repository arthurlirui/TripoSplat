"""Quaternion utilities — NumPy and PyTorch implementations.

Provides a single source of truth for quaternion ↔ rotation matrix conversion
and quaternion composition, shared across ``triposplat.py``, ``gaussian_model.py``,
``skinning.py``, and ``alignment.py``.

NumPy API
---------
- ``quat_to_matrix(q)``  — (N, 4) → (N, 3, 3)
- ``matrix_to_quat(R)``  — (N, 3, 3) → (N, 4)

PyTorch API
-----------
- ``rotation_matrix_to_quaternion(R)`` — (..., 3, 3) → (..., 4)
- ``quaternion_compose(q1, q2)``      — Hamilton product with broadcasting
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# NumPy implementation
# ═══════════════════════════════════════════════════════════════════════════

def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert (N, 4) quaternions (w,x,y,z) to (N, 3, 3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], axis=-1).reshape(-1, 3, 3)
    return R


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert (N, 3, 3) rotation matrices to (N, 4) quaternions (w,x,y,z).

    Uses the Shepperd method with four branches for numerical stability.
    """
    N = R.shape[0]
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = np.zeros((N, 4), dtype=R.dtype)

    # ── Branch 1: trace > 0 (the common case) ──
    s = np.sqrt(np.maximum(trace + 1, 0)) * 2
    q[:, 0] = 0.25 * s
    # Use np.divide with a where mask to avoid NaN when s == 0 (those rows
    # are overwritten by later branches anyway).
    q[:, 1] = np.divide(R[:, 2, 1] - R[:, 1, 2], s, where=(s != 0),
                        out=np.zeros(N, dtype=R.dtype))
    q[:, 2] = np.divide(R[:, 0, 2] - R[:, 2, 0], s, where=(s != 0),
                        out=np.zeros(N, dtype=R.dtype))
    q[:, 3] = np.divide(R[:, 1, 0] - R[:, 0, 1], s, where=(s != 0),
                        out=np.zeros(N, dtype=R.dtype))

    # ── Branch 2: r00 is the largest diagonal ──
    m01 = (R[:, 0, 0] >= R[:, 1, 1]) & (R[:, 0, 0] >= R[:, 2, 2]) & (s == 0)
    if m01.any():
        s1 = np.sqrt(np.maximum(1 + R[m01, 0, 0] - R[m01, 1, 1] - R[m01, 2, 2], 0)) * 2
        q[m01, 0] = (R[m01, 2, 1] - R[m01, 1, 2]) / s1
        q[m01, 1] = 0.25 * s1
        q[m01, 2] = (R[m01, 0, 1] + R[m01, 1, 0]) / s1
        q[m01, 3] = (R[m01, 0, 2] + R[m01, 2, 0]) / s1

    # ── Branch 3: r11 is the largest diagonal ──
    m11 = (R[:, 1, 1] > R[:, 0, 0]) & (R[:, 1, 1] >= R[:, 2, 2]) & (s == 0)
    if m11.any():
        s2 = np.sqrt(np.maximum(1 + R[m11, 1, 1] - R[m11, 0, 0] - R[m11, 2, 2], 0)) * 2
        q[m11, 0] = (R[m11, 0, 2] - R[m11, 2, 0]) / s2
        q[m11, 1] = (R[m11, 0, 1] + R[m11, 1, 0]) / s2
        q[m11, 2] = 0.25 * s2
        q[m11, 3] = (R[m11, 1, 2] + R[m11, 2, 1]) / s2

    # ── Branch 4: r22 is the largest diagonal ──
    m21 = (R[:, 2, 2] > R[:, 0, 0]) & (R[:, 2, 2] > R[:, 1, 1]) & (s == 0)
    if m21.any():
        s3 = np.sqrt(np.maximum(1 + R[m21, 2, 2] - R[m21, 0, 0] - R[m21, 1, 1], 0)) * 2
        q[m21, 0] = (R[m21, 1, 0] - R[m21, 0, 1]) / s3
        q[m21, 1] = (R[m21, 0, 2] + R[m21, 2, 0]) / s3
        q[m21, 2] = (R[m21, 1, 2] + R[m21, 2, 1]) / s3
        q[m21, 3] = 0.25 * s3

    return q / np.linalg.norm(q, axis=-1, keepdims=True)


# ═══════════════════════════════════════════════════════════════════════════
# PyTorch implementation (batched, supports arbitrary leading dims)
# ═══════════════════════════════════════════════════════════════════════════

def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions (w, x, y, z).

    Supports batched input: ``(..., 3, 3) → (..., 4)``.
    Uses the Shepperd method with four branches.
    """
    shape = R.shape[:-2]  # leading batch dims
    R = R.reshape(-1, 3, 3)

    r00, r01, r02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    r10, r11, r12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    r20, r21, r22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    trace = r00 + r11 + r22
    q = torch.zeros(R.shape[0], 4, dtype=R.dtype, device=R.device)

    # Branch 1: trace > 0
    mask1 = trace > 0
    if mask1.any():
        S = torch.sqrt(trace[mask1] + 1.0) * 2.0
        q[mask1, 0] = 0.25 * S
        q[mask1, 1] = (r21[mask1] - r12[mask1]) / S
        q[mask1, 2] = (r02[mask1] - r20[mask1]) / S
        q[mask1, 3] = (r10[mask1] - r01[mask1]) / S

    # Branch 2: r00 largest
    mask2 = (~mask1) & (r00 >= r11) & (r00 >= r22)
    if mask2.any():
        S = torch.sqrt(1.0 + r00[mask2] - r11[mask2] - r22[mask2]) * 2.0
        q[mask2, 0] = (r21[mask2] - r12[mask2]) / S
        q[mask2, 1] = 0.25 * S
        q[mask2, 2] = (r01[mask2] + r10[mask2]) / S
        q[mask2, 3] = (r02[mask2] + r20[mask2]) / S

    # Branch 3: r11 largest
    mask3 = (~mask1) & (~mask2) & (r11 >= r22)
    if mask3.any():
        S = torch.sqrt(1.0 + r11[mask3] - r00[mask3] - r22[mask3]) * 2.0
        q[mask3, 0] = (r02[mask3] - r20[mask3]) / S
        q[mask3, 1] = (r01[mask3] + r10[mask3]) / S
        q[mask3, 2] = 0.25 * S
        q[mask3, 3] = (r12[mask3] + r21[mask3]) / S

    # Branch 4: r22 largest
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if mask4.any():
        S = torch.sqrt(1.0 + r22[mask4] - r00[mask4] - r11[mask4]) * 2.0
        q[mask4, 0] = (r10[mask4] - r01[mask4]) / S
        q[mask4, 1] = (r02[mask4] + r20[mask4]) / S
        q[mask4, 2] = (r12[mask4] + r21[mask4]) / S
        q[mask4, 3] = 0.25 * S

    q = F.normalize(q.reshape(*shape, 4), dim=-1)
    return q


def quaternion_compose(
    q1: torch.Tensor,  # (..., 4)
    q2: torch.Tensor,  # (..., 4)
) -> torch.Tensor:
    """Hamilton product q1 ⊗ q2 with broadcasting.

    Supports any leading batch dimensions that broadcast together.
    Examples:
        q1=(4,),     q2=(N,4)  → (N,4)
        q1=(N,4),    q2=(N,4)  → (N,4)
        q1=(B,N,4),  q2=(B,N,4)  → (B,N,4)
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)
