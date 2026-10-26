"""Joint indices and body-pose layout for SMPL and SMPL-X.

Both skeletons share the same body joints for indices ``0..21``; they differ
only in what comes after (SMPL: two hand joints, SMPL-X: jaw/eyes/fingers).
That shared prefix lets an action be written once and applied to either model.

``body_pose`` never contains the root joint, so joint ``j`` occupies the
3-vector at ``body_pose[3*(j-1) : 3*j]``.  Writing a pose at the wrong slot is
silent — the model still moves, just the wrong limb — which is why these
indices live in one place instead of being spelled out per call site.
"""

from __future__ import annotations

# ── Shared body joints (identical in SMPL 24 and SMPL-X 55) ─────────────────
PELVIS = 0
L_HIP = 1
R_HIP = 2
SPINE1 = 3
L_KNEE = 4
R_KNEE = 5
SPINE2 = 6
L_ANKLE = 7
R_ANKLE = 8
SPINE3 = 9
L_FOOT = 10
R_FOOT = 11
NECK = 12
L_COLLAR = 13
R_COLLAR = 14
HEAD = 15
L_SHOULDER = 16
R_SHOULDER = 17
L_ELBOW = 18
R_ELBOW = 19
L_WRIST = 20
R_WRIST = 21

# ── SMPL-only joints ────────────────────────────────────────────────────────
SMPL_NUM_JOINTS = 24
SMPL_L_HAND = 22
SMPL_R_HAND = 23
# 23 joints excluding the root → 69 axis-angle values.
SMPL_BODY_POSE_DIM = 69

# ── SMPL-X-only joints ──────────────────────────────────────────────────────
SMPLX_NUM_JOINTS = 55
SMPLX_JAW = 22
SMPLX_L_EYE = 23
SMPLX_R_EYE = 24
# 21 body joints excluding the root → 63 axis-angle values.
SMPLX_BODY_POSE_DIM = 63

# The shared body prefix covers joints 1..21 = 21 joints.
NUM_BODY_JOINTS = 21
BODY_POSE_DIM = 3 * NUM_BODY_JOINTS  # 63

BODY_JOINT_NAMES = {
    L_HIP: "L_Hip", R_HIP: "R_Hip", SPINE1: "Spine1",
    L_KNEE: "L_Knee", R_KNEE: "R_Knee", SPINE2: "Spine2",
    L_ANKLE: "L_Ankle", R_ANKLE: "R_Ankle", SPINE3: "Spine3",
    L_FOOT: "L_Foot", R_FOOT: "R_Foot", NECK: "Neck",
    L_COLLAR: "L_Collar", R_COLLAR: "R_Collar", HEAD: "Head",
    L_SHOULDER: "L_Shoulder", R_SHOULDER: "R_Shoulder",
    L_ELBOW: "L_Elbow", R_ELBOW: "R_Elbow",
    L_WRIST: "L_Wrist", R_WRIST: "R_Wrist",
}


def body_pose_slot(joint: int) -> int:
    """Start index of *joint* inside a ``body_pose`` vector.

    Args:
        joint: SMPL/SMPL-X joint index, 1..21 (the root is not posed here).

    Returns:
        Index of the first of the joint's three axis-angle components.
    """
    if not 1 <= joint <= NUM_BODY_JOINTS:
        raise ValueError(
            f"joint {joint} has no body_pose slot; expected 1..{NUM_BODY_JOINTS}."
        )
    return 3 * (joint - 1)


def body_pose_dim(n_joints: int) -> int:
    """Number of axis-angle values expected for a skeleton with *n_joints*."""
    if n_joints >= SMPLX_NUM_JOINTS:
        return SMPLX_BODY_POSE_DIM
    return SMPL_BODY_POSE_DIM


def match_body_pose(body_pose, expected_dim: int):
    """Pad or slice a ``body_pose`` to *expected_dim* along the last axis.

    Poses are authored in the shared 63-dim body layout.  SMPL additionally
    poses two hand joints (slots 21, 22), which are left at rest; SMPL-X
    carries jaw/eye pose in separate arguments, so nothing is lost by slicing.

    Args:
        body_pose: ``(..., D)`` tensor or ``None``.
        expected_dim: target size of the last axis (63 or 69).

    Returns:
        ``(..., expected_dim)`` tensor, or ``None`` when *body_pose* is None.
    """
    if body_pose is None:
        return None
    import torch

    d = body_pose.shape[-1]
    if d == expected_dim:
        return body_pose
    if d > expected_dim:
        return body_pose[..., :expected_dim].contiguous()
    pad = torch.zeros(
        *body_pose.shape[:-1], expected_dim - d,
        dtype=body_pose.dtype, device=body_pose.device,
    )
    return torch.cat([body_pose, pad], dim=-1)


# Default shoulder swing that brings the arms down from the T-pose.
#
# Both SMPL and SMPL-X ship a T-pose (arms horizontal): the neutral template
# spans ~1.75 m across the shoulders with the wrists at y=+0.22.  A
# photographed subject stands with arms down, spanning ~1.1 m.  Fitting a
# T-pose to such a cloud therefore drives the scale down until the arms fit
# inside the torso, which shrinks everything else with them.  Posing the
# shoulders down first removes that bias.
#
# Sign is determined empirically from forward kinematics on SMPL_NEUTRAL:
# with L=-s, R=+s about Z the wrists move to y=-0.04 and x=±0.36 (arms down,
# close to the body).  The opposite signs raise the arms overhead.
#
# Magnitude chosen by sweeping the shoulder angle and measuring the fit on
# ``static/000001``: trimmed symmetric Chamfer was 1.83% of body height at
# 0.6 rad, 1.12% at 1.0, and 0.95% at 1.1, flattening after that (0.96% at
# 1.4).  1.1 rad also leaves the arms slightly clear of the torso, which is
# how a standing subject is normally photographed.
A_POSE_SHOULDER_SWING = 1.1


def a_pose_body_pose(device="cuda", swing: float = A_POSE_SHOULDER_SWING):
    """Body pose (63-dim) with the arms lowered from the T-pose.

    The left and right shoulders need opposite signs about Z; equal signs
    raise one arm and lower the other, which measures as a *taller* mesh
    rather than a narrower one.
    """
    import torch

    bp = torch.zeros(BODY_POSE_DIM, device=device)
    bp[body_pose_slot(L_SHOULDER) + 2] = -swing
    bp[body_pose_slot(R_SHOULDER) + 2] = swing
    return bp
