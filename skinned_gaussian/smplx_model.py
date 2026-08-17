"""SMPL-X model wrapper — 55 joints with face, hands, and expression.

Extends SMPLWrapper to support:
  - 55 joints (body 22 + head/jaw/eyes 3 + hands 30)
  - 100-dimensional expression blendshapes
  - Jaw and eye pose parameters
  - FLAME-compatible facial animation
"""

from __future__ import annotations

import pickle
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from .smpl_model import SMPLWrapper


class SMPLXWrapper:
    """SMPL-X model with 55 joints, expression, and face/hand support.

    Joint order (verified by position):
      0-12: body (pelvis → spine3 → neck)
      13-14: L/R collar
      15: head
      16-21: L/R shoulder → wrist
      22: jaw (controls mouth)
      23-24: L/R eye (controls gaze)
      25-54: finger joints (15 per hand)
    """

    NUM_JOINTS = 55
    NUM_VERTICES = 10475
    NUM_BETAS = 300       # shape params
    NUM_EXPRESSION = 100  # expression params
    NUM_BODY_JOINTS = 22  # joints 0-21 (body pose = 21*3 = 63)
    NUM_JAW_PARAMS = 3    # jaw pose
    NUM_EYE_PARAMS = 6    # L eye (3) + R eye (3)

    JOINT_NAMES = [
        'pelvis', 'L_hip', 'R_hip', 'spine1', 'L_knee', 'R_knee',
        'spine2', 'L_ankle', 'R_ankle', 'spine3', 'L_foot', 'R_foot',
        'neck', 'L_collar', 'R_collar', 'head',
        'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow', 'L_wrist', 'R_wrist',
        'jaw', 'L_eye', 'R_eye',
        'L_thumb1', 'L_thumb2', 'L_thumb3',
        'L_index1', 'L_index2', 'L_index3',
        'L_middle1', 'L_middle2', 'L_middle3',
        'L_ring1', 'L_ring2', 'L_ring3',
        'L_pinky1', 'L_pinky2', 'L_pinky3',
        'R_thumb1', 'R_thumb2', 'R_thumb3',
        'R_index1', 'R_index2', 'R_index3',
        'R_middle1', 'R_middle2', 'R_middle3',
        'R_ring1', 'R_ring2', 'R_ring3',
        'R_pinky1', 'R_pinky2', 'R_pinky3',
    ]

    def __init__(self, model_path: str, device: str = 'cuda'):
        """Load SMPL-X from .npz file.

        Args:
            model_path: Path to SMPLX_NEUTRAL_2020.npz
            device: torch device.
        """
        self.device = torch.device(device)
        data = np.load(model_path, allow_pickle=True)

        def to_t(x):
            if hasattr(x, 'toarray'):
                x = x.toarray()
            return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)

        self.v_template = to_t(data['v_template'])       # (10475, 3)
        self.weights = to_t(data['weights'])              # (10475, 55)
        self.shapedirs = to_t(data['shapedirs'])          # (10475, 3, 400)
        self.posedirs = to_t(data['posedirs'])            # (10475, 3, 486)
        self.J_regressor = to_t(data['J_regressor'])      # (55, 10475)
        self.faces = torch.from_numpy(np.asarray(data['f'], dtype=np.int64)).to(self.device)

        # kintree: row 0 = parent, row 1 = child
        kt = np.asarray(data['kintree_table'], dtype=np.int64)
        self.parents = {}
        for i in range(self.NUM_JOINTS):
            p = int(kt[0, i])
            self.parents[i] = p if p != 4294967295 else -1

        # Split shapedirs into shape (300) and expression (100).
        self.shapedirs_shape = self.shapedirs[:, :, :300]     # (10475, 3, 300)
        self.shapedirs_expr = self.shapedirs[:, :, 300:]      # (10475, 3, 100)

        # Alignment (for scene-space placement).
        self._align_scale = 1.0
        self._align_transl = torch.zeros(3, device=self.device)
        self._aligned = False

        print(f"SMPL-X loaded: {self.NUM_VERTICES} vertices, {self.NUM_JOINTS} joints, "
              f"{self.NUM_BETAS} shape, {self.NUM_EXPRESSION} expression, "
              f"{self.faces.shape[0]} faces")

    def align_to_scene(self, scale: float, transl: torch.Tensor):
        """Scale + translate to match scene coordinates."""
        self._align_scale = scale
        self._align_transl = transl.to(self.device)
        self.v_template = self.v_template * scale + self._align_transl
        self.shapedirs_shape = self.shapedirs_shape * scale
        self.shapedirs_expr = self.shapedirs_expr * scale
        self._aligned = True
        print(f"  SMPL-X aligned: scale={scale:.3f}, transl={transl.tolist()}")

    def get_rest_mesh(self, betas=None, expression=None):
        """Return rest-pose vertices with shape + expression blendshapes.

        Args:
            betas: (300,) shape parameters.
            expression: (100,) expression parameters.

        Returns:
            vertices (10475, 3), faces (20908, 3)
        """
        if betas is None:
            betas = torch.zeros(self.NUM_BETAS, device=self.device)
        if expression is None:
            expression = torch.zeros(self.NUM_EXPRESSION, device=self.device)

        v = self.v_template + torch.einsum('vdb,b->vd', self.shapedirs_shape, betas)
        v = v + torch.einsum('vdb,b->vd', self.shapedirs_expr, expression)
        return v, self.faces

    def get_rest_joints(self, betas=None, expression=None):
        """Return rest-pose joint locations (55, 3)."""
        v, _ = self.get_rest_mesh(betas, expression)
        return self.J_regressor @ v

    def get_skinning_weights(self):
        """Return SMPL-X skinning weights (10475, 55)."""
        return self.weights

    def get_joint_transforms(
        self,
        betas=None,
        body_pose=None,      # (63,) = 21 joints × 3 (excl. root)
        global_orient=None,   # (3,)
        transl=None,
        jaw_pose=None,        # (3,) jaw rotation
        leye_pose=None,       # (3,) left eye
        reye_pose=None,       # (3,) right eye
        expression=None,      # (100,) expression coefficients
    ):
        """Compute per-joint RELATIVE transformation matrices for LBS.

        Returns T_k = G_k @ G_rest^{-1} (55, 4, 4).
        For rest pose, T_k = I (identity).
        """
        if betas is None:
            betas = torch.zeros(self.NUM_BETAS, device=self.device)
        if body_pose is None:
            body_pose = torch.zeros(63, device=self.device)
        if global_orient is None:
            global_orient = torch.zeros(3, device=self.device)
        if transl is None:
            transl = torch.zeros(3, device=self.device)
        if jaw_pose is None:
            jaw_pose = torch.zeros(3, device=self.device)
        if leye_pose is None:
            leye_pose = torch.zeros(3, device=self.device)
        if reye_pose is None:
            reye_pose = torch.zeros(3, device=self.device)
        if expression is None:
            expression = torch.zeros(self.NUM_EXPRESSION, device=self.device)

        # Build full pose (55 joints × 3 = 165).
        full_pose = torch.zeros(self.NUM_JOINTS, 3, device=self.device)
        full_pose[0] = global_orient          # root
        full_pose[1:22] = body_pose.view(21, 3)  # body (excl. root)
        full_pose[22] = jaw_pose              # jaw
        full_pose[23] = leye_pose             # L eye
        full_pose[24] = reye_pose             # R eye
        # Joints 25-54 (fingers) = zeros (no finger animation in this demo)

        # Posed transforms.
        G_posed = self._forward_kinematics(betas, expression, full_pose, transl)
        # Rest transforms (zero pose, zero expression, zero transl).
        G_rest = self._forward_kinematics(
            betas, torch.zeros_like(expression),
            torch.zeros_like(full_pose), torch.zeros_like(transl))
        # Relative transforms.
        G_rest_inv = self._invert_transforms(G_rest)
        T = torch.bmm(G_posed, G_rest_inv)
        return T

    def _forward_kinematics(self, betas, expression, full_pose, transl):
        """Compute absolute per-joint world transforms (55, 4, 4)."""
        v_shaped = self.v_template
        v_shaped = v_shaped + torch.einsum('vdb,b->vd', self.shapedirs_shape, betas)
        v_shaped = v_shaped + torch.einsum('vdb,b->vd', self.shapedirs_expr, expression)
        J = self.J_regressor @ v_shaped  # (55, 3)

        rot_mats = SMPLWrapper._batch_rodrigues(full_pose)  # (55, 3, 3)

        transforms = torch.zeros(self.NUM_JOINTS, 4, 4, device=self.device)
        transforms[:, 3, 3] = 1.0

        for j in range(self.NUM_JOINTS):
            R_j = rot_mats[j]
            parent = self.parents[j]

            if parent == -1:
                transforms[j, :3, :3] = R_j
                transforms[j, :3, 3] = J[j] + transl
            else:
                R_parent = transforms[parent, :3, :3]
                t_parent = transforms[parent, :3, 3]
                t_local = J[j] - R_j @ J[parent]
                transforms[j, :3, :3] = R_parent @ R_j
                transforms[j, :3, 3] = R_parent @ t_local + t_parent

        return transforms

    @staticmethod
    def _invert_transforms(G):
        """Invert (55, 4, 4) rigid transforms."""
        G_inv = G.clone()
        R = G[:, :3, :3]
        t = G[:, :3, 3]
        G_inv[:, :3, :3] = R.transpose(-2, -1)
        G_inv[:, :3, 3] = -torch.bmm(R.transpose(-2, -1), t.unsqueeze(-1)).squeeze(-1)
        return G_inv

    def get_expression_displacements(self, expression):
        """Compute per-vertex expression displacement.

        Δv_expr = shapedirs_expr @ expression  →  (10475, 3)

        This can be interpolated to Gaussians for expression-driven deformation.
        """
        return torch.einsum('vdb,b->vd', self.shapedirs_expr, expression)
