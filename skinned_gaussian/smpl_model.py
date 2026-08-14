"""SMPL model wrapper — loads SMPL .pkl and provides joint transforms.

Supports the original SMPL .pkl format (v_template, weights, posedirs,
shapedirs, J_regressor, kintree_table, f) without requiring chumpy.
"""

from __future__ import annotations

import pickle
import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


class SMPLWrapper:
    """Lightweight SMPL model that provides per-joint transformation matrices.

    Does not depend on the `smplx` package — directly loads the SMPL .pkl
    and implements LBS internally.
    """

    NUM_JOINTS = 24
    NUM_VERTICES = 6890
    NUM_BETAS = 10  # default shape params
    NUM_POSE_PARAMS = 72  # 24 joints × 3 axis-angle

    def __init__(self, model_path: str, gender: str = 'neutral', device: str = 'cuda'):
        """Load SMPL model from a .pkl file.

        Args:
            model_path: Path to the SMPL .pkl file (e.g. SMPL_NEUTRAL.pkl).
            gender: 'neutral', 'male', or 'female' (for reference only).
            device: torch device.
        """
        self.device = torch.device(device)
        self.gender = gender

        with open(model_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        # Helper: convert to dense numpy array (handles chumpy + sparse).
        def to_numpy(x):
            if hasattr(x, 'toarray'):  # scipy sparse matrix
                return np.asarray(x.toarray(), dtype=np.float32)
            elif hasattr(x, 'r'):  # chumpy object
                return np.asarray(x.r, dtype=np.float32)
            else:
                return np.asarray(x, dtype=np.float32)

        # Core SMPL components — convert to torch tensors.
        self.v_template = torch.from_numpy(
            to_numpy(data['v_template'])
        ).to(self.device)  # (6890, 3)

        self.weights = torch.from_numpy(
            to_numpy(data['weights'])
        ).to(self.device)  # (6890, 24) skinning weights

        self.posedirs = torch.from_numpy(
            to_numpy(data['posedirs'])
        ).to(self.device)  # (6890, 3, 207) pose blend shapes

        self.shapedirs = torch.from_numpy(
            to_numpy(data['shapedirs'])
        ).to(self.device)  # (6890, 3, 10) shape blend shapes

        self.J_regressor = torch.from_numpy(
            to_numpy(data['J_regressor'])
        ).to(self.device)  # (24, 6890) joint regressor

        self.faces = torch.from_numpy(
            np.asarray(data['f'], dtype=np.int64)
        ).to(self.device)  # (13776, 3) mesh faces

        kintree = np.asarray(data['kintree_table'], dtype=np.int64)
        self.kintree_table = torch.from_numpy(kintree).to(self.device)  # (2, 24)
        # parents[i] = parent joint of joint i (-1 for root).
        self.parents = kintree[0].copy()  # (24,) numpy
        self.parents[0] = -1  # root has no parent

        print(f"SMPL loaded: {self.NUM_VERTICES} vertices, {self.NUM_JOINTS} joints, "
              f"{self.faces.shape[0]} faces, gender={gender}")

        # Alignment transform (applied to v_template so joints match scene space).
        self._align_scale = 1.0
        self._align_transl = torch.zeros(3, device=self.device)
        self._aligned = False

    def align_to_scene(self, scale: float, transl: torch.Tensor):
        """Permanently scale + translate the SMPL model to match the scene.

        This modifies v_template (and thus joint positions) so that
        get_joint_transforms returns transforms in the scene coordinate system.

        Args:
            scale: uniform scale factor.
            transl: (3,) translation vector.
        """
        self._align_scale = scale
        self._align_transl = transl.to(self.device)
        self.v_template = self.v_template * scale + self._align_transl
        # shapedirs must also be scaled (they are offsets added to v_template).
        self.shapedirs = self.shapedirs * scale
        self._aligned = True
        print(f"  SMPL aligned: scale={scale:.3f}, transl={transl.tolist()}")

    # ── Rest pose ──────────────────────────────────────────────────────

    def get_rest_mesh(self, betas: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return rest-pose vertices and faces.

        Args:
            betas: (10,) or (B, 10) shape parameters. If None, uses zeros.

        Returns:
            vertices (6890, 3) or (B, 6890, 3), faces (13776, 3)
        """
        if betas is None:
            betas = torch.zeros(self.NUM_BETAS, device=self.device)
        # Shape blend: v = v_template + shapedirs @ betas
        if betas.ndim == 1:
            v_shaped = self.v_template + torch.einsum('vdb,b->vd', self.shapedirs, betas)
        else:
            v_shaped = self.v_template + torch.einsum('vdb,nb->nvd', self.shapedirs, betas)
        return v_shaped, self.faces

    def get_rest_joints(self, betas: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return rest-pose joint locations (24, 3)."""
        v_shaped, _ = self.get_rest_mesh(betas)
        # J = J_regressor @ v_shaped
        return self.J_regressor @ v_shaped  # (24, 3)

    def get_skinning_weights(self) -> torch.Tensor:
        """Return SMPL skinning weights (6890, 24)."""
        return self.weights

    # ── Forward kinematics ─────────────────────────────────────────────

    def get_joint_transforms(
        self,
        betas: Optional[torch.Tensor] = None,
        body_pose: Optional[torch.Tensor] = None,
        global_orient: Optional[torch.Tensor] = None,
        transl: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-joint **relative** transformation matrices for LBS.

        Returns T_k = G_k @ G_rest_k^{-1} where:
          - G_k is the posed world transform of joint k
          - G_rest_k is the rest-pose world transform of joint k

        For rest pose, T_k = I (identity), so LBS deformation is identity.
        For a posed frame, T_k encodes the rotation/translation relative
        to the rest pose, which is what LBS needs to deform the mesh/Gaussians.

        Args:
            betas: (10,) shape parameters.
            body_pose: (69,) body pose (23 joints × 3 axis-angle, excluding root).
            global_orient: (3,) global orientation (root joint).
            transl: (3,) global translation.

        Returns:
            (24, 4, 4) per-joint relative transformation matrices.
        """
        if betas is None:
            betas = torch.zeros(self.NUM_BETAS, device=self.device)
        if body_pose is None:
            body_pose = torch.zeros(69, device=self.device)
        if global_orient is None:
            global_orient = torch.zeros(3, device=self.device)
        if transl is None:
            transl = torch.zeros(3, device=self.device)

        # Compute posed and rest transforms, then return relative.
        G_posed = self._forward_kinematics(betas, body_pose, global_orient, transl)
        G_rest = self._forward_kinematics(betas, torch.zeros_like(body_pose),
                                          torch.zeros_like(global_orient),
                                          torch.zeros_like(transl))
        # T_k = G_posed @ G_rest^{-1}
        G_rest_inv = self._invert_transforms(G_rest)
        T = torch.bmm(G_posed, G_rest_inv)  # (24, 4, 4)
        return T

    def _forward_kinematics(self, betas, body_pose, global_orient, transl):
        """Compute absolute per-joint world transforms G_k (24, 4, 4)."""
        full_pose = torch.cat([global_orient, body_pose])  # (72,)
        v_shaped = self.v_template + torch.einsum('vdb,b->vd', self.shapedirs, betas)
        J = self.J_regressor @ v_shaped  # (24, 3)
        rot_mats = self._batch_rodrigues(full_pose.view(24, 3))  # (24, 3, 3)

        transforms = torch.zeros(self.NUM_JOINTS, 4, 4, device=self.device)
        transforms[:, 3, 3] = 1.0

        for j in range(self.NUM_JOINTS):
            R_j = rot_mats[j]
            if j == 0:
                transforms[j, :3, :3] = R_j
                transforms[j, :3, 3] = J[j] + transl
            else:
                parent = int(self.parents[j])
                R_parent = transforms[parent, :3, :3]
                t_parent = transforms[parent, :3, 3]
                t_local = J[j] - R_j @ J[parent]
                transforms[j, :3, :3] = R_parent @ R_j
                transforms[j, :3, 3] = R_parent @ t_local + t_parent

        return transforms

    @staticmethod
    def _invert_transforms(G):
        """Invert (24, 4, 4) rigid transformation matrices (rotation + translation)."""
        G_inv = G.clone()
        R = G[:, :3, :3]      # (24, 3, 3)
        t = G[:, :3, 3]       # (24, 3)
        # For rigid transforms: G^{-1} = [[R^T, -R^T @ t], [0, 1]]
        G_inv[:, :3, :3] = R.transpose(-2, -1)
        G_inv[:, :3, 3] = -torch.bmm(R.transpose(-2, -1), t.unsqueeze(-1)).squeeze(-1)
        return G_inv

    def get_vertex_transforms(
        self,
        joint_transforms: torch.Tensor,
        betas: Optional[torch.Tensor] = None,
        body_pose: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute posed vertices via LBS.

        Args:
            joint_transforms: (24, 4, 4) from get_joint_transforms.
            betas, body_pose: for pose blend shapes.

        Returns:
            (6890, 3) posed vertices.
        """
        if betas is None:
            betas = torch.zeros(self.NUM_BETAS, device=self.device)
        if body_pose is None:
            body_pose = torch.zeros(69, device=self.device)

        full_pose = torch.cat([torch.zeros(3, device=self.device), body_pose])  # (72,)
        v_shaped, _ = self.get_rest_mesh(betas)

        # Pose blend shapes.
        # posedirs: (6890, 3, 207), 207 = 23*9
        # For each of the 23 body joints, the 9 blend shape coeffs are:
        # [1, px, py, pz, px*px, px*py, px*pz, py*py, py*pz] (but SMPL uses a
        # specific expansion). The standard smplx lbs() does:
        #   pose_feat = (R_rest - I) flattened, where R_rest = rodrigues(pose)
        # For simplicity, we use the direct posedirs formulation:
        #   pose_blends = posedirs @ (pose_feature)
        # where pose_feature is built from the axis-angle pose.
        pose_feat = full_pose[1:]  # (69,) = 23*3, skip root
        # Build (23, 9) pose feature: for each joint (px,py,pz):
        # [px, py, pz, px*px, px*py, px*pz, py*py, py*pz, pz*pz]
        # But SMPL's posedirs expect a specific ordering. The standard smplx
        # implementation computes: pose_feature = (rot_mats - I).reshape(N, -1)
        # For our purposes, we use the simpler approximation:
        rot_mats_pose = self._batch_rodrigues(pose_feat.view(23, 3))  # (23, 3, 3)
        I = torch.eye(3, device=self.device).unsqueeze(0).expand(23, -1, -1)
        pose_feature = (rot_mats_pose - I).reshape(-1)  # (207,)
        pose_blends = torch.einsum('vdp,p->vd', self.posedirs, pose_feature)

        # Apply pose blend shapes before LBS.
        v_posed_shape = v_shaped + pose_blends  # (6890, 3)

        # LBS: blend joint transforms with skinning weights.
        T_blend = torch.einsum('vk,kij->vij', self.weights, joint_transforms)  # (6890, 4, 4)
        v_homog = torch.cat([v_posed_shape, torch.ones(6890, 1, device=self.device)], dim=-1)
        v_posed = torch.einsum('vij,vj->vi', T_blend, v_homog)  # (6890, 4)
        return v_posed[:, :3]  # (6890, 3)

    # ── Utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _batch_rodrigues(axis_angle: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle (N, 3) to rotation matrices (N, 3, 3).

        Uses the Rodrigues formula.
        """
        N = axis_angle.shape[0]
        theta = axis_angle.norm(dim=-1, keepdim=True)  # (N, 1)
        # Avoid division by zero.
        theta_safe = theta.clamp_min(1e-8)
        k = axis_angle / theta_safe  # (N, 3) rotation axis (unit)

        K = torch.zeros(N, 3, 3, device=axis_angle.device)
        K[:, 0, 1] = -k[:, 2]
        K[:, 0, 2] = k[:, 1]
        K[:, 1, 0] = k[:, 2]
        K[:, 1, 2] = -k[:, 0]
        K[:, 2, 0] = -k[:, 1]
        K[:, 2, 1] = k[:, 0]

        I = torch.eye(3, device=axis_angle.device).unsqueeze(0).expand(N, -1, -1)
        sin_t = torch.sin(theta).unsqueeze(-1)  # (N, 1, 1)
        cos_t = torch.cos(theta).unsqueeze(-1)
        R = I + sin_t * K + (1 - cos_t) * (K @ K)

        # When theta ≈ 0, R should be identity.
        zero_mask = (theta.squeeze(-1) < 1e-8)
        if zero_mask.any():
            R[zero_mask] = I[zero_mask]

        return R  # (N, 3, 3)
