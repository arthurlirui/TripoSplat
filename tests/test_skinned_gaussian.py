"""Tests for SMPL/SMPL-X skinned Gaussian animation.

Covers the pieces whose failure is silent rather than loud: LBS that is not
the identity at rest, a body model that is placed in the wrong frame, and
skinning weights that land on the wrong limb.  The animation still renders in
all of those cases, so they need assertions rather than eyeballing.

GPU tests are skipped when CUDA is unavailable; the pure-geometry tests run
anywhere.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skinned_gaussian import joints
from skinned_gaussian.pose_io import generate_demo_poses
from skinned_gaussian.skinning import lbs_deform_gaussians

SMPL_PKL = "E:/Code/mmforall/model/assets/SMPL/SMPL_NEUTRAL.pkl"
SMPLX_NPZ = "E:/Code/mmforall/model/assets/SMPLX/SMPLX_NEUTRAL_2020.npz"

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA")
needs_smpl = pytest.mark.skipif(
    not os.path.exists(SMPL_PKL), reason=f"missing {SMPL_PKL}")
needs_smplx = pytest.mark.skipif(
    not os.path.exists(SMPLX_NPZ), reason=f"missing {SMPLX_NPZ}")


# ── Joint layout ────────────────────────────────────────────────────────────

def test_body_pose_slot_matches_documented_index():
    """body_pose excludes the root, so joint j starts at 3*(j-1)."""
    assert joints.body_pose_slot(1) == 0
    assert joints.body_pose_slot(joints.L_SHOULDER) == 45  # joint 16
    assert joints.body_pose_slot(joints.R_SHOULDER) == 48  # joint 17
    assert joints.body_pose_slot(joints.R_ELBOW) == 54     # joint 19
    assert joints.body_pose_slot(joints.L_WRIST) == 57     # joint 20


def test_body_pose_slot_rejects_root_and_out_of_range():
    for bad in (0, 22, -1, 100):
        with pytest.raises(ValueError):
            joints.body_pose_slot(bad)


def test_match_body_pose_pads_and_slices():
    """Both skeletons must accept either 63- or 69-dim body poses."""
    bp63 = torch.arange(63, dtype=torch.float32).reshape(1, 63)
    padded = joints.match_body_pose(bp63, 69)
    assert padded.shape == (1, 69)
    # Original values preserved, hand slots left at rest.
    assert torch.equal(padded[..., :63], bp63)
    assert torch.all(padded[..., 63:] == 0)

    bp69 = torch.arange(2 * 69, dtype=torch.float32).reshape(2, 69)
    cut = joints.match_body_pose(bp69, 63)
    assert cut.shape == (2, 63)
    assert torch.equal(cut, bp69[..., :63])

    same = torch.zeros(3, 63)
    assert torch.equal(joints.match_body_pose(same, 63), same)
    assert joints.match_body_pose(None, 63) is None


def test_demo_poses_are_shared_layout():
    """Demo poses are authored once, at 63 dims, for either skeleton."""
    seq = generate_demo_poses(n_frames=4, device="cpu")
    assert seq["body_pose"].shape == (4, joints.BODY_POSE_DIM)
    assert seq["body_pose"].shape[1] == joints.SMPLX_BODY_POSE_DIM
    assert seq["n_frames"] == 4


# ── LBS ────────────────────────────────────────────────────────────────────

def test_lbs_is_identity_at_rest_pose():
    """Rest-pose joint transforms must leave every Gaussian untouched.

    This is the check that catches a wrong LBS convention: if T_k is absolute
    rather than relative, the whole cloud is translated to the origin and the
    render is empty, which looks like a rigging bug rather than a maths one.
    """
    torch.manual_seed(0)
    N, K = 500, 24
    means = torch.randn(N, 3)
    scales = torch.rand(N, 3) * 0.05 + 0.01
    quats = torch.nn.functional.normalize(torch.randn(N, 4), dim=-1)
    weights = torch.softmax(torch.randn(N, K), dim=-1)
    identity = torch.eye(4).expand(K, 4, 4).contiguous()

    m, s, q = lbs_deform_gaussians(means, scales, quats, weights, identity)

    assert torch.allclose(m, means, atol=1e-6)
    assert torch.allclose(s, scales, atol=1e-5)
    # q and -q are the same rotation.
    dot = (q * quats).sum(-1).abs()
    assert torch.allclose(dot, torch.ones(N), atol=1e-5)


def test_lbs_rigid_translation_follows_weights():
    """A pure translation on one joint moves exactly the Gaussians on it."""
    N, K = 10, 3
    means = torch.zeros(N, 3)
    scales = torch.ones(N, 3)
    quats = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(N, 4).contiguous()

    # All Gaussians bound rigidly to joint 1.
    weights = torch.zeros(N, K)
    weights[:, 1] = 1.0

    T = torch.eye(4).expand(K, 4, 4).clone()
    T[1, :3, 3] = torch.tensor([1.0, 2.0, 3.0])

    m, _, _ = lbs_deform_gaussians(means, scales, quats, weights, T)
    assert torch.allclose(m, torch.tensor([1.0, 2.0, 3.0]).expand(N, 3), atol=1e-6)


def test_lbs_rotation_preserves_rigid_distances():
    """LBS of a uniformly-weighted rigid body must be an isometry."""
    N = 64
    torch.manual_seed(1)
    means = torch.randn(N, 3)
    scales = torch.full((N, 3), 0.01)
    quats = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(N, 4).contiguous()

    K = 2
    weights = torch.zeros(N, K)
    weights[:, 0] = 1.0  # one joint only → no blending, must be rigid

    angle = 0.7
    c, s = math.cos(angle), math.sin(angle)
    T = torch.eye(4).expand(K, 4, 4).clone()
    T[0, :3, :3] = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    m, _, _ = lbs_deform_gaussians(means, scales, quats, weights, T)

    before = torch.cdist(means, means)
    after = torch.cdist(m, m)
    # float32 polar decomposition: relative agreement, not bit equality.
    assert torch.allclose(before, after, rtol=1e-4, atol=2e-3)


def test_lbs_scale_composes_multiplicatively():
    """A blended uniform scale multiplies the canonical scale."""
    N, K = 4, 1
    means = torch.zeros(N, 3)
    scales = torch.full((N, 3), 0.5)
    quats = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(N, 4).contiguous()
    weights = torch.ones(N, K)

    T = torch.eye(4).expand(K, 4, 4).clone()
    T[0, :3, :3] *= 2.0

    _, s, _ = lbs_deform_gaussians(means, scales, quats, weights, T)
    assert torch.allclose(s, torch.full((N, 3), 1.0), atol=1e-4)


# ── Skeleton forward kinematics ────────────────────────────────────────────

@needs_cuda
@needs_smpl
def test_smpl_rest_transforms_are_identity():
    from skinned_gaussian.smpl_model import SMPLWrapper

    smpl = SMPLWrapper(SMPL_PKL, device="cuda")
    T = smpl.get_joint_transforms()
    assert T.shape == (24, 4, 4)
    assert (T - torch.eye(4, device="cuda")).abs().max().item() < 1e-6


@needs_cuda
@needs_smpl
def test_smpl_pose_produces_nonidentity_transforms():
    from skinned_gaussian.smpl_model import SMPLWrapper

    smpl = SMPLWrapper(SMPL_PKL, device="cuda")
    bp = torch.zeros(69, device="cuda")
    bp[joints.body_pose_slot(joints.R_SHOULDER)] = 1.0
    T = smpl.get_joint_transforms(body_pose=bp)
    dev = (T - torch.eye(4, device="cuda")).abs().max().item()
    assert dev > 0.1, "posed transforms should differ substantially from rest"


@needs_cuda
@needs_smpl
def test_smpl_accepts_sixty_three_dim_pose():
    """The shared layout must not crash SMPL's 69-dim path."""
    from skinned_gaussian.smpl_model import SMPLWrapper

    smpl = SMPLWrapper(SMPL_PKL, device="cuda")
    T = smpl.get_joint_transforms(body_pose=torch.zeros(63, device="cuda"))
    assert T.shape == (24, 4, 4)


@needs_cuda
@needs_smpl
def test_smpl_vertex_transforms_run_and_move_vertices():
    """get_vertex_transforms had an off-by-one pose slice (71 vs 69 values)."""
    from skinned_gaussian.smpl_model import SMPLWrapper

    smpl = SMPLWrapper(SMPL_PKL, device="cuda")
    bp = torch.zeros(69, device="cuda")
    bp[joints.body_pose_slot(joints.R_SHOULDER) + 2] = 0.8

    T = smpl.get_joint_transforms(body_pose=bp)
    rest, _ = smpl.get_rest_mesh()
    posed = smpl.get_vertex_transforms(T, body_pose=bp)

    assert posed.shape == rest.shape
    assert (posed - rest).abs().max().item() > 1e-3, "pose had no effect"


@needs_cuda
@needs_smplx
def test_smplx_accepts_both_pose_widths():
    from skinned_gaussian.smplx_model import SMPLXWrapper

    sx = SMPLXWrapper(SMPLX_NPZ, device="cuda")
    for dim in (63, 69):
        T = sx.get_joint_transforms(body_pose=torch.zeros(dim, device="cuda"))
        assert T.shape == (55, 4, 4)


@needs_cuda
@needs_smplx
def test_smplx_rest_pose_is_identity():
    from skinned_gaussian.smplx_model import SMPLXWrapper

    sx = SMPLXWrapper(SMPLX_NPZ, device="cuda")
    T = sx.get_joint_transforms()
    assert (T - torch.eye(4, device="cuda")).abs().max().item() < 1e-6


# ── Body alignment ────────────────────────────────────────────────────────

def test_umeyama_recovers_known_similarity():
    from skinned_gaussian.alignment import umeyama

    rng = np.random.default_rng(0)
    src = rng.normal(size=(200, 3))

    angle = 0.9
    c, s = math.cos(angle), math.sin(angle)
    R_true = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    scale_true = 2.5
    t_true = np.array([1.0, -2.0, 0.5])
    dst = scale_true * (src @ R_true.T) + t_true

    scale, R, t = umeyama(src, dst)
    assert scale == pytest.approx(scale_true, rel=1e-6)
    assert np.allclose(R, R_true, atol=1e-6)
    assert np.allclose(t, t_true, atol=1e-6)


def test_umeyama_never_returns_a_reflection():
    """A mirrored target must not be matched by a reflection.

    The standard SVD solution flips sign when det(U)det(V)<0; without the fix
    the "rotation" is improper and the model comes out inside-out.
    """
    from skinned_gaussian.alignment import umeyama

    rng = np.random.default_rng(1)
    src = rng.normal(size=(100, 3))
    # Mirror across X, then add noise so the fit is not exact.
    dst = src * np.array([-1.0, 1.0, 1.0]) + rng.normal(scale=1e-3, size=src.shape)

    _, R, _ = umeyama(src, dst)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


def test_rotation_between_aligns_vectors():
    from skinned_gaussian.alignment import _rotation_between

    a = np.array([0.0, 1.0, 0.0])
    for b in ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.3, 0.5, 0.8]):
        b = np.asarray(b, dtype=float)
        b /= np.linalg.norm(b)
        R = _rotation_between(a, b)
        assert np.allclose(R @ a, b, atol=1e-8)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-8)


def test_rotation_between_handles_antiparallel():
    from skinned_gaussian.alignment import _rotation_between

    a = np.array([0.0, 1.0, 0.0])
    R = _rotation_between(a, -a)
    assert np.allclose(R @ a, -a, atol=1e-8)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


def test_estimate_up_axis_finds_long_axis():
    from skinned_gaussian.alignment import estimate_up_axis

    torch.manual_seed(0)
    # A tall cloud along Z.
    pts = torch.randn(4000, 3) * torch.tensor([0.2, 0.2, 1.0])
    up, diag = estimate_up_axis(pts)

    assert abs(abs(up[2]) - 1.0) < 0.05
    assert diag["elongation"] > 2.0


def test_estimate_up_axis_rejects_degenerate_cloud():
    from skinned_gaussian.alignment import estimate_up_axis

    with pytest.raises(ValueError):
        estimate_up_axis(torch.zeros(2, 3))


def test_estimate_up_axis_flags_isotropic_cloud():
    from skinned_gaussian.alignment import estimate_up_axis

    torch.manual_seed(0)
    pts = torch.randn(3000, 3)
    _, diag = estimate_up_axis(pts)
    assert "warning" in diag


def test_trimmed_chamfer_is_zero_for_identical_clouds():
    from skinned_gaussian.alignment import trimmed_chamfer

    rng = np.random.default_rng(0)
    pts = rng.normal(size=(200, 3))
    assert trimmed_chamfer(pts, pts) == pytest.approx(0.0, abs=1e-9)


def test_trimmed_chamfer_ignores_outliers():
    """Trimming is what makes the score robust to hair and loose clothing."""
    from skinned_gaussian.alignment import trimmed_chamfer

    rng = np.random.default_rng(0)
    model = rng.normal(size=(500, 3))
    scene = model.copy()
    # Corrupt 10% badly; the trimmed score must stay near zero.
    n_out = 50
    scene[:n_out] = rng.normal(scale=100.0, size=(n_out, 3))
    assert trimmed_chamfer(model, scene, trim_frac=0.8) < 0.5


def test_fit_recovers_known_transform_on_synthetic_cloud():
    """End-to-end synthetic check: place a body, then fit it back."""
    from skinned_gaussian.alignment import (
        _posed_vertices, fit_body_to_gaussians)
    from skinned_gaussian.smpl_model import SMPLWrapper

    if not os.path.exists(SMPL_PKL):
        pytest.skip(f"missing {SMPL_PKL}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    smpl = SMPLWrapper(SMPL_PKL, device=device)

    # True placement: scale 1.4, rotate 37° about Y, translate.
    true_R = np.array([
        [math.cos(0.65), 0.0, math.sin(0.65)],
        [0.0, 1.0, 0.0],
        [-math.sin(0.65), 0.0, math.cos(0.65)],
    ])
    true_scale = 1.4
    true_t = np.array([0.5, -0.2, 1.0])

    verts = _posed_vertices(smpl, None, joints.a_pose_body_pose(device))
    verts = verts[::4]  # subsample for speed
    scene = true_scale * (verts @ true_R.T) + true_t
    # Thin shell of points around the surface, as a Gaussian cloud would be.
    means = torch.from_numpy(scene).float() + torch.randn(len(scene), 3) * 0.01

    fit = fit_body_to_gaussians(
        means, smpl, n_sample_scene=8000, n_sample_model=1500, verbose=False)

    # Note: the fit searches an A-pose, so the scale should match closely; the
    # rotation may differ by a yaw multiple only if the body is symmetric.
    assert fit.scale == pytest.approx(true_scale, rel=0.15)
    assert fit.chamfer_model_to_scene < 0.1 * true_scale * 1.7

    # The recovered transform must actually reproduce the cloud.
    mapped = fit.apply(torch.from_numpy(verts).float())
    err = (mapped - means).norm(dim=-1).median().item()
    assert err < 0.15


# ── SkinnedGaussians container ────────────────────────────────────────────

def _make_skinned(n=32, k=24, device="cpu", **kw):
    from skinned_gaussian.skinned_gaussians import SkinnedGaussians

    torch.manual_seed(0)
    # Broadcast a single bone across each Gaussian when n > k, so every
    # Gaussian has a valid (n, k) weight row either way.
    weights = torch.zeros(n, k)
    for i in range(n):
        weights[i, i % k] = 1.0
    return SkinnedGaussians.from_attributes(
        means=torch.randn(n, 3),
        scales_raw=torch.zeros(n, 3),
        quats=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4).clone(),
        opacities_raw=torch.zeros(n),
        sh0=torch.zeros(n, 1, 3),
        skin_weights=weights,
        n_joints=k,
        device=device,
        **kw,
    )


def test_skin_weights_are_normalized():
    g = _make_skinned()
    sums = g.skin_weights.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_training_setup_weights_creates_optimizer():
    """Previously raised AttributeError: no self.optimizers."""
    g = _make_skinned(weights_optimizable=True)
    opt = g.training_setup_weights()
    assert opt is not None
    assert "skin_weights" in g.optimizers


def test_training_setup_weights_is_noop_when_not_optimizable():
    g = _make_skinned(weights_optimizable=False)
    assert g.training_setup_weights() is None


def test_weight_regularization_is_zero_at_init():
    g = _make_skinned(weights_optimizable=True)
    assert g.weight_regularization_loss().item() == pytest.approx(0.0, abs=1e-9)


def test_weight_regularization_is_zero_when_frozen():
    g = _make_skinned(weights_optimizable=False)
    assert g.weight_regularization_loss().item() == pytest.approx(0.0)


def test_deform_is_identity_at_rest():
    g = _make_skinned(k=3)
    T = torch.eye(4).expand(3, 4, 4).contiguous()
    m, s, _ = g.deform(T)
    assert torch.allclose(m, g.means, atol=1e-6)


def test_expression_basis_displaces_means():
    from skinned_gaussian.skinned_gaussians import SkinnedGaussians

    n, k, n_expr = 16, 3, 4
    basis = torch.zeros(n, n_expr, 3)
    basis[:, 0, 0] = 1.0  # first basis moves +X
    weights = torch.zeros(n, k)
    for i in range(n):
        weights[i, i % k] = 1.0
    g = SkinnedGaussians.from_attributes(
        means=torch.zeros(n, 3), scales_raw=torch.zeros(n, 3),
        quats=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n, 4).clone(),
        opacities_raw=torch.zeros(n), sh0=torch.zeros(n, 1, 3),
        skin_weights=weights, n_joints=k, device="cpu",
        expr_basis=basis)

    T = torch.eye(4).expand(k, 4, 4).contiguous()
    expr = torch.zeros(n_expr)
    expr[0] = 2.0

    m, _, _ = g.deform(T, expression=expr)
    assert torch.allclose(m[:, 0], torch.full((n,), 2.0), atol=1e-5)
    assert torch.allclose(m[:, 1:], torch.zeros(n, 2), atol=1e-5)


def test_ply_roundtrip_preserves_rig(tmp_path):
    """A saved rig must reload with the same weights and expression bases."""
    from skinned_gaussian.skinned_gaussians import SkinnedGaussians

    n, k, n_expr = 24, 5, 3
    torch.manual_seed(0)
    basis = torch.randn(n, n_expr, 3)
    g = SkinnedGaussians.from_attributes(
        means=torch.randn(n, 3), scales_raw=torch.randn(n, 3) * 0.1,
        quats=torch.nn.functional.normalize(torch.randn(n, 4), dim=-1),
        opacities_raw=torch.randn(n) * 0.1, sh0=torch.randn(n, 1, 3),
        skin_weights=torch.softmax(torch.randn(n, k), dim=-1),
        n_joints=k, device="cpu", expr_basis=basis)

    path = str(tmp_path / "rig.ply")
    g.save_ply(path)
    loaded = SkinnedGaussians.load_ply(path, device="cpu")

    assert loaded.n_gaussians == n
    assert loaded.n_joints == k
    assert torch.allclose(loaded.means, g.means, atol=1e-5)
    assert torch.allclose(loaded.skin_weights, g.skin_weights, atol=1e-4)
    assert loaded._expr_basis is not None
    assert loaded._expr_basis.shape == (n, n_expr, 3)
    assert torch.allclose(loaded._expr_basis, basis, atol=1e-5)


def test_ply_roundtrip_without_expression_basis(tmp_path):
    from skinned_gaussian.skinned_gaussians import SkinnedGaussians

    g = _make_skinned(n=12, k=4)
    path = str(tmp_path / "rig.ply")
    g.save_ply(path)
    loaded = SkinnedGaussians.load_ply(path, device="cpu")
    assert loaded._expr_basis is None
    assert loaded.n_joints == 4


def test_load_ply_rejects_unskinned_file(tmp_path):
    from plyfile import PlyData, PlyElement
    from skinned_gaussian.skinned_gaussians import SkinnedGaussians

    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.zeros(3, dtype=[(nm, "f4") for nm in names])
    PlyData([PlyElement.describe(data, "vertex")]).write(str(tmp_path / "plain.ply"))

    with pytest.raises(ValueError, match="bone_weight"):
        SkinnedGaussians.load_ply(str(tmp_path / "plain.ply"), device="cpu")


# ── Rigging ────────────────────────────────────────────────────────────────

@needs_cuda
@needs_smpl
def test_rigging_assigns_weights_to_arm_joints():
    """Arms must not collapse onto torso bones.

    With the T-pose template, a subject standing arms-down has no Gaussian
    within reach of an elbow or wrist vertex, so every arm Gaussian inherits
    torso weights and the arms cannot be animated.  Measured before the fix:
    0 Gaussians on each of joints 18-21.
    """
    from skinned_gaussian.rigging import rig_gaussians_to_smpl
    from skinned_gaussian.smpl_model import SMPLWrapper

    smpl = SMPLWrapper(SMPL_PKL, device="cuda")
    # Put the model in scene-ish scale so distances are realistic.
    smpl.align_to_scene(scale=1.0, transl=torch.zeros(3, device="cuda"))

    verts, _ = smpl.get_rest_mesh()
    pts = verts[::6].float() + torch.randn(len(verts[::6]), 3, device="cuda") * 0.005
    n = pts.shape[0]

    quats = torch.zeros(n, 4, device="cuda")
    quats[:, 0] = 1.0  # identity rotation

    skinned = rig_gaussians_to_smpl(
        pts, torch.zeros(n, 3, device="cuda"), quats,
        torch.zeros(n, device="cuda"), torch.zeros(n, 1, 3, device="cuda"),
        smpl, n_neighbours=4, temperature=0.05, device="cuda")

    dom = skinned.skin_weights.argmax(-1)
    arm_joints = (18, 19, 20, 21)
    arm_count = sum(int((dom == j).sum()) for j in arm_joints)
    assert arm_count > 0, "no Gaussian is dominated by an elbow or wrist joint"


@needs_cuda
@needs_smpl
def test_estimate_rig_pose_prefers_a_non_tpose_swing():
    """The sweep must find a swing whose arms reach the cloud."""
    from skinned_gaussian.rigging import estimate_rig_pose
    from skinned_gaussian.smpl_model import SMPLWrapper

    smpl = SMPLWrapper(SMPL_PKL, device="cuda")
    smpl.align_to_scene(scale=1.0, transl=torch.zeros(3, device="cuda"))

    # Build a cloud from the arms-down pose, then check the sweep finds it.
    from skinned_gaussian.rigging import _posed_vertices

    bp_true = joints.a_pose_body_pose("cuda")
    verts = _posed_vertices(smpl, None, bp_true)
    pts = verts[::5]

    bp, diag = estimate_rig_pose(smpl, pts)
    assert abs(diag["swing_rad"]) > 0.5, "sweep chose a near-T-pose"
    assert diag["arm_median_distance"] < 0.1
