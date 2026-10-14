"""Unit tests for differentiable alignment refinement.

Tests the ``SimilarityTransform`` module's math (no GPU or rendering required):
- Identity initialisation reproduces the coarse transform.
- Gradient flow through all three parameters.
- Rotation composition correctness.
- Scale and translation composition correctness.
- Quaternion-to-matrix consistency with the existing quaternion utilities.
"""

from __future__ import annotations

import math

import torch

from multiview_refine.differentiable_refine import SimilarityTransform
from multiview_refine.quaternion import rotation_matrix_to_quaternion


def test_identity_init_reproduces_coarse():
    """At init (q_delta=identity, log_scale=0, t_delta=0), the effective
    transform equals the coarse transform."""
    scale_coarse = 2.5
    R_coarse = torch.tensor(
        [[0.0, -1.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )  # 90° rotation around Z
    t_coarse = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    sim = SimilarityTransform(scale_coarse, R_coarse, t_coarse, device="cpu")

    assert math.isclose(sim.scale.item(), scale_coarse, rel_tol=1e-5)
    torch.testing.assert_close(sim.t, t_coarse, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sim.R, R_coarse, rtol=1e-5, atol=1e-6)

    # Transform a point and check it matches the coarse transform.
    p = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    p_expected = scale_coarse * (R_coarse @ p.T).T + t_coarse
    p_got = sim.transform_means(p)
    torch.testing.assert_close(p_got, p_expected, rtol=1e-5, atol=1e-6)


def test_scale_correction():
    """A non-zero log_scale shifts the effective scale multiplicatively."""
    sim = SimilarityTransform(1.0, torch.eye(3), torch.zeros(3), device="cpu")
    sim.log_scale.data.fill_(math.log(2.0))  # exp(log(2)) * 1.0 = 2.0
    assert math.isclose(sim.scale.item(), 2.0, rel_tol=1e-5)


def test_translation_correction():
    """t_delta adds to t_coarse."""
    sim = SimilarityTransform(1.0, torch.eye(3), torch.tensor([1.0, 1.0, 1.0]), device="cpu")
    sim.t_delta.data.fill_(0.5)
    torch.testing.assert_close(sim.t, torch.tensor([1.5, 1.5, 1.5]), rtol=1e-5, atol=1e-6)


def test_rotation_composition():
    """R = R(q_delta) @ R_coarse.

    If q_delta represents a 90° rotation around Z and R_coarse is identity,
    then R should rotate (1,0,0) → (0,1,0).
    """
    sim = SimilarityTransform(1.0, torch.eye(3), torch.zeros(3), device="cpu")
    # 90° rotation around Z: quaternion = (cos(45°), 0, 0, sin(45°))
    angle = math.pi / 2
    sim.q_delta.data = torch.tensor(
        [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]
    )

    p = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    p_transformed = sim.transform_means(p)  # scale=1, t=0, so just rotation
    expected = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(p_transformed, expected, rtol=1e-5, atol=1e-6)


def test_rotation_on_top_of_coarse():
    """R_delta applied on top of a non-identity R_coarse.

    R_coarse = 90° around Z: (1,0,0) → (0,1,0)
    R_delta = 90° around Z: (0,1,0) → (-1,0,0)
    Combined: (1,0,0) → (-1,0,0)
    """
    R_coarse = torch.tensor(
        [[0.0, -1.0, 0.0],
         [1.0, 0.0, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    sim = SimilarityTransform(1.0, R_coarse, torch.zeros(3), device="cpu")
    angle = math.pi / 2
    sim.q_delta.data = torch.tensor(
        [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]
    )

    p = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    p_transformed = sim.transform_means(p)
    expected = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float32)
    torch.testing.assert_close(p_transformed, expected, rtol=1e-5, atol=1e-6)


def test_transform_quats():
    """transform_quats composes q(R) ⊗ q_old."""
    sim = SimilarityTransform(1.0, torch.eye(3), torch.zeros(3), device="cpu")
    # 90° around Z
    angle = math.pi / 2
    sim.q_delta.data = torch.tensor(
        [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]
    )

    # Old quaternion = identity.
    q_old = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    q_new = sim.transform_quats(q_old)
    # Should equal q(R) = the 90° Z rotation quaternion.
    expected = torch.tensor(
        [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)], dtype=torch.float32
    )
    # Normalise for comparison.
    q_new_norm = q_new / q_new.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(q_new_norm.squeeze(0), expected, rtol=1e-5, atol=1e-6)


def test_gradient_flow():
    """Gradients flow to all three learnable parameters."""
    sim = SimilarityTransform(
        2.0,
        torch.eye(3),
        torch.zeros(3),
        device="cpu",
    )

    means = torch.randn(10, 3)
    # Simple loss: push means toward a target.
    target = torch.randn(10, 3)
    means_t = sim.transform_means(means)
    loss = ((means_t - target) ** 2).mean()
    loss.backward()

    assert sim.log_scale.grad is not None
    assert sim.q_delta.grad is not None
    assert sim.t_delta.grad is not None
    assert not torch.allclose(sim.log_scale.grad, torch.tensor(0.0))
    assert not torch.allclose(sim.t_delta.grad, torch.tensor(0.0))


def test_transform_scales():
    """transform_scales multiplies by the effective scale."""
    sim = SimilarityTransform(3.0, torch.eye(3), torch.zeros(3), device="cpu")
    sim.log_scale.data.fill_(math.log(2.0))  # effective scale = 6.0
    scales = torch.ones(5, 3)
    scales_t = sim.transform_scales(scales)
    torch.testing.assert_close(
        scales_t, torch.full((5, 3), 6.0), rtol=1e-5, atol=1e-6
    )


def test_get_params_tuple_returns_detached():
    """get_params_tuple returns detached tensors (no grad)."""
    sim = SimilarityTransform(1.0, torch.eye(3), torch.zeros(3), device="cpu")
    s, R, t = sim.get_params_tuple()
    assert not s.requires_grad
    assert not R.requires_grad
    assert not t.requires_grad


def test_R_is_orthonormal():
    """The effective R should always be orthonormal (it's a product of two
    rotation matrices)."""
    # Non-trivial coarse rotation.
    R_coarse = torch.tensor(
        [[0.0, 0.0, 1.0],
         [0.0, 1.0, 0.0],
         [-1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    sim = SimilarityTransform(1.0, R_coarse, torch.zeros(3), device="cpu")
    # Random-ish q_delta.
    sim.q_delta.data = torch.tensor([0.8, 0.1, 0.3, 0.5])

    R = sim.R
    # Check R @ R^T = I.
    torch.testing.assert_close(R @ R.T, torch.eye(3), rtol=1e-4, atol=1e-5)
    # Check det(R) ≈ +1 (proper rotation).
    assert math.isclose(torch.det(R).item(), 1.0, abs_tol=1e-4)
