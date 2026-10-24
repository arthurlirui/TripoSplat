"""Tests for the persistent transform format (multiview_refine.transform_io).

The point of this module is that a stored transform cannot silently be wrong:
loading validates orthonormality, handedness, and the upright invariant.
Several tests below feed it the exact matrices that caused real bugs.
"""

import json
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiview_refine.coord_registry import MODEL2SCENE
from multiview_refine.transform_io import (
    GaussianTransform,
    save_transform,
    load_transform,
    convert_ply,
    MODEL_UP,
    SCENE_UP,
)

I3 = np.eye(3)


# ── round trip ──────────────────────────────────────────────────────────────

def test_roundtrip_preserves_values(tmp_path):
    tfm = GaussianTransform.from_align(1.77, MODEL2SCENE, [0.02, 0.13, -0.05],
                                       name="t", init_view="00.png")
    p = save_transform(str(tmp_path / "t.json"), tfm)
    back = load_transform(p)

    assert back.scale == pytest.approx(tfm.scale, rel=1e-9)
    np.testing.assert_allclose(back.R, tfm.R, atol=1e-9)
    np.testing.assert_allclose(back.t, tfm.t, atol=1e-9)
    assert back.init_view == "00.png"
    assert back.source_frame == tfm.source_frame
    assert back.target_frame == tfm.target_frame


def test_saved_file_is_self_describing(tmp_path):
    tfm = GaussianTransform.from_align(1.5, MODEL2SCENE, [0, 0, 0])
    p = save_transform(str(tmp_path / "t.json"), tfm)
    d = json.load(open(p))
    for key in ("schema", "version", "source_frame", "target_frame",
                "scale", "R", "t", "convention", "up"):
        assert key in d, f"missing {key}"
    assert d["convention"] == "xyz = scale * (R @ xyz_model) + t"


# ── validation rejects the real bugs ───────────────────────────────────────

def test_rejects_the_transposed_model2scene_bug():
    """The exact matrix that left every render 180° upside down."""
    buggy = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    tfm = GaussianTransform.from_align(1.77, buggy, [0, 0, 0])
    with pytest.raises(ValueError, match="upside down|tipped over|expected scene up"):
        tfm.validate()


def test_rejects_reflection():
    """det = -1 mirrors the model; must not pass as a rotation."""
    R = np.diag([1.0, -1.0, 1.0]) @ MODEL2SCENE.numpy()
    tfm = GaussianTransform.from_align(1.0, R, [0, 0, 0])
    with pytest.raises(ValueError, match="reflection"):
        tfm.validate()


def test_rejects_non_orthonormal_matrix():
    """A matrix that also scales would double-apply the scale factor."""
    R = np.array(MODEL2SCENE.numpy()) * 2.0
    tfm = GaussianTransform.from_align(1.0, R, [0, 0, 0])
    with pytest.raises(ValueError, match="not orthonormal"):
        tfm.validate()


def test_rejects_nonpositive_scale():
    tfm = GaussianTransform.from_align(0.0, MODEL2SCENE, [0, 0, 0])
    with pytest.raises(ValueError, match="scale must be"):
        tfm.validate()


def test_rejects_nan():
    tfm = GaussianTransform.from_align(1.0, MODEL2SCENE, [float("nan"), 0, 0])
    with pytest.raises(ValueError, match="non-finite"):
        tfm.validate()


def test_upright_invariant_holds_for_the_real_matrix():
    up_scene = np.asarray(MODEL2SCENE.numpy(), dtype=np.float64) @ np.array(MODEL_UP)
    np.testing.assert_allclose(up_scene, np.array(SCENE_UP), atol=1e-6)
    GaussianTransform.from_align(1.76, MODEL2SCENE, [0, 0, 0]).validate()  # no raise


# ── legacy compatibility ───────────────────────────────────────────────────

@pytest.mark.parametrize("rot_key,tr_key", [("R", "t"), ("rotation", "translation")])
def test_load_accepts_legacy_key_names(tmp_path, rot_key, tr_key):
    """Older scripts wrote rotation/translation, newer ones R/t."""
    d = {
        "scale": 1.77,
        rot_key: MODEL2SCENE.numpy().tolist(),
        tr_key: [0.02, 0.13, -0.05],
        "init_view": "00.png",
    }
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps(d))
    tfm = load_transform(str(p))
    assert tfm.scale == pytest.approx(1.77)
    np.testing.assert_allclose(tfm.t, [0.02, 0.13, -0.05], atol=1e-9)


def test_load_rejects_missing_fields(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"scale": 1.0}))
    with pytest.raises(ValueError, match="missing rotation"):
        load_transform(str(p))


def test_load_rejects_corrupt_transform_on_disk(tmp_path):
    """A flipped transform written to disk must fail at LOAD, not at render."""
    p = tmp_path / "flipped.json"
    p.write_text(json.dumps({
        "scale": 1.0,
        "R": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        "t": [0, 0, 0],
    }))
    with pytest.raises(ValueError):
        load_transform(str(p))


# ── math: apply / inverse / compose ────────────────────────────────────────

def test_apply_matches_manual_formula():
    s, t = 1.8, np.array([0.1, 0.2, 0.3])
    R = MODEL2SCENE.numpy().astype(np.float64)
    tfm = GaussianTransform.from_align(s, R, t)

    means = torch.tensor([[0.0, 0.0, 0.5], [0.1, -0.2, 0.3]])
    out, _, _ = tfm.apply(means)
    expect = s * (means.numpy() @ R.T) + t
    np.testing.assert_allclose(out.numpy(), expect, atol=1e-6)


def test_apply_scales_and_rotates_quats():
    tfm = GaussianTransform.from_align(2.0, MODEL2SCENE, [0, 0, 0])
    means = torch.zeros(1, 3)
    scales = torch.ones(1, 3)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    m, s, q = tfm.apply(means, scales, quats)
    np.testing.assert_allclose(s.numpy(), [[2.0, 2.0, 2.0]], atol=1e-6)
    assert q is not None and q.shape == (1, 4)
    # identity Gaussian rotated by MODEL2SCENE
    from multiview_refine.quaternion import quat_to_matrix
    Rq = quat_to_matrix(q.numpy())[0]
    np.testing.assert_allclose(Rq, MODEL2SCENE.numpy(), atol=1e-5)


def test_apply_leaves_optional_args_alone():
    tfm = GaussianTransform.from_align(1.0, MODEL2SCENE, [0, 0, 0])
    m, s, q = tfm.apply(torch.zeros(2, 3))
    assert s is None and q is None


def test_inverse_roundtrip():
    tfm = GaussianTransform.from_align(1.8, MODEL2SCENE, [0.1, 0.2, 0.3])
    means = torch.randn(10, 3)
    fwd, _, _ = tfm.apply(means)
    back, _, _ = tfm.inverse().apply(fwd)
    np.testing.assert_allclose(back.numpy(), means.numpy(), atol=1e-5)


def test_compose_matches_sequential_application():
    a = GaussianTransform.from_align(1.5, MODEL2SCENE, [0.1, 0.0, 0.0])
    b = GaussianTransform.from_align(1.2, np.eye(3), [0.0, 0.3, 0.0])
    means = torch.randn(5, 3)

    # a first, then b  ==  b.compose(a)
    once, _, _ = a.apply(means)
    twice, _, _ = b.apply(once)

    composed, _, _ = b.compose(a).apply(means)
    np.testing.assert_allclose(composed.numpy(), twice.numpy(), atol=1e-5)


def test_compose_preserves_frames():
    a = GaussianTransform.from_align(1.0, MODEL2SCENE, [0, 0, 0])
    b = GaussianTransform.from_align(1.0, np.eye(3), [0, 0, 0])
    c = a.compose(b)
    assert c.source_frame == b.source_frame
    assert c.target_frame == a.target_frame


# ── subject height sanity ──────────────────────────────────────────────────

def test_metrics_survive_and_summarise(tmp_path):
    tfm = GaussianTransform.from_align(
        1.77, MODEL2SCENE, [0, 0, 0],
        metrics={"subject_height_m": 1.76, "iou": 0.735, "upright": 0.80})
    back = load_transform(save_transform(str(tmp_path / "m.json"), tfm))
    assert back.metrics["subject_height_m"] == pytest.approx(1.76)
    s = back.summary()
    assert "1.76m" in s and "0.735" in s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
