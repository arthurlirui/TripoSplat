"""Tests for the uprightness gate (guards against MODEL2SCENE flip regressions)."""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiview_refine.orientation import (
    silhouette_orientation_stats,
    is_upright,
    uprightness_report,
    assert_upright,
)


def make_standing_mask(h=200, w=120):
    """A crude standing figure: narrow head band, wide shoulders, split legs."""
    m = np.zeros((h, w), bool)
    cx = w // 2
    # Head (narrow, single run).
    m[10:45, cx - 14:cx + 14] = True
    # Shoulders/torso (wide).
    m[45:70, cx - 45:cx + 45] = True
    m[70:130, cx - 38:cx + 38] = True
    # Legs: two separate runs near the bottom.
    m[130:190, cx - 34:cx - 6] = True
    m[130:190, cx + 6:cx + 34] = True
    return m


def test_stats_detect_upright_figure():
    st = silhouette_orientation_stats(make_standing_mask())
    assert st["valid"]
    assert st["band_ratio"] < 1.0, st
    assert st["top_runs"] == 1


def test_stats_detect_inverted_figure():
    st = silhouette_orientation_stats(make_standing_mask()[::-1])
    assert st["valid"]
    assert st["band_ratio"] > 1.0, st
    assert st["top_runs"] >= 2


def test_is_upright():
    mask = make_standing_mask()
    assert is_upright(mask)
    assert not is_upright(mask[::-1])


def test_empty_mask_is_not_judged():
    st = silhouette_orientation_stats(np.zeros((10, 10), bool))
    assert not st["valid"]
    assert is_upright(np.zeros((10, 10), bool))  # nothing to judge


def test_uprightness_report_flags_flipped_majority():
    good = [make_standing_mask() for _ in range(4)]
    bad = [make_standing_mask()[::-1] for _ in range(4)]

    ok = uprightness_report(good)
    assert ok["frac_upright"] == 1.0
    assert not ok["flip_suspected"]

    flipped = uprightness_report(bad)
    assert flipped["frac_upright"] == 0.0
    assert flipped["flip_suspected"]


def test_assert_upright_raises_on_flip():
    bad = [make_standing_mask()[::-1] for _ in range(4)]
    with pytest.raises(AssertionError, match="Uprightness check failed"):
        assert_upright(bad)
    # Upright input must not raise.
    assert_upright([make_standing_mask() for _ in range(4)])


def test_assert_upright_reports_model2scene_hint():
    """The failure message must point at MODEL2SCENE so the cause is obvious."""
    bad = [make_standing_mask()[::-1] for _ in range(2)]
    with pytest.raises(AssertionError) as exc:
        assert_upright(bad)
    assert "MODEL2SCENE" in str(exc.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
