"""Upright sanity check for silhouettes.

Silhouette **IoU alone cannot detect a 180° flip** of a standing figure: an
upright person mask and its vertical mirror still overlap substantially
(measured empirically on this project's captures: mean self-IoU 0.53 for the
vertical flip).  That is exactly how the ``MODEL2SCENE`` transposition bug
survived a green IoU of 0.55 while every render was upside down.

This module adds a cheap, orientation-aware gate.  A standing person mask has
three robust properties:

* the top band (head/shoulders) is **narrower** than the bottom band
  (legs/feet) — ``band_ratio < 1``;
* the head is a **single connected run** across the top band, whereas the
  bottom band may split into two legs;
* the head touches only a small fraction of the mask's maximal horizontal
  extent.

Calibration on ``static/000001/mvs/images`` (full-body captures):

    upright : band_ratio = 0.32 … 0.64   (top_runs = 1)
    flipped : band_ratio = 1.57 … 3.13   (top_runs = 2 … 3)

so a threshold of 1.0 separates the two classes with a wide margin.
"""

from __future__ import annotations

import math

import numpy as np


def _band_extent(mask_band: np.ndarray) -> int:
    """Horizontal extent (px) of the foreground inside a horizontal band."""
    cols = np.where(mask_band.any(axis=0))[0]
    return int(cols.max() - cols.min() + 1) if len(cols) else 0


def _num_runs(row: np.ndarray) -> int:
    """Number of contiguous foreground runs along a 1-D binary row."""
    padded = np.concatenate([[0], row.astype(np.int8), [0]])
    return int(np.sum(np.diff(padded) == 1))


def silhouette_orientation_stats(mask: np.ndarray) -> dict:
    """Describe the vertical structure of a binary foreground mask.

    Args:
        mask: (H, W) array-like; non-zero is foreground.

    Returns:
        dict with ``band_ratio`` (top band extent / bottom band extent),
        ``top_runs``, ``bot_runs``, and the raw extents.  ``band_ratio`` is
        ``inf`` when the bottom band is empty and the top is not.
    """
    m = np.asarray(mask) > 0
    rows = np.where(m.any(axis=1))[0]
    if len(rows) == 0:
        return dict(band_ratio=float("nan"), top_runs=0, bot_runs=0,
                    top_extent=0, bot_extent=0, valid=False)

    y0, y1 = rows.min(), rows.max()
    h = y1 - y0 + 1
    band_h = max(1, h // 8)

    top = m[y0:y0 + band_h]
    bot = m[y1 - band_h + 1:y1 + 1]

    top_ext = _band_extent(top)
    bot_ext = _band_extent(bot)
    if bot_ext == 0:
        ratio = float("inf") if top_ext else float("nan")
    else:
        ratio = top_ext / bot_ext

    # A shape with equal top and bottom widths (a cylinder, a featureless
    # blob, or a subject viewed along its axis) carries no vertical asymmetry,
    # so it says nothing about orientation.  Such a mask must not be counted
    # as "inverted" — that turns a non-verdict into a false alarm.
    ambiguous = bool(math.isfinite(ratio) and abs(ratio - 1.0) < 0.10)

    return dict(
        band_ratio=float(ratio),
        top_runs=_num_runs(top[top.shape[0] // 2]),
        bot_runs=_num_runs(bot[bot.shape[0] // 2]),
        top_extent=top_ext,
        bot_extent=bot_ext,
        ambiguous=ambiguous,
        valid=True,
    )


def is_upright(mask: np.ndarray, max_band_ratio: float = 1.0) -> bool:
    """Heuristic: is this standing-figure silhouette the right way up?

    A mask whose top band is wider than its bottom band is almost certainly
    inverted (head-down).  Empty masks and masks with no vertical asymmetry
    return ``True`` — nothing to judge, so no accusation.
    """
    st = silhouette_orientation_stats(mask)
    if not st["valid"] or st.get("ambiguous", False):
        return True
    return st["band_ratio"] < max_band_ratio


def uprightness_report(pred_masks: list[np.ndarray],
                       real_masks: list[np.ndarray] | None = None,
                       max_band_ratio: float = 1.0) -> dict:
    """Aggregate an uprightness verdict over a set of rendered silhouettes.

    Args:
        pred_masks: rendered binary silhouettes (one per view).
        real_masks: optional GT silhouettes for a reference comparison.
        max_band_ratio: threshold passed to :func:`is_upright`.

    Returns:
        dict with the fraction of upright renders, per-view stats, and a
        ``flip_suspected`` flag that fires when the majority of views look
        inverted.
    """
    pred_stats = [silhouette_orientation_stats(m) for m in pred_masks]
    judged = [s for s in pred_stats if s["valid"] and not s.get("ambiguous", False)]
    n_up = sum(1 for s in judged if s["band_ratio"] < max_band_ratio)
    frac_up = (n_up / len(judged)) if judged else float("nan")

    report = dict(
        n_views=len(pred_masks),
        n_judged=len(judged),
        n_ambiguous=sum(1 for s in pred_stats if s.get("ambiguous", False)),
        frac_upright=float(frac_up),
        flip_suspected=bool(len(judged) > 0 and frac_up < 0.5),
        per_view_band_ratio=[s["band_ratio"] for s in pred_stats],
    )

    if real_masks is not None:
        ref = [silhouette_orientation_stats(m) for m in real_masks]
        ref_judged = [s for s in ref if s["valid"] and not s.get("ambiguous", False)]
        report["real_frac_upright"] = float(
            np.mean([s["band_ratio"] < max_band_ratio for s in ref_judged])
        ) if ref_judged else float("nan")

    return report


def assert_upright(pred_masks: list[np.ndarray],
                   real_masks: list[np.ndarray] | None = None,
                   max_band_ratio: float = 1.0,
                   min_frac: float = 0.5) -> None:
    """Raise ``AssertionError`` when the rendered silhouettes look inverted.

    Call this before reporting alignment metrics so a flipped model cannot
    silently produce a good-looking IoU score.
    """
    rep = uprightness_report(pred_masks, real_masks, max_band_ratio)
    if rep["n_judged"] == 0:
        return
    if rep["frac_upright"] < min_frac:
        raise AssertionError(
            "Uprightness check failed: only "
            f"{rep['frac_upright']:.0%} of {rep['n_judged']} rendered silhouettes "
            f"are the right way up (threshold {min_frac:.0%}). "
            f"Per-view top/bottom band ratio: "
            f"{[round(r, 2) for r in rep['per_view_band_ratio']]}. "
            "A ratio > 1 means the top band is wider than the bottom one, "
            "which indicates a vertically flipped model — check MODEL2SCENE."
        )
