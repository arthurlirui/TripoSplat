# ContinualGS — Continual-Learning Drivable 3D Gaussian Avatars

Reference implementation for the ICME 2027 submission *ContinualGS:
Continual-Learning Drivable 3D Gaussian Avatars from Monocular Video
Streams*. See `latex/continual_avatar/` for the paper.

## Idea in one paragraph

A drivable 3D-Gaussian-Splatting (3DGS) avatar is **warm-started** from a
single monocular frame using a feed-forward Gaussian generator
(TripoSplat) + SMPL registration + skeleton rigging, giving a usable
avatar within seconds. The remaining monocular video is then ingested
**clip by clip as a continual-learning problem**: each clip is optimized
against the avatar while two complementary mechanisms mitigate
catastrophic forgetting of previously observed viewpoints — an **EWC
prior** (per-Gaussian Fisher-information importance that penalizes changes
to well-constrained parameters) and a **viewpoint-stratified replay
buffer** (episodic re-rendering of representative historical frames). A
**pose-conditioned selective reactivation** policy steers densification
toward under-observed body regions, improving sample efficiency on
streaming data.

## Package layout

```
continual_avatar/
├── __init__.py        # public API
├── frame.py           # MonocularFrame + SMPL regressor protocol
├── warm_start.py      # single-frame → drivable avatar (Sec. 3.2)
├── ewc.py             # Fisher importance + EWC penalty (Sec. 3.3.2)
├── replay.py          # viewpoint-stratified replay buffer (Sec. 3.3.3)
├── reactivation.py    # per-Gaussian visibility + densification mod. (Sec. 3.3.4)
└── continual.py       # ContinualRefiner: the online loop (Algorithm 1)
run_continual_avatar.py  # end-to-end runner
tests/test_continual_avatar.py  # smoke tests (no gsplat/CUDA needed)
```

## Quick start

```bash
conda activate rtg132   # torch 2.12+cu132, gsplat 1.5.3
python run_continual_avatar.py \
    --video_frames static/000001/mvs/images \
    --transforms static/000001/mvs/transforms_test.json \
    --init_frame 00.png \
    --smpl_model_path E:/Code/mmforall/model/assets/SMPL/SMPL_NEUTRAL.pkl \
    --clip_size 16 --steps_per_clip 500 \
    --output_dir continual_output
```

> The runner ships with a **placeholder SMPL regressor** — to run
> end-to-end, plug a concrete backend (e.g. SMPLer-X / YOLO11-pose) into
> `build_smpl_regressor()` in `run_continual_avatar.py`. The
> feed-forward generator uses `TripoSplatPipeline` from `triposplat.py`.

## Smoke tests

Logic tests for EWC, the replay buffer, selective reactivation, and
refiner construction run **without gsplat/CUDA**:

```bash
PYTHONPATH=. conda run -n rtg132 python tests/test_continual_avatar.py
```

## Integration with the existing repo

`continual_avatar` reuses the project's existing modules rather than
duplicating them:

- `skinned_gaussian.smpl_model.SMPLWrapper` — SMPL forward kinematics &
  skinning weights.
- `skinned_gaussian.skinned_gaussians.SkinnedGaussians` — the canonical
  LBS avatar representation (deform / render_posed / weight-regularization).
- `multiview_refine.gsplat_wrapper.render_gaussians` — differentiable 3DGS
  rasterization (called transitively via `SkinnedGaussians.render_posed`).
- `multiview_refine.cameras` — camera loading for monocular frames.
- `multiview_refine.ply_io` — PLY I/O for warm-start attribute extraction.

gsplat's `DefaultStrategy` is wired to `SkinnedGaussians` via a small
training manager inside `continual.py` (`_SkinnedTrainingManager`), since
`SkinnedGaussians` is a plain `nn.Module` and does not itself own a
densification strategy.
