# Photo-Space Registration — Design Document

## Problem statement

Given a generated Gaussian-Splatting (GS) model and a set of real camera
photographs with known poses, compute the GS model's 3D **world coordinates**
by registering in **camera-photo space** — i.e. by comparing what the model
*renders* against what the cameras actually *photograph*, using image content
rather than silhouettes alone. The result must support **real-time multi-view
updates** of the Gaussian content.

## Why a new algorithm

The existing aligners (`multiview_refine/alignment.py`) use silhouette IoU
and a known canonical orientation. They work for initial placement but:

1. **Silhouettes ignore texture.** Front and back of a symmetric figure have
   near-identical silhouettes; IoU cannot distinguish a 180° yaw, and the
   `orientation.py` uprightness check only catches vertical flips.
2. **They are global and offline.** A full grid search over (R, s) cannot
   run every time a new frame arrives.

Photo-space registration works directly in the pixel grid of the real
photographs, leveraging texture — the same cue the human visual system uses
to localise objects — and supports incremental, streaming refinement.

## Algorithm overview

```
real photo_i  ─┐                        ┌─ matched (pixel_i, model_pt)
render_i      ─┤─ feature match ────────┤
real photo_j  ─┤                        └─ matched (pixel_j, model_pt)
render_j      ─┘

(pixel_i, pixel_j) ──triangulate──▶ scene_pt   (uses real camera poses)
(model_pt, scene_pt) pairs ──Umeyama──▶ (scale, R, t)
```

### Step 1 — Feature extraction (photo space)

For each camera view, detect 2D feature points (SIFT or ORB) in **both**
the real photograph and a render of the model from the same pose.

- **Real photo** features live in the photograph's pixel space.
- **Render** features are detectable because the GS model has texture.

### Step 2 — Render → 3D back-link (the key step)

The critical innovation that makes this "photo-space" registration rather
than silhouette matching: for each rendered feature pixel, we read back
**which Gaussian** (by row index) produced it.

This is done by `render_index_map()`, which renders each Gaussian's
normalised row index as its "colour" and reads back the blended value.
A blended index is accurate when one Gaussian dominates a pixel — the
common case at feature keypoints on a surface. The result is a **2D → 3D
correspondence**: render pixel → 3D Gaussian centre (model frame).

### Step 3 — Feature matching and cross-view tracks

Match real-photo features ↔ render features per view (Lowe ratio test).
Each match gives `(real_pixel, model_3D_point)`.

Then group matches across views by **model-point identity** (rounded 3D
coordinate). A model point seen in ≥2 views forms a **track**:
`{(view_i, real_pixel_i), (view_j, real_pixel_j), ...}` with a shared
`model_3D_point`.

### Step 4 — Triangulation into scene 3D

For each track, triangulate the real-photo pixels across ≥2 views using
the **known real camera poses** (linear DLT). This yields a **scene 3D
point** — where the real-world feature actually is.

An epipolar consistency check rejects descriptor mismatches before
triangulation, and points behind any camera are discarded.

Now we have matched 3D pairs: `(model_3D_point, scene_3D_point)`.

### Step 5 — Similarity transform (Umeyama + RANSAC)

Solve for the similarity transform mapping model points → scene points:

```
scene = scale * (R @ model) + t
```

This is the **Umeyama** closed-form solution (rigid + uniform scale),
wrapped in **RANSAC** to reject outlier correspondences from mismatches
or index-map blending errors.

The result `(scale, R, t)` gives every Gaussian its **world coordinate**:
`xyz_scene = scale * (R @ xyz_model) + t`.

## Real-time incremental registration

`IncrementalPhotoRegistrar` supports streaming frames:

- Each new frame is matched and its features added to a growing track store.
- The transform is recomputed every `refine_every` frames (or when enough
  tracks accumulate).
- A **low-pass filter** (exponential smoothing on scale/translation +
  SLERP on rotation) prevents jitter from noisy per-frame correspondences.
- A **degeneracy guard** rejects updates with implausible scale (collapse
  to zero or explosion), preserving the last good estimate.

This is the entry point for "real-time update of Gaussian content": once
the transform is stable, downstream code splats new frames' Gaussians
directly into the scene frame.

## Mathematical details

### Umeyama similarity transform

Given matched point sets `M = {m_i}` (model) and `S = {s_i}` (scene):

1. Compute centroids: `μ_m = mean(M)`, `μ_s = mean(S)`.
2. Center: `M_c = M - μ_m`, `S_c = S - μ_s`.
3. Cross-covariance: `H = M_c^T S_c / N`.
4. SVD: `H = U D V^T`.
5. Correct reflection: `S = diag(1,1,det(VU))`.
6. `R = V S U^T`.
7. `scale = trace(D S) / var(M)`.
8. `t = μ_s - scale * (R @ μ_m)`.

This is the least-squares optimal similarity transform.

### Triangulation (linear DLT)

For a point seen in two cameras with projection matrices `P_a = K_a [R_a|t_a]`
and `P_b = K_b [R_b|t_b]`, the 3D point `X` satisfies:

```
x_a P_a[2] - P_a[0] = 0    (from u_a)
y_a P_a[2] - P_a[1] = 0    (from v_a)
x_b P_b[2] - P_b[0] = 0    (from u_b)
y_b P_b[2] - P_b[1] = 0    (from v_b)
```

This gives a 4×N system `AX = 0` solved by SVD (smallest singular vector).

## How it composes with the existing pipeline

```
                  ┌─────────────────────────┐
TripoSplat init → │ alignment.py            │ → coarse (s, R, t)
  (single image)  │ (silhouette IoU /       │   (model → scene)
                  │  analytic / c2w)        │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
  real photos +   │ photo_space_registration │ → refined (s, R, t)
  camera poses →  │ (feature correspondences)│   (texture-aware)
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ differentiable_refine.py │ → final (s, R, t)
                  │ (RGB + silhouette grad)  │   (sub-pixel)
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
  streaming →     │ IncrementalPhotoRegistrar│ → live (s, R, t)
  new frames      │ (track + smooth)         │   (real-time)
                  └─────────────────────────┘
```

Photo-space registration sits **between** the coarse silhouette aligner
and the differentiable RGB refinement. It can also run standalone when
the silhouette aligner is unavailable. The incremental registrar handles
the streaming case for continuous learning.

## Coordinate conventions

- **Model points**: TripoSplat AABB frame (same as `alignment.py`).
- **Scene points**: Y-up world frame (`coord_registry.py`).
- **Cameras**: OpenCV convention (`cameras.py`).
- **Returned `(scale, R, t)`**: `xyz_scene = scale * (R @ xyz_model) + t`,
  identical convention to the rest of the package.

## Files

| File | Purpose |
|------|---------|
| `multiview_refine/photo_space_registration.py` | Algorithm implementation |
| `tests/test_photo_space_registration.py` | Tests (synthetic, known-transform recovery) |

## API summary

```python
# One-shot registration from a batch of cameras
from multiview_refine.photo_space_registration import register_photo_space
result = register_photo_space(means, scales, quats, opacities, cameras,
                              descriptor="sift")
# result.scale, result.R, result.t, result.n_correspondences, ...

# Real-time streaming registration
from multiview_refine.photo_space_registration import IncrementalPhotoRegistrar
reg = IncrementalPhotoRegistrar(means, scales, quats, opacities,
                                descriptor="orb", refine_every=3)
for cam in streaming_cameras:
    reg.add_frame(cam)
    scale, R, t = reg.transform
```

## Test results

All 6 tests pass:

| Test | Verifies |
|------|----------|
| `test_umeyama_recovery` | Umeyama recovers known (s, R, t) to 1e-6 |
| `test_kabsch_ransac_outliers` | RANSAC rejects 40% outliers |
| `test_triangulation` | DLT triangulation recovers known 3D point to 1e-4 |
| `test_render_index_map` | Index map returns correct Gaussian at visible pixels |
| `test_end_to_end_identity` | Full pipeline recovers identity transform (scale≈0.91) |
| `test_incremental_registrar` | Streaming registrar accumulates tracks, guards degeneracy |
```
