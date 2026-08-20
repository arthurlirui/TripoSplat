# pointsplat_bridge

Bridge PointSplat (zju3dv, ECCV 2026) — a feed-forward multi-view human-Gaussian
reconstruction model — into TripoSplat's existing SMPL-X skinning pipeline.

## Why a bridge

PointSplat pins Python 3.10 + torch 2.4 + xformers 0.0.27 + numpy 1.24.4 +
pytorch_lightning 2.4 + gsplat 1.5.x. TripoSplat's own runtime typically uses
torch 2.6 / gsplat 0.1.0 on Python 3.11. These cannot coexist in one process, so
the bridge drives PointSplat as a **child process** in a dedicated conda env and
exchanges data through the standard 3DGS PLY file that PointSplat writes
(`test.save_gaussian=true`). TripoSplat then reads that PLY back with
`multiview_refine.ply_io.load_gaussians_ply` and runs the usual
align → rig → animate → render flow.

```
                     conda env: pointsplat              conda env: <your TripoSplat env>
 multi-view images ──►  PointSplat  ──►  PLY file  ──►  load_gaussians_ply
 (THuman2.0 /          (torch 2.4,       (3DGS)         → SMPL-X align
  DNA-Rendering)        gsplat 1.5.x,                   → rig_gaussians_to_smplx
                        xformers, …)                    → LBS + expression blendshape
                                                         → gsplat render → MP4
```

## 1. Create the PointSplat conda env (one-time)

```bash
conda create -n pointsplat python=3.10 -y
conda activate pointsplat
cd third_party/pointsplat
pip install -r requirements.txt
```

This installs torch 2.4 (cu124), xformers 0.0.27, gsplat 1.5.x,
pytorch_lightning 2.4, hydra-core 1.3.2, numpy 1.24.4, etc. Building xformers
can take a while; the pinned wheel usually avoids compilation.

Verify:

```bash
conda run -n pointsplat python -c "import torch,gsplat,xformers,pytorch_lightning,hydra; print('torch',torch.__version__,'gsplat',gsplat.__version__,'cuda',torch.cuda.is_available())"
```

## 2. Download the checkpoint

```bash
hf download Yujie0012/PointSplat_pretrained_weights pointsplat_mixed.pt \
  --local-dir pretrained/pointsplat
```

Expected layout:

```
pretrained/pointsplat/pointsplat_mixed.pt
```

## 3. Get example data

THuman2.0 (smaller, recommended first try):

```bash
hf download Yujie0012/PointSplat_example_data THuman2_0_example.zip \
  --repo-type dataset --local-dir third_party/pointsplat/datasets
unzip third_party/pointsplat/datasets/THuman2_0_example.zip \
  -d third_party/pointsplat/datasets
```

DNA-Rendering:

```bash
hf download Yujie0012/PointSplat_example_data DNA_Rendering_example.zip \
  --repo-type dataset --local-dir third_party/pointsplat/datasets
unzip third_party/pointsplat/datasets/DNA_Rendering_example.zip \
  -d third_party/pointsplat/datasets
```

## 4. Run the end-to-end demo

From the repo root, using TripoSplat's own env (e.g. `gs`):

```bash
conda activate gs            # or whichever env runs your TripoSplat
export PYTHONUTF8=1
python run_pointsplat_demo.py \
  --dataset_root third_party/pointsplat/datasets/THuman2_0_example \
  --experiment pointsplat_thuman \
  --conda_env pointsplat \
  --smpl_model_path E:/Code/mmforall/model/assets/SMPLX/SMPLX_NEUTRAL_2020.npz \
  --output_dir pointsplat_demo \
  --n_frames 240 --fps 30
```

Outputs land in `pointsplat_demo/`:
- `pointsplat_demo.mp4` — orbit-camera animation with SMPL-X body actions
  (wave → both arms raise → body turn + reach → return), plus jaw/expression.
- `frame_000/060/120/180.png` — key frame screenshots.

PointSplat's own artifacts (the generated PLY, intermediate images) are written
to `pointsplat_bridge/runs/<timestamp>/` unless `--ps_output_dir` is set.

### Reusing a previously generated PLY

Once PointSplat has produced a PLY you can iterate on the animation without
re-running it:

```bash
python run_pointsplat_demo.py \
  --reuse_ply pointsplat_bridge/runs/<timestamp>/gaussians/<scene>.ply \
  --smpl_model_path <...> --output_dir pointsplat_demo
```

## 5. Using your own multi-view data

PointSplat's dataset loaders expect specific on-disk layouts. Prepare your data
to match the experiment you choose:

- **THuman2.0** (`pointsplat_thuman`): `val/<scene>_<view>/{img/2.jpg,
  mask/2.png, parm/2_intrinsic.npy, parm/2_extrinsic.npy}`. See
  [GPS-Gaussian dataset prep](https://github.com/aipixel/GPS-Gaussian#dataset-preparation).
- **DNA-Rendering** (`pointsplat_renbody`): `scenes/<scene>/transforms.json` +
  `images/<view>/<timestep>.webp` + `masks/<scene>/fmasks/<view>/<timestep>.png`.
  See [Diffuman4D](https://github.com/zju3dv/Diffuman4D#processed-dna-rendering-dataset).

If you have a different capture format, either convert it to one of the above,
or write a small dataset adapter inside `third_party/pointsplat/src/dataset/`
and add a matching Hydra config — outside the scope of this bridge.

## Module overview

- `inference.py` — `reconstruct_gaussians(...)`: shells out to the `pointsplat`
  conda env, runs `python -m src.main` with Hydra overrides that force
  `test.save_gaussian=true`, then locates and returns the output PLY path.
  Never imports torch/PointSplat itself, so it's importable from TripoSplat's
  runtime.

- `run_pointsplat_demo.py` (repo root) — entry point. Step 1 calls the bridge;
  steps 2–7 reuse the load/align/rig/animate/render flow from
  `run_real_body_demo.py` (shared via `import run_real_body_demo`).

## Troubleshooting

- **"PointSplat checkpoint not found"** — run the `hf download` command in §2.
- **"PointSplat exited with code N"** — the bridge streams PointSplat's stdout
  through; read it. Common causes: wrong `--conda_env` name, CUDA OOM (lower
  `dataset.view_sampler.num_context_views` via an override), or a dataset layout
  that doesn't match `--experiment`.
- **gsplat version mismatch inside the pointsplat env** — the model uses the
  newer `gsplat.rasterization` + `gsplat.strategy` API (1.5.x). The pinned
  `gsplat` in `requirements.txt` is correct; do not downgrade it.
- **TripoSplat render side fails on the PLY** — the PLY PointSplat writes is
  standard 3DGS; if `load_gaussians_ply` complains about missing fields, check
  that you're not mixing PLYs from a different writer.
