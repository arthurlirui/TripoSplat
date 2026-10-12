# triposplat_cpp — C++ Stage-A inference for TripoSplat

A C++ reimplementation of TripoSplat's **Stage-A** generator: a single RGB(A)
image in, a bundle of 3D Gaussians out (written as standard 3DGS `.ply` and
32-byte `.splat` files). It is a drop-in, dependency-light replacement for the
Python `TripoSplatPipeline.run` path, intended to be linked into other C++
applications on Windows x64.

> **Scope.** Only the *generative* stage (image → Gaussians) is ported. The
> downstream Stage-B animation path (SMPL/SMPL-X rigging, LBS deformation,
> gsplat rasterization) is **not** included — the `.ply`/`.splat` output is the
> standard 3DGS format and can be consumed by any downstream renderer.

## How it works

The five neural networks in the Python pipeline (DinoV3 image encoder, Flux2
VAE encoder, latent flow-matching model, octree probability decoder, elastic
Gaussian decoder) are exported **once** to TorchScript `.pt` files from Python
(`scripts/export_torchscript.py`). The C++ library loads them with libtorch's
`torch::jit::load` and runs the same forward passes, while the orchestration
logic — image preprocessing, the Euler + CFG flow sampler, the 8-level octree
systematic sampler, and Gaussian attribute assembly — is reimplemented natively
in C++.

BiRefNet (background removal) is **not** exported because it uses
`torchvision.ops.deform_conv2d`, which has no libtorch C++ binding. Instead the
caller pre-cuts the image to RGBA with `scripts/rmbg_preprocess.py` and feeds
that to the C++ pipeline. This keeps the C++ dependency surface to libtorch
alone.

## Prerequisites

| Component | Version |
|---|---|
| libtorch (CUDA, C++) | matches your Python torch (e.g. 2.x cu121) |
| CUDA Toolkit | the one your libtorch was built against |
| CMake | ≥ 3.24 |
| Compiler | MSVC v143 (Visual Studio 2022), C++17 |
| Python (export step only) | the existing TripoSplat conda env (torch, safetensors, torchvision) |

`stb_image` and `nlohmann::json` are fetched automatically by CMake
(`FetchContent`) on first configure — no manual vendoring needed.

## Build & run

### 1. Export the TorchScript weights (one-time, from Python)

```bash
# from the repo root, in the TripoSplat conda env
python scripts/export_torchscript.py --out cpp/weights --seed 42 --num-gaussians 262144 --steps 20
```

This writes `cpp/weights/{dinov3,vae_encoder,flow_model,octree,gs_decoder}.pt`
and self-verifies each traced module reproduces the original (fp16, `allclose`).

### 2. (If using background removal) pre-cut the input image

```bash
python scripts/rmbg_preprocess.py input.jpg input_rgba.png
```

### 3. Build the C++ library

```bash
cd cpp
cmake -B build -G "Visual Studio 17 2022" -A x64 \
  -DCMAKE_PREFIX_PATH="<path-to-libtorch>"
cmake --build build --config Release
```

The static library `triposplat` and the `run_example` executable are produced
under `build/Release/`.

### 4. Run the end-to-end example

```bash
# from the repo root (so cpp/weights resolves)
./cpp/build/Release/run_example.exe input_rgba.png output \
    --weights cpp/weights --device cuda --seed 42 --steps 20 --num-gaussians 262144
```

This writes `output/gaussians.ply` and `output/gaussians.splat`.

## API

The public surface is three headers under `include/triposplat/`:

```cpp
#include "triposplat/TripoSplat.hpp"
#include "triposplat/IO.hpp"

triposplat::TripoSplatConfig cfg;
cfg.weights_dir = "cpp/weights";
cfg.device = torch::kCUDA;

triposplat::TripoSplatPipeline pipeline(cfg);

// `image` accepts (H,W,3) uint8 RGB (with remove_background=false) or
// (H,W,4) uint8 RGBA (with remove_background=true).
triposplat::Gaussian g = pipeline.run(
    image, /*seed=*/42, /*steps=*/20, /*guidance_scale=*/3.0f,
    /*shift=*/3.0f, /*num_gaussians=*/262144, /*erode_radius=*/1,
    /*remove_background=*/true);

triposplat::save_ply("gaussians.ply", g);
triposplat::save_splat("gaussians.splat", g);
```

| Header | Contents |
|---|---|
| `TripoSplat.hpp` | `TripoSplatConfig`, `TripoSplatPipeline`, `TripoSplatError` |
| `Gaussian.hpp` | `Gaussian` struct (means/scales/quats/opacities/sh0/aabb) |
| `IO.hpp` | `save_ply`, `save_splat` |
| `Export.h` | `TRIPOSPLAT_API` macro (shared-lib export control) |

## Module layout

| Source | Python counterpart | Responsibility |
|---|---|---|
| `src/Preprocess.cpp` | `triposplat.preprocess_image` | resize / alpha erosion / bbox crop / black composite |
| `src/Encoders.cpp` | `triposplat.encode_image` | DinoV3 + Flux2 VAE forward via traced modules |
| `src/FlowSampler.cpp` | `triposplat.FlowEulerCfgSampler.sample` | Euler + CFG 20-step loop |
| `src/Decoder.cpp` | `model.OctreeGaussianDecoder.decode` + `triposplat._build_gaussians` | octree systematic sampling, gs forward, attribute assembly |
| `src/GaussianIO.cpp` | `Gaussian.save_ply` / `save_splat` | binary PLY + 32-byte SPLAT writers (raw conversions + coordinate transform) |
| `src/Pipeline.cpp` | `TripoSplatPipeline.run` | orchestration |

## Numerical parity

The export script verifies each traced module against the original Python
module (`torch.allclose`, fp16). The C++ orchestration mirrors the Python logic
1:1 (same schedule, same octree loop, same activations, same coordinate
transform), so with a fixed `seed` the C++ output matches the Python output to
fp16 precision. Image-resampling differences (PIL LANCZOS vs. libtorch bilinear)
introduce sub-pixel input variance, so exact bit-equality is not guaranteed,
but the generated Gaussians are visually equivalent.

## Limitations

- BiRefNet background removal runs in Python only (deform_conv2d porting risk).
  Feed an RGBA cutout to the C++ pipeline.
- The depth-guided background reconstruction (`reconstruct_bg=True`, Depth
  Anything V2) is not ported — it is a secondary feature.
- Stage B (SMPL rigging, LBS, gsplat rasterization) is out of scope; consume
  the `.ply`/`.splat` output with any 3DGS renderer.
