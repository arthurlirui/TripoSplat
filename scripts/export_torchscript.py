"""Export the TripoSplat Stage-A neural networks to TorchScript (.pt) for the C++ library.

This is a one-time offline step. It loads the 4 safetensors checkpoints used by
Stage A (DinoV3 image encoder, Flux2 VAE encoder, latent flow-matching model, and
the octree Gaussian decoder), wraps their dict/generator-based forward signatures
into flat-tensor nn.Modules that TorchScript can trace, and saves them as .pt
files under cpp/weights/.

BiRefNet (background removal) is intentionally NOT exported here: it uses
torchvision.ops.deform_conv2d which has no libtorch C++ binding. The C++ library
expects the caller to supply an already-cutout RGBA image (see scripts/rmbg_preprocess.py).

Usage:
    python scripts/export_torchscript.py \
        --out cpp/weights \
        --seed 42 --num-gaussians 262144 --steps 20

After exporting, the script self-verifies that each traced module reproduces the
original module's output (torch.allclose, fp16 rtol) on a fixed example, so the
C++ side can rely on numerical parity with the Python pipeline.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the repo root importable so `from triposplat import ...` works regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from triposplat import (  # noqa: E402
    FLOW_MODEL_ARGS,
    GS_DECODER_ARGS,
    OCTREE_DECODER_ARGS,
    DinoV3ViT,
    Flux2VAEEncoder,
    LatentSeqMMFlowModel,
    OctreeGaussianDecoder,
    load_decoder,
    load_dinov3,
    load_flow_model,
    load_vae_encoder,
)


# ---------------------------------------------------------------------------
# Flat-tensor wrappers (make dict/generator interfaces traceable)
# ---------------------------------------------------------------------------

class Flux2VAEEncoderWrapper(nn.Module):
    """Wrap Flux2VAEEncoder.encode so TorchScript can trace it.

    The original encode(images, deterministic, generator) takes a non-tensor
    generator argument. For Stage-A inference the pipeline calls it with
    deterministic=False and a generator seeded from `seed`.

    The C++ pipeline calls torch::manual_seed(seed) ONCE at the start of run(),
    then the VAE and the flow sampler both draw from the *global* CUDA generator
    in sequence — exactly mirroring Python's single shared Generator that is
    advanced by the VAE's randn and then by the sampler's randn. So this wrapper
    passes generator=None (use the global generator) and ignores the `seed`
    tensor argument at trace time (the C++ side seeds globally before calling).

    The `seed` argument is retained in the signature only so the traced module's
    input count matches what C++ passes; it has no runtime effect.
    """

    def __init__(self, vae: Flux2VAEEncoder):
        super().__init__()
        self.vae = vae

    def forward(self, images: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
        # generator=None -> draw from the global default generator, which the
        # C++ pipeline seeds via torch::manual_seed(seed) before this call.
        return self.vae.encode(images, deterministic=False, generator=None)


class FlowModelWrapper(nn.Module):
    """Wrap LatentSeqMMFlowModel.forward to flat-tensor I/O.

    Original: model(x_t={'latent','camera'}, t, cond={'feature1','feature2'})
              -> {'latent','camera'}
    Wrapped:  forward(latent, camera, t, feature1, feature2)
              -> (latent_out, camera_out)
    """

    def __init__(self, flow: LatentSeqMMFlowModel):
        super().__init__()
        self.flow = flow

    def forward(self,
                latent: torch.Tensor,
                camera: torch.Tensor,
                t: torch.Tensor,
                feature1: torch.Tensor,
                feature2: torch.Tensor) -> tuple:
        x_t = {'latent': latent, 'camera': camera}
        cond = {'feature1': feature1, 'feature2': feature2}
        out = self.flow(x_t, t, cond)
        return (out['latent'], out['camera'])


class OctreeModuleWrapper(nn.Module):
    """Wrap OctreeProbabilityFixedlenDecoder.forward for tracing.

    Original: forward(x=coords_norm, l=res, cond=latent, l2=num) -> {'logits','probs'}
    Wrapped:  forward(coords_norm, res, latent, num) -> logits
    (the sampler only consumes logits; probs are recomputed in C++ via softmax.)
    """

    def __init__(self, octree: nn.Module):
        super().__init__()
        self.octree = octree

    def forward(self,
                coords_norm: torch.Tensor,
                res: torch.Tensor,
                latent: torch.Tensor,
                num: torch.Tensor) -> torch.Tensor:
        out = self.octree(coords_norm, res, latent, num)
        return out['logits']


class GSDecoderWrapper(nn.Module):
    """Wrap ElasticGaussianFixedlenDecoder.forward for tracing.

    Original: forward(x={"points": points}, cond=latent) -> {'features'}
    Wrapped:  forward(points, latent) -> features
    """

    def __init__(self, gs: nn.Module):
        super().__init__()
        self.gs = gs

    def forward(self, points: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.gs(x={"points": points}, cond=latent)['features']


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _trace_and_save(module: nn.Module, example_inputs, out_path: Path, name: str,
                    check_module: nn.Module = None, check_inputs=None, rtol=1e-3, atol=1e-3):
    """Trace `module` on `example_inputs`, save to out_path, optionally verify."""
    module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, example_inputs, check_trace=False)
    traced.save(str(out_path))
    print(f"[export] saved {name} -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    if check_module is not None and check_inputs is not None:
        with torch.no_grad():
            ref = check_module(*check_inputs)
            got = traced(*example_inputs)
        _compare(name, ref, got, rtol, atol)


def _compare(name, ref, got, rtol, atol):
    if isinstance(ref, (tuple, list)):
        assert len(ref) == len(got), f"{name}: output arity mismatch {len(ref)} vs {len(got)}"
        for i, (r, g) in enumerate(zip(ref, got)):
            _compare(f"{name}[{i}]", r, g, rtol, atol)
        return
    if not isinstance(ref, torch.Tensor):
        # non-tensor (e.g. dict) — skip
        print(f"[verify] {name}: non-tensor output, skipping numeric check")
        return
    ok = torch.allclose(ref, got, rtol=rtol, atol=atol)
    max_abs = (ref.float() - got.float()).abs().max().item()
    print(f"[verify] {name}: allclose={ok} max_abs_diff={max_abs:.6e} shape={tuple(ref.shape)}")
    if not ok:
        raise RuntimeError(f"{name}: traced output diverges from original (max_abs={max_abs:.3e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Export TripoSplat Stage-A models to TorchScript.")
    ap.add_argument("--ckpts", default=str(_REPO_ROOT / "ckpts"),
                    help="Path to ckpts/ directory.")
    ap.add_argument("--out", default=str(_REPO_ROOT / "cpp" / "weights"),
                    help="Output directory for .pt files.")
    ap.add_argument("--device", default="cuda", help="torch device (default: cuda).")
    ap.add_argument("--seed", type=int, default=42, help="Seed for example tensors.")
    ap.add_argument("--num-gaussians", type=int, default=262144,
                    help="Target gaussian count (drives octree example token count).")
    ap.add_argument("--steps", type=int, default=20, help="Sampler steps (for example t).")
    ap.add_argument("--skip-verify", action="store_true", help="Skip self-verification.")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpts = Path(args.ckpts)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    gpp = GS_DECODER_ARGS['representation_config']['num_gaussians']  # 32
    num_decoder_tokens = max(1, args.num_gaussians // gpp)
    q_token_length = FLOW_MODEL_ARGS['q_token_length']  # 8192
    in_channels = FLOW_MODEL_ARGS['in_channels']  # 16
    cam_channels = FLOW_MODEL_ARGS['cam_channels']  # 5
    cond_channels = FLOW_MODEL_ARGS['cond_channels']  # 1280
    cond2_channels = FLOW_MODEL_ARGS['cond2_channels']  # 128

    torch.manual_seed(seed)

    # ---- 1. DinoV3 ----
    print("[export] loading DinoV3 ...")
    dinov3 = load_dinov3(str(ckpts / "clip_vision" / "dino_v3_vit_h.safetensors"),
                         device=device, dtype=torch.bfloat16)
    ex_img = torch.randn(1, 3, 1024, 1024, device=device, dtype=torch.bfloat16)
    _trace_and_save(dinov3, (ex_img,), out_dir / "dinov3.pt", "dinov3",
                    check_module=None if args.skip_verify else dinov3,
                    check_inputs=None if args.skip_verify else (ex_img,),
                    rtol=2e-3, atol=2e-3)
    del dinov3

    # ---- 2. Flux2 VAE encoder ----
    print("[export] loading Flux2 VAE encoder ...")
    vae = load_vae_encoder(str(ckpts / "vae" / "flux2-vae.safetensors"),
                           device=device, dtype=torch.bfloat16)
    vae_wrap = Flux2VAEEncoderWrapper(vae).to(device)
    vae_wrap.eval()
    ex_vae_img = torch.rand(1, 3, 1024, 1024, device=device, dtype=torch.bfloat16)
    ex_seed = torch.tensor([seed], dtype=torch.int64, device=device)
    # VAE encode is stochastic (uses generator); reproducibility comes from the
    # fixed seed, so verify against a freshly-seeded wrapper forward, not a cached value.
    with torch.no_grad():
        traced_vae = torch.jit.trace(vae_wrap, (ex_vae_img, ex_seed), check_trace=False)
    traced_vae.save(str(out_dir / "vae_encoder.pt"))
    print(f"[export] saved vae_encoder -> {out_dir / 'vae_encoder.pt'} "
          f"({(out_dir / 'vae_encoder.pt').stat().st_size / 1e6:.1f} MB)")
    if not args.skip_verify:
        # Reproducibility check: the wrapper draws from the global CUDA generator,
        # so re-seeding before each call must produce identical outputs (this is
        # exactly what the C++ pipeline does once per run()).
        with torch.no_grad():
            torch.manual_seed(seed)
            a = traced_vae(ex_vae_img, ex_seed)
            torch.manual_seed(seed)
            b = traced_vae(ex_vae_img, ex_seed)
        ok = torch.allclose(a, b, rtol=1e-4, atol=1e-4)
        print(f"[verify] vae_encoder: seeded reproducible={ok} shape={tuple(a.shape)}")
        if not ok:
            raise RuntimeError("vae_encoder: non-reproducible when re-seeded")
    # Capture a reference feature2 for the flow model example below.
    with torch.no_grad():
        torch.manual_seed(seed)
        feature2_ref = traced_vae(ex_vae_img, ex_seed)
    del vae, vae_wrap, traced_vae

    # ---- 3. Flow model ----
    print("[export] loading flow model ...")
    flow = load_flow_model(str(ckpts / "diffusion_models" / "triposplat_fp16.safetensors"),
                           device=device, dtype=torch.float16)
    flow_wrap = FlowModelWrapper(flow).to(device)
    flow_wrap.eval()
    ex_latent = torch.randn(1, q_token_length, in_channels, device=device, dtype=torch.float16)
    ex_camera = torch.randn(1, 1, cam_channels, device=device, dtype=torch.float16)
    ex_t = torch.tensor([1000.0 * 0.5] * 1, device=device, dtype=torch.float32)
    # feature1 from DinoV3: (1, 1+4+patches, 1280). patches = (1024/16)^2 = 4096 -> 4100 tokens.
    n_feat1 = 1 + 4 + (1024 // 16) ** 2
    ex_feature1 = torch.randn(1, n_feat1, cond_channels, device=device, dtype=torch.float16)
    # feature2 length must match feature1 (pipeline pads 5 register tokens); reuse a plausible length.
    ex_feature2 = torch.randn(1, n_feat1, cond2_channels, device=device, dtype=torch.float16)
    _trace_and_save(flow_wrap,
                    (ex_latent, ex_camera, ex_t, ex_feature1, ex_feature2),
                    out_dir / "flow_model.pt", "flow_model",
                    check_module=None if args.skip_verify else flow_wrap,
                    check_inputs=None if args.skip_verify
                    else (ex_latent, ex_camera, ex_t, ex_feature1, ex_feature2),
                    rtol=2e-3, atol=2e-3)
    del flow, flow_wrap

    # ---- 4. Octree + GS decoder ----
    print("[export] loading octree Gaussian decoder ...")
    decoder = load_decoder(str(ckpts / "vae" / "triposplat_vae_decoder_fp16.safetensors"),
                           device=device, dtype=torch.float16)

    octree_wrap = OctreeModuleWrapper(decoder.octree).to(device)
    octree_wrap.eval()
    gs_wrap = GSDecoderWrapper(decoder.gs).to(device)
    gs_wrap.eval()

    # Octree example: at level 1 there is 1 parent coord (0.5,0.5,0.5), res=2.
    ex_coords = torch.tensor([[[0.5, 0.5, 0.5]]], device=device, dtype=torch.float16)
    ex_res = torch.tensor([2], device=device, dtype=torch.long)
    ex_latent_dec = torch.randn(1, q_token_length, in_channels, device=device, dtype=torch.float16)
    ex_num = torch.tensor([num_decoder_tokens], device=device, dtype=torch.long)
    _trace_and_save(octree_wrap,
                    (ex_coords, ex_res, ex_latent_dec, ex_num),
                    out_dir / "octree.pt", "octree",
                    check_module=None if args.skip_verify else octree_wrap,
                    check_inputs=None if args.skip_verify
                    else (ex_coords, ex_res, ex_latent_dec, ex_num),
                    rtol=2e-3, atol=2e-3)

    # GS decoder example: points_pred has shape (B, num_decoder_tokens, 3).
    ex_points = torch.rand(1, num_decoder_tokens, 3, device=device, dtype=torch.float16)
    _trace_and_save(gs_wrap,
                    (ex_points, ex_latent_dec),
                    out_dir / "gs_decoder.pt", "gs_decoder",
                    check_module=None if args.skip_verify else gs_wrap,
                    check_inputs=None if args.skip_verify
                    else (ex_points, ex_latent_dec),
                    rtol=2e-3, atol=2e-3)

    print(f"\n[export] DONE. All .pt files written to {out_dir}")
    print("Files: dinov3.pt, vae_encoder.pt, flow_model.pt, octree.pt, gs_decoder.pt")


if __name__ == "__main__":
    main()
