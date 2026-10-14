"""CLI entry point for multi-view Gaussian refinement.

Usage example:

    python run_refine.py \\
        --image_dir static/000001/mvs/images \\
        --transforms static/000001/mvs/transforms_train.json \\
        --init_view 00.png \\
        --num_iterations 30000 \\
        --image_scale 0.5 \\
        --output_dir output/refine_output

This runs the full pipeline:
  1. TripoSplat generates initial Gaussians from --init_view.
  2. Loads multi-view cameras from --transforms.
  3. Aligns init Gaussians to the scene coordinate system (depth + ICP).
  4. Optimizes via gsplat differentiable rasterization.
  5. Saves refined PLY / SPLAT to --output_dir.
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Multi-view refinement of TripoSplat-initialized 3D Gaussians."
    )
    # I/O.
    parser.add_argument("--image_dir", required=True,
                        help="Directory containing multi-view images.")
    parser.add_argument("--transforms", required=True,
                        help="Path to Nerfstudio transforms_{train,val,test}.json.")
    parser.add_argument("--init_view", required=True,
                        help="Init-view image filename (e.g. 00.png).")
    parser.add_argument("--output_dir", default="output/refine_output",
                        help="Output directory for refined PLY/SPLAT.")

    # TripoSplat.
    parser.add_argument("--ckpt_path", default="ckpts/diffusion_models/triposplat_fp16.safetensors")
    parser.add_argument("--decoder_path", default="ckpts/vae/triposplat_vae_decoder_fp16.safetensors")
    parser.add_argument("--dinov3_path", default="ckpts/clip_vision/dino_v3_vit_h.safetensors")
    parser.add_argument("--flux2_vae_encoder_path", default="ckpts/vae/flux2-vae.safetensors")
    parser.add_argument("--rmbg_path", default="ckpts/background_removal/birefnet.safetensors")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_init_gaussians", type=int, default=262144,
                        help="Number of Gaussians for TripoSplat init (32768–262144).")

    # Optimization.
    parser.add_argument("--num_iterations", type=int, default=30000)
    parser.add_argument("--image_scale", type=float, default=0.5,
                        help="Downscale factor for training images.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_l1", type=float, default=0.8)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--sh_degree", type=int, default=0,
                        help="Spherical harmonics degree (0 = DC only).")

    # Densification.
    parser.add_argument("--refine_start_iter", type=int, default=500)
    parser.add_argument("--refine_stop_iter", type=int, default=15000)
    parser.add_argument("--refine_every", type=int, default=100)
    parser.add_argument("--reset_every", type=int, default=3000)
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0002)

    # Misc.
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="Save intermediate PLY every N iterations (0 = disabled).")
    parser.add_argument("--background", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Background RGB color for rendering.")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output.")

    args = parser.parse_args()

    # Import TripoSplat.
    from triposplat import TripoSplatPipeline
    from multiview_refine.refine import refine_from_multiview

    # Build the pipeline.
    print("Loading TripoSplat pipeline...")
    pipe = TripoSplatPipeline(
        ckpt_path=args.ckpt_path,
        decoder_path=args.decoder_path,
        dinov3_path=args.dinov3_path,
        flux2_vae_encoder_path=args.flux2_vae_encoder_path,
        rmbg_path=args.rmbg_path,
        device=args.device,
    )

    # Run refinement.
    model = refine_from_multiview(
        pipe=pipe,
        image_dir=args.image_dir,
        transforms_json=args.transforms,
        init_view=args.init_view,
        num_init_gaussians=args.num_init_gaussians,
        num_iterations=args.num_iterations,
        image_scale=args.image_scale,
        output_dir=args.output_dir,
        seed=args.seed,
        lambda_l1=args.lambda_l1,
        lambda_dssim=args.lambda_dssim,
        background_color=tuple(args.background),
        refine_start_iter=args.refine_start_iter,
        refine_stop_iter=args.refine_stop_iter,
        refine_every=args.refine_every,
        reset_every=args.reset_every,
        densify_grad_threshold=args.densify_grad_threshold,
        sh_degree=args.sh_degree,
        save_interval=args.save_interval,
        verbose=not args.quiet,
    )

    print(f"\nRefinement complete. Final Gaussian count: {model.n_gaussians}")
    print(f"Outputs saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
