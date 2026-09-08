"""Batch pipeline: for each timestamp folder, generate Gaussians via TripoSplat.

Reads multi-view image folders from a base directory (e.g. F:\Data\warm_2-mvs),
where each subfolder (000000, 000001, ...) contains mvs/images/ with 00.png etc.

For each timestamp:
  1. Use 00.png as input to TripoSplat → generate Gaussian splatting model.
  2. Save the .ply file to output_dir/<timestamp>/gaussians.ply.

Usage:
    conda activate rtg132
    python run_batch_generate.py --base_dir F:/Data/warm_2-mvs --output_dir gaussian_output
    python run_batch_generate.py --base_dir F:/Data/warm_2-mvs --output_dir gaussian_output --start 000000 --end 000010
"""

import argparse
import os
import sys
import time
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def find_timestamp_folders(base_dir):
    """Find all timestamp folders (e.g. 000000, 000001, ...) in base_dir.

    Each folder must contain mvs/images/00.png.
    """
    folders = []
    for name in sorted(os.listdir(base_dir)):
        img_path = os.path.join(base_dir, name, 'mvs', 'images', '00.png')
        if os.path.isfile(img_path):
            folders.append(name)
    return folders


def generate_gaussians(pipe, image_path, output_ply, num_gaussians=131072):
    """Run TripoSplat on a single image and save Gaussians as PLY.

    Args:
        pipe: TripoSplatPipeline instance.
        image_path: Path to input image (e.g. .../000000/mvs/images/00.png).
        output_ply: Path to save the output .ply file.
        num_gaussians: Number of Gaussians to generate.

    Returns:
        True if successful, False otherwise.
    """
    try:
        from triposplat import Gaussian
        import torch
        import numpy as np
        import math

        SH_C0 = 0.28209479177387814

        # Run TripoSplat.
        gaussian, prepared = pipe.run(
            image_path,
            num_gaussians=num_gaussians,
            remove_background=True,
            show_progress=False,
        )

        # Extract raw attributes (in TripoSplat model frame).
        means = gaussian.get_xyz.detach().float().cpu()
        scales_activated = gaussian.get_scaling.detach().float().cpu()
        scales_raw = torch.log(scales_activated.clamp_min(1e-8))
        quats = (gaussian._rotation + gaussian.rots_bias[None, :]).detach().float().cpu()
        opacities = gaussian.get_opacity.detach().squeeze(-1).float().cpu()
        opacities_raw = torch.log(opacities.clamp(1e-6, 1 - 1e-6))
        sh0 = gaussian._features_dc.detach().float().cpu()

        N = means.shape[0]

        # Build PLY.
        from plyfile import PlyData, PlyElement

        means_np = means.numpy()
        normals = np.zeros_like(means_np)
        f_dc = sh0[:, 0, :].numpy()  # (N, 3)
        opac_np = opacities_raw.numpy().reshape(-1, 1)
        scales_np = scales_raw.numpy()
        rots_np = torch.nn.functional.normalize(quats, dim=-1).numpy()

        names = ['x', 'y', 'z', 'nx', 'ny', 'nz',
                 'f_dc_0', 'f_dc_1', 'f_dc_2',
                 'opacity', 'scale_0', 'scale_1', 'scale_2',
                 'rot_0', 'rot_1', 'rot_2', 'rot_3']

        data = np.concatenate([
            means_np, normals, f_dc, opac_np, scales_np, rots_np
        ], axis=1)

        dtype = [(n, 'f4') for n in names]
        elements = np.empty(N, dtype=dtype)
        elements[:] = list(map(tuple, data))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el], byte_order='<', text=False).write(output_ply)

        return True, N

    except Exception as e:
        print(f"    ERROR: {e}")
        return False, 0


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate Gaussian splatting models from multi-view image folders."
    )
    parser.add_argument('--base_dir', default='F:/Data/warm_2-mvs',
                        help='Base directory containing timestamp folders (000000, 000001, ...)')
    parser.add_argument('--output_dir', default='gaussian_output',
                        help='Output directory for generated .ply files')
    parser.add_argument('--init_view', default='00.png',
                        help='Input image filename within each timestamp folder')
    parser.add_argument('--num_gaussians', type=int, default=131072,
                        help='Number of Gaussians to generate (32768-262144)')
    parser.add_argument('--start', default=None,
                        help='Start timestamp (e.g. 000000). Default: first available.')
    parser.add_argument('--end', default=None,
                        help='End timestamp (e.g. 000010). Default: last available.')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip timestamps that already have output .ply')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    # ═══ Find all timestamp folders ═══
    print("=" * 60)
    print("Batch Gaussian Splatting Generation")
    print("=" * 60)
    print(f"\nBase directory: {args.base_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Init view: {args.init_view}")
    print(f"Num Gaussians: {args.num_gaussians}")

    all_folders = find_timestamp_folders(args.base_dir)
    if not all_folders:
        print(f"\nERROR: No valid timestamp folders found in {args.base_dir}")
        print(f"  Expected structure: {args.base_dir}/000000/mvs/images/{args.init_view}")
        return

    # Filter by start/end.
    folders = all_folders
    if args.start:
        folders = [f for f in folders if f >= args.start]
    if args.end:
        folders = [f for f in folders if f <= args.end]

    print(f"\nFound {len(all_folders)} timestamp folders total, {len(folders)} to process")
    if len(folders) > 0:
        print(f"  Range: {folders[0]} .. {folders[-1]}")

    # ═══ Load TripoSplat pipeline (once) ═══
    print(f"\n{'=' * 60}")
    print("Loading TripoSplat pipeline...")
    print("=" * 60)
    from triposplat import TripoSplatPipeline
    pipe = TripoSplatPipeline(
        ckpt_path='ckpts/diffusion_models/triposplat_fp16.safetensors',
        decoder_path='ckpts/vae/triposplat_vae_decoder_fp16.safetensors',
        dinov3_path='ckpts/clip_vision/dino_v3_vit_h.safetensors',
        flux2_vae_encoder_path='ckpts/vae/flux2-vae.safetensors',
        rmbg_path='ckpts/background_removal/birefnet.safetensors',
        device=args.device,
    )
    print("Pipeline loaded.\n")

    # ═══ Process each timestamp ═══
    success_count = 0
    skip_count = 0
    fail_count = 0
    total_start = time.time()

    for i, ts in enumerate(folders):
        img_path = os.path.join(args.base_dir, ts, 'mvs', 'images', args.init_view)
        out_dir = os.path.join(args.output_dir, ts)
        out_ply = os.path.join(out_dir, 'gaussians.ply')

        # Skip existing.
        if args.skip_existing and os.path.isfile(out_ply):
            print(f"[{i+1}/{len(folders)}] {ts}: SKIP (already exists)")
            skip_count += 1
            continue

        os.makedirs(out_dir, exist_ok=True)

        print(f"[{i+1}/{len(folders)}] {ts}: Generating from {img_path}")
        t0 = time.time()
        ok, n_gs = generate_gaussians(pipe, img_path, out_ply, args.num_gaussians)
        elapsed = time.time() - t0

        if ok:
            file_size = os.path.getsize(out_ply) / (1024 * 1024)  # MB
            print(f"  -> {n_gs} Gaussians, {file_size:.1f} MB, {elapsed:.1f}s")
            print(f"  Saved: {out_ply}")
            success_count += 1
        else:
            fail_count += 1

    # ═══ Summary ═══
    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Done! Processed {len(folders)} timestamps in {total_elapsed:.1f}s")
    print(f"  Success: {success_count}")
    print(f"  Skipped: {skip_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Output:  {args.output_dir}/<timestamp>/gaussians.ply")
    print("=" * 60)


if __name__ == "__main__":
    main()
