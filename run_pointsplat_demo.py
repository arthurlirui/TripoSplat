"""End-to-end demo: multi-view images → PointSplat → SMPL-X skinned animation.

Pipeline
--------
1. Run PointSplat (in a dedicated conda env) on a multi-view human dataset to
   produce a compact static human Gaussian cloud, written to a 3DGS PLY.
2. Load that PLY with TripoSplat's :func:`load_gaussians_ply`.
3. Load SMPL-X and align it to the Gaussian cloud (height ratio + centroid).
4. Rig the Gaussians to SMPL-X (55 joints + expression blendshape bases) via
   :func:`rig_gaussians_to_smplx`.
5. Generate a body-action pose sequence (wave → raise → turn → return) and an
   orbit camera.
6. Render the driven animation with gsplat and write an MP4.

Steps 2–6 reuse the exact flow from ``run_real_body_demo.py``; this script only
swaps the Gaussian *source* (PointSplat feed-forward instead of a pre-baked PLY)
and shares the camera/action helpers from that module.

Usage
-----
    # 1. Build the pointsplat conda env once (see pointsplat_bridge/README.md)
    # 2. Download the checkpoint:
    hf download Yujie0012/PointSplat_pretrained_weights pointsplat_mixed.pt \
        --local-dir pretrained/pointsplat
    # 3. Download example data (THuman2.0):
    hf download Yujie0012/PointSplat_example_data THuman2_0_example.zip \
        --repo-type dataset --local-dir third_party/pointsplat/datasets
    unzip third_party/pointsplat/datasets/THuman2_0_example.zip \
        -d third_party/pointsplat/datasets
    # 4. Run:
    export PYTHONUTF8=1
    python run_pointsplat_demo.py \
        --dataset_root third_party/pointsplat/datasets/THuman2_0_example \
        --experiment pointsplat_thuman \
        --smpl_model_path E:/Code/mmforall/model/assets/SMPLX/SMPLX_NEUTRAL_2020.npz \
        --output_dir output/pointsplat_demo --n_frames 240 --fps 30
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# On torch>=2.7, gsplat 1.5.x's JIT loader passes a stale _jit_compile
# signature and fails even when a compiled .pyd is cached. Preload the cached
# extension if available so ``gsplat.rasterization`` works. No-op otherwise.
try:
    import pointsplat_bridge._gsplat_preload  # noqa: F401
except Exception:
    pass

SH_C0 = 0.28209479177387814


def main():
    parser = argparse.ArgumentParser(
        description="PointSplat → SMPL-X skinned Gaussian animation demo"
    )
    # PointSplat front-end
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Multi-view dataset root passed to PointSplat as dataset.roots",
    )
    parser.add_argument(
        "--experiment",
        default="pointsplat_thuman",
        choices=["pointsplat_thuman", "pointsplat_renbody"],
        help="PointSplat Hydra experiment; must match dataset_root layout",
    )
    parser.add_argument(
        "--conda_env",
        default="pointsplat",
        help="Conda env with PointSplat deps (see pointsplat_bridge/README.md)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="pointsplat_mixed.pt path (default pretrained/pointsplat/)",
    )
    parser.add_argument(
        "--ps_output_dir",
        default=None,
        help="Where PointSplat writes outputs (default pointsplat_bridge/runs/<ts>)",
    )
    parser.add_argument(
        "--reuse_ply",
        default=None,
        help="Skip PointSplat and reuse a previously generated PLY at this path",
    )
    # SMPL-X / rendering back-end
    parser.add_argument("--smpl_model_path", required=True)
    parser.add_argument("--output_dir", default="output/pointsplat_demo")
    parser.add_argument("--n_frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ═══ 1. PointSplat reconstruction (or reuse a PLY) ═══
    print("=" * 70)
    print("PointSplat → SMPL-X skinned Gaussian animation demo")
    print("=" * 70)

    if args.reuse_ply:
        ply_path = args.reuse_ply
        print(f"\nStep 1: Reusing existing PLY {ply_path}")
    else:
        print(f"\nStep 1: Run PointSplat on {args.dataset_root}")
        from pointsplat_bridge import reconstruct_gaussians

        result = reconstruct_gaussians(
            args.dataset_root,
            experiment=args.experiment,
            conda_env=args.conda_env,
            checkpoint=args.checkpoint,
            output_dir=args.ps_output_dir,
        )
        ply_path = result.ply_path
        print(f"  PointSplat produced {ply_path}")

    # ═══ 2–7: reuse the real-body demo's load/align/rig/animate/render flow ═══
    # Import the helpers from run_real_body_demo rather than duplicating them.
    import run_real_body_demo as rbd

    print(f"\nStep 2: Load Gaussians from {ply_path}")
    from multiview_refine.ply_io import load_gaussians_ply

    means, scales_raw, quats, opac_raw, sh0, _ = load_gaussians_ply(ply_path, device)
    opacities = torch.sigmoid(opac_raw)
    print(f"  {len(means)} Gaussians")
    print(f"  Y range: [{means[:,1].min():.3f}, {means[:,1].max():.3f}]")

    # Prune dark opaque floaters (same heuristic as run_real_body_demo).
    colors_eval = (sh0[:, 0, :] * SH_C0 + 0.5).clamp(0, 1)
    keep = ~((colors_eval.max(dim=-1).values < 0.01) & (opacities > 0.5))
    print(f"  Pruning {(~keep).sum().item()} dark floaters → {keep.sum().item()} remaining")
    means, scales_raw, quats, opac_raw, sh0 = [
        x[keep] for x in [means, scales_raw, quats, opac_raw, sh0]
    ]

    # Step 3: Load SMPL-X + align (height ratio + centroid).
    print(f"\nStep 3: Load SMPL-X + align to Gaussian cloud")
    from skinned_gaussian.smplx_model import SMPLXWrapper

    smplx = SMPLXWrapper(args.smpl_model_path, device=args.device)
    rest_verts, _ = smplx.get_rest_mesh()
    smpl_h = (rest_verts[:, 1].max() - rest_verts[:, 1].min()).item()
    gs_h = (means[:, 1].max() - means[:, 1].min()).item()
    scale = gs_h / smpl_h
    gc = means.median(0).values
    smplx.align_to_scene(scale=scale, transl=gc)
    print(f"  Scale: {scale:.3f}, Centroid: {gc.tolist()}")

    # Step 4: Rig Gaussians to SMPL-X.
    print(f"\nStep 4: Rig {len(means)} Gaussians to SMPL-X (55 joints + expression)")
    from skinned_gaussian.rigging import rig_gaussians_to_smplx

    skinned = rig_gaussians_to_smplx(
        means, scales_raw, quats, opac_raw, sh0, smplx,
        n_neighbours=4, temperature=0.1, device=args.device,
        weights_optimizable=False, n_expression_bases=100,
    )

    # Step 4b: Verify rest = identity.
    rest_T = smplx.get_joint_transforms()
    means_rest, _, _ = skinned.deform(rest_T)
    diff = (means_rest - means).abs().max()
    print(f"\n  Rest pose diff = {diff:.6f} ({'✓' if diff < 0.01 else '✗'})")

    # Step 5: Generate actions + orbit cameras.
    print(f"\nStep 5: Generate {args.n_frames} action frames + orbit cameras")
    pose_seq = rbd.generate_body_actions(args.n_frames, device)
    cameras = rbd.make_orbit_cameras(gc, args.n_frames, device)

    # Step 6: Render.
    print(f"\nStep 6: Render {args.n_frames} frames (orbit + actions)")
    from multiview_refine.gsplat_wrapper import render_gaussians
    from multiview_refine.video_io import create_video_writer

    bg = torch.zeros(3, device=device)
    W, H = cameras[0].width, cameras[0].height
    video_path = os.path.join(args.output_dir, "pointsplat_demo.mp4")
    vw, _ = create_video_writer(video_path, "mp4v", args.fps, (W, H))

    t0 = time.time()
    for i in range(args.n_frames):
        T = smplx.get_joint_transforms(
            betas=pose_seq["betas"],
            body_pose=pose_seq["body_pose"][i],
            global_orient=pose_seq["global_orient"][i],
            transl=pose_seq["transl"][i],
            jaw_pose=pose_seq["jaw_pose"][i],
            leye_pose=pose_seq["leye_pose"][i],
            reye_pose=pose_seq["reye_pose"][i],
            expression=pose_seq["expression"][i],
        )
        with torch.no_grad():
            rgb, alpha, meta = skinned.render_posed(
                T, cameras[i], background=bg, sh_degree=0,
                expression=pose_seq["expression"][i],
            )
        frame = cv2.cvtColor(
            (rgb.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        vw.write(frame)

        if i % 30 == 0 or i == 0:
            elapsed = time.time() - t0
            means_i, _, _ = skinned.deform(T, expression=pose_seq["expression"][i])
            deform = (means_i - means_rest).abs().max().item()
            print(
                f"  frame {i:4d}/{args.n_frames}: rgb={rgb.mean():.4f}, "
                f"deform={deform:.4f}, {elapsed:.1f}s"
            )

    vw.release()
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  Video: {os.path.getsize(video_path) // 1024} KB")

    # Step 7: Save key frames.
    print(f"\nStep 7: Save key frames")
    key_indices = [i for i in [0, 60, 120, 180] if i < args.n_frames]
    for idx in key_indices:
        T = smplx.get_joint_transforms(
            body_pose=pose_seq["body_pose"][idx],
            global_orient=pose_seq["global_orient"][idx],
            jaw_pose=pose_seq["jaw_pose"][idx],
            expression=pose_seq["expression"][idx],
        )
        with torch.no_grad():
            rgb, _, _ = skinned.render_posed(
                T, cameras[idx], background=bg, sh_degree=0,
                expression=pose_seq["expression"][idx],
            )
        frame = cv2.cvtColor(
            (rgb.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR,
        )
        cv2.imwrite(os.path.join(args.output_dir, f"frame_{idx:03d}.png"), frame)
        print(f"    frame_{idx:03d}.png")

    print(f"\n{'=' * 70}")
    print(f"Done! Outputs in {args.output_dir}/:")
    print(f"  pointsplat_demo.mp4 — {args.n_frames} frames, rotating camera + body actions")
    print(f"  frame_000/060/120/180.png — key frame screenshots")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
