"""End-to-end: TripoSplat init → align → multi-view optimize → compare.

This script demonstrates the full pipeline:
  1. TripoSplat generates initial Gaussians from a single image.
  2. Align Gaussians to the scene coordinate system using init-view camera pose.
  3. Render aligned (un-optimized) Gaussians from all views → measure PSNR.
  4. Optimize with gsplat differentiable rasterization (10000 iters).
  5. Render optimized Gaussians → measure PSNR improvement.
  6. Save comparison video (init | optimized | GT).

Usage:
    export PYTHONUTF8=1
    python run_e2e_test.py \\
        --init_image static/000001/mvs/images/00.png \\
        --transforms static/000001/mvs/transforms_test.json \\
        --output_dir output/e2e_output \\
        --num_gaussians 131072 \\
        --num_iterations 10000 \\
        --image_scale 0.5
"""

import argparse
import os
import sys
import math
import time
import random

import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SH_C0 = 0.28209479177387814


def main():
    parser = argparse.ArgumentParser(description="End-to-end TripoSplat + multi-view refinement test.")
    parser.add_argument('--init_image', required=True)
    parser.add_argument('--transforms', required=True)
    parser.add_argument('--output_dir', default='output/e2e_output')
    parser.add_argument('--num_gaussians', type=int, default=131072)
    parser.add_argument('--num_iterations', type=int, default=10000)
    parser.add_argument('--image_scale', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ═══════════════════════════════════════════════════════════════════
    # Phase 1: TripoSplat generate initial Gaussians
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("Phase 1: TripoSplat — generate initial Gaussians from single image")
    print("=" * 70)

    from triposplat import TripoSplatPipeline
    pipe = TripoSplatPipeline(
        ckpt_path='ckpts/diffusion_models/triposplat_fp16.safetensors',
        decoder_path='ckpts/vae/triposplat_vae_decoder_fp16.safetensors',
        dinov3_path='ckpts/clip_vision/dino_v3_vit_h.safetensors',
        flux2_vae_encoder_path='ckpts/vae/flux2-vae.safetensors',
        rmbg_path='ckpts/background_removal/birefnet.safetensors',
        device='cuda',
    )
    gaussian, prepared = pipe.run(
        args.init_image, num_gaussians=args.num_gaussians,
        remove_background=True, show_progress=True)

    # Extract raw attributes (in TripoSplat model frame).
    means = gaussian.get_xyz.detach().float().to(device)
    scales = gaussian.get_scaling.detach().float().to(device)
    quats = (gaussian._rotation + gaussian.rots_bias[None, :]).detach().float().to(device)
    opacities = gaussian.get_opacity.detach().squeeze(-1).float().to(device)
    sh0 = gaussian._features_dc.detach().float().to(device)
    print(f"  {len(means)} Gaussians in TripoSplat model frame")
    print(f"  Means range: [{means.min():.3f}, {means.max():.3f}], centroid: {means.mean(0).tolist()}")

    # ═══════════════════════════════════════════════════════════════════
    # Phase 2: Load cameras + align coordinate system
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("Phase 2: Load multi-view cameras + align coordinate system")
    print("=" * 70)

    from multiview_refine.cameras import load_cameras, find_camera_by_filename
    from multiview_refine.gsplat_wrapper import render_gaussians
    from multiview_refine.losses import compute_psnr
    from multiview_refine.coord_registry import MODEL2SCENE
    from multiview_refine.quaternion import quaternion_compose, rotation_matrix_to_quaternion
    from multiview_refine.alignment import estimate_similarity_transform_multiview

    cameras = load_cameras(args.transforms, device=device,
                           image_scale=args.image_scale, load_image=True)
    init_cam = find_camera_by_filename(cameras, os.path.basename(args.init_image))
    print(f"  {len(cameras)} cameras at {cameras[0].width}x{cameras[0].height}")
    print(f"  Init camera position: {init_cam.position.tolist()}")

    # ── Pre-transform: model → Y-up scene ──
    # TripoSplat stores Gaussians in AABB [-0.5, 0.5]³.  MODEL2SCENE maps
    # (x,y,z) → (x, z, −y) to the Y-up scene frame (the same world frame as
    # the SfM cameras).  This is consistent with coord_registry and
    # run_pipeline.py / run_pipeline_verify.py.
    R_model2scene = MODEL2SCENE.to(device)
    means_scene = means @ R_model2scene.T
    quats_scene = quaternion_compose(rotation_matrix_to_quaternion(R_model2scene), quats)
    print(f"  Applied MODEL2SCENE (model → Y-up scene)")

    # ── Multi-view silhouette alignment ──
    # Delegate to the shared alignment module, seeding the yaw search from
    # the init camera's c2w forward direction for a faster, targeted search.
    forward = init_cam.c2w[:3, 2]
    init_yaw = math.atan2(float(forward[0]), float(forward[2]))

    scale, R, t = estimate_similarity_transform_multiview(
        tensors=(means_scene, scales, quats_scene, opacities, sh0),
        cameras=cameras,
        init_yaw=init_yaw,
        alpha_threshold=0.1,
        verbose=True,
    )

    # Total transform: scale * (R @ R_model2scene @ means) + t
    means_aligned = scale * (means_scene @ R.T) + t
    scales_aligned = scales * scale
    R_total = R @ R_model2scene
    q_total = rotation_matrix_to_quaternion(R_total)
    quats_aligned = quaternion_compose(q_total, quats)

    print(f"  scale={scale:.4f}, t={t.tolist()}")
    print(f"  Aligned means range: [{means_aligned.min():.3f}, {means_aligned.max():.3f}]")

    # ═══════════════════════════════════════════════════════════════════
    # Phase 3: Evaluate un-optimized aligned Gaussians
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("Phase 3: Evaluate aligned (un-optimized) Gaussians")
    print("=" * 70)

    bg = torch.zeros(3, device=device)
    psnrs_init = []
    for i, cam in enumerate(cameras):
        with torch.no_grad():
            rgb, _, _ = render_gaussians(means_aligned, scales_aligned, quats_aligned,
                                         sh0, opacities, cam, sh_degree=0, background=bg)
        p = compute_psnr(rgb, cam.image).item()
        psnrs_init.append(p)
    print(f"  Aligned (un-optimized) PSNR:")
    print(f"    Mean={np.mean(psnrs_init):.2f}, Max={np.max(psnrs_init):.2f}, Min={np.min(psnrs_init):.2f}")

    # Save aligned init PLY.
    from run_pipeline import save_gaussians_ply
    save_gaussians_ply(
        os.path.join(args.output_dir, 'aligned_init.ply'),
        means_aligned,
        torch.log(scales_aligned.clamp_min(1e-8)),
        quats_aligned,
        torch.log(opacities.clamp(1e-6, 1-1e-6)),
        sh0,
        torch.zeros(len(means), 0, 3, device=device))

    # ═══════════════════════════════════════════════════════════════════
    # Phase 4: Multi-view optimization
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"Phase 4: Multi-view optimization ({args.num_iterations} iterations)")
    print("=" * 70)

    from multiview_refine.gaussian_model import RefineableGaussians
    from multiview_refine.cameras import compute_cameras_extent, split_train_test
    from multiview_refine.losses import compute_loss
    from gsplat.strategy import DefaultStrategy

    # Create RefineableGaussians from aligned attributes.
    model = RefineableGaussians(sh_degree=0, device=device)
    model._means = torch.nn.Parameter(means_aligned.contiguous())
    model._scales = torch.nn.Parameter(torch.log(scales_aligned.clamp_min(1e-8)).contiguous())
    model._quats = torch.nn.Parameter(quats_aligned.contiguous())
    model._opacities = torch.nn.Parameter(torch.log(opacities.clamp(1e-6, 1-1e-6)).contiguous())
    model._sh0 = torch.nn.Parameter(sh0.contiguous())
    model._shN = torch.nn.Parameter(torch.zeros(len(means), 0, 3, device=device), requires_grad=False)

    extent = compute_cameras_extent(cameras)
    strategy = DefaultStrategy(
        prune_opa=0.005, grow_grad2d=0.0002, grow_scale3d=0.01,
        grow_scale2d=0.05, prune_scale3d=0.1, prune_scale2d=0.15,
        refine_start_iter=500, refine_stop_iter=15000, refine_every=100,
        reset_every=3000, verbose=False,
    )
    model.training_setup(scene_scale=extent, strategy=strategy)

    train_cams, test_cams = split_train_test(cameras, os.path.basename(args.init_image))
    print(f"  Train: {len(train_cams)}, Test: {len(test_cams)}")

    t0 = time.time()
    for it in range(1, args.num_iterations + 1):
        model.train()
        model.update_learning_rate(it)
        cam = random.choice(train_cams)
        rgb, alpha, meta = render_gaussians(
            model.means, model.scales, model.raw_quats,
            model.get_colors(), model.opacities, cam,
            background=bg, sh_degree=0)
        loss = compute_loss(rgb, cam.image)
        info = dict(meta); info['step'] = it
        model.step_pre_backward(info)
        loss.backward()
        model.step_post_backward(info, step=it, packed=True)
        model.optimizer_step()
        model.optimizer_zero_grad()

        if it % 1000 == 0 or it == 1:
            with torch.no_grad():
                bright_cam = max(train_cams, key=lambda c: c.image.mean().item())
                br, _, _ = render_gaussians(
                    model.means, model.scales, model.raw_quats,
                    model.get_colors(), model.opacities, bright_cam,
                    background=bg, sh_degree=0)
                psnr_bright = compute_psnr(br, bright_cam.image).item()
            elapsed = time.time() - t0
            print(f"  iter {it:5d}: loss={loss.item():.4f}, psnr_bright={psnr_bright:.2f}, "
                  f"n_gs={model.n_gaussians}, {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\n  Optimization done in {elapsed:.1f}s. Final: {model.n_gaussians} Gaussians")

    # Save refined PLY.
    model.save_ply(os.path.join(args.output_dir, 'refined.ply'))
    model.save_splat(os.path.join(args.output_dir, 'refined.splat'))

    # ═══════════════════════════════════════════════════════════════════
    # Phase 5: Evaluate optimized Gaussians
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("Phase 5: Evaluate optimized Gaussians")
    print("=" * 70)

    psnrs_opt = []
    for i, cam in enumerate(cameras):
        with torch.no_grad():
            rgb, _, _ = render_gaussians(
                model.means, model.scales, model.raw_quats,
                model.get_colors(), model.opacities, cam,
                background=bg, sh_degree=0)
        p = compute_psnr(rgb, cam.image).item()
        psnrs_opt.append(p)

    # Print comparison table.
    print(f"\n  {'cam':>4} {'idx':>4} {'PSNR_init':>10} {'PSNR_opt':>10} {'delta':>8}")
    print(f"  {'-'*40}")
    for i in range(0, len(cameras), 4):
        d = psnrs_opt[i] - psnrs_init[i]
        print(f"  {i:4d} {cameras[i].cam_idx:4d} {psnrs_init[i]:10.2f} {psnrs_opt[i]:10.2f} {d:+8.2f}")
    print(f"  {'-'*40}")
    print(f"  {'mean':>9} {np.mean(psnrs_init):10.2f} {np.mean(psnrs_opt):10.2f} "
          f"{np.mean(psnrs_opt)-np.mean(psnrs_init):+8.2f}")
    print(f"  {'max':>9} {np.max(psnrs_init):10.2f} {np.max(psnrs_opt):10.2f}")
    print(f"  {'min':>9} {np.min(psnrs_init):10.2f} {np.min(psnrs_opt):10.2f}")

    # ═══════════════════════════════════════════════════════════════════
    # Phase 6: Comparison video
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("Phase 6: Render comparison video (init | optimized | GT)")
    print("=" * 70)

    W, H = cameras[0].width, cameras[0].height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(os.path.join(args.output_dir, 'comparison.mp4'), fourcc, 8, (W * 3, H))

    for i, cam in enumerate(cameras):
        with torch.no_grad():
            # Init (aligned, un-optimized).
            rgb_init, _, _ = render_gaussians(
                means_aligned, scales_aligned, quats, sh0, opacities, cam,
                sh_degree=0, background=bg)
            # Optimized.
            rgb_opt, _, _ = render_gaussians(
                model.means, model.scales, model.raw_quats,
                model.get_colors(), model.opacities, cam,
                sh_degree=0, background=bg)

        init_bgr = cv2.cvtColor((rgb_init.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        opt_bgr  = cv2.cvtColor((rgb_opt.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        gt_bgr   = cv2.cvtColor((cam.image.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        frame = np.concatenate([init_bgr, opt_bgr, gt_bgr], axis=1)
        cv2.putText(frame, f'Init PSNR={psnrs_init[i]:.1f}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(frame, f'Optimized PSNR={psnrs_opt[i]:.1f}', (W+10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(frame, 'GT', (2*W+10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        vw.write(frame)

    vw.release()
    print(f"  Saved comparison.mp4: init | optimized | GT")
    print(f"\n{'=' * 70}")
    print("Done!")
    print(f"  Outputs in {args.output_dir}/:")
    print(f"    aligned_init.ply  — aligned un-optimized Gaussians")
    print(f"    refined.ply       — optimized Gaussians")
    print(f"    refined.splat     — web viewer format")
    print(f"    comparison.mp4    — init | optimized | GT comparison")
    print(f"  PSNR: {np.mean(psnrs_init):.2f} → {np.mean(psnrs_opt):.2f} "
          f"(+{np.mean(psnrs_opt)-np.mean(psnrs_init):.2f} dB)")


if __name__ == "__main__":
    main()
