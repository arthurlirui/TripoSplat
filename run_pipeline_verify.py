"""Full pipeline coordinate verification: run step1+2+3 with fresh data and
save intermediate comparison images at every stage.

Usage:
    conda run -n rtg132 python run_pipeline_verify.py [--init_view 00.png]
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import cv2
from PIL import Image as PILImage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Preload cached gsplat CUDA extension before any gsplat import.
from pointsplat_bridge._gsplat_preload import preload_gsplat_cuda
preload_gsplat_cuda()

from multiview_refine.ply_io import load_gaussians_ply, save_gaussians_ply
from multiview_refine.cameras import Camera, load_cameras, invert_se3, find_camera_by_filename
from multiview_refine.gsplat_wrapper import render_gaussians
from multiview_refine.coord_registry import MODEL2SCENE
from multiview_refine.quaternion import quaternion_compose, rotation_matrix_to_quaternion
from multiview_refine.alignment import estimate_similarity_transform_multiview, _apply_transform
from multiview_refine.video_io import create_video_writer

SH_C0 = 0.28209479177387814


def tensor_to_bgr(t, alpha=None):
    """(3, H, W) float tensor → BGR uint8 numpy."""
    img = (t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if alpha is not None:
        a = (alpha.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        img = np.concatenate([img, np.stack([a] * 3, axis=-1)], axis=1)
    return img


def add_label(img, label, color=(0, 255, 255)):
    h, w = img.shape[:2]
    bar = np.zeros((40, w, 3), dtype=np.uint8)
    cv2.putText(bar, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return np.concatenate([bar, img], axis=0)


def save_side_by_side(render_img, gt_img, path, label_r="Render", label_g="GT"):
    """Save render (left) vs GT (right) side by side with labels."""
    r = add_label(render_img, label_r, (0, 255, 0))
    g = add_label(gt_img, label_g, (0, 0, 255))
    h = max(r.shape[0], g.shape[0])
    w = r.shape[1] + g.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:r.shape[0], :r.shape[1]] = r
    canvas[:g.shape[0], r.shape[1]:] = g
    cv2.imwrite(path, canvas)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Full pipeline coordinate verification")
    parser.add_argument("--init_view", default="00.png")
    parser.add_argument("--image_dir", default="static/000001/mvs/images")
    parser.add_argument("--transforms", default="static/000001/mvs/transforms_train.json")
    parser.add_argument("--output_dir", default="output/pipeline_verify_output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_gaussians", type=int, default=131072)
    parser.add_argument("--image_scale", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 0: Save init image and GT comparison
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("Stage 0: Save init image and GT views")
    print("=" * 70)

    init_img_path = os.path.join(args.image_dir, args.init_view)
    init_img = cv2.imread(init_img_path, cv2.IMREAD_UNCHANGED)
    # RGBA → BGR with alpha
    if init_img.shape[-1] == 4:
        bgr = init_img[:, :, :3]
        alpha = init_img[:, :, 3]
        init_disp = np.concatenate([bgr, np.stack([alpha] * 3, axis=-1)], axis=1)
    else:
        init_disp = init_img
    cv2.imwrite(os.path.join(args.output_dir, "00_init_image.png"), init_disp)
    print(f"  Init image: {init_img_path}, shape={init_img.shape}")

    # Save a few GT views
    for i in [0, 8, 16, 24]:
        gt_path = os.path.join(args.image_dir, f"{i:02d}.png")
        if os.path.exists(gt_path):
            gt_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
            if gt_img.shape[-1] == 4:
                gt_disp = np.concatenate([gt_img[:, :, :3], np.stack([gt_img[:, :, 3]] * 3, axis=-1)], axis=1)
            else:
                gt_disp = gt_img
            cv2.imwrite(os.path.join(args.output_dir, f"00_gt_cam{i:02d}.png"), gt_disp)

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 1: Run TripoSplat on init image (fresh generation)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Stage 1: Run TripoSplat on init image")
    print("=" * 70)

    from triposplat import TripoSplatPipeline
    pipe = TripoSplatPipeline(
        ckpt_path="ckpts/diffusion_models/triposplat_fp16.safetensors",
        decoder_path="ckpts/vae/triposplat_vae_decoder_fp16.safetensors",
        dinov3_path="ckpts/clip_vision/dino_v3_vit_h.safetensors",
        flux2_vae_encoder_path="ckpts/vae/flux2-vae.safetensors",
        rmbg_path="ckpts/background_removal/birefnet.safetensors",
        device=args.device,
    )

    gaussian, prepared = pipe.run(
        init_img_path,
        num_gaussians=args.num_gaussians,
        remove_background=True,
        show_progress=True,
    )

    # Save the preprocessed image
    prepared.save(os.path.join(args.output_dir, "01_preprocessed.png"))
    print(f"  Preprocessed image saved")

    # Extract raw attributes (in model frame)
    means_model = gaussian.get_xyz.detach().float()
    scales_act = gaussian.get_scaling.detach().float()
    quats_raw = (gaussian._rotation + gaussian.rots_bias[None, :]).detach().float()
    opac_act = gaussian.get_opacity.detach().squeeze(-1).float()
    sh0 = gaussian._features_dc.detach().float()
    shN = torch.zeros(means_model.shape[0], 0, 3, dtype=torch.float32)

    print(f"  Generated {means_model.shape[0]} Gaussians (model frame)")
    print(f"  Means range: x=[{means_model[:,0].min():.3f}, {means_model[:,0].max():.3f}]")
    print(f"               y=[{means_model[:,1].min():.3f}, {means_model[:,1].max():.3f}]")
    print(f"               z=[{means_model[:,2].min():.3f}, {means_model[:,2].max():.3f}]")

    # Save initial PLY (model frame)
    scales_raw = torch.log(scales_act.clamp_min(1e-8))
    opac_raw = torch.log(opac_act.clamp(1e-6, 1 - 1e-6))
    ply_path = os.path.join(args.output_dir, "01_initial_gaussians_model.ply")
    save_gaussians_ply(ply_path, means_model, scales_raw, quats_raw, opac_raw, sh0, shN)

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 2: Apply MODEL2SCENE transform and save comparison renders
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Stage 2: Apply MODEL2SCENE transform (model → Y-up scene)")
    print("=" * 70)

    R_m2s = MODEL2SCENE.to(device)
    means_scene = means_model @ R_m2s.T
    q_R = rotation_matrix_to_quaternion(R_m2s)
    quats_scene = quaternion_compose(q_R, quats_raw)

    print(f"  After MODEL2SCENE:")
    print(f"  Means range: x=[{means_scene[:,0].min():.3f}, {means_scene[:,0].max():.3f}]")
    print(f"               y=[{means_scene[:,1].min():.3f}, {means_scene[:,1].max():.3f}]")
    print(f"               z=[{means_scene[:,2].min():.3f}, {means_scene[:,2].max():.3f}]")

    # Save scene-frame PLY
    ply_path = os.path.join(args.output_dir, "02_gaussians_scene_frame.ply")
    save_gaussians_ply(ply_path, means_scene, scales_raw, quats_scene, opac_raw, sh0, shN)

    # Load cameras (OpenCV convention from load_cameras)
    cameras = load_cameras(args.transforms, device=device, image_scale=args.image_scale, load_image=True)
    init_cam = find_camera_by_filename(cameras, args.init_view)
    print(f"  Loaded {len(cameras)} cameras, init cam = {args.init_view}")

    # Render from all cameras WITHOUT alignment (just MODEL2SCENE)
    # This shows the raw model→scene alignment without similarity transform.
    bg = torch.zeros(3, device=device)
    for i in [0, 8, 16, 24]:
        if i >= len(cameras):
            continue
        cam = cameras[i]
        with torch.no_grad():
            rgb, alpha, _ = render_gaussians(
                means_scene, scales_act, quats_scene, sh0, opac_act,
                cam, background=bg, sh_degree=0)
        render_img = tensor_to_bgr(rgb, alpha)
        gt_img = tensor_to_bgr(cam.image) if cam.image is not None else np.zeros_like(render_img)
        save_side_by_side(render_img, gt_img,
                          os.path.join(args.output_dir, f"02_no_align_cam{i:02d}.png"),
                          f"MODEL2SCENE only cam{i}", f"GT cam{i}")

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 3: Run alignment (estimate_similarity_transform_multiview)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Stage 3: Run alignment (estimate_similarity_transform_multiview)")
    print("=" * 70)

    # Prepare tensors for alignment (in scene frame after MODEL2SCENE)
    scales_act_al = scales_act
    opac_al = opac_act

    scale, R, t = estimate_similarity_transform_multiview(
        tensors=(means_scene, scales_act_al, quats_scene, opac_al, sh0),
        cameras=cameras,
        init_yaw=None,  # Let it do full search
        alpha_threshold=0.1,
        verbose=True,
    )

    print(f"\n  Alignment result: scale={scale:.4f}")
    print(f"  R:\n{R.cpu().numpy()}")
    print(f"  t: {t.cpu().numpy()}")

    # Apply full transform: means_aligned = scale * (R @ means_scene) + t
    means_aligned, scales_aligned, quats_aligned = _apply_transform(
        means_scene, scales_act_al, quats_scene, scale, R, t)
    scales_aligned_raw = torch.log(scales_aligned.clamp_min(1e-8))

    print(f"  Aligned means range: x=[{means_aligned[:,0].min():.3f}, {means_aligned[:,0].max():.3f}]")
    print(f"                       y=[{means_aligned[:,1].min():.3f}, {means_aligned[:,1].max():.3f}]")
    print(f"                       z=[{means_aligned[:,2].min():.3f}, {means_aligned[:,2].max():.3f}]")

    # Save aligned PLY
    ply_path = os.path.join(args.output_dir, "03_aligned_gaussians.ply")
    save_gaussians_ply(ply_path, means_aligned, scales_aligned_raw, quats_aligned, opac_raw, sh0, shN)

    # Save alignment params
    params = {
        "scale": float(scale),
        "R": R.cpu().numpy().tolist(),
        "t": t.cpu().numpy().tolist(),
        "MODEL2SCENE": R_m2s.cpu().numpy().tolist(),
        "init_view": args.init_view,
    }
    with open(os.path.join(args.output_dir, "alignment_params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 4: Render aligned Gaussians from all cameras vs GT
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Stage 4: Render aligned Gaussians vs GT (all views)")
    print("=" * 70)

    comparison_frames = []
    for i, cam in enumerate(cameras):
        with torch.no_grad():
            rgb, alpha, _ = render_gaussians(
                means_aligned, scales_aligned, quats_aligned, sh0, opac_act,
                cam, background=bg, sh_degree=0)
        render_img = tensor_to_bgr(rgb, alpha)
        gt_img = tensor_to_bgr(cam.image) if cam.image is not None else np.zeros_like(render_img)

        # Save individual comparison
        canvas = save_side_by_side(render_img, gt_img,
                                   os.path.join(args.output_dir, f"04_aligned_vs_gt_cam{i:02d}.png"),
                                   f"Aligned render cam{i}", f"GT cam{i}")
        comparison_frames.append(canvas)

        # Print stats
        gt_alpha = cam.alpha if cam.alpha is not None else None
        if gt_alpha is not None:
            gt_cov = (gt_alpha > 0.5).float().mean().item()
            pred_cov = (alpha > 0.5).float().mean().item()
            print(f"  cam{i:02d}: GT coverage={gt_cov:.3f}, Render coverage={pred_cov:.3f}, "
                  f"PSNR region: render_mean={rgb.mean().item():.4f}")

    # Save comparison video
    video_path = os.path.join(args.output_dir, "04_aligned_vs_gt.mp4")
    if comparison_frames:
        H, W = comparison_frames[0].shape[:2]
        vw, video_path = create_video_writer(video_path, "mp4v", 5, (W, H))
        for frame in comparison_frames:
            vw.write(frame)
        vw.release()
        print(f"  Saved video: {video_path}")

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 5: Load existing refined.ply and compare
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Stage 5: Compare with existing refined.ply")
    print("=" * 70)

    refined_path = "refine_output/refined.ply"
    if os.path.exists(refined_path):
        means_ref, scales_ref_raw, quats_ref, opac_ref_raw, sh0_ref, _ = load_gaussians_ply(
            refined_path, device=device)
        scales_ref = scales_ref_raw.exp()
        opac_ref = torch.sigmoid(opac_ref_raw)
        print(f"  Loaded {means_ref.shape[0]} refined Gaussians")

        for i in [0, 8, 16, 24]:
            if i >= len(cameras):
                continue
            cam = cameras[i]
            with torch.no_grad():
                rgb, alpha, _ = render_gaussians(
                    means_ref, scales_ref, quats_ref, sh0_ref, opac_ref,
                    cam, background=bg, sh_degree=0)
            render_img = tensor_to_bgr(rgb, alpha)
            gt_img = tensor_to_bgr(cam.image) if cam.image is not None else np.zeros_like(render_img)
            save_side_by_side(render_img, gt_img,
                              os.path.join(args.output_dir, f"05_refined_vs_gt_cam{i:02d}.png"),
                              f"refined.ply cam{i}", f"GT cam{i}")

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 6: Summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Stage 6: Summary")
    print("=" * 70)
    print(f"  Output directory: {args.output_dir}/")
    print(f"  Key files:")
    print(f"    00_init_image.png          — Raw init image (RGBA)")
    print(f"    00_gt_cam*.png             — Ground truth views")
    print(f"    01_preprocessed.png        — After background removal + crop")
    print(f"    01_initial_gaussians_model.ply — Stage-A output (model frame)")
    print(f"    02_gaussians_scene_frame.ply   — After MODEL2SCENE (Y-up scene)")
    print(f"    02_no_align_cam*.png       — Render with MODEL2SCENE only, no similarity")
    print(f"    03_aligned_gaussians.ply   — After similarity alignment")
    print(f"    04_aligned_vs_gt_cam*.png  — Aligned render vs GT")
    print(f"    04_aligned_vs_gt.mp4       — All views comparison video")
    print(f"    05_refined_vs_gt_cam*.png  — Existing refined.ply vs GT")
    print(f"    alignment_params.json      — Estimated (scale, R, t)")


if __name__ == "__main__":
    main()
