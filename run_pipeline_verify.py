"""Full pipeline coordinate verification: run step1+2+3 with fresh data and
save intermediate comparison images at every stage.

Usage:
    conda run -n rtg132 python run_pipeline_verify.py [--init_view 00.png]
"""

import argparse
import json
import math
import os
import re
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
from multiview_refine.alignment import (
    estimate_similarity_transform_multiview,
    align_from_init_view,
    _apply_transform,
)
from multiview_refine.video_io import create_video_writer
from multiview_refine.orientation import uprightness_report, assert_upright

SH_C0 = 0.28209479177387814


def _resolve_image_path_from_camera(cam, transforms_json: str) -> str | None:
    """Resolve the on-disk image path for a camera's ``file_path``.

    ``file_path`` in transforms.json is extension-less (e.g. ``images/02``).
    """
    from multiview_refine.cameras import _resolve_image_path
    base_dir = os.path.dirname(os.path.abspath(transforms_json))
    if not cam.file_path:
        return None
    return _resolve_image_path(base_dir, cam.file_path)


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
    parser.add_argument("--init_view", default="00.png",
                        help="Image used to initialise TripoSplat.  Must be a "
                             "FULL-BODY view whose alpha does not touch the image "
                             "border — images/02,05,16 etc. are cropped close-ups "
                             "and produce a torso-only model with no legs.")
    parser.add_argument("--image_dir", default="static/000001/mvs/images")
    parser.add_argument("--transforms", default="static/000001/mvs/transforms_train.json")
    parser.add_argument("--output_dir", default="output/pipeline_verify_output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_gaussians", type=int, default=131072)
    parser.add_argument("--image_scale", type=float, default=0.5)
    parser.add_argument("--refined_ply", default=None,
                        help="Optional refined PLY to compare in stage 5.")
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

    # Warn if the init image is a cropped close-up.  TripoSplat can only
    # hallucinate what it can see: a torso-only input yields a torso-only
    # avatar (no legs), which then cannot be aligned to full-body GT.
    if init_img.shape[-1] == 4:
        _a = init_img[:, :, 3] > 128
        if _a.any():
            _touch = [bool(_a[0].any()), bool(_a[-1].any()),
                      bool(_a[:, 0].any()), bool(_a[:, -1].any())]
            if any(_touch):
                edges = [n for n, t in zip(("top", "bottom", "left", "right"), _touch) if t]
                print(f"  ⚠️  init image foreground touches the {'/'.join(edges)} border — "
                      "this is a cropped view. The generated avatar will be missing "
                      "whatever is cut off (e.g. no legs). Prefer a full-body view.")
            else:
                print("  ✅ init image contains a complete, uncropped subject")

    # Save a few GT views.
    # IMPORTANT: cameras[i] is NOT images/{i:02d}.  transforms_train.json
    # starts at images/02, so the camera-list index and the filename drift
    # apart.  Stages 2 and 4 below render `cameras[i]` and label the panel
    # "cam{i}", so the stage-0 reference images MUST be pulled from the same
    # camera's file_path — otherwise the side-by-side pairs two different
    # viewpoints and the comparison is meaningless.
    _cam_list = load_cameras(args.transforms, device=device,
                             image_scale=1.0, load_image=False)
    for i in [0, 8, 16, 24]:
        if i >= len(_cam_list):
            continue
        gt_path = _resolve_image_path_from_camera(_cam_list[i], args.transforms)
        if gt_path is None or not os.path.exists(gt_path):
            print(f"  [warn] GT for camera {i} ({_cam_list[i].file_path}) not found")
            continue
        gt_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        if gt_img.shape[-1] == 4:
            gt_disp = np.concatenate([gt_img[:, :, :3], np.stack([gt_img[:, :, 3]] * 3, axis=-1)], axis=1)
        else:
            gt_disp = gt_img
        fname = os.path.basename(_cam_list[i].file_path)
        cv2.imwrite(os.path.join(args.output_dir, f"00_gt_cam{i:02d}.png"), gt_disp)
        print(f"  GT cam{i:02d} <- {fname}.png")

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
    # The init view is usually NOT in transforms_train.json — a good full-body
    # frame is typically held out into transforms_val.  Its pose is essential
    # for the analytic alignment, so search every split before giving up.
    init_cam = None
    split_path = args.transforms
    for _split in ("val", "test", "train"):
        _p = re.sub(r"transforms_(train|val|test)\.json$", f"transforms_{_split}.json",
                    args.transforms)
        if not os.path.exists(_p):
            continue
        try:
            _cams = load_cameras(_p, device=device, image_scale=args.image_scale,
                                 load_image=True)
            init_cam = find_camera_by_filename(_cams, args.init_view)
            split_path = _p
            print(f"  Init view {args.init_view} found in {os.path.basename(_p)} "
                  f"(idx {init_cam.cam_idx}, position={init_cam.position.tolist()})")
            break
        except ValueError:
            continue
    if init_cam is None:
        print(f"  ⚠️  Init view {args.init_view} is not calibrated in any split — "
              "analytic alignment unavailable, will fall back to silhouette search.")

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
        gt_name = os.path.basename(cam.file_path) or f"cam{i}"
        save_side_by_side(render_img, gt_img,
                          os.path.join(args.output_dir, f"02_no_align_cam{i:02d}.png"),
                          f"MODEL2SCENE only cam{i}", f"GT {gt_name}")

    # ═══════════════════════════════════════════════════════════════════════
    # Stage 3: Run alignment (estimate_similarity_transform_multiview)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Stage 3: Run alignment (estimate_similarity_transform_multiview)")
    print("=" * 70)

    # Prepare tensors for alignment
    scales_act_al = scales_act
    opac_al = opac_act

    # ── Primary: analytic alignment anchored on the generation view ──
    # The Gaussians were generated FROM args.init_view, so that view's camera
    # defines their canonical pose.  Since MODEL2SCENE puts the avatar facing a
    # known scene axis, the rotation is solvable in closed form, and the scale
    # follows from the subject's apparent pixel height.  This replaces the
    # silhouette grid search, which empirically preferred inflated scales
    # (a 2.6 m "person") and could not tell front from back.
    scale = R = t = None
    if init_cam is not None:
        print("  Using analytic init-view alignment (no silhouette search)")
        scale, R, t = align_from_init_view(
            means_model,            # raw AABB frame; MODEL2SCENE folds into R
            init_cam,
            cameras,
            init_mask=init_cam.alpha,
            verbose=True,
        )
    else:
        # Fallback: the init view has no calibration in transforms_train.json.
        print("  Init view not calibrated — falling back to silhouette search")
        fwd = None
        scale, R, t = estimate_similarity_transform_multiview(
            tensors=(means_scene, scales_act_al, quats_scene, opac_al, sh0),
            cameras=cameras,
            init_yaw=None,
            alpha_threshold=0.1,
            verbose=True,
        )

    print(f"\n  Alignment result: scale={scale:.4f}")
    print(f"  R:\n{R.cpu().numpy()}")
    print(f"  t: {t.cpu().numpy()}")

    # Apply full transform.  The analytic path returns R in the *model* frame
    # (already folding in MODEL2SCENE), so apply it to means_model; the
    # fallback path expects the MODEL2SCENE-rotated means.
    if init_cam is not None:
        means_aligned, scales_aligned, quats_aligned = _apply_transform(
            means_model, scales_act_al, quats_raw, scale, R, t)
    else:
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
    pred_masks, real_masks = [], []
    for i, cam in enumerate(cameras):
        with torch.no_grad():
            rgb, alpha, _ = render_gaussians(
                means_aligned, scales_aligned, quats_aligned, sh0, opac_act,
                cam, background=bg, sh_degree=0)
        render_img = tensor_to_bgr(rgb, alpha)
        gt_img = tensor_to_bgr(cam.image) if cam.image is not None else np.zeros_like(render_img)

        # Collect silhouettes for the uprightness gate.
        pred_masks.append((alpha.squeeze(0) > 0.1).cpu().numpy())
        if cam.alpha is not None:
            real_masks.append((cam.alpha.squeeze(0) > 0.5).cpu().numpy())

        # Save individual comparison.  Label with the real filename so the
        # panel can be traced back to an actual image.
        gt_name = os.path.basename(cam.file_path) or f"cam{i}"
        canvas = save_side_by_side(render_img, gt_img,
                                   os.path.join(args.output_dir, f"04_aligned_vs_gt_cam{i:02d}.png"),
                                   f"Aligned render cam{i}", f"GT {gt_name}")
        comparison_frames.append(canvas)

        # Print stats
        gt_alpha = cam.alpha if cam.alpha is not None else None
        if gt_alpha is not None:
            gt_cov = (gt_alpha > 0.5).float().mean().item()
            pred_cov = (alpha > 0.5).float().mean().item()
            print(f"  cam{i:02d} ({gt_name}): GT coverage={gt_cov:.3f}, "
                  f"Render coverage={pred_cov:.3f}")

    # ── Uprightness gate ──
    # Silhouette IoU cannot reject a 180° flip (an upright figure and its
    # vertical mirror overlap ~0.53), so check orientation explicitly before
    # trusting any IoU number.
    rep = uprightness_report(pred_masks, real_masks or None)
    print(f"\n  Uprightness: {rep['frac_upright']:.0%} of {rep['n_judged']} rendered "
          f"silhouettes are head-up"
          + (f" (GT reference: {rep['real_frac_upright']:.0%})"
             if real_masks else ""))
    with open(os.path.join(args.output_dir, "uprightness.json"), "w") as f:
        json.dump(rep, f, indent=2)
    try:
        assert_upright(pred_masks, real_masks or None)
        print("  ✅ Orientation check passed")
    except AssertionError as exc:
        print(f"  ❌ {exc}")

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
    # Stage 5 (+7): Load refined PLY and compare against the same views
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Stage 5: Compare refined Gaussians against GT")
    print("=" * 70)

    # `refined_ply` is either an explicit --refined_ply or the per-run output
    # of run_alignment_refine_test.py.  The old hard-coded
    # "refine_output/refined.ply" was missing the leading "output/", so this
    # whole stage silently no-oped and the refined model was never rendered.
    refined_candidates = [
        args.refined_ply,
        os.path.join("output", "alignment_refine_test", "refined_aligned.ply"),
        os.path.join("output", "refine_output", "refined.ply"),
    ]
    refined_path = next((p for p in refined_candidates if p and os.path.exists(p)), None)
    if refined_path is None:
        print("  [skip] no refined PLY found. Looked for:")
        for p in refined_candidates:
            if p:
                print(f"           {p}")
    else:
        means_ref, scales_ref_raw, quats_ref, opac_ref_raw, sh0_ref, _ = load_gaussians_ply(
            refined_path, device=device)
        scales_ref = scales_ref_raw.exp()
        opac_ref = torch.sigmoid(opac_ref_raw)
        print(f"  Loaded {means_ref.shape[0]} refined Gaussians from {refined_path}")

        ref_masks = []
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
            gt_name = os.path.basename(cam.file_path) or f"cam{i}"
            save_side_by_side(render_img, gt_img,
                              os.path.join(args.output_dir, f"05_refined_vs_gt_cam{i:02d}.png"),
                              f"refined cam{i}", f"GT {gt_name}")
            ref_masks.append((alpha.squeeze(0) > 0.1).cpu().numpy())

        rep_ref = uprightness_report(ref_masks)
        print(f"  Uprightness (refined): {rep_ref['frac_upright']:.0%} head-up, "
              f"flip_suspected={rep_ref['flip_suspected']}")

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
