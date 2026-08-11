"""Core multi-view refinement orchestration.

This module ties together the full pipeline:
  1. Run TripoSplat on a single init-view image to get initial Gaussians.
  2. Load multi-view cameras + images from a Nerfstudio transforms JSON.
  3. Estimate a 7-DoF similarity transform to align the init Gaussians to
     the scene coordinate system (depth + silhouette ICP).
  4. Optimize the Gaussians via gsplat differentiable rasterization with
     adaptive density control (gsplat DefaultStrategy).
  5. Export the refined Gaussians to PLY / SPLAT.
"""

from __future__ import annotations

import os
import random
import time
from typing import Optional

import torch

from .cameras import (
    Camera,
    load_cameras,
    find_camera_by_filename,
    compute_cameras_extent,
    split_train_test,
)
from .gaussian_model import RefineableGaussians
from .gsplat_wrapper import render_gaussians
from .losses import compute_loss, compute_psnr
from .alignment import estimate_similarity_transform


def refine_from_multiview(
    pipe,
    image_dir: str,
    transforms_json: str,
    init_view: str,
    num_init_gaussians: int = 262144,
    num_iterations: int = 30000,
    image_scale: float = 0.5,
    output_dir: str = "refine_output",
    seed: int = 42,
    lambda_l1: float = 0.8,
    lambda_dssim: float = 0.2,
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    refine_start_iter: int = 500,
    refine_stop_iter: int = 15000,
    refine_every: int = 100,
    reset_every: int = 3000,
    densify_grad_threshold: float = 0.0002,
    sh_degree: int = 0,
    save_interval: int = 1000,
    verbose: bool = True,
) -> RefineableGaussians:
    """Run the full multi-view refinement pipeline.

    Args:
        pipe: A ``TripoSplatPipeline`` instance (for init generation + depth).
        image_dir: Directory containing the multi-view images.
        transforms_json: Path to ``transforms_{train,val,test}.json``.
        init_view: Filename of the init-view image (e.g. ``"00.png"``).
        num_init_gaussians: Number of Gaussians for TripoSplat init generation.
        num_iterations: Total optimization iterations.
        image_scale: Downscale factor for training images (1.0 = full res).
        output_dir: Where to save outputs (PLY, SPLAT, comparison images).
        seed: Random seed.
        lambda_l1, lambda_dssim: Loss weights.
        background_color: RGB background for rendering (tuple in [0,1]).
        refine_start_iter, refine_stop_iter: Densification start/stop iterations.
        refine_every: Densification interval (iterations).
        reset_every: Opacity reset interval (iterations).
        densify_grad_threshold: Gradient threshold for clone/split.
        sh_degree: SH degree (0 = DC only).
        save_interval: Save intermediate PLY every N iterations.
        verbose: Print progress.

    Returns:
        The refined :class:`RefineableGaussians`.
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    device = pipe.device if hasattr(pipe, "device") else "cuda"

    # ── Phase 1: Single-image initialization ──
    if verbose:
        print("=" * 60)
        print("Phase 1: TripoSplat single-image initialization")
        print("=" * 60)

    init_image_path = os.path.join(image_dir, init_view)
    tripo_gaussian, prepared = pipe.run(
        init_image_path,
        num_gaussians=num_init_gaussians,
        remove_background=True,
        show_progress=verbose,
    )
    if verbose:
        print(f"  Generated {tripo_gaussian.get_xyz.shape[0]} initial Gaussians")

    # ── Phase 2: Load multi-view cameras + images ──
    if verbose:
        print("\n" + "=" * 60)
        print("Phase 2: Loading multi-view cameras")
        print("=" * 60)

    cameras = load_cameras(
        transforms_json,
        base_dir=os.path.dirname(os.path.abspath(transforms_json)),
        device=device,
        image_scale=image_scale,
        load_image=True,
    )
    if verbose:
        print(f"  Loaded {len(cameras)} cameras (image_scale={image_scale})")
        print(f"  Image resolution: {cameras[0].width}x{cameras[0].height}")

    init_camera = find_camera_by_filename(cameras, init_view)
    if verbose:
        print(f"  Init camera: cam_idx={init_camera.cam_idx}, file_path={init_camera.file_path}")

    # ── Phase 3: Multi-view silhouette alignment ──
    if verbose:
        print("\n" + "=" * 60)
        print("Phase 3: Multi-view silhouette alignment")
        print("  Render init Gaussians from all views → compare with real images")
        print("=" * 60)

    # Get background removal function for silhouette extraction.
    rmbg_fn = None
    if hasattr(pipe, "rmbg") and pipe.rmbg is not None:
        rmbg_fn = pipe.rmbg.remove_background

    scale, R, t = estimate_similarity_transform(
        tripo_gaussian,
        init_camera,
        init_camera.image,
        depth_estimator=None,
        rmbg_fn=rmbg_fn,
        cameras=cameras,
        alpha_threshold=0.1,
        verbose=verbose,
    )

    # ── Phase 4: Create refineable Gaussians ──
    if verbose:
        print("\n" + "=" * 60)
        print("Phase 4: Creating refineable Gaussians")
        print("=" * 60)

    model = RefineableGaussians.from_tripo_splat(
        tripo_gaussian, (scale, R, t), sh_degree=sh_degree, device=device
    )
    if verbose:
        print(f"  {model.n_gaussians} Gaussians in scene coordinates")

    cameras_extent = compute_cameras_extent(cameras)
    if verbose:
        print(f"  Cameras extent (scene scale): {cameras_extent:.4f}")

    # Configure the densification strategy.
    from gsplat.strategy import DefaultStrategy
    strategy = DefaultStrategy(
        prune_opa=0.005,
        grow_grad2d=densify_grad_threshold,
        grow_scale3d=0.01,
        grow_scale2d=0.05,
        prune_scale3d=0.1,
        prune_scale2d=0.15,
        refine_start_iter=refine_start_iter,
        refine_stop_iter=refine_stop_iter,
        refine_every=refine_every,
        reset_every=reset_every,
        verbose=verbose,
    )
    model.training_setup(scene_scale=cameras_extent, strategy=strategy)

    # ── Phase 5: Optimization loop ──
    if verbose:
        print("\n" + "=" * 60)
        print("Phase 5: Multi-view optimization")
        print("=" * 60)
        print(f"  Iterations: {num_iterations}")
        print(f"  Densification: iter {refine_start_iter}–{refine_stop_iter}, every {refine_every}")
        print(f"  Opacity reset: every {reset_every}")

    train_cameras, test_cameras = split_train_test(cameras, init_view)
    if verbose:
        print(f"  Train cameras: {len(train_cameras)}, Test cameras: {len(test_cameras)}")

    bg = torch.tensor(background_color, dtype=torch.float32, device=device)

    for iteration in range(1, num_iterations + 1):
        model.train()
        model.update_learning_rate(iteration)

        # Random training view.
        cam = random.choice(train_cameras)
        gt = cam.image  # (3, H, W)

        # Render.
        colors = model.get_colors()  # (N, K, 3) SH coeffs
        rgb, alpha, meta = render_gaussians(
            model.means,
            model.scales,       # activated
            model.raw_quats,    # raw (normalised internally by gsplat)
            colors,
            model.opacities,    # activated
            cam,
            background=bg,
            sh_degree=sh_degree,
        )

        # Loss.
        loss = compute_loss(rgb, gt, lambda_l1=lambda_l1, lambda_dssim=lambda_dssim)

        # Strategy: pre-backward (retain grad on means2d).
        info = dict(meta)
        info["step"] = iteration
        model.step_pre_backward(info)

        # Backward.
        loss.backward()

        # Strategy: post-backward (densification + opacity reset).
        model.step_post_backward(info, step=iteration, packed=True)

        # Optimizer step.
        model.optimizer_step()
        model.optimizer_zero_grad()

        # Logging.
        if verbose and (iteration % 500 == 0 or iteration == 1):
            with torch.no_grad():
                # Evaluate on a test camera (or a random train camera if no test).
                eval_cam = test_cameras[0] if test_cameras else cam
                eval_rgb, _, _ = render_gaussians(
                    model.means, model.scales, model.raw_quats,
                    model.get_colors(), model.opacities,
                    eval_cam, background=bg, sh_degree=sh_degree,
                )
                eval_gt = eval_cam.image
                psnr_val = compute_psnr(eval_rgb, eval_gt).item()
                print(
                    f"  iter {iteration:5d}: loss={loss.item():.4f}, "
                    f"psnr={psnr_val:.2f}, n_gs={model.n_gaussians}"
                )

        # Save intermediate.
        if save_interval > 0 and iteration % save_interval == 0:
            model.save_ply(os.path.join(output_dir, f"refined_iter{iteration}.ply"))

    # ── Phase 6: Export ──
    if verbose:
        print("\n" + "=" * 60)
        print("Phase 6: Exporting refined Gaussians")
        print("=" * 60)

    final_ply = os.path.join(output_dir, "refined.ply")
    final_splat = os.path.join(output_dir, "refined.splat")
    model.save_ply(final_ply)
    model.save_splat(final_splat)

    if verbose:
        print(f"  PLY: {final_ply} ({model.n_gaussians} Gaussians)")
        print(f"  SPLAT: {final_splat}")
        print("\nDone!")

    return model
