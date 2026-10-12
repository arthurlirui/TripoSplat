"""Run ContinualGS end-to-end on a monocular video stream.

Usage (in the rtg132 conda env with gsplat installed):

    python run_continual_avatar.py \\
        --video_frames static/000001/mvs/images \\
        --transforms static/000001/mvs/transforms_test.json \\
        --init_frame 00.png \\
        --smpl_model_path E:/Code/mmforall/model/assets/SMPL/SMPL_NEUTRAL.pkl \\
        --clip_size 16 --steps_per_clip 500 \\
        --output_dir continual_output

The script (1) warm-starts a drivable avatar from the first frame via the
TripoSplat feed-forward generator, (2) partitions the remaining frames
into clips, and (3) runs the ContinualRefiner loop, saving the avatar PLY
and a replay snapshot after each clip.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from continual_avatar import (
    ContinualRefiner, ReplayBuffer, EWCPrior, VisibilityAccumulator,
    warm_start_avatar,
)
from continual_avatar.continual import RefinerConfig


def make_camera_for_frame(transforms_json, image_dir, filename, image_scale=0.5, device="cuda"):
    """Load a single camera matching ``filename`` from a transforms JSON."""
    from multiview_refine.cameras import load_cameras, find_camera_by_filename
    cams = load_cameras(transforms_json, base_dir=image_dir, device=device,
                        image_scale=image_scale, load_image=True)
    return find_camera_by_filename(cams, filename)


def build_smpl_regressor(device):
    """Build a YOLO11-pose + SMPL regressor backend.

    Returns a callable(image_hwc) -> SMPL params dict. This is a thin shim
    around run_yolo_skeleton_demo-style detection; for a real run, plug in
    SMPLer-X. Here we return a placeholder regressor that raises if used.
    """
    def _regressor(image_hwc):
        raise NotImplementedError(
            "Plug in a concrete SMPLRegressor (e.g. SMPLer-X) here. "
            "See continual_avatar.frame.SMPLRegressor.")
    return _regressor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_frames", required=True, help="dir of monocular frames")
    ap.add_argument("--transforms", required=True, help="transforms_*.json")
    ap.add_argument("--init_frame", default="00.png")
    ap.add_argument("--smpl_model_path", required=True)
    ap.add_argument("--clip_size", type=int, default=16)
    ap.add_argument("--steps_per_clip", type=int, default=500)
    ap.add_argument("--num_init_gaussians", type=int, default=262144)
    ap.add_argument("--image_scale", type=float, default=0.5)
    ap.add_argument("--output_dir", default="continual_output")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    # ── 1. Build TripoSplat generator ──
    from triposplat import TripoSplatPipeline
    pipe = TripoSplatPipeline(device=device)

    def generator(image_path, **kw):
        return pipe.run(image_path, **kw)[0]

    # ── 2. SMPL model ──
    from skinned_gaussian.smpl_model import SMPLWrapper
    smpl = SMPLWrapper(args.smpl_model_path, device=device)

    # ── 3. Warm-start ──
    init_path = os.path.join(args.video_frames, args.init_frame)
    print(f"[warm-start] generating from {init_path}")
    avatar, info = warm_start_avatar(
        generator, dict(image_path=init_path,
                        num_gaussians=args.num_init_gaussians,
                        remove_background=True),
        smpl, device=device)
    print(f"[warm-start] {info}")
    avatar.save_ply(os.path.join(args.output_dir, "warm_start.ply"))

    # ── 4. Build frame list (after the init frame) ──
    import glob
    frames = sorted(glob.glob(os.path.join(args.video_frames, "*.png")))
    frames = [f for f in frames if os.path.basename(f) != args.init_frame]
    if not frames:
        print("No additional frames; done.")
        return

    # ── 5. SMPL regressor for each frame ──
    regressor = build_smpl_regressor(device)
    from continual_avatar.frame import estimate_frame_smpl

    # ── 6. Continual refinement ──
    cfg = RefinerConfig(steps_per_clip=args.steps_per_clip, device=device)
    refiner = ContinualRefiner(avatar, smpl, cfg)
    refiner.setup()

    # partition into clips
    clips = [frames[i:i + args.clip_size] for i in range(0, len(frames), args.clip_size)]
    for ci, clip_paths in enumerate(clips, 1):
        print(f"\n[clip {ci}/{len(clips)}] {len(clip_paths)} frames")
        clip_frames = []
        for fp in clip_paths:
            cam = make_camera_for_frame(args.transforms, args.video_frames,
                                        os.path.basename(fp),
                                        image_scale=args.image_scale, device=device)
            # NOTE: requires a real SMPL regressor; placeholder raises here.
            try:
                f = estimate_frame_smpl(regressor, _load_image(fp), cam, device=device)
            except NotImplementedError as e:
                print(f"  [skip] {e}")
                continue
            clip_frames.append(f)
        if not clip_frames:
            continue
        stats = refiner.refine_clip(clip_frames)
        print(f"  -> {stats}")
        refiner.avatar.save_ply(os.path.join(args.output_dir, f"avatar_clip{ci}.ply"))

    refiner.avatar.save_ply(os.path.join(args.output_dir, "avatar_final.ply"))
    print(f"\nDone. Outputs in {args.output_dir}")


def _load_image(path):
    import numpy as np
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"))


if __name__ == "__main__":
    main()
