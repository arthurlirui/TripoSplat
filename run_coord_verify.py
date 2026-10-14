"""Coordinate-system verification: render intermediate results as images.

Produces a grid of images showing each coordinate transformation stage so
the user can visually verify correctness:

  1. Refined Gaussians from real cameras (OpenCV scene frame — baseline)
  2. SMPL rest skeleton overlaid on Gaussians (with SMPL2OPENCV rotation)
  3. SMPL rest skeleton WITHOUT rotation (old behavior — shows the misalignment)
  4. SplatShop-converted Gaussians rendered from a SplatShop orbit camera
  5. Side-by-side: OpenCV vs SplatShop world-up comparison

Usage:
    conda run -n rtg132 python run_coord_verify.py
"""

import os
import sys
import math
import argparse

import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Preload cached gsplat CUDA extension before any gsplat import.
from pointsplat_bridge._gsplat_preload import preload_gsplat_cuda
preload_gsplat_cuda()

from multiview_refine.ply_io import load_gaussians_ply
from multiview_refine.cameras import Camera, load_cameras, invert_se3
from multiview_refine.gsplat_wrapper import render_gaussians
from multiview_refine.coord_registry import (
    SMPL2SCENE,
    SCENE2SPLATSHOP_WORLD,
    OPENCV2SPLATSHOP_CAM,
    transform_means,
    transform_quats,
)

SH_C0 = 0.28209479177387814


def tensor_to_cvimg(rgb_tensor, alpha=None):
    """Convert (3, H, W) float tensor to BGR uint8 numpy for cv2."""
    img = (rgb_tensor.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if alpha is not None:
        a = (alpha.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        img = np.concatenate([img, np.stack([a] * 3, axis=-1)], axis=1)
    return img


def add_label(img, label, color=(0, 255, 255)):
    """Draw a label bar at the top of the image."""
    h, w = img.shape[:2]
    bar = np.zeros((40, w, 3), dtype=np.uint8)
    cv2.putText(bar, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return np.concatenate([bar, img], axis=0)


def make_orbit_camera(centre, radius, azim, elev, fx, W, H, device, up=None):
    """Create an OpenCV-convention orbit camera."""
    if up is None:
        up = np.array([0.0, -1.0, 0.0])  # OpenCV: Y-down
    pos = centre + np.array([
        radius * math.cos(elev) * math.sin(azim),
        -radius * math.sin(elev),
        radius * math.cos(elev) * math.cos(azim),
    ], dtype=np.float32)
    forward = centre - pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    down = np.cross(right, forward)
    down = down / np.linalg.norm(down)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = pos
    c2w_t = torch.from_numpy(c2w).to(device)
    w2c = invert_se3(c2w_t)
    return Camera(
        w2c=w2c, c2w=c2w_t, position=torch.from_numpy(pos).to(device),
        fx=fx, fy=fx, cx=W / 2, cy=H / 2, width=W, height=H,
    )


def draw_joints_on_image(img, joints_2d, parents, color=(0, 255, 0)):
    """Draw skeleton joints and bones on a BGR image."""
    for j in range(len(joints_2d)):
        if joints_2d[j] is None:
            continue
        cv2.circle(img, joints_2d[j], 4, color, -1)
        cv2.circle(img, joints_2d[j], 4, (255, 255, 255), 1)
        p = parents[j]
        if p >= 0 and joints_2d[p] is not None:
            cv2.line(img, joints_2d[p], joints_2d[j], color, 2)


def project_joints(joints_3d, cam):
    """Project (J, 3) world-space joints to 2D pixel coordinates."""
    pts = []
    for j in range(joints_3d.shape[0]):
        p = torch.cat([joints_3d[j], torch.tensor([1.0], device=joints_3d.device)])
        cp = (cam.w2c @ p)[:3]
        if cp[2] > 0.01:
            x = int(cam.fx * cp[0] / cp[2] + cam.cx)
            y = int(cam.fy * cp[1] / cp[2] + cam.cy)
            pts.append((x, y))
        else:
            pts.append(None)
    return pts


def main():
    parser = argparse.ArgumentParser(description="Coordinate verification with image outputs.")
    parser.add_argument("--ply", default="refine_output/refined.ply")
    parser.add_argument("--transforms", default="static/000001/mvs/transforms_test.json")
    parser.add_argument("--smpl_model_path", default="E:/Code/mmforall/model/assets/SMPL/SMPL_NEUTRAL.pkl")
    parser.add_argument("--output_dir", default="coord_verify_output")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ═══════════════════════════════════════════════════════════════════════
    # 1. Load refined Gaussians (OpenCV scene frame)
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("1. Loading refined Gaussians (OpenCV scene frame)")
    print("=" * 60)
    means, scales_raw, quats, opac_raw, sh0, shN = load_gaussians_ply(args.ply, device=device)
    scales = scales_raw.exp()
    opac = torch.sigmoid(opac_raw)
    print(f"  {len(means)} Gaussians")
    print(f"  Means range: x=[{means[:,0].min():.3f}, {means[:,0].max():.3f}]")
    print(f"               y=[{means[:,1].min():.3f}, {means[:,1].max():.3f}]")
    print(f"               z=[{means[:,2].min():.3f}, {means[:,2].max():.3f}]")
    gc = means.median(0).values
    print(f"  Centroid: {gc.tolist()}")

    # ═══════════════════════════════════════════════════════════════════════
    # 2. Load real cameras and render from a few viewpoints
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("2. Rendering from real cameras (OpenCV scene frame)")
    print("=" * 60)
    cameras = load_cameras(args.transforms, device=device, image_scale=0.5, load_image=True)
    print(f"  {len(cameras)} cameras loaded")

    bg = torch.zeros(3, device=device)
    real_renders = []
    for i in [0, 8, 16, 24]:
        if i >= len(cameras):
            continue
        with torch.no_grad():
            rgb, alpha, _ = render_gaussians(
                means, scales, quats, sh0, opac, cameras[i],
                background=bg, sh_degree=0)
        img = tensor_to_cvimg(rgb, alpha)
        img = add_label(img, f"Real cam {i} — OpenCV scene frame")
        real_renders.append(img)
        # Also get GT
        if cameras[i].image is not None:
            gt_img = tensor_to_cvimg(cameras[i].image)
            gt_img = add_label(gt_img, f"GT cam {i}")
            real_renders.append(gt_img)
        cv2.imwrite(os.path.join(args.output_dir, f"01_opencv_render_cam{i:02d}.png"), img)

    # ═══════════════════════════════════════════════════════════════════════
    # 3. Load SMPL and align WITHOUT rotation (correct — both Y-up)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("3. SMPL alignment WITHOUT rotation (correct — both Y-up)")
    print("=" * 60)
    from skinned_gaussian.smpl_model import SMPLWrapper

    smpl_correct = SMPLWrapper(args.smpl_model_path, device=device)
    rest_verts, _ = smpl_correct.get_rest_mesh()
    smpl_height = (rest_verts[:, 1].max() - rest_verts[:, 1].min()).item()
    gs_height = (means[:, 1].max() - means[:, 1].min()).item()
    smpl_scale = gs_height / smpl_height if smpl_height > 0.01 else 1.0

    smpl_correct.align_to_scene(scale=smpl_scale, transl=gc)
    rest_joints_correct = smpl_correct.get_rest_joints()
    print(f"  SMPL scale: {smpl_scale:.4f}")
    print(f"  Correct joints Y range: [{rest_joints_correct[:,1].min():.3f}, {rest_joints_correct[:,1].max():.3f}]")
    print(f"  Gaussian Y range:       [{means[:,1].min():.3f}, {means[:,1].max():.3f}]")

    # ═══════════════════════════════════════════════════════════════════════
    # 4. Load SMPL and align WITH wrong rotation (SMPL2OPENCV=diag(1,-1,1) — upside-down)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("4. SMPL alignment WITH diag(1,-1,1) rotation (WRONG — upside-down)")
    print("=" * 60)
    wrong_rot = torch.tensor([[1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]], dtype=torch.float32)
    smpl_wrong = SMPLWrapper(args.smpl_model_path, device=device)
    smpl_wrong.align_to_scene(scale=smpl_scale, rotation=wrong_rot.to(device), transl=gc)
    rest_joints_wrong = smpl_wrong.get_rest_joints()
    print(f"  Wrong joints Y range:   [{rest_joints_wrong[:,1].min():.3f}, {rest_joints_wrong[:,1].max():.3f}]")

    # ═══════════════════════════════════════════════════════════════════════
    # 5. Render SMPL skeleton overlay from orbit cameras
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("5. Rendering SMPL skeleton overlays")
    print("=" * 60)

    centre = gc.cpu().numpy()
    radius = float((means - gc).norm(dim=-1).max().item()) * 2.5
    W, H = 800, 600
    fx = 600.0

    overlay_images = []
    for az_deg in [0, 90, 180, 270]:
        az = math.radians(az_deg)
        elev = math.radians(5)
        cam = make_orbit_camera(centre, radius, az, elev, fx, W, H, device)

        # Render Gaussians
        with torch.no_grad():
            rgb, alpha, _ = render_gaussians(
                means, scales, quats, sh0, opac, cam,
                background=bg, sh_degree=0)
        img_base = tensor_to_cvimg(rgb)

        # Overlay WITHOUT rotation (green skeleton — correct, both Y-up)
        img_correct = img_base.copy()
        joints_2d_correct = project_joints(rest_joints_correct, cam)
        draw_joints_on_image(img_correct, joints_2d_correct, smpl_correct.parents, color=(0, 255, 0))
        img_correct = add_label(img_correct, f"WITHOUT rotation (az={az_deg}°) — green=correct (both Y-up)", (0, 255, 0))
        cv2.imwrite(os.path.join(args.output_dir, f"02_smpl_correct_az{az_deg:03d}.png"), img_correct)
        overlay_images.append(img_correct)

        # Overlay WITH wrong rotation (red skeleton — upside-down)
        img_wrong = img_base.copy()
        joints_2d_wrong = project_joints(rest_joints_wrong, cam)
        draw_joints_on_image(img_wrong, joints_2d_wrong, smpl_wrong.parents, color=(0, 0, 255))
        img_wrong = add_label(img_wrong, f"WITH diag(1,-1,1) rotation (az={az_deg}°) — red=upside-down", (0, 0, 255))
        cv2.imwrite(os.path.join(args.output_dir, f"03_smpl_wrong_az{az_deg:03d}.png"), img_wrong)

        # Side-by-side comparison
        h = max(img_correct.shape[0], img_wrong.shape[0])
        w = img_correct.shape[1] + img_wrong.shape[1]
        comparison = np.zeros((h, w, 3), dtype=np.uint8)
        comparison[:img_correct.shape[0], :img_correct.shape[1]] = img_correct
        comparison[:img_wrong.shape[0], img_correct.shape[1]:] = img_wrong
        cv2.imwrite(os.path.join(args.output_dir, f"04_smpl_comparison_az{az_deg:03d}.png"), comparison)

    # ═══════════════════════════════════════════════════════════════════════
    # 6. SplatShop conversion: Y-up scene → Z-up world
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("6. SplatShop conversion (Y-up scene → Z-up world)")
    print("=" * 60)

    means_ss = transform_means(means, SCENE2SPLATSHOP_WORLD.to(means))
    quats_ss = transform_quats(quats, SCENE2SPLATSHOP_WORLD.to(quats))
    print(f"  Scene Y range:    [{means[:,1].min():.3f}, {means[:,1].max():.3f}]")
    print(f"  SplatShop Z range: [{means_ss[:,2].min():.3f}, {means_ss[:,2].max():.3f}]")
    print(f"  (Y-up body height should become Z-up height)")

    # Create a SplatShop-convention orbit camera (Z-up world, OpenGL camera)
    gc_ss = means_ss.median(0).values
    centre_ss = gc_ss.cpu().numpy()
    radius_ss = float((means_ss - gc_ss).norm(dim=-1).max().item()) * 2.5

    splatshop_images = []
    for az_deg in [0, 90, 180, 270]:
        az = math.radians(az_deg)
        # SplatShop: Z-up world. Camera orbits in the XY plane at slight Z offset.
        # Position in Z-up frame: X=sin(az)*r, Y=cos(az)*r, Z=small_positive
        pos_ss = centre_ss + np.array([
            radius_ss * math.sin(az),
            radius_ss * math.cos(az),
            radius_ss * 0.15,  # slight elevation above centre
        ], dtype=np.float32)

        # Build OpenGL c2w in SplatShop (Z-up) frame: +X right, +Y up, -Z forward
        forward_ss = centre_ss - pos_ss
        forward_ss = forward_ss / np.linalg.norm(forward_ss)
        # In Z-up world, "up" for the camera is +Z
        up_world = np.array([0.0, 0.0, 1.0])
        right_ss = np.cross(forward_ss, up_world)
        right_ss = right_ss / np.linalg.norm(right_ss)
        up_cam_ss = np.cross(right_ss, forward_ss)
        up_cam_ss = up_cam_ss / np.linalg.norm(up_cam_ss)

        # OpenGL c2w: columns = right, up, -forward, position
        c2w_gl = np.eye(4, dtype=np.float32)
        c2w_gl[:3, 0] = right_ss
        c2w_gl[:3, 1] = up_cam_ss
        c2w_gl[:3, 2] = -forward_ss  # OpenGL: -Z forward
        c2w_gl[:3, 3] = pos_ss

        # Convert to OpenCV for gsplat: flip Y and Z camera axes
        c2w_cv = c2w_gl.copy()
        c2w_cv[:3, 1] *= -1  # flip Y column
        c2w_cv[:3, 2] *= -1  # flip Z column
        c2w_cv_t = torch.from_numpy(c2w_cv).to(device)
        w2c_cv = invert_se3(c2w_cv_t)

        cam_ss = Camera(
            w2c=w2c_cv, c2w=c2w_cv_t,
            position=torch.from_numpy(pos_ss).to(device),
            fx=fx, fy=fx, cx=W / 2, cy=H / 2, width=W, height=H,
        )

        with torch.no_grad():
            rgb_ss, alpha_ss, _ = render_gaussians(
                means_ss, scales, quats_ss, sh0, opac, cam_ss,
                background=bg, sh_degree=0)
        img_ss = tensor_to_cvimg(rgb_ss, alpha_ss)
        img_ss = add_label(img_ss, f"SplatShop Z-up (az={az_deg}°) — body should be upright", (255, 255, 0))
        cv2.imwrite(os.path.join(args.output_dir, f"05_splatshop_az{az_deg:03d}.png"), img_ss)
        splatshop_images.append(img_ss)

    # ═══════════════════════════════════════════════════════════════════════
    # 7. Create summary grids
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("7. Creating summary grids")
    print("=" * 60)

    # Grid 1: OpenCV real camera renders
    if real_renders:
        rows = []
        for i in range(0, len(real_renders), 2):
            pair = real_renders[i:i+2]
            h = max(p.shape[0] for p in pair)
            w = sum(p.shape[1] for p in pair)
            row = np.zeros((h, w, 3), dtype=np.uint8)
            x = 0
            for p in pair:
                row[:p.shape[0], x:x+p.shape[1]] = p
                x += p.shape[1]
            rows.append(row)
        grid = np.vstack(rows)
        cv2.imwrite(os.path.join(args.output_dir, "grid_01_opencv_renders.png"), grid)
        print(f"  Saved grid_01_opencv_renders.png ({grid.shape[1]}x{grid.shape[0]})")

    # Grid 2: SMPL overlay comparison (with rot vs no rot)
    if overlay_images:
        grid2 = np.hstack(overlay_images[:2])  # az=0 and az=90
        cv2.imwrite(os.path.join(args.output_dir, "grid_02_smpl_overlays.png"), grid2)
        print(f"  Saved grid_02_smpl_overlays.png")

    # Grid 3: SplatShop renders
    if splatshop_images:
        grid3 = np.hstack(splatshop_images[:2])
        cv2.imwrite(os.path.join(args.output_dir, "grid_03_splatshop.png"), grid3)
        print(f"  Saved grid_03_splatshop.png")

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {args.output_dir}/")
    print(f"  01_opencv_render_cam*.png  — Gaussians from real cameras (Y-up scene)")
    print(f"  02_smpl_correct_az*.png    — SMPL skeleton WITHOUT rotation (correct, both Y-up)")
    print(f"  03_smpl_wrong_az*.png      — SMPL skeleton WITH diag(1,-1,1) (wrong, upside-down)")
    print(f"  04_smpl_comparison_az*.png — Side-by-side: correct vs wrong")
    print(f"  05_splatshop_az*.png       — SplatShop Z-up conversion")
    print(f"  grid_*.png                 — Summary grids")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
