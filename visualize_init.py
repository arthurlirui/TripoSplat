"""Visualize the initial Gaussians, camera poses, and training images.

Shows two scenes side by side (or sequentially):
  1. TripoSplat initial Gaussians (in their normalized AABB frame) + the
     init-view camera position.
  2. Aligned Gaussians in the real scene coordinate frame + all multi-view
     camera poses + training images as billboards (textured quads at each
     camera's position).

Controls:
  - Mouse: rotate / pan / zoom
  - R:    reset viewpoint
  - +/-:  change point size
  - I:    toggle training image billboards
  - C:    toggle camera frustums
  - P:    toggle point cloud
  - Q/ESC: quit

Usage:
    python visualize_init.py \\
        --ply output/refine_output/refined.ply \\
        --transforms static/000001/mvs/transforms_train.json \\
        --image_dir static/000001/mvs/images \\
        --image_scale 0.25
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "."))

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from multiview_refine.cameras import load_cameras, find_camera_by_filename

SH_C0 = 0.28209479177387814


# ── PLY loading ────────────────────────────────────────────────────────────

def load_ply_colors(ply_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load means, RGB colors, and opacities from a 3DGS PLY file.

    Returns:
        means (N, 3), colors (N, 3) in [0, 1], opacities (N,) in [0, 1].
    """
    from plyfile import PlyData
    plydata = PlyData.read(ply_path)
    v = plydata["vertex"]
    data = np.stack([v[name] for name in v.data.dtype.names], axis=1)
    names = list(v.data.dtype.names)

    def col(name):
        return data[:, names.index(name)]

    means = np.stack([col("x"), col("y"), col("z")], axis=-1)
    f_dc = np.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], axis=-1)
    colors = (f_dc * SH_C0 + 0.5).clip(0, 1)
    opacities = 1.0 / (1.0 + np.exp(-col("opacity")))  # sigmoid
    return means, colors, opacities


# ── Camera geometry ────────────────────────────────────────────────────────

def make_frustum_lines(c2w: np.ndarray, fx: float, fy: float,
                       cx: float, cy: float, W: int, H: int,
                       frustum_scale: float = 0.3) -> np.ndarray:
    """Create line segments for a camera frustum (5 lines: 4 edges + apex).

    Args:
        c2w: (4, 4) camera-to-world in OpenCV convention (+Y down, +Z forward).
        fx, fy, cx, cy: intrinsics.
        W, H: image dimensions.
        frustum_scale: size of the frustum relative to image plane.

    Returns:
        (10, 3) array of line endpoints (5 line segments × 2 points).
    """
    # Camera centre in world.
    centre = c2w[:3, 3]

    # Four corners of the image plane in camera space.
    # Using the pinhole model: x_cam = (u - cx) * z / fx, y_cam = (v - cy) * z / fy
    z = frustum_scale
    corners_cam = np.array([
        [(0 - cx) * z / fx, (0 - cy) * z / fy, z],          # top-left
        [(W - cx) * z / fx, (0 - cy) * z / fy, z],          # top-right
        [(W - cx) * z / fx, (H - cy) * z / fy, z],          # bottom-right
        [(0 - cx) * z / fx, (H - cy) * z / fy, z],          # bottom-left
    ], dtype=np.float32)

    # Transform to world: p_world = R @ p_cam + t
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    corners_world = (R @ corners_cam.T).T + t

    # Build line segments: centre→each corner + corners in order (frustum edges)
    lines = np.array([
        centre, corners_world[0],  # centre → top-left
        centre, corners_world[1],  # centre → top-right
        centre, corners_world[2],  # centre → bottom-right
        centre, corners_world[3],  # centre → bottom-left
        corners_world[0], corners_world[1],  # top edge
        corners_world[1], corners_world[2],  # right edge
        corners_world[2], corners_world[3],  # bottom edge
        corners_world[3], corners_world[0],  # left edge
    ], dtype=np.float32)
    return lines


def make_frustum_line_set(cameras: list, frustum_scale: float = 0.3) -> o3d.geometry.LineSet:
    """Build a LineSet of all camera frustums."""
    all_points = []
    all_lines = []
    all_colors = []
    for i, cam in enumerate(cameras):
        c2w = cam.c2w.cpu().numpy()
        lines = make_frustum_lines(
            c2w, cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height,
            frustum_scale=frustum_scale,
        )
        base = len(all_points)
        all_points.extend(lines.tolist())
        # 8 line segments per camera
        for j in range(8):
            all_lines.append([base + j * 2, base + j * 2 + 1])
        # Color: green for all, red for init camera
        color = [1.0, 0.0, 0.0] if i == 0 else [0.0, 1.0, 0.0]
        all_colors.extend([color] * 8)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(all_points)
    ls.lines = o3d.utility.Vector2iVector(all_lines)
    ls.colors = o3d.utility.Vector3dVector(all_colors)
    return ls


def make_image_billboard(c2w: np.ndarray, image: np.ndarray,
                         fx: float, fy: float, cx: float, cy: float,
                         W: int, H: int, depth: float = 0.3) -> o3d.geometry.TriangleMesh:
    """Create a textured quad (billboard) showing the training image at the camera position.

    The quad is placed at distance `depth` in front of the camera, sized to
    match the image aspect ratio.
    """
    # Four corners in camera space (same as frustum but filled quad)
    corners_cam = np.array([
        [(0 - cx) * depth / fx, (0 - cy) * depth / fy, depth],
        [(W - cx) * depth / fx, (0 - cy) * depth / fy, depth],
        [(W - cx) * depth / fx, (H - cy) * depth / fy, depth],
        [(0 - cx) * depth / fx, (H - cy) * depth / fy, depth],
    ], dtype=np.float32)

    R = c2w[:3, :3]
    t = c2w[:3, 3]
    corners_world = (R @ corners_cam.T).T + t

    # Build a TriangleMesh with vertex colors sampled from the image.
    # Open3D 0.19 doesn't support per-triangle UV easily, so we use a
    # grid of vertices to approximate the image texture via vertex colors.
    mesh = o3d.geometry.TriangleMesh()
    # Use a grid of n×n vertices for better color representation
    n = 8
    h_img, w_img = image.shape[:2]
    vertices = []
    vertex_colors = []
    for vi in range(n + 1):
        for ui in range(n + 1):
            u = int(ui * (w_img - 1) / n)
            v_ = int(vi * (h_img - 1) / n)
            # Camera-space corner for this grid point
            x_cam = (u - cx) * depth / fx
            y_cam = (v_ - cy) * depth / fy
            p_cam = np.array([x_cam, y_cam, depth], dtype=np.float32)
            p_world = R @ p_cam + t
            vertices.append(p_world)
            vertex_colors.append(image[v_, u] / 255.0)

    # Build triangles for the grid
    triangles = []
    for vi in range(n):
        for ui in range(n):
            idx = vi * (n + 1) + ui
            idx_r = idx + 1
            idx_d = idx + (n + 1)
            idx_dr = idx_d + 1
            triangles.append([idx, idx_r, idx_d])
            triangles.append([idx_r, idx_dr, idx_d])

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh.compute_vertex_normals()
    return mesh


# ── Point cloud ────────────────────────────────────────────────────────────

def make_point_cloud(means: np.ndarray, colors: np.ndarray,
                     opacities: np.ndarray | None = None,
                     min_opacity: float = 0.05) -> o3d.geometry.PointCloud:
    """Create an Open3D PointCloud from Gaussian means and colors."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(means)

    if opacities is not None:
        # Modulate color by opacity for visualization
        opac = np.clip(opacities, min_opacity, 1.0)[:, None]
        colors = colors * opac

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# ── Main visualization ────────────────────────────────────────────────────

def run_offscreen_render(means, colors, cameras, output_path, image_dir,
                          image_scale, show_billboards=True, show_frustums=True,
                          width=1920, height=1080):
    """Render a view to an image file using the legacy Visualizer (Windows-compatible)."""

    # Build geometries
    pcd = make_point_cloud(means, colors)
    frustums = make_frustum_line_set(cameras, frustum_scale=0.3)

    all_geometries = [pcd, frustums]

    if show_billboards:
        for i, cam in enumerate(cameras):
            if cam.image is None:
                continue
            img_np = (cam.image.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            c2w = cam.c2w.cpu().numpy()
            bb = make_image_billboard(
                c2w, img_np, cam.fx, cam.fy, cam.cx, cam.cy,
                cam.width, cam.height, depth=0.3,
            )
            all_geometries.append(bb)

    # Use the legacy Visualizer for Windows compatibility
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    for geom in all_geometries:
        vis.add_geometry(geom)

    # Set viewpoint
    ctr = vis.get_view_control()
    gc = means.mean(axis=0)
    ctr.set_front([1, 1, 1])
    ctr.set_up([0, -1, 0])
    ctr.set_lookat(gc.tolist())
    ctr.set_zoom(0.5)

    # Point size
    render_opt = vis.get_render_option()
    render_opt.point_size = 3
    render_opt.background_color = np.array([0.1, 0.1, 0.1])
    render_opt.line_width = 2

    vis.poll_events()
    vis.update_renderer()

    # Capture image
    img = vis.capture_screen_float_buffer(do_render=True)
    img = (np.asarray(img) * 255).astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    vis.destroy_window()
    print(f"Saved: {output_path}")


def run_gui_visualization(means, colors, opacities, cameras, image_dir, image_scale):
    """Run the interactive Open3D visualization (Windows-compatible legacy visualizer)."""

    # Build geometries
    pcd = make_point_cloud(means, colors, opacities)
    frustums = make_frustum_line_set(cameras, frustum_scale=0.3)

    geometries = [pcd, frustums]

    # Image billboards
    for i, cam in enumerate(cameras):
        if cam.image is None:
            continue
        img_np = (cam.image.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        c2w = cam.c2w.cpu().numpy()
        bb = make_image_billboard(
            c2w, img_np, cam.fx, cam.fy, cam.cx, cam.cy,
            cam.width, cam.height, depth=0.3,
        )
        geometries.append(bb)

    # Coordinate frame at the Gaussian centroid
    gc = means.mean(axis=0)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=gc)
    geometries.append(coord_frame)

    print(f"\nVisualizing {len(means)} Gaussians + {len(cameras)} cameras")
    print("Controls:")
    print("  Mouse left:  rotate")
    print("  Mouse middle/shift: pan")
    print("  Mouse wheel: zoom")
    print("  Q/ESC: quit")

    # Use the legacy visualizer with key callbacks
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Gaussian Splatting Visualizer", 1600, 900)

    for geom in geometries:
        vis.add_geometry(geom)

    # Render options
    render_opt = vis.get_render_option()
    render_opt.point_size = 3
    render_opt.background_color = np.array([0.1, 0.1, 0.1])
    render_opt.line_width = 2

    # Set initial viewpoint
    ctr = vis.get_view_control()
    ctr.set_front([1, 1, 1])
    ctr.set_up([0, -1, 0])
    ctr.set_lookat(gc.tolist())
    ctr.set_zoom(0.4)

    # Key callbacks
    def reset_view(vis):
        ctr = vis.get_view_control()
        ctr.set_front([1, 1, 1])
        ctr.set_up([0, -1, 0])
        ctr.set_lookat(gc.tolist())
        ctr.set_zoom(0.4)
        return False

    def increase_point_size(vis):
        opt = vis.get_render_option()
        opt.point_size = min(opt.point_size + 1, 20)
        return False

    def decrease_point_size(vis):
        opt = vis.get_render_option()
        opt.point_size = max(opt.point_size - 1, 1)
        return False

    def save_screenshot(vis):
        import time
        path = f"output/refine_output/screenshot_{int(time.time())}.png"
        vis.capture_screen_image(path)
        print(f"Saved: {path}")
        return False

    vis.register_key_callback(ord("R"), reset_view)
    vis.register_key_callback(ord("+"), increase_point_size)
    vis.register_key_callback(ord("="), increase_point_size)
    vis.register_key_callback(ord("-"), decrease_point_size)
    vis.register_key_callback(ord("S"), save_screenshot)

    print("  R: reset view  +/-: point size  S: screenshot  Q/ESC: quit")
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize initial Gaussians, camera poses, and training images."
    )
    parser.add_argument("--ply", default="output/refine_output/refined.ply",
                        help="Path to the Gaussian PLY file.")
    parser.add_argument("--transforms", default="static/000001/mvs/transforms_train.json",
                        help="Path to transforms.json.")
    parser.add_argument("--image_dir", default=None,
                        help="Directory with training images (defaults to transforms dir).")
    parser.add_argument("--image_scale", type=float, default=0.25,
                        help="Downscale factor for loaded images.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offscreen", action="store_true",
                        help="Save a rendered image instead of opening GUI.")
    parser.add_argument("--offscreen_output", default="output/refine_output/visualization.png",
                        help="Output path for offscreen render.")
    parser.add_argument("--frustum_scale", type=float, default=0.3,
                        help="Size of camera frustums.")
    parser.add_argument("--prune_dark", action="store_true", default=True,
                        help="Prune dark opaque Gaussians for better visualization.")
    args = parser.parse_args()

    # ── Load Gaussians ──
    print(f"Loading Gaussians from {args.ply}...")
    means, colors, opacities = load_ply_colors(args.ply)
    print(f"  {len(means)} Gaussians")
    print(f"  means range: [{means.min():.3f}, {means.max():.3f}]")
    print(f"  colours range: [{colors.min():.3f}, {colors.max():.3f}]")
    print(f"  opacity range: [{opacities.min():.3f}, {opacities.max():.3f}]")

    if args.prune_dark:
        dark = colors.max(axis=-1) < 0.01
        high_opac = opacities > 0.5
        prune = dark & high_opac
        keep = ~prune
        print(f"  Pruning {prune.sum()} dark opaque Gaussians ({100*prune.mean():.1f}%)")
        means, colors, opacities = means[keep], colors[keep], opacities[keep]
        print(f"  Remaining: {len(means)}")

    # ── Load cameras ──
    print(f"\nLoading cameras from {args.transforms}...")
    base_dir = os.path.dirname(os.path.abspath(args.transforms))
    if args.image_dir is None:
        args.image_dir = base_dir
    cameras = load_cameras(args.transforms, base_dir=base_dir,
                           device=args.device, image_scale=args.image_scale,
                           load_image=True)
    print(f"  {len(cameras)} cameras loaded")
    print(f"  Camera positions range: "
          f"x=[{min(c.position[0].item() for c in cameras):.2f}, {max(c.position[0].item() for c in cameras):.2f}], "
          f"y=[{min(c.position[1].item() for c in cameras):.2f}, {max(c.position[1].item() for c in cameras):.2f}], "
          f"z=[{min(c.position[2].item() for c in cameras):.2f}, {max(c.position[2].item() for c in cameras):.2f}]")

    # ── Run visualization ──
    if args.offscreen:
        print("\nRendering offscreen image...")
        run_offscreen_render(
            means, colors, cameras, args.offscreen_output,
            args.image_dir, args.image_scale,
            show_billboards=True, show_frustums=True,
        )
    else:
        print("\nStarting GUI visualization...")
        print("Controls:")
        print("  Mouse: rotate / pan / zoom")
        print("  Q/ESC: quit")
        print("  Use checkboxes to toggle point cloud, frustums, billboards")
        run_gui_visualization(means, colors, opacities, cameras,
                              args.image_dir, args.image_scale)


if __name__ == "__main__":
    main()
