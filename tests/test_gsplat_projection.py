"""Verify gsplat projection correctness with a synthetic camera + points.

Places Gaussians at known 3D positions, renders them, and checks that the
rendered peaks land at the expected pixel coordinates computed independently
via the pinhole model: u = fx * x / z + cx, v = fy * y / z + cy.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import math
import torch

from multiview_refine.cameras import Camera, invert_se3
from multiview_refine.gsplat_wrapper import render_rgb, render_alpha


def make_camera(W, H, fx, fy, cx, cy, position, look_at, up=(0, 1, 0), device="cuda"):
    """Create an OpenCV-convention camera looking at a target point.

    OpenCV: +X right, +Y down, +Z forward.
    """
    position = torch.tensor(position, dtype=torch.float32, device=device)
    look_at = torch.tensor(look_at, dtype=torch.float32, device=device)
    up = torch.tensor(up, dtype=torch.float32, device=device)

    forward = look_at - position
    forward = forward / forward.norm()
    right = torch.cross(forward, up)
    right = right / right.norm()
    down = torch.cross(right, forward)
    down = down / down.norm()

    c2w = torch.eye(4, dtype=torch.float32, device=device)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = position

    w2c = invert_se3(c2w)
    return Camera(
        w2c=w2c, c2w=c2w, position=position,
        fx=fx, fy=fy, cx=cx, cy=cy, width=W, height=H,
    )


def expected_pixel(cam, point3d):
    """Pinhole: u = fx * x_cam / z_cam + cx, v = fy * y_cam / z_cam + cy."""
    p_view = cam.w2c[:3, :3] @ point3d + cam.w2c[:3, 3]
    u = cam.fx * p_view[0] / p_view[2] + cam.cx
    v = cam.fy * p_view[1] / p_view[2] + cam.cy
    return u.item(), v.item()


def test_point(cam, point3d, label):
    """Render a single Gaussian at point3d and check the peak pixel."""
    W, H = cam.width, cam.height
    means = point3d.unsqueeze(0)
    scales = torch.tensor([[0.05, 0.05, 0.05]], dtype=torch.float32, device=cam.device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=cam.device)
    colors = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=cam.device)
    opacities = torch.tensor([1.0], dtype=torch.float32, device=cam.device)

    # Use raw colours (sh_degree=None)
    img = render_rgb(means, scales, quats, colors, opacities, cam, sh_degree=None)
    assert img.shape == (3, H, W), f"Unexpected shape {img.shape}"

    peak_idx = torch.argmax(img[0].flatten()).item()
    peak_y, peak_x = peak_idx // W, peak_idx % W

    exp_u, exp_v = expected_pixel(cam, point3d)
    error = math.sqrt((peak_x - exp_u) ** 2 + (peak_y - exp_v) ** 2)
    status = "✓" if error < 3.0 else "✗"
    print(f"  {status} {label}: peak ({peak_x}, {peak_y})  expected ({exp_u:.1f}, {exp_v:.1f})  error={error:.2f}px")
    return error < 3.0


def main():
    device = "cuda"
    W, H = 800, 600
    fx = fy = 500.0
    cx, cy = W / 2, H / 2

    # Camera at [0, 0, 5] looking at origin, up = world +Y
    cam = make_camera(W, H, fx, fy, cx, cy,
                      position=[0, 0, 5], look_at=[0, 0, 0], up=[0, 1, 0], device=device)

    print("=== Projection accuracy test ===")
    ok1 = test_point(cam, torch.tensor([0.0, 0.0, 0.0], device=device), "center")
    ok2 = test_point(cam, torch.tensor([1.0, -0.5, 0.0], device=device), "off-center (1, -0.5, 0)")
    ok3 = test_point(cam, torch.tensor([-0.8, 0.3, 0.5], device=device), "off-center (-0.8, 0.3, 0.5)")

    print("\n=== Alpha render test ===")
    means = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    scales = torch.tensor([[0.1, 0.1, 0.1]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    opacities = torch.tensor([1.0], device=device)
    alpha = render_alpha(means, scales, quats, opacities, cam)
    print(f"  alpha shape: {alpha.shape}, max: {alpha.max().item():.4f}")

    print("\n=== Gradient flow test ===")
    from multiview_refine.gsplat_wrapper import render_gaussians
    means_g = torch.tensor([[0.0, 0.0, 0.0]], device=device, requires_grad=True)
    grad_colors = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    rgb, _, _ = render_gaussians(means_g, scales, quats, grad_colors, opacities, cam, sh_degree=None)
    loss = rgb.sum()
    loss.backward()
    grad_ok = means_g.grad is not None and means_g.grad.abs().sum() > 0
    print(f"  means.grad = {means_g.grad}")
    print(f"  gradient flows to means: {'✓' if grad_ok else '✗'}")

    print("\n=== SH degree 0 test ===")
    sh_colors = torch.tensor([[[0.5, 0.0, 0.0]]], device=device)  # (1, 1, 3)
    img_sh = render_rgb(means, scales, quats, sh_colors, opacities, cam, sh_degree=0)
    peak_val = img_sh[0, 299, 399].item()
    expected_sh = 0.5 * 0.28209479177387814 + 0.5  # SH DC coefficient * C0 + 0.5
    sh_ok = abs(peak_val - expected_sh) < 0.01
    print(f"  SH peak R: {peak_val:.4f}  expected: {expected_sh:.4f}  {'✓' if sh_ok else '✗'}")

    all_ok = ok1 and ok2 and ok3 and grad_ok and sh_ok
    print(f"\n{'='*40}")
    print(f"Overall: {'ALL PASSED ✓' if all_ok else 'FAILED ✗'}")
    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
