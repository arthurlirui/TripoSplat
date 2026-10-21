"""Audit coord_transform.convert for world/camera sync consistency.

Invariant: converting splat + cameras to ANY target convention must not change
what the renderer sees. So we convert, then render with the converted cameras
(re-expressed back to the renderer's OpenCV expectation) and compare pixels.
"""
import sys, math
import numpy as np
import torch

sys.path.insert(0, 'E:/Code/TripoSplat')
from pointsplat_bridge._gsplat_preload import preload_gsplat_cuda
preload_gsplat_cuda()

from multiview_refine.cameras import Camera, opengl_c2w_to_opencv_c2w, invert_se3
from multiview_refine.gsplat_wrapper import render_gaussians
from multiview_refine.coord_transform import convert

dev = 'cuda'
W = H = 180
f = 280.0


def make_cam(pos, look=(0, 0, 0)):
    pos = np.array(pos, float); look = np.array(look, float)
    z = pos - look; z /= np.linalg.norm(z)
    x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w_gl = np.eye(4); c2w_gl[:3, 0] = x; c2w_gl[:3, 1] = y
    c2w_gl[:3, 2] = z; c2w_gl[:3, 3] = pos
    c2w = opengl_c2w_to_opencv_c2w(torch.tensor(c2w_gl, dtype=torch.float32, device=dev))
    w2c = invert_se3(c2w)
    return Camera(w2c=w2c, c2w=c2w, position=c2w[:3, 3], fx=f, fy=f,
                  cx=W / 2, cy=H / 2, width=W, height=H)


torch.manual_seed(1)
N = 500
means = ((torch.rand(N, 3, device=dev) - 0.5) * 0.5)
scales_log = torch.log(torch.full((N, 3), 0.012, device=dev))
quats = torch.nn.functional.normalize(torch.randn(N, 4, device=dev), dim=-1)
opac = torch.full((N,), 0.9, device=dev)
sh0 = torch.zeros(N, 1, 3, device=dev); sh0[..., 0] = 0.8; sh0[..., 1] = 0.35

cams = [make_cam([0, 0.3, 2.2]), make_cam([1.6, 0.2, 1.6]), make_cam([-1.5, 0.4, 1.4])]

with torch.no_grad():
    base = [render_gaussians(means, scales_log.exp(), quats, sh0, opac, c,
                             background=torch.zeros(3, device=dev), sh_degree=0)[0]
            for c in cams]

print('=' * 72)
print('convert: does the render survive a convention change?')
print('=' * 72)

for target in ('opencv', 'opengl', 'splatshop'):
    try:
        out = convert(means, scales_log, quats, cams, target=target,
                                src_world_up='y', device=dev)
        m2 = out['means']; s2 = out['scales'].exp(); q2 = out['quats']
        diffs = []
        for i, c2w_np in enumerate(out['cameras']):
            c2w = torch.tensor(c2w_np, dtype=torch.float32, device=dev)
            if target == 'opencv':
                c2w_cv = c2w
            else:
                # exported as OpenGL camera: flip back to OpenCV for gsplat
                c2w_cv = c2w @ torch.diag(torch.tensor([1., -1., -1., 1.], device=dev))
            w2c = invert_se3(c2w_cv)
            cam2 = Camera(w2c=w2c, c2w=c2w_cv, position=c2w_cv[:3, 3],
                          fx=f, fy=f, cx=W / 2, cy=H / 2, width=W, height=H)
            with torch.no_grad():
                r2 = render_gaussians(m2, s2, q2, sh0, opac, cam2,
                                      background=torch.zeros(3, device=dev), sh_degree=0)[0]
            diffs.append(float((base[i] - r2).abs().max()))
        mx = max(diffs)
        status = 'OK' if mx < 1e-3 else 'MISMATCH'
        print(f'  target={target:10s} per-view max diff={[f"{d:.2e}" for d in diffs]}  -> {status}')
    except Exception as e:
        print(f'  target={target:10s} ERROR: {type(e).__name__}: {e}')

print()
print('=' * 72)
print('export_cameras_json: file_path naming assumption')
print('=' * 72)
print("  writes file_path = 'images/{i:02d}' from the frame INDEX.")
print("  -> For a filtered/reordered camera list this no longer matches the real")
print("     image name. Same class of bug as the run_pipeline_verify index drift.")
