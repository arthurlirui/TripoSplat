"""Is the registration yaw failure caused by the MODEL2SCENE fix?

Builds EXACTLY the synthetic scene from
tests/test_registration.py::test_registration_recovers_synthetic_transform,
then runs register_gaussians_to_scene twice: once with the corrected
MODEL2SCENE and once with the old (transposed) matrix patched in.
"""
import sys, math
import numpy as np
import torch

sys.path.insert(0, 'E:/Code/TripoSplat')
from pointsplat_bridge._gsplat_preload import preload_gsplat_cuda
preload_gsplat_cuda()

import multiview_refine.registration as reg
from multiview_refine.coord_registry import MODEL2SCENE
from multiview_refine.gsplat_wrapper import render_alpha
from multiview_refine.quaternion import quaternion_compose, rotation_matrix_to_quaternion
from multiview_refine.registration import rotation_y, register_gaussians_to_scene

dev = 'cuda'
OLD = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # pre-fix
NEW = MODEL2SCENE.clone()
print(f'NEW MODEL2SCENE =\n{NEW}\nOLD =\n{OLD}\n')

# ── rebuild the test's synthetic scene verbatim ──
sys.path.insert(0, 'E:/Code/TripoSplat/tests')
from test_registration import orbit_cameras

torch.manual_seed(0)
N = 4000
z = torch.rand(N) - 0.5
r = 0.03 + 0.09 * (0.5 - z)
theta = torch.rand(N) * 2 * math.pi
xy = torch.stack([torch.cos(theta) * r, torch.sin(theta) * r * 0.6], dim=1)
means = torch.cat([xy, z[:, None]], dim=1)
scales = torch.full((N, 3), 0.015)
quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(N, 1)
opac = torch.full((N,), 0.95)

s_true, yaw_true = 1.4, math.radians(-70.0)


def run(M2S, label):
    reg.MODEL2SCENE = M2S          # patch the module-level reference
    if hasattr(reg, 'MODEL2SCENE'):
        pass
    R_true = (torch.from_numpy(rotation_y(yaw_true)).float().to(dev) @ M2S.to(dev))
    t_true = torch.tensor([0.1, 0.0, -0.15], device=dev)

    means_w = s_true * (means.to(dev) @ R_true.T) + t_true
    q_w = quaternion_compose(rotation_matrix_to_quaternion(R_true), quats.to(dev))

    cams = [c.to(dev) for c in orbit_cameras(10, radius=4.0, height=0.5)]
    for cam in cams:
        with torch.no_grad():
            a = render_alpha(means_w, scales.to(dev) * s_true, q_w, opac.to(dev), cam)
        cam.alpha = a.detach()

    res = register_gaussians_to_scene(
        means.to(dev), scales.to(dev), quats.to(dev), opac.to(dev), cams, verbose=False)
    yaw_err = abs(((math.degrees(res.yaw) - math.degrees(yaw_true) + 180) % 360) - 180)
    print(f'[{label}]')
    print(f'  scale = {res.scale:.4f}  (true {s_true}, err {abs(res.scale-s_true)/s_true*100:.1f}%)')
    print(f'  yaw   = {math.degrees(res.yaw):8.2f}deg  (true {math.degrees(yaw_true):.2f}, err {yaw_err:.1f})')
    print(f'  -> {"PASS" if yaw_err < 10 and abs(res.scale-s_true)/s_true < 0.08 else "FAIL"}')
    return dict(label=label, scale=res.scale, yaw_err=yaw_err)


a = run(NEW, 'with FIXED MODEL2SCENE')
b = run(OLD, 'with OLD (transposed) MODEL2SCENE')
print()
print('CONCLUSION:', 'both fail -> pre-existing, NOT caused by the fix'
      if a['yaw_err'] > 10 and b['yaw_err'] > 10 else
      ('only the fixed one fails -> REGRESSION' if a['yaw_err'] > 10 else
       'only the old one fails -> the fix actually improves it'))
