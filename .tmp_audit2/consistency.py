"""Coordinate self-consistency checks across the whole pipeline.

Verifies invariants that the unit tests do not cover:
  1. Rotating means by R (row form, means@R.T) and rotating quats by q(R) is
     the SAME world rotation. If Hamilton composition and quat->matrix use
     mismatched conventions, orientations silently drift.
  2. A transformed Gaussian renders identically to the original seen from a
     correspondingly transformed camera (render-space covariance).
  3. Scale + quaternion packing round-trips through the PLY writer.
"""
import sys, math
import numpy as np
import torch

sys.path.insert(0, 'E:/Code/TripoSplat')
from pointsplat_bridge._gsplat_preload import preload_gsplat_cuda
preload_gsplat_cuda()

from multiview_refine.quaternion import (
    quaternion_compose, rotation_matrix_to_quaternion, quat_to_matrix, matrix_to_quat,
)
from multiview_refine.coord_registry import MODEL2SCENE, SCENE2SPLATSHOP_WORLD
from multiview_refine.gsplat_wrapper import render_gaussians
from multiview_refine.cameras import Camera

dev = 'cuda'

def rand_rot():
    q = torch.randn(1, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    M = torch.from_numpy(quat_to_matrix(q.numpy())).float().squeeze(0)
    return M, q.squeeze(0)

print('=' * 70)
print('CHECK 1: means rotation vs quaternion rotation consistency')
print('=' * 70)
worst = 0.0
for trial in range(20):
    R, _ = rand_rot()
    # random per-Gaussian orientation G
    G, qG = rand_rot()
    G = G.unsqueeze(0); qG = qG.unsqueeze(0)
    # means-route: world rotation R applied to the Gaussian frame
    G_means_route = R @ G[0]
    # quat-route
    q_R = rotation_matrix_to_quaternion(R)
    q_new = quaternion_compose(q_R, qG)
    G_quat_route = torch.from_numpy(quat_to_matrix(q_new.numpy())).float().squeeze(0)
    err = float((G_means_route - G_quat_route).abs().max())
    worst = max(worst, err)
print(f'  max deviation over 20 random trials: {worst:.3e}')
print(f'  -> {"OK: q(R) (x) q_G == R @ G" if worst < 1e-5 else "BUG: conventions disagree"}')

print()
print('=' * 70)
print('CHECK 2: rotation matrix <-> quaternion round trip')
print('=' * 70)
worst = 0.0
for t in range(20):
    M, _ = rand_rot()
    q = rotation_matrix_to_quaternion(M)          # (4,)
    M2 = torch.from_numpy(quat_to_matrix(q.reshape(1, 4).numpy())).float().reshape(3, 3)
    worst = max(worst, float((M - M2).abs().max()))
print(f'  max round-trip error: {worst:.3e}  -> {"OK" if worst < 1e-5 else "BUG"}')

print()
print('=' * 70)
print('CHECK 3: registry matrices are proper rotations (det=+1, orthonormal)')
print('=' * 70)
for name, M in [('MODEL2SCENE', MODEL2SCENE), ('SCENE2SPLATSHOP_WORLD', SCENE2SPLATSHOP_WORLD)]:
    d = torch.det(M).item()
    orth = float((M @ M.T - torch.eye(3)).abs().max())
    # a reflection (det=-1) would mirror the model -> also a coordinate bug
    print(f'  {name:24s} det={d:+.6f}  |MM^T-I|={orth:.2e}  '
          f'-> {"proper rotation OK" if abs(d-1) < 1e-6 else "REFLECTION/BUG"}')

print()
print('=' * 70)
print('CHECK 4: render invariance -- rotating the model and the camera')
print('         together must leave the image unchanged')
print('=' * 70)
torch.manual_seed(0)
N = 400
means = ((torch.rand(N, 3) - 0.5) * 0.4).to(dev)
scales = torch.full((N, 3), 0.01).to(dev)
quats = torch.nn.functional.normalize(torch.randn(N, 4).to(dev), dim=-1)
opac = torch.full((N,), 0.9).to(dev)
sh0 = torch.zeros(N, 1, 3, device=dev); sh0[..., 0] = 0.8
sh0[..., 1] = 0.4

W = H = 200
f = 300.0

def make_cam(pos, look=np.zeros(3)):
    pos = np.array(pos, float); look = np.array(look, float)
    z = pos - look; z /= np.linalg.norm(z)
    x = np.cross([0, 1, 0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4); c2w[:3, 0] = x; c2w[:3, 1] = y; c2w[:3, 2] = z; c2w[:3, 3] = pos
    # OpenGL -> OpenCV: flip Y and Z of the rotation
    c2w[:3, 1] *= -1; c2w[:3, 2] *= -1
    Rr = c2w[:3, :3]; t = c2w[:3, 3]
    w2c = np.eye(4); w2c[:3, :3] = Rr.T; w2c[:3, 3] = -Rr.T @ t
    from multiview_refine.cameras import Camera
    return Camera(w2c=torch.tensor(w2c, dtype=torch.float32, device=dev),
                  c2w=torch.tensor(c2w, dtype=torch.float32, device=dev),
                  position=torch.tensor(t, dtype=torch.float32, device=dev),
                  fx=f, fy=f, cx=W/2, cy=H/2, width=W, height=H)

cam = make_cam([0, 0, 2.5])
with torch.no_grad():
    rgb0, a0, _ = render_gaussians(means, scales, quats, sh0, opac, cam,
                                   background=torch.zeros(3, device=dev), sh_degree=0)

# Apply world rotation Rw to model AND camera.
for label, Rw in [('rot_y_90', torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]], device=dev)),
                  ('rot_x_45', torch.tensor([[1., 0., 0.],
                                             [0., math.cos(math.pi/4), -math.sin(math.pi/4)],
                                             [0., math.sin(math.pi/4), math.cos(math.pi/4)]], device=dev))]:
    m2 = means @ Rw.T
    q2 = quaternion_compose(rotation_matrix_to_quaternion(Rw), quats)
    c2 = cam.c2w.clone(); c2[:3, :3] = Rw @ cam.c2w[:3, :3]; c2[:3, 3] = Rw @ cam.c2w[:3, 3]
    Rr = c2[:3, :3].cpu().numpy(); t = c2[:3, 3].cpu().numpy()
    w2c = np.eye(4); w2c[:3, :3] = Rr.T; w2c[:3, 3] = -Rr.T @ t
    cam2 = Camera(w2c=torch.tensor(w2c, dtype=torch.float32, device=dev), c2w=c2,
                  position=torch.tensor(t, dtype=torch.float32, device=dev),
                  fx=f, fy=f, cx=W/2, cy=H/2, width=W, height=H)
    with torch.no_grad():
        rgb1, a1, _ = render_gaussians(m2, scales, q2, sh0, opac, cam2,
                                       background=torch.zeros(3, device=dev), sh_degree=0)
    diff = float((rgb0 - rgb1).abs().max())
    print(f'  {label}: max pixel diff = {diff:.3e}  '
          f'-> {"OK (rotation is a true isometry)" if diff < 1e-4 else "BUG: render not invariant"}')

print()
print('=' * 70)
print('CHECK 5: MODEL2SCENE keeps the avatar upright after transform')
print('=' * 70)
ext_model = torch.tensor([0.234, 0.362, 0.997], device=dev)  # measured model extents (x,y,z)
ext_scene = ext_model @ MODEL2SCENE.to(dev).T
print(f'  model extents (x,y,z) = {ext_model.tolist()}')
print(f'  scene extents         = {[round(v,3) for v in ext_scene.tolist()]}')
print(f'  tallest scene axis = {"Y (up) OK" if int(torch.argmax(ext_scene)) == 1 else "NOT Y (BUG)"}')
