"""Audit the skinned_gaussian / SMPL coordinate path end-to-end."""
import sys, numpy as np, torch

sys.path.insert(0, 'E:/Code/TripoSplat')
from pointsplat_bridge._gsplat_preload import preload_gsplat_cuda
preload_gsplat_cuda()

from skinned_gaussian.smplx_model import SMPLXWrapper
from multiview_refine.gsplat_wrapper import render_gaussians
from multiview_refine.cameras import Camera, opengl_c2w_to_opencv_c2w, invert_se3
from PIL import Image

SMPLX = 'E:/Code/mmforall/model/assets/SMPLX/SMPLX_NEUTRAL_2020.npz'
dev = 'cuda'

print('=' * 72)
print('AUDIT: SMPL-X frame vs scene frame')
print('=' * 72)

sx = SMPLXWrapper(SMPLX, device=dev)
v = sx.get_rest_mesh(betas=torch.zeros(sx.NUM_BETAS, device=dev))[0].cpu().numpy()
feet = float(v[v[:, 1] < np.percentile(v[:, 1], 5)][:, 1].mean())
head = float(v[v[:, 1] > np.percentile(v[:, 1], 95)][:, 1].mean())
print(f'  SMPL-X rest mesh: feetY={feet:.3f}  headY={head:.3f}  -> '
      f'{"upright (Y-up) OK" if head > feet else "INVERTED (BUG)"}')
print(f'  height = {head-feet:.3f} m')
print()
print('  NOTE: SMPL2SCENE is identity, and every demo calls align_to_scene(scale=..., transl=...)')
print('        WITHOUT rotation. That is self-consistent ONLY because the SMPL-X template is')
print('        already Y-up — same as the scene. Do not "fix" it with a diag(1,-1,1) flip.')

print()
print('=' * 72)
print('AUDIT: rigging / skinning round-trip at rest pose')
print('=' * 72)
try:
    from skinned_gaussian.skinning import lbs_deform_gaussians
    means = v[:2000].copy()
    quats = torch.tensor(np.tile([1., 0, 0, 0], (2000, 1)), dtype=torch.float32, device=dev)
    scales = torch.full((2000, 3), 0.01, device=dev)
    W_J = sx.get_rest_joints(betas=torch.zeros(sx.NUM_BETAS, device=dev))
    # identity pose -> joint_transforms should be identity
    T = torch.eye(4, device=dev).repeat(sx.NUM_JOINTS, 1, 1)
    weights = sx.get_skinning_weights()
    w = torch.zeros(2000, sx.NUM_JOINTS, device=dev)
    w[:, 0] = 1.0
    mp, sp, qp = lbs_deform_gaussians(
        torch.tensor(means, device=dev), scales, quats, w, T)
    dpos = float((mp - torch.tensor(means, device=dev)).abs().max())
    print(f'  rest-pose LBS invariance: max |mean| shift = {dpos:.3e} -> '
          f'{"OK" if dpos < 1e-5 else "BUG"}')
except Exception as e:
    print(f'  skinning audit error: {type(e).__name__}: {e}')

print()
print('=' * 72)
print('AUDIT: demo scripts load PLY from a stale relative path')
print('=' * 72)
import os
for p in ['scratch_output/refined_pruned.ply', 'output/scratch_output/refined_pruned.ply']:
    print(f'  {p:44s} exists={os.path.exists(p)}')
print('  -> demos default to the nonexistent path; run only works with an explicit --gaussians.')
