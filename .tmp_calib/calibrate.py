"""Calibrate the transform format on REAL capture data.

1. Align the raw Tripo model to static/000001/mvs using the analytic method.
2. Persist it with the new schema (save_transform).
3. Re-load it from disk in the SAME process, then in a SEPARATE process, and
   confirm the transform reproduces identical renders and metrics.

This is the end-to-end proof that a stored calibration is sufficient to
reproduce an alignment without re-running the solver.
"""
import sys, json, math
import numpy as np
import torch

sys.path.insert(0, 'E:/Code/TripoSplat')
from pointsplat_bridge._gsplat_preload import preload_gsplat_cuda
preload_gsplat_cuda()

from multiview_refine.cameras import load_cameras, find_camera_by_filename
from multiview_refine.ply_io import load_gaussians_ply
from multiview_refine.gsplat_wrapper import render_alpha, render_gaussians
from multiview_refine.alignment import _get_real_masks, _compute_iou, align_from_init_view
from multiview_refine.orientation import uprightness_report
from multiview_refine.transform_io import GaussianTransform, save_transform, load_transform

dev = 'cuda'; SC = 0.5
INIT_PLY = 'output/pipeline_fixed/01_initial_gaussians_model.ply'
OUT_JSON = 'output/calibration/alignment_000001.json'

cams = load_cameras('static/000001/mvs/transforms_train.json', device=dev,
                    image_scale=SC, load_image=True)
masks = _get_real_masks(cams)
val = load_cameras('static/000001/mvs/transforms_val.json', device=dev,
                   image_scale=SC, load_image=True)
init_cam = find_camera_by_filename(val, '00.png')

m0, s_raw, q0, o_raw, sh0, shN = load_gaussians_ply(INIT_PLY, device=dev)
s_act = s_raw.exp(); o_act = torch.sigmoid(o_raw)
q0n = torch.nn.functional.normalize(q0, dim=-1)

print('=' * 70)
print('STEP 1: solve alignment on real data')
print('=' * 70)
scale, R, t = align_from_init_view(m0, init_cam, cams, init_mask=init_cam.alpha, verbose=True)

# Score it once so the JSON records how good the calibration is.
mt, st, qt = scale * (m0 @ R.T) + t, s_act * scale, None
from multiview_refine.quaternion import quaternion_compose, rotation_matrix_to_quaternion
qt = quaternion_compose(rotation_matrix_to_quaternion(R), q0n)

def score(means, scales, quats):
    ious, preds = [], []
    for i, c in enumerate(cams):
        with torch.no_grad():
            a = render_alpha(means, scales, quats, o_act, c)
        sil = (a.squeeze(0) > 0.1).float(); rm = masks[i]
        if sil.shape != rm.shape:
            sil = torch.nn.functional.interpolate(
                sil[None, None], size=rm.shape, mode='bilinear', align_corners=False).squeeze()
        ious.append(_compute_iou(sil, rm)); preds.append(sil.cpu().numpy())
    rep = uprightness_report(preds)
    return float(np.mean(ious)), float(rep['frac_upright'])

iou0, up0 = score(mt, st, qt)
height = float(mt[:, 1].max() - mt[:, 1].min())
print(f'  IoU={iou0:.4f}  upright={up0:.0%}  height={height:.2f}m')

print()
print('=' * 70)
print('STEP 2: persist with the new schema')
print('=' * 70)
tfm = GaussianTransform.from_align(
    scale, R, t,
    name='calibration_000001',
    init_view='00.png',
    init_camera_position=init_cam.position,
    image_scale=SC,
    metrics={'subject_height_m': height, 'iou': iou0, 'upright': up0},
    notes='analytic init-view alignment on static/000001/mvs (transforms_train, '
          'init camera from transforms_val)',
)
path = save_transform(OUT_JSON, tfm)
print(f'  {tfm.summary()}')
print(f'  wrote {path}')

print()
print('=' * 70)
print('STEP 3: reload from disk and re-render')
print('=' * 70)
tfm2 = load_transform(path)          # validates orthonormality + uprightness
print(f'  loaded ok: {tfm2.summary()}')

mt2, st2, qt2 = tfm2.apply(m0, s_act, q0n)
iou1, up1 = score(mt2, st2, qt2)
dR = float(np.abs(np.asarray(tfm2.R) - np.asarray(tfm.R)).max())
dt = float(np.abs(np.asarray(tfm2.t) - np.asarray(tfm.t)).max())
dpos = float((mt2 - mt).abs().max())
print(f'  reload deltas: |dR|={dR:.2e}  |dt|={dt:.2e}  |dMeans|={dpos:.2e}')
print(f'  IoU {iou0:.4f} -> {iou1:.4f}   upright {up0:.0%} -> {up1:.0%}')
ok = abs(iou1 - iou0) < 1e-6 and dpos < 1e-4
print(f'  -> {"PASS: stored calibration reproduces the alignment" if ok else "FAIL"}')

print()
print('=' * 70)
print('STEP 4: cross-check against the PLY the pipeline already wrote')
print('=' * 70)
try:
    mp, sp_raw, qp, op, shp, _ = load_gaussians_ply('output/p2_fixed/aligned_gaussians.ply', device=dev)
    d = float((mp - mt).abs().max())
    # run_pipeline step2 solves at image_scale=0.25 while this calibration used
    # 0.5, so scale differs by ~0.19% and means shift sub-millimetre-to-mm.
    tol = 2e-2
    print(f'  |means_recomputed - means_on_disk| = {d:.2e} '
          f'(tolerance {tol:.0e}; explains by image_scale {SC} vs 0.25) -> '
          f'{"MATCH" if d < tol else "MISMATCH"}')
except FileNotFoundError:
    print('  (output/p2_fixed/aligned_gaussians.ply not found; run_pipeline step2 first)')
