"""Fresh process: load ONLY the JSON calibration and verify it against real GT.

Proves a stored transform is self-sufficient — no solver, no prior state.
"""
import sys
import numpy as np
import torch

sys.path.insert(0, 'E:/Code/TripoSplat')
from pointsplat_bridge._gsplat_preload import preload_gsplat_cuda
preload_gsplat_cuda()

from multiview_refine.cameras import load_cameras
from multiview_refine.ply_io import load_gaussians_ply
from multiview_refine.gsplat_wrapper import render_alpha, render_gaussians
from multiview_refine.alignment import _get_real_masks, _compute_iou
from multiview_refine.orientation import uprightness_report
from multiview_refine.transform_io import load_transform

dev = 'cuda'; SC = 0.5
JSON = 'output/calibration/alignment_000001.json'
RAW = 'output/pipeline_fixed/01_initial_gaussians_model.ply'

print('=' * 70)
print('INDEPENDENT PROCESS: load calibration JSON only')
print('=' * 70)

tfm = load_transform(JSON)            # validates on load
print(f'  {tfm.summary()}')
print(f'  source={tfm.source_frame}  target={tfm.target_frame}  init_view={tfm.init_view}')
print(f'  convention: xyz = scale * (R @ xyz_model) + t')

m0, s_raw, q0, o_raw, sh0, _ = load_gaussians_ply(RAW, device=dev)
s_act = s_raw.exp(); o_act = torch.sigmoid(o_raw)
q0n = torch.nn.functional.normalize(q0, dim=-1)

# The whole point: apply the STORED transform, don't solve anything.
mt, st, qt = tfm.apply(m0, s_act, q0n)

height = float(mt[:, 1].max() - mt[:, 1].min())
print(f'\n  applied -> subject height {height:.2f} m, '
      f'Y extent {height:.3f}, top axis = {["X","Y","Z"][int(torch.argmax(mt.max(0).values-mt.min(0).values))]}')

cams = load_cameras('static/000001/mvs/transforms_train.json', device=dev,
                    image_scale=SC, load_image=True)
masks = _get_real_masks(cams)
bg = torch.zeros(3, device=dev)

ious, psnrs, preds = [], [], []
for i, c in enumerate(cams):
    with torch.no_grad():
        a = render_alpha(mt, st, qt, o_act, c)
        rgb, _, _ = render_gaussians(mt, st, qt, sh0, o_act, c, background=bg, sh_degree=0)
    sil = (a.squeeze(0) > 0.1).float(); rm = masks[i]
    if sil.shape != rm.shape:
        sil = torch.nn.functional.interpolate(
            sil[None, None], size=rm.shape, mode='bilinear', align_corners=False).squeeze()
    ious.append(_compute_iou(sil, rm)); preds.append(sil.cpu().numpy())
    mse = torch.nn.functional.mse_loss(rgb, c.image)
    psnrs.append(float(10 * torch.log10(1.0 / mse)))

rep = uprightness_report(preds, [(m > 0.5).cpu().numpy() for m in masks])
rec_iou = float(np.mean(ious)); rec_psnr = float(np.mean(psnrs))

print(f'\n  IoU   = {rec_iou:.4f}   (recorded {tfm.metrics.get("iou"):.4f})')
print(f'  PSNR  = {rec_psnr:.2f} dB')
print(f'  upright = {rep["frac_upright"]:.0%}  (recorded {tfm.metrics.get("upright"):.0%})')
print(f'  flip_suspected = {rep["flip_suspected"]}')

ok = (abs(rec_iou - tfm.metrics["iou"]) < 1e-6
      and abs(height - tfm.metrics["subject_height_m"]) < 1e-3
      and not rep["flip_suspected"])
print(f'\n  -> {"PASS: JSON alone reproduces the calibration exactly" if ok else "FAIL"}')
