"""Prototype 2: given up axis (+Y prior), scale and translation, is yaw identifiable?

Tests the IoU-vs-yaw curve on synthetic ground truth, and checks whether the
true yaw is the global maximum (i.e. front/back is not ambiguous).
"""
import sys, os, math
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pointsplat_bridge._gsplat_preload import preload_gsplat_cuda
preload_gsplat_cuda()
from multiview_refine.cameras import load_cameras
from multiview_refine.ply_io import load_gaussians_ply
from multiview_refine.gsplat_wrapper import render_alpha
from multiview_refine.coord_registry import MODEL2SCENE
from multiview_refine.quaternion import quaternion_compose, rotation_matrix_to_quaternion

dev = "cuda"
cams = load_cameras("static/000001/mvs/transforms_train.json", device=dev, image_scale=0.5, load_image=True)
m, s_raw, q, o, sh0, _ = load_gaussians_ply("output/gaussian_output/000001/gaussians.ply", device=dev)
scales, opac = s_raw.exp(), torch.sigmoid(o)
qn = torch.nn.functional.normalize(q, dim=-1)


def rot_y(a):
    c, sn = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0, sn], [0, 1, 0], [-sn, 0, c]], dtype=torch.float32, device=dev)


S_STAR = 1.70
R_STAR = rot_y(math.radians(35.0)) @ MODEL2SCENE.to(dev)
T_STAR = torch.tensor([0.05, 0.02, -0.08], device=dev)
H_MODEL = float((m @ R_STAR.T)[:, 1].max() - (m @ R_STAR.T)[:, 1].min())

means_gt = S_STAR * (m @ R_STAR.T) + T_STAR
q_gt = quaternion_compose(rotation_matrix_to_quaternion(R_STAR), qn)
masks = []
for c in cams:
    with torch.no_grad():
        a = render_alpha(means_gt, scales * S_STAR, q_gt, opac, c)
    masks.append((a.squeeze(0) > 0.5))

clipped = []
for mk in masks:
    ys, xs = torch.nonzero(mk, as_tuple=True)
    clipped.append(bool(ys.min() == 0 or ys.max() == mk.shape[0] - 1
                        or xs.min() == 0 or xs.max() == mk.shape[1] - 1))
usable = [i for i in range(len(cams)) if not clipped[i]]
print(f"unclipped views: {len(usable)}/{len(cams)}")

# ---- per-view metric height estimate (robust) ----
heights = []
for i in usable:
    mk = masks[i]
    ys, xs = torch.nonzero(mk, as_tuple=True)
    c = cams[i]
    z = float((c.c2w[:3, :3].T @ (torch.tensor([0.0, 0.5, 0.0], device=dev) - c.position))[2])
    # true distance from camera plane to subject centre is what matters
    z_true = float((c.c2w[:3, :3].T @ (means_gt.mean(0) - c.position))[2])
    bh = float(ys.max() - ys.min() + 1)
    heights.append(bh * z_true / c.fy)
print("per-view metric height: median=%.3f  min=%.3f max=%.3f (true %.3f)" %
      (float(np.median(heights)), min(heights), max(heights), S_STAR * H_MODEL))

# ---- yaw sweep with scale/up fixed, translation from observed centre ----
gc = m.mean(0)
MU_WORLD = means_gt.mean(0)  # what a correct centroid triangulation would give


def eval_yaw(yaw_deg, scale, mset, real_masks, use_centroid=True):
    R = rot_y(math.radians(yaw_deg)) @ MODEL2SCENE.to(dev)
    gc_r = gc @ R.T
    t = MU_WORLD - scale * gc_r
    mt = scale * (m @ R.T) + t
    st = scales * scale
    qt = quaternion_compose(rotation_matrix_to_quaternion(R), qn)
    vals = []
    for i in mset:
        with torch.no_grad():
            a = render_alpha(mt, st, qt, opac, cams[i])
        sil = a.squeeze(0) > 0.1
        r = real_masks[i]
        inter = (sil & r).sum().float()
        un = (sil | r).sum().float()
        vals.append(float(inter / un) if un > 0 else 0.0)
    return float(np.mean(vals))


s_est = 1.773
print("\n yaw | IoU(unclipped)")
curve = []
for yaw in range(-180, 180, 10):
    v = eval_yaw(yaw, s_est, usable, masks)
    curve.append((yaw, v))
    star = "  <== true 35 deg" if yaw == 30 else ""
    print(f" {yaw:5d} | {v:.4f}{star}")
best = max(curve, key=lambda t: t[1])
print(f"\nbest yaw={best[0]} IoU={best[1]:.4f} | true=35.0")
