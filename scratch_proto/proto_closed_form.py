"""Prototype: closed-form 7-DoF registration from multi-view silhouettes (synthetic GT check).

Stages
  1. up axis  u  : triangulate per-view silhouette top/bottom points, difference.
  2. scale       : vertical-line fit of silhouette extremes along u -> metric height.
  3. centroid    : triangulate mask centroids -> translation.
  4. yaw/roll    : from u (2 DoF) + facing prior / silhouette asymmetry (1 DoF).
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
TRUE_H = S_STAR * H_MODEL
print(f"GT: s={S_STAR:.3f} t={T_STAR.tolist()} metric height={TRUE_H:.3f}")

means_gt = S_STAR * (m @ R_STAR.T) + T_STAR
q_gt = quaternion_compose(rotation_matrix_to_quaternion(R_STAR), qn)
sc_gt = scales * S_STAR

masks = []
for c in cams:
    with torch.no_grad():
        a = render_alpha(means_gt, sc_gt, q_gt, opac, c)
    masks.append((a.squeeze(0) > 0.5).cpu().numpy())

# ---------------- observations ----------------
obs = []
for i, (c, mk) in enumerate(zip(cams, masks)):
    ys, xs = np.nonzero(mk)
    if len(ys) < 50:
        continue
    clipped = ys.min() == 0 or ys.max() == mk.shape[0] - 1 or xs.min() == 0 or xs.max() == mk.shape[1] - 1
    obs.append(dict(i=i, c=c, ys=ys, xs=xs, clipped=clipped,
                    cen=np.array([xs.mean(), ys.mean()])))
usable = [O for O in obs if not O["clipped"]]
print(f"usable (unclipped) views: {len(usable)}/{len(obs)}")


def unproject(c, u, v):
    d = np.array([(u - c.cx) / c.fx, (v - c.cy) / c.fy, 1.0])
    d /= np.linalg.norm(d)
    C = c.position.detach().cpu().double().numpy()
    Rw = c.c2w[:3, :3].detach().cpu().double().numpy()
    return C, Rw @ d


def tri(rays):
    A, b = [], []
    for C, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A.append(P); b.append(P @ C)
    A, b = np.vstack(A), np.concatenate(b)
    X, *_ = np.linalg.lstsq(A, b, rcond=None)
    res = np.array([np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (X - C)) for C, d in rays])
    return X, res


# ---------------- stage 1: up axis ----------------
tops, bots = [], []
for O in usable:
    c, ys, xs = O["c"], O["ys"], O["xs"]
    ymin, ymax = ys.min(), ys.max()
    # take the median column of all rows within 1px of the extreme, more robust than mean
    tops.append(unproject(c, np.median(xs[ys <= ymin + 1]), ymin))
    bots.append(unproject(c, np.median(xs[ys >= ymax - 1]), ymax))
Ptop, r_top = tri(tops)
Pbot, r_bot = tri(bots)
u_up = (Ptop - Pbot); u_up /= np.linalg.norm(u_up)
if u_up[1] < 0:
    u_up = -u_up
err_up = math.degrees(math.acos(min(1.0, float(u_up @ np.array([0, 1.0, 0])))))
print(f"stage1 up axis {u_up.round(4)}  err={err_up:.3f} deg  resid top={r_top.mean():.4f} bot={r_bot.mean():.4f}")

# ---------------- stage 2: scale from vertical extent ----------------
p0, _ = tri([unproject(O["c"], O["cen"][0], O["cen"][1]) for O in usable])


def fit_offset(p0, u, pairs, want_top):
    A, b = [], []
    for (ray, is_top) in pairs:
        if is_top != want_top:
            continue
        C, d = ray
        P = np.eye(3) - np.outer(d, d)
        A.append((P @ u).reshape(3, 1))
        b.append((-P @ (p0 - C)).reshape(3, 1))
    A, b = np.vstack(A), np.vstack(b)
    a, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(a[0, 0])


pairs = []
for O in usable:
    c, ys, xs = O["c"], O["ys"], O["xs"]
    ymin, ymax = ys.min(), ys.max()
    pairs.append((unproject(c, np.median(xs[ys <= ymin + 1]), ymin), True))
    pairs.append((unproject(c, np.median(xs[ys >= ymax - 1]), ymax), False))
a_top = fit_offset(p0, u_up, pairs, True)
a_bot = fit_offset(p0, u_up, pairs, False)
H_est = a_top - a_bot
s_est = H_est / H_MODEL
print(f"stage2 H_est={H_est:.3f} (true {TRUE_H:.3f})  scale={s_est:.4f} (true {S_STAR:.4f}) "
      f"err={100*abs(s_est-S_STAR)/S_STAR:.2f}%")

# ---------------- stage 3: translation ----------------
# top/bottom world points from the fitted line, midpoint = subject centre
mid = p0 + 0.5 * (a_top + a_bot) * u_up
# model's up-axis extent centre in its own frame
mu_model = np.array([0.0, 0.5 * (H_MODEL * 1.0), 0.0])
gc = (m.mean(0)).detach().cpu().double().numpy()
print(f"stage3 line midpoint={mid.round(3)} (true t={T_STAR.cpu().numpy().round(3)})")
print(f"        gaussian centroid (model frame)={gc.round(3)}")
t_est = mid - s_est * gc
print(f"        t_est={t_est.round(3)}  err={np.linalg.norm(t_est-T_STAR.cpu().numpy()):.3f} m")
