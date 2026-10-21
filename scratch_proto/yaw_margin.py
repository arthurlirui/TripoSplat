import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from pointsplat_bridge._gsplat_preload import preload_gsplat_cuda
preload_gsplat_cuda()
from multiview_refine.cameras import load_cameras
from multiview_refine.ply_io import load_gaussians_ply
from multiview_refine.registration import register_gaussians_to_scene, _iou_against_masks, extract_observations, select_views, rotation_y
dev='cuda'
cams = load_cameras("static/000001/mvs/transforms_train.json", device=dev, image_scale=0.5, load_image=True)
m,s_raw,q,o,sh0,_ = load_gaussians_ply("output/gaussian_output/000001/gaussians.ply", device=dev)
scales=s_raw.exp(); opac=torch.sigmoid(o); quats=torch.nn.functional.normalize(q,dim=-1)
res = register_gaussians_to_scene(m, scales, quats, opac, cams, colors=sh0, verbose=False)
obs,_ = select_views(extract_observations(cams), 6)
R_up = torch.from_numpy(np.array(res.diagnostics['yaw'].get('x',0)) if False else np.eye(3)).float()
# rebuild R_up from result: R = Ry(yaw) @ R_up  => R_up = Ry(-yaw) @ R
Ry = torch.from_numpy(rotation_y(res.yaw)).float().to(dev)
R_up = Ry.T @ res.R
mc = m.mean(0); centre=torch.from_numpy(res.subject_centre).float().to(dev)
print(" yaw | IoU")
best=[]
for y in range(-180,180,10):
    R = torch.from_numpy(rotation_y(math.radians(y))).float().to(dev) @ R_up
    t = centre - res.scale*(mc @ R.T)
    v=_iou_against_masks(m,scales,quats,opac,res.scale,R,t,obs,0.1)
    best.append((y,v))
    mark=" <== chosen" if abs(y-math.degrees(res.yaw))<6 else ""
    print(f"{y:5d} | {v:.4f}{mark}")
best.sort(key=lambda e:-e[1]); print("\ntop3:", [(y,round(v,4)) for y,v in best[:3]], "chosen", round(math.degrees(res.yaw),1))
# flip margin
alt = [(y,v) for y,v in best if abs(((y-math.degrees(res.yaw)+180)%360)-180)>90]
print("best flipped (>90 deg away):", alt[0] if alt else None)
