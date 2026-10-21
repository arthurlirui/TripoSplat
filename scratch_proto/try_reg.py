import sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from pointsplat_bridge._gsplat_preload import preload_gsplat_cuda
preload_gsplat_cuda()
from multiview_refine.cameras import load_cameras
from multiview_refine.ply_io import load_gaussians_ply
from multiview_refine.registration import register_gaussians_to_scene, verify_registration

dev='cuda'
cams = load_cameras("static/000001/mvs/transforms_train.json", device=dev, image_scale=0.5, load_image=True)
m,s_raw,q,o,sh0,_ = load_gaussians_ply("output/gaussian_output/000001/gaussians.ply", device=dev)
scales=s_raw.exp(); opac=torch.sigmoid(o); quats=torch.nn.functional.normalize(q,dim=-1)
t0=time.time()
res = register_gaussians_to_scene(m, scales, quats, opac, cams, colors=sh0, verbose=True)
print("time %.1fs"%(time.time()-t0))
print("scale=%.4f height=%.3f yaw=%.1f"%(res.scale,res.metric_height,math.degrees(res.yaw)))
v = verify_registration(res, m, scales, quats, opac, cams)
print({k:(round(x,4) if isinstance(x,float) else x) for k,x in v.items() if k!="per_view_iou"})
