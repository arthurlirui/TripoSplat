import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from pointsplat_bridge._gsplat_preload import preload_gsplat_cuda
preload_gsplat_cuda()
from multiview_refine.coord_registry import MODEL2SCENE
from multiview_refine.gsplat_wrapper import render_alpha
from multiview_refine.quaternion import quaternion_compose, rotation_matrix_to_quaternion
from multiview_refine.registration import *
from multiview_refine.registration import _image_up_direction, _observed_extent_along
sys.path.insert(0,'tests')
from test_registration import orbit_cameras

dev='cuda'; torch.manual_seed(0)
N=3000
pts=torch.randn(N,3); pts[:,1]=pts[:,1]*0.5; pts=torch.clamp(pts,-0.5,0.5)
means=(pts*torch.tensor([0.25,1.0,0.25])).to(dev)
scales=torch.full((N,3),0.02,device=dev)
quats=torch.tensor([[1.,0,0,0]],device=dev).repeat(N,1)
opac=torch.full((N,),0.95,device=dev)
print("model y extent", float(means[:,1].min()), float(means[:,1].max()))

s_true,yaw=1.4,math.radians(-70)
R_true=torch.from_numpy(rotation_y(yaw)).float().to(dev)@MODEL2SCENE.to(dev)
t_true=torch.tensor([0.1,0,-0.15],device=dev)
w=s_true*(means@R_true.T)+t_true
q_w=quaternion_compose(rotation_matrix_to_quaternion(R_true),quats)
cams=[c.to(dev) for c in orbit_cameras(10,4.0,0.5)]
for c in cams:
    with torch.no_grad(): a=render_alpha(w,scales*s_true,q_w,opac,c)
    c.alpha=a.detach()

res=register_gaussians_to_scene(means,scales,quats,opac,cams,verbose=True)
print("scale",res.scale,"true",s_true,"height",res.metric_height,"model_h",res.model_height)
print("yaw",math.degrees(res.yaw),"true",math.degrees(yaw))
print("centre",res.subject_centre.round(3),"true",t_true.cpu().numpy().round(3))

# what's the rendered metric height at truth?
obs,_=select_views(extract_observations(cams),3)
up=np.array([0,1.0,0])
c=res.subject_centre
print("\nview | elevation-ish | obs_extent | pred_extent_at_true_H")
for o in obs[:6]:
    d=_image_up_direction(o.cam,c,up)
    if d is None: print(o.view_idx,"no dir"); continue
    e_obs=_observed_extent_along(o,d)
    H=1.0*s_true
    u,v,z=project_points(o.cam,np.stack([c+0.5*H*up,c-0.5*H*up]))
    pred=math.hypot(u[0]-u[1],v[0]-v[1])
    print(f"{o.view_idx:4d} | {e_obs:8.1f} | {pred:8.1f}  ratio {e_obs/pred:.3f}")
