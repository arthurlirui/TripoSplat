import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from pointsplat_bridge._gsplat_preload import preload_gsplat_cuda
preload_gsplat_cuda()
from multiview_refine.coord_registry import MODEL2SCENE
from multiview_refine.gsplat_wrapper import render_alpha
from multiview_refine.quaternion import quaternion_compose, rotation_matrix_to_quaternion
from multiview_refine.registration import *
from multiview_refine.registration import _projection_jacobian
sys.path.insert(0,'tests')
from test_registration import orbit_cameras
dev='cuda'; torch.manual_seed(0)
N=3000
pts=torch.randn(N,3); pts[:,1]*=0.5; pts=torch.clamp(pts,-0.5,0.5)
means=(pts*torch.tensor([0.25,1.0,0.25])).to(dev)
scales=torch.full((N,3),0.02,device=dev); quats=torch.tensor([[1.,0,0,0]],device=dev).repeat(N,1)
opac=torch.full((N,),0.95,device=dev)
s,yaw=1.4,math.radians(-70)
R=torch.from_numpy(rotation_y(yaw)).float().to(dev)@MODEL2SCENE.to(dev)
t=torch.tensor([0.1,0,-0.15],device=dev)
w=s*(means@R.T)+t; qw=quaternion_compose(rotation_matrix_to_quaternion(R),quats)
cams=[c.to(dev) for c in orbit_cameras(10,4.0,0.5)]
for c in cams:
    with torch.no_grad(): a=render_alpha(w,scales*s,qw,opac,c)
    c.alpha=a.detach()
print("cam_up per view (world):", [-c.c2w[:3,1].cpu().numpy().round(2) for c in cams[:3]])
obs=extract_observations(cams)
print("n obs",len(obs))
for o in obs:
    ys,xs=np.nonzero(o.mask)
    X=np.stack([xs-xs.mean(),ys-ys.mean()]).astype(float)
    C=X@X.T/len(xs); ev,evec=np.linalg.eigh(C)
    d=evec[:,int(np.argmax(ev))]
    elong=math.sqrt(ev.max()/max(ev.min(),1e-12))
    bbox=(xs.max()-xs.min()+1, ys.max()-ys.min()+1)
    M=_projection_jacobian(o.cam)
    row=d[1]*M[0]-d[0]*M[1]; row/=np.linalg.norm(row)
    print(f"v{o.view_idx} d={d.round(3)} elong={elong:.2f} bbox(w,h)={bbox} row={row.round(3)} row@Y={row[1]:+.3f}")
