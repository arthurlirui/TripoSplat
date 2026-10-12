"""Optimized SMPL registration via multi-view reprojection minimization.

Pipeline:
  1. Render Gaussians from 32 real camera views.
  2. YOLO11-pose detects 2D keypoints from rendered images.
  3. DLT triangulation → 3D joint initial positions.
  4. Adam optimization: minimize multi-view reprojection + 3D joint loss.
     - Stage 1: optimize transl + global_orient (50 steps)
     - Stage 2: optimize all params (150 steps)
  5. Rig Gaussians to optimized SMPL + render verification.

Usage:
    conda activate rtg132
    python run_optimized_registration.py [--gaussians_ply PATH] [--cameras_json PATH] \
        [--smpl_model_path PATH] [--camsys_path PATH]
"""

import torch, sys, numpy as np, cv2, os, math, time, argparse
sys.path.insert(0, '.')

# Add external camsys repo to path if not already importable.
_camsys_default = 'E:/Code/mmforall'
if _camsys_default not in sys.path:
    sys.path.insert(0, _camsys_default)

from multiview_refine.ply_io import load_gaussians_ply
from skinned_gaussian.smpl_model import SMPLWrapper
from skinned_gaussian.rigging import rig_gaussians_to_smpl
from multiview_refine.gsplat_wrapper import render_gaussians
from multiview_refine.cameras import load_cameras
from ultralytics import YOLO

device = 'cuda'
SH_C0 = 0.28209479177387814
COCO_TO_SMPL = {0:15, 5:16, 6:17, 7:18, 8:19, 9:20, 10:21, 11:1, 12:2, 13:4, 14:5, 15:7, 16:8}
COCO_NAMES = ['nose','L_eye','R_eye','L_ear','R_ear','L_shoulder','R_shoulder',
              'L_elbow','R_elbow','L_wrist','R_wrist','L_hip','R_hip',
              'L_knee','R_knee','L_ankle','R_ankle']
JN = ['pelvis','L_hip','R_hip','spine1','L_knee','R_knee','spine2',
      'L_ankle','R_ankle','spine3','L_foot','R_foot','neck','L_collar',
      'R_collar','head','L_shoulder','R_shoulder','L_elbow','R_elbow',
      'L_wrist','R_wrist','L_hand','R_hand']


def project_joints(joints_3d, cam):
    """Project (24, 3) SMPL joints to 2D pixel coords using camera."""
    # joints_3d: (1, 24, 3) or (24, 3)
    if joints_3d.ndim == 2:
        joints_3d = joints_3d.unsqueeze(0)
    N, J, _ = joints_3d.shape
    homog = torch.cat([joints_3d, torch.ones(N, J, 1, device=joints_3d.device)], dim=-1)  # (N, J, 4)
    cam_pts = torch.einsum('njk,ik->nji', homog, cam.w2c)[:, :, :3]  # (N, J, 3)
    z = cam_pts[..., 2].clamp_min(1e-6)
    u = cam.fx * cam_pts[..., 0] / z + cam.cx
    v = cam.fy * cam_pts[..., 1] / z + cam.cy
    return torch.stack([u, v], dim=-1)  # (N, J, 2)


def main():
    parser = argparse.ArgumentParser(description='Optimized SMPL registration via multi-view reprojection minimization')
    parser.add_argument('--gaussians_ply', default='scratch_output/refined_pruned.ply',
                        help='Path to the input Gaussian PLY file')
    parser.add_argument('--cameras_json', default='static/000001/mvs/transforms_test.json',
                        help='Path to the Nerfstudio transforms JSON')
    parser.add_argument('--smpl_model_path', default='E:/Code/mmforall/model/assets/SMPL/SMPL_NEUTRAL.pkl',
                        help='Path to the SMPL model file')
    parser.add_argument('--camsys_path', default='E:/Code/mmforall',
                        help='Path to the external camsys repo (contains camsys.smpl_forward)')
    args = parser.parse_args()

    # Lazy-import camsys.SMPLModel from the external repo path.
    if args.camsys_path not in sys.path:
        sys.path.insert(0, args.camsys_path)
    from camsys.smpl_forward import SMPLModel

    # ═══ 1. Load Gaussians ═══
    print('=' * 60)
    print('Step 1: Load Gaussians')
    print('=' * 60)
    means, scales_raw, quats, opac_raw, sh0, shN = load_gaussians_ply(
        args.gaussians_ply, device)
    opac = torch.sigmoid(opac_raw)
    colors = (sh0[:, 0, :] * SH_C0 + 0.5).clamp(0, 1)
    keep = ~((colors.max(-1).values < 0.01) & (opac > 0.5))
    means, scales_raw, quats, opac_raw, sh0 = [x[keep] for x in [means, scales_raw, quats, opac_raw, sh0]]
    opac_flat = torch.sigmoid(opac_raw)
    print(f'  {len(means)} Gaussians')

    # ═══ 2. Render from real cameras ═══
    print(f'\n{"=" * 60}')
    print('Step 2: Render from 32 camera views')
    print('=' * 60)
    cams = load_cameras(args.cameras_json,
                        device=device, image_scale=0.5, load_image=True)
    bg = torch.zeros(3, device=device)
    rendered = []
    for cam in cams:
        with torch.no_grad():
            rgb, _, _ = render_gaussians(means, scales_raw.exp(), quats, sh0,
                                         opac_flat, cam, sh_degree=0, background=bg)
        rendered.append((rgb.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8))
    print(f'  Rendered {len(rendered)} views')

    # ═══ 3. YOLO11-pose ═══
    print("=" * 60)
    print("Step 3: YOLO11-pose detect 2D keypoints")
    print('=' * 60)
    yolo = YOLO('yolo11n-pose.pt')
    kpts_2d = []
    for img in rendered:
        r = yolo(img, verbose=False)[0]
        if r.keypoints is not None and r.keypoints.data.shape[0] > 0:
            best = r.keypoints.data[:, :, 2].mean(-1).argmax()
            k = r.keypoints.data[best].cpu().numpy()
            k[k[:, 2] < 0.3, 2] = 0
        else:
            k = np.zeros((17, 3))
        kpts_2d.append(k)
    kpts_2d = np.stack(kpts_2d)
    n_det = sum(1 for k in kpts_2d if k[:, 2].max() > 0)
    print(f'  Detected in {n_det}/{len(cams)} views')

    # ═══ 4. DLT Triangulation ═══
    print(f'\n{"=" * 60}')
    print('Step 4: DLT triangulation -> 3D joints')
    print('=' * 60)
    joints_3d_obs = np.zeros((17, 3))
    joints_valid = np.zeros(17, dtype=bool)
    for k in range(17):
        valid_v = [v for v in range(len(cams)) if kpts_2d[v, k, 2] > 0]
        if len(valid_v) < 2:
            continue
        A = []
        for v in valid_v:
            cam = cams[v]
            x, y = kpts_2d[v, k, 0], kpts_2d[v, k, 1]
            K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]], dtype=np.float64)
            P = K @ cam.w2c.cpu().numpy().astype(np.float64)[:3, :]
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        joints_3d_obs[k] = Vt[-1, :3] / Vt[-1, 3]
        joints_valid[k] = True

    for k in range(17):
        if joints_valid[k]:
            print(f'  {COCO_NAMES[k]:12s}: ({joints_3d_obs[k,0]:+.3f}, {joints_3d_obs[k,1]:+.3f}, {joints_3d_obs[k,2]:+.3f})')

    # ═══ 5. SMPL Optimization (core) ═══
    print(f'\n{"=" * 60}')
    print('Step 5: SMPL optimization (reprojection + 3D joint loss)')
    print('=' * 60)

    smpl_model = SMPLModel(args.smpl_model_path, device=device)

    # Prepare observation data for optimization.
    # Map COCO keypoints to SMPL joints.
    smpl_joint_indices = sorted(set(COCO_TO_SMPL.values()))  # SMPL joint indices with observations
    coco_to_smpl_list = [(c, s) for c, s in COCO_TO_SMPL.items() if joints_valid[c]]

    # 2D observations: per-view (N_obs_joints, 2) pixel coords + confidences.
    obs_2d = {}  # view -> (smpl_idx, xy, conf)
    for v in range(len(cams)):
        obs_list = []
        for c, s in coco_to_smpl_list:
            if kpts_2d[v, c, 2] > 0:
                obs_list.append((s, kpts_2d[v, c, 0], kpts_2d[v, c, 1], kpts_2d[v, c, 2]))
        if obs_list:
            obs_2d[v] = obs_list

    # 3D observations: (N_joints, 3) + valid mask.
    target_3d = np.zeros((24, 3))
    target_3d_valid = np.zeros(24, dtype=bool)
    for c, s in coco_to_smpl_list:
        target_3d[s] = joints_3d_obs[c]
        target_3d_valid[s] = True
    target_3d_t = torch.tensor(target_3d, dtype=torch.float32, device=device).unsqueeze(0)
    target_3d_valid_t = torch.tensor(target_3d_valid, dtype=torch.bool, device=device)

    # Initialize SMPL params.
    # Pelvis = midpoint of L_hip and R_hip (or first valid joint).
    if target_3d_valid[1] and target_3d_valid[2]:
        pelvis_obs = (target_3d[1] + target_3d[2]) / 2
    else:
        pelvis_obs = target_3d[0] if target_3d_valid[0] else np.zeros(3)

    # Get SMPL rest-pose pelvis.
    _, joints_rest, _ = smpl_model.forward(
        torch.zeros(1, 10, device=device),
        torch.zeros(1, 69, device=device),
        torch.zeros(1, 3, device=device),
        torch.zeros(1, 3, device=device))
    rest_pelvis = joints_rest[0, 0].cpu().numpy()

    # Initialize params.
    transl_init = torch.tensor(pelvis_obs - rest_pelvis, dtype=torch.float32, device=device).unsqueeze(0)
    global_orient = torch.zeros(1, 3, device=device, requires_grad=True)
    body_pose = torch.zeros(1, 69, device=device, requires_grad=True)
    betas = torch.zeros(1, 10, device=device, requires_grad=True)
    transl = transl_init.clone().requires_grad_(True)

    # Loss weights.
    w_2d = 1.0       # reprojection
    w_3d = 10.0      # 3D joint
    w_beta = 0.01    # shape regularization

    # Stage 1: optimize transl + global_orient only.
    print('\n  Stage 1: optimize transl + global_orient (50 steps)')
    opt1 = torch.optim.Adam([transl, global_orient], lr=0.05)
    for it in range(50):
        opt1.zero_grad()
        _, joints, _ = smpl_model.forward(betas, body_pose, global_orient, transl)

        loss = torch.tensor(0.0, device=device)
        # 2D reprojection.
        for v, obs_list in obs_2d.items():
            cam = cams[v]
            proj = project_joints(joints, cam)[0]  # (24, 2)
            for s_idx, ox, oy, conf in obs_list:
                err = (proj[s_idx] - torch.tensor([ox, oy], device=device)) * conf / cam.fx
                loss = loss + (err ** 2).mean()
        loss = loss * w_2d / max(len(obs_2d), 1)

        # 3D joint.
        if target_3d_valid_t.any():
            diff = joints[0, target_3d_valid_t] - target_3d_t[0, target_3d_valid_t]
            loss = loss + w_3d * (diff ** 2).mean()

        loss.backward()
        opt1.step()
        if it % 10 == 0:
            print(f'    iter {it}: loss={loss.item():.6f}, transl={transl[0].detach().tolist()}')

    # Stage 2: optimize all params.
    print('\n  Stage 2: optimize all params (150 steps)')
    opt2 = torch.optim.Adam([global_orient, body_pose, betas, transl], lr=0.025)
    for it in range(150):
        opt2.zero_grad()
        _, joints, _ = smpl_model.forward(betas, body_pose, global_orient, transl)

        loss = torch.tensor(0.0, device=device)
        # 2D reprojection.
        for v, obs_list in obs_2d.items():
            cam = cams[v]
            proj = project_joints(joints, cam)[0]
            for s_idx, ox, oy, conf in obs_list:
                err = (proj[s_idx] - torch.tensor([ox, oy], device=device)) * conf / cam.fx
                loss = loss + (err ** 2).mean()
        loss = loss * w_2d / max(len(obs_2d), 1)

        # 3D joint.
        if target_3d_valid_t.any():
            diff = joints[0, target_3d_valid_t] - target_3d_t[0, target_3d_valid_t]
            loss = loss + w_3d * (diff ** 2).mean()

        # Beta regularization.
        loss = loss + w_beta * (betas ** 2).mean()

        loss.backward()
        opt2.step()
        if it % 25 == 0:
            print(f'    iter {it}: loss={loss.item():.6f}')

    print(f'\n  Optimized params:')
    print(f'    transl: {transl[0].detach().tolist()}')
    print(f'    global_orient: {global_orient[0].detach().tolist()}')
    print(f'    body_pose (non-zero): {[(i, f"{body_pose[0,i]:.3f}") for i in range(69) if abs(body_pose[0,i]) > 0.01]}')
    print(f'    betas: {betas[0].detach().tolist()}')

    # Compute final joint errors.
    with torch.no_grad():
        _, joints_final, _ = smpl_model.forward(betas, body_pose, global_orient, transl)
    print(f'\n  Joint errors (optimized SMPL vs triangulated):')
    errs = []
    for c, s in coco_to_smpl_list:
        obs = target_3d[s]
        fit = joints_final[0, s].cpu().numpy()
        e = np.linalg.norm(obs - fit)
        errs.append(e)
        print(f'    {COCO_NAMES[c]:12s}: obs={obs.round(3).tolist()} fit={fit.round(3).tolist()} err={e:.4f}')
    print(f'  Mean err: {np.mean(errs):.4f}, Max err: {np.max(errs):.4f}')

    # ═══ 6. Rig + render ═══
    print(f'\n{"=" * 60}')
    print('Step 6: Rig Gaussians + render verification')
    print('=' * 60)

    # Use our SMPLWrapper for rigging (same model, in-place align).
    smpl_wrap = SMPLWrapper(args.smpl_model_path, device=device)
    # Get optimized SMPL posed joints from smpl_model (already in world coords).
    # The smpl_model already includes transl, so joints are in world space.
    # We need to align SMPLWrapper to the same space.
    # smpl_model's v_template is in SMPL model space; transl puts it in world.
    # SMPLWrapper's align_to_scene does: v_template * scale + transl.
    # Here scale=1.0 (SMPL is already at correct scale after optimization),
    # and transl = optimized transl (which already accounts for rest pelvis offset).
    smpl_wrap.align_to_scene(scale=1.0, transl=transl[0].detach())

    skinned = rig_gaussians_to_smpl(means, scales_raw, quats, opac_raw, sh0, smpl_wrap,
                                    n_neighbours=4, temperature=0.1, device=device)

    # Get joint transforms from our SMPLWrapper using the optimized pose.
    bp_opt = body_pose[0].detach()
    go_opt = global_orient[0].detach()
    tr_opt = transl[0].detach()

    # For SMPLWrapper, transl is added on top of align_to_scene's transl.
    # But we already aligned with transl, so we need to pass zero transl to get_joint_transforms
    # and the pose should be relative to the aligned space.
    # Actually, smpl_model.forward adds transl to joints. SMPLWrapper.get_joint_transforms
    # also adds transl. Since we aligned with transl already, the "additional" transl
    # in get_joint_transforms should be zero (the body is already placed).
    smpl_T = smpl_wrap.get_joint_transforms(
        betas=torch.zeros(10, device=device),
        body_pose=bp_opt,
        global_orient=go_opt,
        transl=torch.zeros(3, device=device))  # transl already in align_to_scene

    m_p, s_p, q_p = skinned.deform(smpl_T)

    cam = cams[0]
    with torch.no_grad():
        rgb_reg, _, _ = render_gaussians(m_p, s_p, q_p, sh0, skinned.opacities,
                                         cam, sh_degree=0, background=bg)
        rgb_orig, _, _ = render_gaussians(means, scales_raw.exp(), quats, sh0,
                                          opac_flat, cam, sh_degree=0, background=bg)

    # Draw skeleton overlay.
    def draw_skel(img, joints, cam):
        pts = []
        for j in range(24):
            hp = torch.cat([joints[j], torch.tensor([1.], device=device)])
            cp = (cam.w2c @ hp)[:3]
            pts.append((int(cam.fx*cp[0]/cp[2]+cam.cx), int(cam.fy*cp[1]/cp[2]+cam.cy)) if cp[2] > 0.01 else None)
        for j in range(24):
            p = smpl_wrap.parents[j]
            if p < 0 or pts[j] is None or pts[p] is None: continue
            col = (0,255,0) if j < 15 else (0,0,255) if j in [16,18,20] else (255,0,0) if j in [17,19,21] else (0,255,255)
            cv2.line(img, pts[j], pts[p], col, 2)
        for j, pt in enumerate(pts):
            if pt is None: continue
            cv2.circle(img, pt, 5, (0,255,0), -1)
            cv2.circle(img, pt, 5, (255,255,255), 1)
            if j in [0,15,16,17,20,21]:
                cv2.putText(img, JN[j], (pt[0]+8, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)

    # Get posed joints from smpl_model (already in world space).
    posed_joints = joints_final[0].detach()

    orig_img = cv2.cvtColor((rgb_orig.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    reg_img = cv2.cvtColor((rgb_reg.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    reg_skel = reg_img.copy()
    draw_skel(reg_skel, posed_joints, cam)
    gt_img = cv2.cvtColor((cam.image.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    W = orig_img.shape[1]
    canvas = np.concatenate([orig_img, reg_skel, gt_img], axis=1)
    cv2.putText(canvas, 'Original Gaussians', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(canvas, f'Optimized (err={np.mean(errs):.3f})', (W+10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(canvas, 'GT', (2*W+10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    os.makedirs('rigging_check', exist_ok=True)
    cv2.imwrite('rigging_check/optimized_registration.png', canvas)
    cv2.imwrite('rigging_check/optimized_only.png', reg_skel)
    print(f'\nSaved rigging_check/optimized_registration.png')
    print(f'  Mean err: {np.mean(errs):.4f}, Max err: {np.max(errs):.4f}')


if __name__ == '__main__':
    main()
