"""Register SMPL skeleton to Gaussian splatting body via YOLO-pose.

Pipeline:
  1. Render Gaussians from real camera views (32 views).
  2. YOLO11-pose detects 2D keypoints from rendered images.
  3. Multi-view triangulation (DLT) → 3D skeleton joints.
  4. Register SMPL: estimate scale + translation + body_pose from 3D joints.
  5. Rig Gaussians to registered SMPL + render with skeleton overlay.

Usage:
    conda activate rtg132
    python run_skeleton_registration.py [--gaussians_ply PATH] [--cameras_json PATH] \
        [--smpl_model_path PATH]
"""

import torch, sys, numpy as np, cv2, os, math, argparse
sys.path.insert(0, '.')

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


def draw_skeleton(img, joints, cam, smpl_parents):
    pts = []
    for j in range(24):
        hp = torch.cat([joints[j], torch.tensor([1.], device=device)])
        cp = (cam.w2c @ hp)[:3]
        pts.append((int(cam.fx*cp[0]/cp[2]+cam.cx), int(cam.fy*cp[1]/cp[2]+cam.cy)) if cp[2] > 0.01 else None)
    for j in range(24):
        p = smpl_parents[j]
        if p < 0 or pts[j] is None or pts[p] is None: continue
        col = (0,255,0) if j < 15 else (0,0,255) if j in [16,18,20] else (255,0,0) if j in [17,19,21] else (0,255,255)
        cv2.line(img, pts[j], pts[p], col, 2)
    for j, pt in enumerate(pts):
        if pt is None: continue
        cv2.circle(img, pt, 5, (0,255,0), -1)
        cv2.circle(img, pt, 5, (255,255,255), 1)
        if j in [0,15,16,17,20,21]:
            cv2.putText(img, JN[j], (pt[0]+8, pt[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)


def main():
    parser = argparse.ArgumentParser(description='Register SMPL skeleton to Gaussian splatting body via YOLO-pose')
    parser.add_argument('--gaussians_ply', default='scratch_output/refined_pruned.ply',
                        help='Path to the input Gaussian PLY file')
    parser.add_argument('--cameras_json', default='static/000001/mvs/transforms_test.json',
                        help='Path to the Nerfstudio transforms JSON')
    parser.add_argument('--smpl_model_path', default='E:/Code/mmforall/model/assets/SMPL/SMPL_NEUTRAL.pkl',
                        help='Path to the SMPL model file')
    args = parser.parse_args()

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
    print('Step 2: Render Gaussians from 32 camera views')
    print('=' * 60)
    cams = load_cameras(args.cameras_json,
                        device=device, image_scale=0.5, load_image=True)
    bg = torch.zeros(3, device=device)
    rendered = []
    for i, cam in enumerate(cams):
        with torch.no_grad():
            rgb, _, _ = render_gaussians(means, scales_raw.exp(), quats, sh0,
                                         opac_flat, cam, sh_degree=0, background=bg)
        rendered.append((rgb.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8))
    print(f'  Rendered {len(rendered)} views')

    # ═══ 3. YOLO11-pose on rendered images ═══
    print(f'\n{"=" * 60}')
    print('Step 3: YOLO11-pose detect 2D keypoints from rendered images')
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
    print(f'  Detected person in {n_det}/{len(cams)} rendered views')

    # ═══ 4. Triangulate 3D joints ═══
    print(f'\n{"=" * 60}')
    print('Step 4: Multi-view triangulation -> 3D skeleton')
    print('=' * 60)
    joints_3d = np.zeros((17, 3))
    for k in range(17):
        valid_views = [v for v in range(len(cams)) if kpts_2d[v, k, 2] > 0]
        if len(valid_views) < 2:
            continue
        A = []
        for v in valid_views:
            cam = cams[v]
            x, y = kpts_2d[v, k, 0], kpts_2d[v, k, 1]
            K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]], dtype=np.float64)
            P = K @ cam.w2c.cpu().numpy().astype(np.float64)[:3, :]
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        joints_3d[k] = Vt[-1, :3] / Vt[-1, 3]

    for k in range(17):
        if np.linalg.norm(joints_3d[k]) > 0.001:
            print(f'  {COCO_NAMES[k]:12s}: ({joints_3d[k,0]:+.3f}, {joints_3d[k,1]:+.3f}, {joints_3d[k,2]:+.3f})')

    # ═══ 5. Register SMPL from 3D skeleton ═══
    print(f'\n{"=" * 60}')
    print('Step 5: Register SMPL from 3D skeleton')
    print('=' * 60)
    smpl = SMPLWrapper(args.smpl_model_path, device=device)
    rest_j = smpl.get_rest_joints().cpu().numpy()

    # Map COCO joints to SMPL
    obs_j = np.zeros((24, 3))
    valid = np.zeros(24, dtype=bool)
    for c, s in COCO_TO_SMPL.items():
        if np.linalg.norm(joints_3d[c]) > 0.001:
            obs_j[s] = joints_3d[c]
            valid[s] = True

    # Estimate scale from bone length ratios
    bone_pairs = [(16,18),(17,19),(18,20),(19,21),(1,4),(2,5),(4,7),(5,8),(0,12)]
    ratios = []
    for a, b in bone_pairs:
        if valid[a] and valid[b]:
            ol = np.linalg.norm(obs_j[a] - obs_j[b])
            rl = np.linalg.norm(rest_j[a] - rest_j[b])
            if ol > 0.01 and rl > 0.01:
                ratios.append(ol / rl)

    smpl_scale = np.median(ratios) if ratios else 1.0
    obs_pelvis = (obs_j[1] + obs_j[2]) / 2 if valid[1] and valid[2] else obs_j[0]
    # Both SMPL and the scene world are Y-up, so no rotation is needed.
    # transl = obs_pelvis - rest_j[0] * scale  (rest_j[0] is the rest pelvis)
    transl = obs_pelvis - rest_j[0] * smpl_scale

    print(f'  Bone ratios: {[f"{r:.3f}" for r in ratios]}')
    print(f'  Scale: {smpl_scale:.4f}')
    print(f'  Translation: {transl.tolist()}')

    smpl.align_to_scene(scale=smpl_scale,
                        transl=torch.tensor(transl, dtype=torch.float32, device=device))

    # Estimate body_pose from joint angles
    body_pose = np.zeros(69)
    for sh, el, wr in [(16,18,20),(17,19,21)]:
        if valid[sh] and valid[el]:
            d = obs_j[el] - obs_j[sh]
            body_pose[(sh-1)*3] = np.arctan2(-d[1], abs(d[0])+abs(d[2])) * 0.8
            if valid[wr]:
                v1 = obs_j[sh] - obs_j[el]
                v2 = obs_j[wr] - obs_j[el]
                c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                body_pose[(el-1)*3] = (np.pi - np.arccos(np.clip(c, -1, 1))) * 0.8
    for hip, kn, an in [(1,4,7),(2,5,8)]:
        if valid[hip] and valid[kn] and valid[an]:
            v1 = obs_j[hip] - obs_j[kn]
            v2 = obs_j[an] - obs_j[kn]
            c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            body_pose[(kn-1)*3] = (np.pi - np.arccos(np.clip(c, -1, 1))) * 0.5

    # Check registration quality
    aligned_j = smpl.get_rest_joints().cpu().numpy()
    errs = []
    for c, s in COCO_TO_SMPL.items():
        if valid[s]:
            e = np.linalg.norm(obs_j[s] - aligned_j[s])
            errs.append(e)
            print(f'  {COCO_NAMES[c]:12s}: obs={obs_j[s].round(3).tolist()} fit={aligned_j[s].round(3).tolist()} err={e:.4f}')
    print(f'  Mean err: {np.mean(errs):.4f}, Max err: {np.max(errs):.4f}')

    # ═══ 6. Rig + render verification ═══
    print(f'\n{"=" * 60}')
    print('Step 6: Rig Gaussians + render with skeleton overlay')
    print('=' * 60)
    skinned = rig_gaussians_to_smpl(means, scales_raw, quats, opac_raw, sh0, smpl,
                                    n_neighbours=4, temperature=0.1, device=device)

    bp_tensor = torch.tensor(body_pose, dtype=torch.float32, device=device)
    smpl_T = smpl.get_joint_transforms(body_pose=bp_tensor)
    m_p, s_p, q_p = skinned.deform(smpl_T)

    cam = cams[0]
    with torch.no_grad():
        rgb_reg, _, _ = render_gaussians(m_p, s_p, q_p, sh0, skinned.opacities,
                                         cam, sh_degree=0, background=bg)
        rgb_orig, _, _ = render_gaussians(means, scales_raw.exp(), quats, sh0,
                                          opac_flat, cam, sh_degree=0, background=bg)

    # Get posed joint positions
    G = smpl._forward_kinematics(torch.zeros(10, device=device), bp_tensor,
                                 torch.zeros(3, device=device), torch.zeros(3, device=device))
    posed_j = G[:, :3, 3]

    orig_img = cv2.cvtColor((rgb_orig.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    reg_img = cv2.cvtColor((rgb_reg.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    reg_skel = reg_img.copy()
    draw_skeleton(reg_skel, posed_j, cam, smpl.parents)
    gt_img = cv2.cvtColor((cam.image.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    W = orig_img.shape[1]
    canvas = np.concatenate([orig_img, reg_skel, gt_img], axis=1)
    cv2.putText(canvas, 'Original Gaussians', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(canvas, f'Registered (scale={smpl_scale:.3f}, err={np.mean(errs):.3f})',
                (W+10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    cv2.putText(canvas, 'GT', (2*W+10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    os.makedirs('output/rigging_check', exist_ok=True)
    cv2.imwrite('output/rigging_check/registered_skeleton.png', canvas)
    cv2.imwrite('output/rigging_check/registered_only.png', reg_skel)
    print(f'\nSaved output/rigging_check/registered_skeleton.png')
    print(f'  scale={smpl_scale:.4f}, mean_err={np.mean(errs):.4f}')


if __name__ == '__main__':
    main()
