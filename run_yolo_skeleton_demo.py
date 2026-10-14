"""YOLO11-pose → multi-view triangulation → SMPL fitting → Gaussian splatting demo.

Pipeline:
  1. Use YOLO11-pose to detect 2D keypoints from all 32 camera views of 00.png.
  2. Triangulate 2D keypoints across views to get 3D skeleton joints.
  3. Fit SMPL body parameters (global_orient, body_pose, transl) from 3D joints.
  4. Load reconstructed Gaussians from scratch_output, rig to SMPL.
  5. Drive Gaussians with the fitted SMPL pose.
  6. Render from a rotating camera viewpoint.

Usage:
    conda activate rtg132
    python run_yolo_skeleton_demo.py \\
        --data_dir F:/Data/warm_2-mvs/000000/mvs \\
        --gaussians scratch_output/refined_pruned.ply \\
        --smpl_model_path E:/Code/mmforall/model/assets/SMPL/SMPL_NEUTRAL.pkl \\
        --output_dir output/yolo_skeleton_demo --n_frames 240 --fps 30
"""

import argparse
import os
import sys
import math
import time

import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SH_C0 = 0.28209479177387814

# COCO 17 keypoints → SMPL 24 joints mapping
COCO_TO_SMPL = {
    0: 15,    # nose → head
    5: 16,    # L_shoulder → L_shoulder
    6: 17,    # R_shoulder → R_shoulder
    7: 18,    # L_elbow → L_elbow
    8: 19,    # R_elbow → R_elbow
    9: 20,    # L_wrist → L_wrist
    10: 21,   # R_wrist → R_wrist
    11: 1,    # L_hip → L_hip
    12: 2,    # R_hip → R_hip
    13: 4,    # L_knee → L_knee
    14: 5,    # R_knee → R_knee
    15: 7,    # L_ankle → L_ankle
    16: 8,    # R_ankle → R_ankle
}

COCO_NAMES = ['nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear',
              'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow',
              'L_wrist', 'R_wrist', 'L_hip', 'R_hip',
              'L_knee', 'R_knee', 'L_ankle', 'R_ankle']


def detect_2d_keypoints(model, image_paths, conf_threshold=0.3):
    """Detect 2D keypoints from multiple views using YOLO11-pose.

    Returns:
        list of (N_views, 17, 3) arrays: (x, y, confidence) per view.
        Only the best person per view is kept.
    """
    all_kpts = []
    for path in image_paths:
        results = model(path, verbose=False)
        r = results[0]
        if r.keypoints is not None and r.keypoints.data.shape[0] > 0:
            # Take the person with highest mean confidence.
            confs = r.keypoints.data[:, :, 2].mean(dim=-1)  # (N_persons,)
            best = confs.argmax()
            kpts = r.keypoints.data[best].cpu().numpy()  # (17, 3)
            # Zero out low-confidence keypoints.
            kpts[kpts[:, 2] < conf_threshold, 2] = 0
        else:
            kpts = np.zeros((17, 3))
        all_kpts.append(kpts)
    return np.stack(all_kpts)  # (N_views, 17, 3)


def triangulate_multiview(kpts_2d, cameras, device='cuda'):
    """Triangulate 2D keypoints across views to get 3D joints.

    Uses DLT (Direct Linear Transform) triangulation.

    Args:
        kpts_2d: (N_views, 17, 3) — (x, y, conf) per view per keypoint.
        cameras: list of Camera objects with w2c, fx, fy, cx, cy.

    Returns:
        (17, 3) 3D joint positions in world space.
    """
    N_views = len(cameras)
    N_kpts = kpts_2d.shape[1]
    joints_3d = np.zeros((N_kpts, 3))

    for k in range(N_kpts):
        # Collect valid views (confidence > 0).
        valid_views = []
        for v in range(N_views):
            if kpts_2d[v, k, 2] > 0:
                valid_views.append(v)

        if len(valid_views) < 2:
            continue

        # Build DLT equations: for each view, 2 rows.
        A = []
        for v in valid_views:
            cam = cameras[v]
            x, y = kpts_2d[v, k, 0], kpts_2d[v, k, 1]

            # Projection matrix P = K @ w2c (3x4)
            K = np.array([[cam.fx, 0, cam.cx],
                          [0, cam.fy, cam.cy],
                          [0, 0, 1]], dtype=np.float64)
            w2c = cam.w2c.cpu().numpy().astype(np.float64)
            P = K @ w2c[:3, :]  # (3, 4)

            # DLT: x * P[2] - P[0] = 0, y * P[2] - P[1] = 0
            A.append(x * P[2, :] - P[0, :])
            A.append(y * P[2, :] - P[1, :])

        A = np.array(A)  # (2*N_valid, 4)
        if A.shape[0] < 4:
            continue

        # SVD: the solution is the last row of V (smallest singular value).
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1, :3] / Vt[-1, 3]  # homogeneous → 3D
        joints_3d[k] = X

    return joints_3d


def fit_smpl_from_joints(joints_3d_coco, smpl, device='cuda'):
    """Fit SMPL parameters from triangulated 3D COCO joints.

    Uses simple least-squares: estimate translation + global orient + 
    approximate body pose from joint positions.

    Args:
        joints_3d_coco: (17, 3) COCO joints in world space.
        smpl: SMPLWrapper instance.

    Returns:
        dict with 'global_orient', 'body_pose', 'transl', 'betas'.
    """
    # Map COCO joints to SMPL joints.
    smpl_joints_3d = np.zeros((24, 3))
    smpl_valid = np.zeros(24, dtype=bool)
    for coco_idx, smpl_idx in COCO_TO_SMPL.items():
        smpl_joints_3d[smpl_idx] = joints_3d_coco[coco_idx]
        smpl_valid[smpl_idx] = True

    # Get SMPL rest-pose joints.
    rest_joints = smpl.get_rest_joints().cpu().numpy()  # (24, 3)

    # Estimate translation: align SMPL pelvis (J0) to observed pelvis.
    # Pelvis ≈ midpoint of L_hip and R_hip.
    if smpl_valid[1] and smpl_valid[2]:
        observed_pelvis = (smpl_joints_3d[1] + smpl_joints_3d[2]) / 2
    else:
        observed_pelvis = smpl_joints_3d[0]
    transl = observed_pelvis - rest_joints[0]

    # Estimate scale: ratio of observed to rest limb lengths.
    # Use shoulder-to-hip distance.
    if smpl_valid[16] and smpl_valid[1]:
        obs_len = np.linalg.norm(smpl_joints_3d[16] - smpl_joints_3d[1])
        rest_len = np.linalg.norm(rest_joints[16] - rest_joints[1])
        scale = obs_len / max(rest_len, 1e-6)
    else:
        scale = 1.0

    # Estimate global orientation: align SMPL facing direction to observed.
    # Facing direction = cross product of (L_shoulder - R_shoulder) and (pelvis - head).
    if smpl_valid[16] and smpl_valid[17] and smpl_valid[0]:
        obs_right = smpl_joints_3d[16] - smpl_joints_3d[17]  # L - R shoulder
        obs_up = smpl_joints_3d[0] - (smpl_joints_3d[1] + smpl_joints_3d[2]) / 2  # head - pelvis
        obs_forward = np.cross(obs_right, obs_up)
        if np.linalg.norm(obs_forward) > 1e-6:
            obs_forward = obs_forward / np.linalg.norm(obs_forward)
        else:
            obs_forward = np.array([0, 0, 1])

        rest_right = rest_joints[16] - rest_joints[17]
        rest_up = rest_joints[0] - (rest_joints[1] + rest_joints[2]) / 2
        rest_forward = np.cross(rest_right, rest_up)
        if np.linalg.norm(rest_forward) > 1e-6:
            rest_forward = rest_forward / np.linalg.norm(rest_forward)
        else:
            rest_forward = np.array([0, 0, 1])

        # Compute rotation from rest_forward to obs_forward (Rodrigues).
        cross = np.cross(rest_forward, obs_forward)
        sin_a = np.linalg.norm(cross)
        cos_a = np.dot(rest_forward, obs_forward)
        if sin_a < 1e-6:
            if cos_a > 0:
                global_orient = np.zeros(3)
            else:
                global_orient = np.array([0, np.pi, 0])
        else:
            axis = cross / sin_a
            angle = np.arctan2(sin_a, cos_a)
            global_orient = axis * angle
    else:
        global_orient = np.zeros(3)

    # Body pose: estimate from joint angles.
    # Simple approach: compute relative joint angles from observed positions.
    body_pose = np.zeros(69)  # 23 joints × 3

    # Estimate arm angles (shoulders + elbows).
    for side, sh_idx, el_idx, wr_idx, hip_idx in [
        ('L', 16, 18, 20, 1),  # L_shoulder, L_elbow, L_wrist, L_hip
        ('R', 17, 19, 21, 2),  # R_shoulder, R_elbow, R_wrist, R_hip
    ]:
        if smpl_valid[sh_idx] and smpl_valid[el_idx]:
            # Shoulder rotation: direction from shoulder to elbow.
            dir_se = smpl_joints_3d[el_idx] - smpl_joints_3d[sh_idx]
            # Approximate: X rotation = arm raise angle.
            angle_x = math.atan2(-dir_se[1], abs(dir_se[0]) + abs(dir_se[2]))
            body_pose[(sh_idx - 1) * 3] = angle_x * 0.8  # shoulder X

            if smpl_valid[wr_idx]:
                # Elbow bend: angle at elbow.
                v1 = smpl_joints_3d[sh_idx] - smpl_joints_3d[el_idx]
                v2 = smpl_joints_3d[wr_idx] - smpl_joints_3d[el_idx]
                cos_elbow = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                cos_elbow = np.clip(cos_elbow, -1, 1)
                elbow_angle = math.acos(cos_elbow)
                body_pose[(el_idx - 1) * 3] = (np.pi - elbow_angle) * 0.8  # elbow X

    # Estimate knee angles.
    for side, hip_idx, kn_idx, an_idx in [
        ('L', 1, 4, 7),
        ('R', 2, 5, 8),
    ]:
        if smpl_valid[hip_idx] and smpl_valid[kn_idx] and smpl_valid[an_idx]:
            v1 = smpl_joints_3d[hip_idx] - smpl_joints_3d[kn_idx]
            v2 = smpl_joints_3d[an_idx] - smpl_joints_3d[kn_idx]
            cos_knee = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_knee = np.clip(cos_knee, -1, 1)
            knee_angle = math.acos(cos_knee)
            body_pose[(kn_idx - 1) * 3] = (np.pi - knee_angle) * 0.5  # knee X

    result = {
        'global_orient': torch.tensor(global_orient, dtype=torch.float32, device=device),
        'body_pose': torch.tensor(body_pose, dtype=torch.float32, device=device),
        'transl': torch.tensor(transl, dtype=torch.float32, device=device),
        'betas': torch.zeros(10, dtype=torch.float32, device=device),
        'scale': scale,
    }

    print(f"  Fitted SMPL: transl={transl.tolist()}, scale={scale:.3f}")
    print(f"  Global orient: {global_orient.tolist()}")
    print(f"  Body pose (first 6): {body_pose[:6].tolist()}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='F:/Data/warm_2-mvs/000000/mvs',
                        help='Directory with images/ and transforms_*.json')
    parser.add_argument('--gaussians', default='scratch_output/refined_pruned.ply')
    parser.add_argument('--smpl_model_path', default='E:/Code/mmforall/model/assets/SMPL/SMPL_NEUTRAL.pkl')
    parser.add_argument('--output_dir', default='output/yolo_skeleton_demo')
    parser.add_argument('--n_frames', type=int, default=240)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ═══ 1. YOLO11-pose: detect 2D keypoints from all views ═══
    print("=" * 70)
    print("Step 1: YOLO11-pose — detect 2D keypoints from multi-view images")
    print("=" * 70)

    from ultralytics import YOLO
    yolo = YOLO('yolo11n-pose.pt')

    # Load all camera views.
    image_dir = os.path.join(args.data_dir, 'images')
    image_paths = sorted([os.path.join(image_dir, f)
                          for f in os.listdir(image_dir) if f.endswith('.png')])
    print(f"  {len(image_paths)} camera views")

    kpts_2d = detect_2d_keypoints(yolo, image_paths)
    print(f"  Detected 2D keypoints: {kpts_2d.shape} (views × keypoints × xyz)")

    # Save 2D keypoints visualization for first view.
    img0 = cv2.imread(image_paths[0])
    for k in range(17):
        x, y, c = kpts_2d[0, k]
        if c > 0:
            cv2.circle(img0, (int(x), int(y)), 8, (0, 0, 255), -1)
            cv2.putText(img0, COCO_NAMES[k], (int(x) + 10, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(args.output_dir, 'keypoints_2d.png'), img0)
    print(f"  Saved keypoints_2d.png")

    # ═══ 2. Load cameras + triangulate 3D joints ═══
    print(f"\n{'=' * 70}")
    print("Step 2: Multi-view triangulation — 2D → 3D skeleton")
    print("=" * 70)

    from multiview_refine.cameras import load_cameras
    transforms_path = os.path.join(args.data_dir, 'transforms_test.json')
    cameras = load_cameras(transforms_path, device=device, image_scale=1.0, load_image=False)
    print(f"  Loaded {len(cameras)} cameras")

    joints_3d_coco = triangulate_multiview(kpts_2d, cameras, device)
    print(f"  Triangulated 3D joints: {joints_3d_coco.shape}")

    # Print 3D joints.
    for k in range(17):
        if np.linalg.norm(joints_3d_coco[k]) > 0.001:
            print(f"    {COCO_NAMES[k]:12s}: ({joints_3d_coco[k,0]:+.3f}, {joints_3d_coco[k,1]:+.3f}, {joints_3d_coco[k,2]:+.3f})")

    # ═══ 3. Fit SMPL from 3D joints ═══
    print(f"\n{'=' * 70}")
    print("Step 3: Fit SMPL parameters from 3D skeleton")
    print("=" * 70)

    from skinned_gaussian.smpl_model import SMPLWrapper
    smpl = SMPLWrapper(args.smpl_model_path, device=device)

    fitted = fit_smpl_from_joints(joints_3d_coco, smpl, device)

    # Get SMPL posed joints for comparison.
    smpl_T = smpl.get_joint_transforms(
        betas=fitted['betas'],
        body_pose=fitted['body_pose'],
        global_orient=fitted['global_orient'],
        transl=fitted['transl'])
    smpl_joints_posed = smpl_T[:, :3, 3].cpu().numpy()  # (24, 3)

    print(f"\n  SMPL posed joints vs triangulated:")
    for coco_idx, smpl_idx in COCO_TO_SMPL.items():
        obs = joints_3d_coco[coco_idx]
        fit = smpl_joints_posed[smpl_idx]
        if np.linalg.norm(obs) > 0.001:
            err = np.linalg.norm(obs - fit)
            print(f"    {COCO_NAMES[coco_idx]:12s}: obs=({obs[0]:+.3f},{obs[1]:+.3f},{obs[2]:+.3f}) "
                  f"fit=({fit[0]:+.3f},{fit[1]:+.3f},{fit[2]:+.3f}) err={err:.3f}")

    # ═══ 4. Load Gaussians + rig to SMPL ═══
    print(f"\n{'=' * 70}")
    print("Step 4: Load reconstructed Gaussians + rig to SMPL")
    print("=" * 70)

    from multiview_refine.ply_io import load_gaussians_ply
    from skinned_gaussian.rigging import rig_gaussians_to_smpl

    means, scales_raw, quats, opac_raw, sh0, shN = load_gaussians_ply(args.gaussians, device)
    opac = torch.sigmoid(opac_raw)
    colors = (sh0[:, 0, :] * SH_C0 + 0.5).clamp(0, 1)
    keep = ~((colors.max(-1).values < 0.01) & (opac > 0.5))
    print(f"  Loaded {len(means)} Gaussians, pruning {(~keep).sum()} → {keep.sum()} remaining")
    means, scales_raw, quats, opac_raw, sh0 = [x[keep] for x in [means, scales_raw, quats, opac_raw, sh0]]

    # NOTE: Do NOT boost scales — the reconstructed Gaussians are already
    # optimized at the correct scale (PSNR=30 without boost vs 14 with 10x boost).
    # Boosting destroys rendering quality.

    # Optimize SMPL scale by minimizing Gaussian-to-SMPL surface distance.
    gc = means.median(0).values
    from scipy.optimize import minimize_scalar
    from scipy.spatial import cKDTree

    def coverage_score(scale_val):
        test_smpl = SMPLWrapper(args.smpl_model_path, device=device)
        test_smpl.align_to_scene(scale=scale_val, transl=gc)
        v, _ = test_smpl.get_rest_mesh()
        tree = cKDTree(v.cpu().numpy())
        dists, _ = tree.query(means.cpu().numpy(), k=1)
        return dists.mean()

    result = minimize_scalar(coverage_score, bounds=(0.5, 10.0), method='bounded')
    smpl_scale = result.x
    smpl.align_to_scene(scale=smpl_scale, transl=gc)
    print(f"  Optimized SMPL scale: {smpl_scale:.3f} (mean_dist={result.fun:.4f})")

    skinned = rig_gaussians_to_smpl(
        means, scales_raw, quats, opac_raw, sh0, smpl,
        n_neighbours=4, temperature=0.2, device=device)

    # ═══ 5. Render: fitted pose + rotating camera ═══
    print(f"\n{'=' * 70}")
    print("Step 5: Render fitted pose with rotating camera")
    print("=" * 70)

    from multiview_refine.gsplat_wrapper import render_gaussians
    from multiview_refine.cameras import Camera, invert_se3
    from multiview_refine.video_io import create_video_writer

    bg = torch.zeros(3, device=device)
    W, H = 1024, 768

    # Camera orbit around the body.
    # Use a radius and focal length that match the original dataset scale.
    # Original cameras are at distance ~2.6 from centre, fx~1790 (full res).
    # At 1024 width: fx = 1790 * 1024/2048 = 895, radius = 3.0.
    radius = 3.0
    fx_render = 900.0
    elev = math.radians(10)

    # Also render the skeleton overlay.
    vw, _ = create_video_writer(
        os.path.join(args.output_dir, 'yolo_skeleton_demo.mp4'), 'mp4v', args.fps, (W, H))
    vw_side, _ = create_video_writer(
        os.path.join(args.output_dir, 'skeleton_vs_gaussians.mp4'), 'mp4v', args.fps, (W * 2, H))

    t0 = time.time()
    for i in range(args.n_frames):
        az = 2 * math.pi * i / args.n_frames

        # Camera.
        pos = gc + torch.tensor([
            radius * math.cos(elev) * math.sin(az),
            -radius * math.sin(elev),
            radius * math.cos(elev) * math.cos(az),
        ], device=device)
        forward = gc - pos; forward = forward / forward.norm()
        up = torch.tensor([0., -1., 0.], device=device)
        right = torch.cross(forward, up); right = right / right.norm()
        down = torch.cross(right, forward); down = down / down.norm()
        c2w = torch.eye(4, device=device)
        c2w[:3, 0] = right; c2w[:3, 1] = down; c2w[:3, 2] = forward; c2w[:3, 3] = pos
        w2c = invert_se3(c2w)
        cam = Camera(w2c=w2c, c2w=c2w, position=pos,
                     fx=fx_render, fy=fx_render, cx=W/2, cy=H/2, width=W, height=H)

        # Render Gaussians with fitted SMPL pose.
        with torch.no_grad():
            means_p, scales_p, quats_p = skinned.deform(smpl_T)
            rgb, alpha, _ = render_gaussians(
                means_p, scales_p, quats_p, sh0, skinned.opacities,
                cam, sh_degree=0, background=bg)

        gauss_img = cv2.cvtColor(
            (rgb.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
            cv2.COLOR_RGB2BGR)

        # Draw skeleton overlay on Gaussian render.
        smpl_joints_2d = []
        for j in range(24):
            j3d = torch.tensor(smpl_joints_posed[j], device=device)
            j_cam = (w2c @ torch.cat([j3d, torch.tensor([1.], device=device)]))[:3]
            if j_cam[2] > 0.01:
                u = 700 * j_cam[0] / j_cam[2] + W/2
                v = 700 * j_cam[1] / j_cam[2] + H/2
                smpl_joints_2d.append((int(u), int(v)))
            else:
                smpl_joints_2d.append(None)

        # Draw bones.
        for j in range(24):
            p = smpl.parents[j]
            if p < 0 or smpl_joints_2d[j] is None or smpl_joints_2d[p] is None:
                continue
            cv2.line(gauss_img, smpl_joints_2d[j], smpl_joints_2d[p], (0, 255, 0), 2)
        # Draw joints.
        for j, pt in enumerate(smpl_joints_2d):
            if pt is not None:
                cv2.circle(gauss_img, pt, 4, (0, 0, 255), -1)

        cv2.putText(gauss_img, f'Frame {i}/{args.n_frames}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        vw.write(gauss_img)

        # Side-by-side: skeleton-only | gaussians+skeleton
        skel_img = np.zeros((H, W, 3), dtype=np.uint8)
        for j in range(24):
            p = smpl.parents[j]
            if p < 0 or smpl_joints_2d[j] is None or smpl_joints_2d[p] is None:
                continue
            cv2.line(skel_img, smpl_joints_2d[j], smpl_joints_2d[p], (0, 255, 0), 2)
        for j, pt in enumerate(smpl_joints_2d):
            if pt is not None:
                cv2.circle(skel_img, pt, 4, (0, 0, 255), -1)
        side = np.concatenate([skel_img, gauss_img], axis=1)
        vw_side.write(side)

        if i % 30 == 0:
            print(f"  frame {i:4d}/{args.n_frames}: rgb={rgb.mean():.4f}, {time.time()-t0:.1f}s")

    vw.release()
    vw_side.release()
    print(f"\n  Done in {time.time()-t0:.1f}s")
    print(f"  yolo_skeleton_demo.mp4: {os.path.getsize(os.path.join(args.output_dir, 'yolo_skeleton_demo.mp4'))//1024} KB")
    print(f"  skeleton_vs_gaussians.mp4: {os.path.getsize(os.path.join(args.output_dir, 'skeleton_vs_gaussians.mp4'))//1024} KB")

    # Save key frames.
    for idx in [0, 60, 120, 180]:
        az = 2 * math.pi * idx / args.n_frames
        pos = gc + torch.tensor([
            radius * math.cos(elev) * math.sin(az),
            -radius * math.sin(elev),
            radius * math.cos(elev) * math.cos(az),
        ], device=device)
        forward = gc - pos; forward = forward / forward.norm()
        up = torch.tensor([0., -1., 0.], device=device)
        right = torch.cross(forward, up); right = right / right.norm()
        down = torch.cross(right, forward); down = down / down.norm()
        c2w = torch.eye(4, device=device)
        c2w[:3, 0] = right; c2w[:3, 1] = down; c2w[:3, 2] = forward; c2w[:3, 3] = pos
        w2c = invert_se3(c2w)
        cam = Camera(w2c=w2c, c2w=c2w, position=pos,
                     fx=fx_render, fy=fx_render, cx=W/2, cy=H/2, width=W, height=H)
        with torch.no_grad():
            means_p, scales_p, quats_p = skinned.deform(smpl_T)
            rgb, _, _ = render_gaussians(means_p, scales_p, quats_p, sh0, skinned.opacities,
                                         cam, sh_degree=0, background=bg)
        frame = cv2.cvtColor((rgb.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.output_dir, f'frame_{idx:03d}.png'), frame)

    print(f"\n{'=' * 70}")
    print(f"Done! Outputs in {args.output_dir}/:")
    print(f"  yolo_skeleton_demo.mp4     — Gaussians + skeleton overlay (rotating camera)")
    print(f"  skeleton_vs_gaussians.mp4  — skeleton | gaussians side-by-side")
    print(f"  keypoints_2d.png           — YOLO11-pose 2D keypoints visualization")
    print(f"  frame_000/060/120/180.png  — key frame screenshots")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
