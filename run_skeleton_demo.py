"""Visualize SMPL-X skeleton motion as a standalone animation.

Renders the skeleton (joints + bones) for each pose frame, from a
rotating camera viewpoint, and saves as a video. No Gaussian splatting
involved — purely the skeletal motion.

Usage:
    conda activate rtg132
    python run_skeleton_demo.py \\
        --smpl_model_path E:/Code/mmforall/model/assets/SMPLX/SMPLX_NEUTRAL_2020.npz \\
        --output_dir output/skeleton_demo --n_frames 240 --fps 30
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

from skinned_gaussian.smplx_model import SMPLXWrapper
from run_real_body_demo import generate_body_actions

# Joint name colors (body=green, head=yellow, arms=red, hands=blue, legs=cyan)
JOINT_COLORS = {}
BODY_JOINTS = set(range(0, 15))  # pelvis → collars
HEAD_JOINTS = {15, 22, 23, 24}   # head, jaw, eyes
ARM_JOINTS = {16, 17, 18, 19, 20, 21}  # shoulders → wrists
HAND_JOINTS = set(range(25, 55))
LEG_JOINTS = {1, 2, 4, 5, 7, 8, 10, 11}

for j in range(55):
    if j in HEAD_JOINTS:
        JOINT_COLORS[j] = (0, 255, 255)    # yellow
    elif j in ARM_JOINTS:
        JOINT_COLORS[j] = (0, 0, 255)      # red
    elif j in HAND_JOINTS:
        JOINT_COLORS[j] = (255, 0, 0)      # blue
    elif j in LEG_JOINTS:
        JOINT_COLORS[j] = (255, 255, 0)    # cyan
    else:
        JOINT_COLORS[j] = (0, 255, 0)      # green (body)


def project_points(points_3d, camera, device='cuda'):
    """Project 3D points to 2D image coordinates using camera intrinsics + extrinsics."""
    # points_3d: (N, 3) in world space
    # camera: Camera with w2c, fx, fy, cx, cy
    N = points_3d.shape[0]
    homog = torch.cat([points_3d, torch.ones(N, 1, device=device)], dim=-1)  # (N, 4)
    cam_pts = (camera.w2c @ homog.T).T[:, :3]  # (N, 3) in camera space

    # Filter z > 0 (in front of camera)
    z = cam_pts[:, 2].clamp_min(0.01)
    u = camera.fx * cam_pts[:, 0] / z + camera.cx
    v = camera.fy * cam_pts[:, 1] / z + camera.cy
    return torch.stack([u, v, z], dim=-1)  # (N, 3) = (u, v, depth)


def render_skeleton_frame(joints_2d, parents, W, H, joint_radius=5, bone_thickness=2):
    """Render skeleton joints + bones on a 2D image."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    joints_np = joints_2d.cpu().numpy()

    # Draw bones (parent-child lines) first.
    for j in range(len(parents)):
        p = parents[j]
        if p < 0:
            continue
        x1, y1, z1 = joints_np[j]
        x2, y2, z2 = joints_np[p]
        if z1 < 0.01 or z2 < 0.01:
            continue
        # Color by child joint
        color = JOINT_COLORS.get(j, (200, 200, 200))
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, bone_thickness)

    # Draw joints on top.
    for j in range(len(joints_np)):
        x, y, z = joints_np[j]
        if z < 0.01:
            continue
        color = JOINT_COLORS.get(j, (255, 255, 255))
        cv2.circle(img, (int(x), int(y)), joint_radius, color, -1)
        # Outline
        cv2.circle(img, (int(x), int(y)), joint_radius, (255, 255, 255), 1)

    return img


def make_orbit_camera(centre, az, elev, radius, W, H, device='cuda', fx=700):
    """Create a single orbit camera at given azimuth."""
    from multiview_refine.cameras import Camera, invert_se3
    pos = centre + torch.tensor([
        radius * math.cos(elev) * math.sin(az),
        -radius * math.sin(elev),
        radius * math.cos(elev) * math.cos(az),
    ], device=device)
    forward = centre - pos
    forward = forward / forward.norm()
    up = torch.tensor([0., -1., 0.], device=device)
    right = torch.cross(forward, up)
    right = right / right.norm()
    down = torch.cross(right, forward)
    down = down / down.norm()
    c2w = torch.eye(4, device=device)
    c2w[:3, 0] = right; c2w[:3, 1] = down; c2w[:3, 2] = forward; c2w[:3, 3] = pos
    w2c = invert_se3(c2w)
    return Camera(w2c=w2c, c2w=c2w, position=pos,
                  fx=fx, fy=fx, cx=W/2, cy=H/2, width=W, height=H)


def main():
    parser = argparse.ArgumentParser(description="SMPL-X skeleton motion visualization.")
    parser.add_argument('--smpl_model_path', required=True)
    parser.add_argument('--output_dir', default='output/skeleton_demo')
    parser.add_argument('--n_frames', type=int, default=240)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--show_fingers', action='store_true', default=False,
                        help='Show finger joints (25-54). Default: hide for clarity.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ═══ 1. Load SMPL-X ═══
    print("=" * 60)
    print("SMPL-X Skeleton Motion Visualization")
    print("=" * 60)
    print("\nStep 1: Load SMPL-X model")
    smplx = SMPLXWrapper(args.smpl_model_path, device=args.device)

    # ═══ 2. Generate poses ═══
    print(f"\nStep 2: Generate {args.n_frames} action frames")
    pose_seq = generate_body_actions(args.n_frames, device)

    # ═══ 3. Compute joint positions for each frame ═══
    print(f"\nStep 3: Compute joint positions per frame")

    # For each frame, compute posed joint positions using forward kinematics.
    # We need the ABSOLUTE (world) joint positions, not relative transforms.
    # Use get_rest_joints + apply the posed transforms to rest joints.

    all_joints = []  # list of (55, 3) tensors
    rest_joints = smplx.get_rest_joints()  # (55, 3) in rest pose

    for i in range(args.n_frames):
        # Get absolute joint transforms (not relative).
        # We compute FK directly for posed, and extract joint positions from the
        # translation part of each joint's world transform.
        betas = pose_seq['betas']
        body_pose = pose_seq['body_pose'][i]
        global_orient = pose_seq['global_orient'][i]
        transl = pose_seq['transl'][i]
        jaw_pose = pose_seq['jaw_pose'][i]
        leye = pose_seq['leye_pose'][i]
        reye = pose_seq['reye_pose'][i]
        expression = pose_seq['expression'][i]

        # Build full pose.
        full_pose = torch.zeros(55, 3, device=device)
        full_pose[0] = global_orient
        full_pose[1:22] = body_pose.view(21, 3)
        full_pose[22] = jaw_pose
        full_pose[23] = leye
        full_pose[24] = reye

        # Forward kinematics (absolute transforms).
        G = smplx._forward_kinematics(betas, expression, full_pose, transl)
        joint_positions = G[:, :3, 3]  # (55, 3) world positions
        all_joints.append(joint_positions)

    print(f"  Computed {len(all_joints)} frames of joint positions")

    # ═══ 4. Render skeleton animation ═══
    print(f"\nStep 4: Render {args.n_frames} skeleton frames (rotating camera)")

    W, H = 1024, 768
    centre = rest_joints.median(0).values
    radius = 5.0
    elev = math.radians(10)

    # Determine which joints to show.
    if args.show_fingers:
        show_joints = list(range(55))
    else:
        # Hide finger joints (25-54) for clarity.
        show_joints = list(range(25))
    show_set = set(show_joints)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(os.path.join(args.output_dir, 'skeleton_motion.mp4'), fourcc, args.fps, (W, H))

    # Also make a side-by-side: skeleton | mesh
    from multiview_refine.gsplat_wrapper import render_gaussians
    from run_synthetic_demo import sample_gaussians_on_smpl

    # Sample Gaussians for side-by-side mesh rendering.
    print("  Sampling Gaussians for mesh comparison...")
    means, scales_raw, quats, opac_raw, sh0, skin_w = \
        sample_gaussians_on_smpl(smplx, n_gaussians=50000, device=device)
    # Boost scales
    mesh_size = (means.max(0).values - means.min(0).values).norm().item()
    scales_raw = torch.log(torch.full((50000, 3), mesh_size * 0.08, device=device))

    from skinned_gaussian.rigging import rig_gaussians_to_smplx
    skinned = rig_gaussians_to_smplx(
        means, scales_raw, quats, opac_raw, sh0, smplx,
        n_neighbours=4, temperature=0.1, device=device,
        weights_optimizable=False, n_expression_bases=100)

    bg = torch.zeros(3, device=device)
    vw_side = cv2.VideoWriter(os.path.join(args.output_dir, 'skeleton_vs_mesh.mp4'),
                               fourcc, args.fps, (W * 2, H))

    t0 = time.time()
    for i in range(args.n_frames):
        az = 2 * math.pi * i / args.n_frames
        cam = make_orbit_camera(centre, az, elev, radius, W, H, device)

        # Project joints to 2D.
        joints_3d = all_joints[i]
        joints_2d = project_points(joints_3d, cam, device)

        # Render skeleton.
        # Build a subset parents dict for visible joints.
        vis_parents = {}
        for j in show_joints:
            p = smplx.parents.get(j, -1)
            vis_parents[j] = p  # include even if parent not in show_set (will skip drawing)
        skel_img = render_skeleton_frame(joints_2d, vis_parents, W, H,
                                          joint_radius=5, bone_thickness=2)

        # Add frame info.
        cv2.putText(skel_img, f'Frame {i}/{args.n_frames}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(skel_img, f'Azimuth {math.degrees(az):.0f} deg', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        vw.write(skel_img)

        # Side-by-side: skeleton | mesh
        T = smplx.get_joint_transforms(
            betas=pose_seq['betas'],
            body_pose=pose_seq['body_pose'][i],
            global_orient=pose_seq['global_orient'][i],
            transl=pose_seq['transl'][i],
            jaw_pose=pose_seq['jaw_pose'][i],
            leye_pose=pose_seq['leye_pose'][i],
            reye_pose=pose_seq['reye_pose'][i],
            expression=pose_seq['expression'][i])
        with torch.no_grad():
            rgb, _, _ = skinned.render_posed(
                T, cam, background=bg, sh_degree=0,
                expression=pose_seq['expression'][i])
        mesh_img = cv2.cvtColor((rgb.permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        side = np.concatenate([skel_img, mesh_img], axis=1)
        vw_side.write(side)

        if i % 30 == 0:
            elapsed = time.time() - t0
            print(f"  frame {i:4d}/{args.n_frames}: {elapsed:.1f}s")

    vw.release()
    vw_side.release()
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s")
    print(f"  skeleton_motion.mp4: {os.path.getsize(os.path.join(args.output_dir, 'skeleton_motion.mp4'))//1024} KB")
    print(f"  skeleton_vs_mesh.mp4: {os.path.getsize(os.path.join(args.output_dir, 'skeleton_vs_mesh.mp4'))//1024} KB")

    # ═══ 5. Save key frames ═══
    print(f"\nStep 5: Save key frames")
    for idx in [0, 60, 120, 180]:
        az = 2 * math.pi * idx / args.n_frames
        cam = make_orbit_camera(centre, az, elev, radius, W, H, device)
        joints_2d = project_points(all_joints[idx], cam, device)
        vis_parents = {}
        for j in show_joints:
            vis_parents[j] = smplx.parents.get(j, -1)
        img = render_skeleton_frame(joints_2d, vis_parents, W, H, joint_radius=5, bone_thickness=2)
        cv2.putText(img, f'Frame {idx}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imwrite(os.path.join(args.output_dir, f'skeleton_{idx:03d}.png'), img)
        print(f"    skeleton_{idx:03d}.png")

    # ═══ 6. Legend ═══
    legend = np.zeros((200, 400, 3), dtype=np.uint8)
    cv2.putText(legend, 'Joint Color Legend', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    for y, (name, color) in enumerate([
        ('Body (green)', (0, 255, 0)),
        ('Head/Jaw/Eyes (yellow)', (0, 255, 255)),
        ('Arms (red)', (0, 0, 255)),
        ('Hands (blue)', (255, 0, 0)),
        ('Legs (cyan)', (255, 255, 0)),
    ], start=1):
        cv2.circle(legend, (20, y * 35 + 10), 8, color, -1)
        cv2.putText(legend, name, (40, y * 35 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(os.path.join(args.output_dir, 'legend.png'), legend)

    print(f"\n{'='*60}")
    print(f"Done! Outputs in {args.output_dir}/:")
    print(f"  skeleton_motion.mp4     — pure skeleton animation (rotating camera)")
    print(f"  skeleton_vs_mesh.mp4    — skeleton | mesh side-by-side")
    print(f"  skeleton_000/060/120/180.png — key frame screenshots")
    print(f"  legend.png              — joint color legend")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
