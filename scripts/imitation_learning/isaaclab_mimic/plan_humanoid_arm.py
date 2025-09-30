# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# #!/usr/bin/env python3
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Plan and execute a humanoid arm lift with cuRobo.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--enable_pinocchio", action="store_true", default=False)

parser.add_argument("--arm", type=str, default="right", choices=["right", "left"])
parser.add_argument("--goal", type=str, default="up", choices=["up", "forward", "random"])
parser.add_argument("--dz", type=float, default=0.05, help="Upward lift in meters (for 'up' goal).")
parser.add_argument("--dx", type=float, default=0.05, help="Forward motion in meters (for 'forward' goal).")
parser.add_argument("--retime_deg", type=float, default=1.0, help="Joint retime step (deg); 0 disables retiming.")
parser.add_argument("--rest", type=int, default=10, help="Initial rest steps before planning.")
parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
parser.add_argument("--replay_trials", type=int, default=10, help="Number of trials to replay.")
parser.add_argument("--visualize_goal", action="store_true", help="Visualize target EE pose marker.")


# append AppLauncher cli args and parse
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import os
import tempfile
import torch
import yaml

# from isaaclab_mimic.datagen.generation import setup_env_config
import isaaclab.utils.math as PoseUtils

# Controller utils to convert USD->URDF
from isaaclab.controllers import utils as ControllerUtils
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
from isaaclab_mimic.envs.pinocchio_envs.pickplace_gr1t2_mimic_env_cfg import PickPlaceGR1T2MimicEnvCfg
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
from isaaclab_mimic.motion_planners.curobo.curobo_planner_humanoid import HumanoidArmCuroboPlanner

import isaaclab_tasks  # noqa: F401


# def _detect_robot_usd_path(env):
#     """Detect robot USD path from environment configuration."""
#     try:
#         usd_path = env.cfg.scene.robot.spawn.usd_path
#         if isinstance(usd_path, str) and len(usd_path) > 0:
#             return usd_path
#     except Exception:
#         # Default GR1T2 USD path
#         usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Robots/FourierIntelligence/GR-1/GR1T2_fourier_hand_6dof/GR1T2_fourier_hand_6dof.usd"
#     return usd_path


def _tool_link_for_arm(arm: str) -> str:
    """Get tool link name for the specified arm."""
    # Matches the converted GR1T2 URDF link names
    return f"GR1T2_fourier_hand_6dof_{arm}_hand_pitch_link"


def to_python(obj):
    """Convert numpy/torch types to Python native types for YAML serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj


def _build_temp_robot_yaml_from_usd(usd_path: str, arm: str) -> str:
    """Build cuRobo robot configuration YAML from USD file."""
    tmp_dir = tempfile.mkdtemp(prefix="gr1_curobo_")
    print("[PlanHumanoid] Converting USD to URDF...")
    urdf_path, _ = ControllerUtils.convert_usd_to_urdf(usd_path, tmp_dir, force_conversion=True)
    print(f"[PlanHumanoid] URDF: {urdf_path}")

    print("[PlanHumanoid] Creating robot config...")
    try:
        from nvplan.applications.custream.config import create_robot_config
        from nvplan.applications.custream.spheres import load_spheres
    except Exception as e:
        print(f"[PlanHumanoid] Error importing custream: {e}")
        raise e

    print("[PlanHumanoid] Loading spheres...")
    tool_links = [_tool_link_for_arm(arm)]
    try:
        robot_config = create_robot_config(
            urdf_path,
            tool_links=tool_links,
            inactive_joints=[],
            depth=2,
            verbose=False,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[PlanHumanoid] Error creating robot config: {e}")
        raise e

    # Configure sphere generation
    max_spheres = 225 - len(tool_links) * 50
    load_spheres(robot_config, max_spheres=max_spheres, max_link_spheres=int(1e9))
    robot_cfg_dict = robot_config["robot_cfg"]
    robot_cfg_yaml = to_python(robot_cfg_dict)

    def _strip_keys(obj, keys):
        """Remove specified keys from nested dictionary."""
        if isinstance(obj, dict):
            for k in list(obj.keys()):
                if k in keys:
                    obj.pop(k, None)
                else:
                    _strip_keys(obj[k], keys)
        elif isinstance(obj, list):
            for v in obj:
                _strip_keys(v, keys)

    # Remove lock_joints and cspace as they'll be configured dynamically
    _strip_keys(robot_cfg_yaml, {"lock_joints", "cspace"})

    # Ensure ee_link points to the selected arm's tool link
    if "kinematics" in robot_cfg_yaml:
        robot_cfg_yaml["kinematics"]["ee_link"] = _tool_link_for_arm(arm)

    out_dir = tempfile.mkdtemp(prefix="curobo_robot_cfg_")
    out_path = os.path.join(out_dir, "gr1_generated.yml")
    print(f"[PlanHumanoid] Writing robot YAML to {out_path} ...")
    with open(out_path, "w") as f:
        yaml.safe_dump({"robot_cfg": robot_cfg_yaml}, f, sort_keys=False)
    print("[PlanHumanoid] Robot YAML written.")
    return out_path


def rest_with_idle_action(env, steps=10):
    cfg = getattr(env, "cfg", None)
    if cfg is None or not hasattr(cfg, "idle_action"):
        raise AttributeError("[PlanHumanoid] This env has no cfg.idle_action defined.")

    idle = cfg.idle_action
    if not isinstance(idle, torch.Tensor):
        idle = torch.tensor(idle, dtype=torch.float32)
    else:
        idle = idle.to(dtype=torch.float32)
    idle = idle.to(device=env.device)

    act_dim = env.action_manager.total_action_dim
    if idle.shape[-1] != act_dim:
        raise ValueError(f"[PlanHumanoid] Idle action dim mismatch ({idle.shape[-1]} != {act_dim}).")

    if idle.dim() == 1:
        idle_batched = idle.unsqueeze(0).repeat(env.num_envs, 1)
    elif idle.dim() == 2 and idle.size(0) == 1:
        idle_batched = idle.repeat(env.num_envs, 1)
    elif idle.dim() == 2 and idle.size(0) == env.num_envs:
        idle_batched = idle
    else:
        raise ValueError(f"[PlanHumanoid] Unexpected idle_action shape: {tuple(idle.shape)}")

    print(f"[PlanHumanoid] Resting with idle action for {steps} steps...")
    for i in range(steps):
        env.step(idle_batched)
        if (i + 1) % 5 == 0 or i == steps - 1:
            print(f"  Rest step {i + 1}/{steps}")


def generate_goal_pose(current_pose: torch.Tensor, goal_type: str, args) -> torch.Tensor:
    """Generate target pose based on goal type."""
    target_pose = current_pose.clone()

    if goal_type == "up":
        # Move up by dz
        target_pose[2, 3] = target_pose[2, 3] + float(args.dz)
        print(f"[PlanHumanoid] Goal: Move up by {args.dz}m")
    elif goal_type == "forward":
        # Move forward by dx
        target_pose[0, 3] = target_pose[0, 3] + float(args.dx)
        print(f"[PlanHumanoid] Goal: Move forward by {args.dx}m")
    elif goal_type == "random":
        # Random offset in x, y, z
        offset = torch.randn(3) * 0.05  # 5cm standard deviation
        target_pose[:3, 3] = target_pose[:3, 3] + offset.to(target_pose.device)
        print(f"[PlanHumanoid] Goal: Random offset {offset.cpu().numpy()}")

    return target_pose


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # Load environment
    env_name = "Isaac-PickPlace-GR1T2-Abs-Mimic-v0"
    print(f"[PlanHumanoid] Env: {env_name}")

    print("[PlanHumanoid] Building env config...")
    env_cfg = PickPlaceGR1T2MimicEnvCfg()
    env_cfg.num_envs = 1

    planner_cfg = CuroboPlannerCfg()
    planner_cfg.visualize_plan = True
    planner_cfg.visualize_spheres = False
    planner_cfg.debug_planner = args_cli.debug

    # Build robot YAML before the env is created to avoid PhysX invalidation
    usd_path = env_cfg.scene.robot.spawn.usd_path
    print(f"[PlanHumanoid] Detected robot USD: {usd_path}")
    robot_yaml = _build_temp_robot_yaml_from_usd(usd_path, args_cli.arm)
    print(f"[PlanHumanoid] Generated cuRobo robot YAML: {robot_yaml}")

    print("[PlanHumanoid] Creating env...")
    try:
        env = gym.make(env_name, cfg=env_cfg).unwrapped
        env.reset()
    except Exception as e:
        print(f"[PlanHumanoid] Error creating env: {e}")
        raise e
    print("[PlanHumanoid] Env ready.")

    if args_cli.rest > 0:
        rest_with_idle_action(env, steps=args_cli.rest)

    # Finish planner config
    planner_cfg.robot_config_file = robot_yaml
    planner_cfg.robot_name = "gr1"
    planner_cfg.approach_distance = 0.01
    planner_cfg.retreat_distance = 0.01
    planner_cfg.time_dilation_factor = 0.5
    planner_cfg.enable_finetune_trajopt = True
    planner_cfg.ee_link_name = _tool_link_for_arm(args_cli.arm)
    planner_cfg.enable_graph = True
    planner_cfg.enable_graph_attempt = 4
    planner_cfg.max_planning_attempts = 10

    # Configure gripper positions for GR1
    planner_cfg.gripper_open_positions = {}
    planner_cfg.gripper_closed_positions = {}

    # Set up arm-specific configuration
    if args_cli.arm == "right":
        active_joint_substrings = ("right_",)
        hand_link_substrings = ("GR1T2_fourier_hand_6dof_right_",)
    else:
        active_joint_substrings = ("left_",)
        hand_link_substrings = ("GR1T2_fourier_hand_6dof_left_",)

    print("[PlanHumanoid] Creating planner...")
    robot = env.scene["robot"]

    try:
        planner = HumanoidArmCuroboPlanner(
            env=env,
            robot=robot,
            config=planner_cfg,
            env_id=0,
            active_joint_substrings=active_joint_substrings,
            hand_link_substrings=hand_link_substrings,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[PlanHumanoid] Error creating planner: {e}")
        raise e
    print("[PlanHumanoid] Planner ready.")

    # # Get current end-effector pose
    # cu_js = planner._get_current_joint_state_for_curobo()
    # ee_pose_cu = planner.get_ee_pose(cu_js)

    # # Build 4x4 pose: convert quaternion -> rotation matrix (make_pose expects a rot mat)
    # pos = planner._to_env_device(ee_pose_cu.position).reshape(-1, 3)[0]
    # cu_quat = planner._to_env_device(getattr(ee_pose_cu, "quaternion", ee_pose_cu.get_rotation())).reshape(-1, 4)[0]
    # rot = PoseUtils.matrix_from_quat(cu_quat.unsqueeze(0))[0]
    # current_pose: torch.Tensor = PoseUtils.make_pose(pos.unsqueeze(0), rot.unsqueeze(0))[0]

    # # Map from cuRobo EE frame to env controller EE frame (left/right)
    # eef_name = args_cli.arm  # "left" or "right"
    # env_eef_pose = env.get_robot_eef_pose(eef_name)[0].to(device=env.device, dtype=torch.float32)

    # # Env origin extraction
    # env_origin = env.scene.env_origins[0].to(device=env.device, dtype=torch.float32)
    # print(f"[PlanHumanoid] Env origin: {env_origin}")
    # env_eef_pos_env = env_eef_pose.clone()
    # env_eef_pos_env[:3, 3] = env_eef_pos_env[:3, 3] - env_origin[:3]

    # site_from_curobo = torch.linalg.solve(current_pose, env_eef_pos_env)

    # print("[PlanHumanoid] Current EE pose:")
    # print("Position: {current_pose[:3, 3].cpu().numpy()}")
    # print("Rotation:\n{current_pose[:3, :3].cpu().numpy()}")

    #    # Generate target pose based on goal type (in cuRobo frame)
    # target_pose = generate_goal_pose(current_pose, args_cli.goal, args_cli)
    # print(f"[PlanHumanoid] Target EE pos (cuRobo frame): {target_pose[:3, 3].cpu().numpy()}")

    # # Also compute target in env controller site frame for visualization/execution
    # target_pose_env_site = (target_pose @ site_from_curobo).clone()

    # # Configure retiming
    # step_size = np.deg2rad(args_cli.retime_deg) if args_cli.retime_deg > 0 else None

    # print(f"target_pose: {target_pose}")
    # print("[PlanHumanoid] Planning...")
    # try:
    #     ok = planner.update_world_and_plan_motion(
    #         target_pose=target_pose_env_site, # target_pose_env_site
    #         expected_attached_object=None,
    #         env_id=0,
    #         step_size=step_size,
    #         enable_retiming=step_size is not None,
    #     )
    # except Exception as e:
    #     import traceback

    #     traceback.print_exc()
    #     print(f"[PlanHumanoid] Error planning: {e}")
    #     raise e
    # Get current end-effector pose in cuRobo (tool) frame
    cu_js = planner._get_current_joint_state_for_curobo()
    ee_pose_cu = planner.get_ee_pose(cu_js)

    # Build 4x4 pose for cuRobo tool frame
    pos = planner._to_env_device(ee_pose_cu.position).reshape(-1, 3)[0]
    cu_quat = planner._to_env_device(getattr(ee_pose_cu, "quaternion", ee_pose_cu.get_rotation())).reshape(-1, 4)[0]
    rot = PoseUtils.matrix_from_quat(cu_quat.unsqueeze(0))[0]
    current_pose: torch.Tensor = PoseUtils.make_pose(pos.unsqueeze(0), rot.unsqueeze(0))[0]

    eef_name = args_cli.arm  # "left" or "right"
    ctrl_site_world = env.get_robot_eef_pose(eef_name)[0].to(device=env.device, dtype=torch.float32)

    env_origin = env.scene.env_origins[0].to(device=env.device, dtype=torch.float32)
    ctrl_site_env = ctrl_site_world.clone()
    ctrl_site_env[:3, 3] = ctrl_site_env[:3, 3] - env_origin[:3]

    site_from_curobo = torch.linalg.solve(current_pose, ctrl_site_env)

    # Sanity: current tool -> site should reconstruct
    recon_env = (current_pose @ site_from_curobo).clone()
    pos_err = torch.linalg.vector_norm(recon_env[:3, 3] - ctrl_site_env[:3, 3]).item()
    rot_err_mat = recon_env[:3, :3].T @ ctrl_site_env[:3, :3]
    rot_err_trace = torch.clamp((torch.trace(rot_err_mat) - 1.0) / 2.0, -1.0, 1.0)
    rot_err_rad = torch.acos(rot_err_trace).item()
    print(f"[PlanHumanoid] Mapping check | pos_err={pos_err:.4e} m | rot_err={rot_err_rad:.4e} rad")

    print("[PlanHumanoid] Current EE pose:")
    print(f"Position: {current_pose[:3, 3].cpu().numpy()}")
    print(f"Rotation:\n{current_pose[:3, :3].cpu().numpy()}")

    target_site_world = ctrl_site_world.clone()
    if args_cli.goal == "up":
        target_site_world[2, 3] = target_site_world[2, 3] + float(args_cli.dz)
        print(f"[PlanHumanoid] Goal: Move up by {args_cli.dz}m (world)")
    elif args_cli.goal == "forward":
        target_site_world[0, 3] = target_site_world[0, 3] + float(args_cli.dx)
        print(f"[PlanHumanoid] Goal: Move forward by {args_cli.dx}m (world)")
    elif args_cli.goal == "random":
        offset = torch.randn(3, device=env.device, dtype=torch.float32) * 0.05
        target_site_world[:3, 3] = target_site_world[:3, 3] + offset
        print(f"[PlanHumanoid] Goal: Random world offset {offset.cpu().numpy()}")

    target_site_env = target_site_world.clone()
    target_site_env[:3, 3] = target_site_env[:3, 3] - env_origin[:3]

    site_inv = torch.linalg.inv(site_from_curobo)
    target_pose = (target_site_env @ site_inv).clone()

    target_pose_env_site = target_site_env.clone()
    # target_pose_env_site_world = target_site_world.clone()

    step_size = np.deg2rad(args_cli.retime_deg) if args_cli.retime_deg > 0 else None

    print(f"target_pose (tool frame, cuRobo): {target_pose}")
    print("[PlanHumanoid] Planning...")
    try:
        ok = planner.update_world_and_plan_motion(
            target_pose=target_pose,
            expected_attached_object=None,
            env_id=0,
            step_size=step_size,
            enable_retiming=step_size is not None,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[PlanHumanoid] Error planning: {e}")
        raise e

    print(f"[PlanHumanoid] Plan success: {ok}")
    if not ok:
        print("Planning failed.")
        return

    # Visualize goal pose
    if args_cli.visualize_goal:
        try:
            # Use a fresh instancer prim path each run to avoid stale Prototypes rel
            viz_path_goal = "/World/Visuals/goal_pose_marker"
            viz_path_ee = "/World/Visuals/ee_pose_marker"

            frame_cfg_goal = FRAME_MARKER_CFG.copy()
            frame_cfg_goal.markers["frame"].scale = (0.1, 0.1, 0.1)
            frame_cfg_goal = frame_cfg_goal.replace(prim_path=viz_path_goal)

            frame_cfg_ee = FRAME_MARKER_CFG.copy()
            frame_cfg_ee.markers["frame"].scale = (0.08, 0.08, 0.08)
            frame_cfg_ee = frame_cfg_ee.replace(prim_path=viz_path_ee)

            goal_pose_visualizer = VisualizationMarkers(frame_cfg_goal)
            ee_pose_visualizer = VisualizationMarkers(frame_cfg_ee)

            goal_pos = target_pose_env_site[:3, 3].detach().to(dtype=torch.float32)
            goal_quat = PoseUtils.quat_from_matrix(target_pose_env_site[:3, :3].unsqueeze(0))[0].detach().to(dtype=torch.float32)

            # Current EE pose marker at time of planning
            cur_eef_pose = env.get_robot_eef_pose(eef_name)[0].to(device=env.device, dtype=torch.float32)
            cur_pos = cur_eef_pose[:3, 3].detach()
            cur_quat = PoseUtils.quat_from_matrix(cur_eef_pose[:3, :3].unsqueeze(0))[0].detach()

            goal_pose_visualizer.visualize(translations=goal_pos.unsqueeze(0), orientations=goal_quat.unsqueeze(0))
            ee_pose_visualizer.visualize(translations=cur_pos.unsqueeze(0), orientations=cur_quat.unsqueeze(0))
        except Exception as e:
            print(f"[PlanHumanoid] Goal visualization failed: {e}")
            # Do not exit the app on viz failure


    # Get planned poses
    print(f"Current plan in joint space: {planner.current_plan}")
    planned_poses = planner.get_planned_poses()
    print(f"[PlanHumanoid] Generated {len(planned_poses)} waypoints")

    # Visualize planned path as a spline in world space (env-site frame)
    try:
        from isaacsim.util.debug_draw import _debug_draw
        draw = _debug_draw.acquire_debug_draw_interface()
        waypoint_points = []
        for _p in planned_poses:
            if _p.dim() == 3 and _p.size(0) == 1:
                _p = _p[0]
            _p = _p.to(device=env.device, dtype=torch.float32)
            # Ensure valid homogeneous transform before conversion
            _p_env = _p.clone()
            # _p[3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device, dtype=torch.float32)
            _p_env = (_p @ site_from_curobo).clone()
            # # Ensure valid homogeneous transform after conversion as well
            _p_env[3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device, dtype=torch.float32)
            waypoint_points.append((
                float(_p_env[0, 3].item()),
                float(_p_env[1, 3].item()),
                float(_p_env[2, 3].item()),
            ))
        if len(waypoint_points) >= 2:
            # color: cyan, thickness: 6, cyclic: False
            draw.draw_lines_spline(waypoint_points, (0.0, 0.8, 1.0, 1.0), 6, False)
    except Exception as e:
        print(f"[PlanHumanoid] Debug draw for planned path failed: {e}")

    if len(planned_poses) == 0:
        print("[PlanHumanoid] No waypoints generated!")
        return

    # Execute the plan
    print(f"[PlanHumanoid] Executing {len(planned_poses)} waypoints...")

    # Idle action (device/dtype aligned)
    idle = env.cfg.idle_action
    if not isinstance(idle, torch.Tensor):
        idle = torch.tensor(idle, dtype=torch.float32)
    else:
        idle = idle.to(dtype=torch.float32)
    idle = idle.to(device=env.device)

    # Helper: build a 4x4 pose from idle slices for a given arm
    def _pose_from_idle(_idle: torch.Tensor, arm: str) -> torch.Tensor:
        if arm == "left":
            pos = _idle[0:3]
            quat = _idle[3:7]
        else:
            pos = _idle[7:10]
            quat = _idle[10:14]
        rot = PoseUtils.matrix_from_quat(quat.unsqueeze(0))[0]
        return PoseUtils.make_pose(pos.unsqueeze(0), rot.unsqueeze(0))[0].to(env.device)

    fixed_arm = "left" if args_cli.arm == "right" else "right"
    fixed_pose = _pose_from_idle(idle, fixed_arm)

    # Execute waypoints
    for _ in range(args_cli.replay_trials):
        print(f"[PlanHumanoid] Replaying trial {_ + 1}/{args_cli.replay_trials}")
        env.reset()
        for idx, target_ee_pose in enumerate(planned_poses):
            # import pdb; pdb.set_trace()
            # sanitize target pose to 4x4 homogeneous, device/dtype
            if target_ee_pose.dim() == 3 and target_ee_pose.size(0) == 1:
                target_ee_pose = target_ee_pose[0]
            target_ee_pose = target_ee_pose.to(device=env.device, dtype=torch.float32)
            target_ee_pose = (target_ee_pose @ site_from_curobo).clone()
            target_ee_pose[:3, 3] = target_ee_pose[:3, 3] + env_origin[:3]
            target_ee_pose[3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device, dtype=torch.float32)

            if args_cli.arm == "right":
                target_dict = {"left": fixed_pose, "right": target_ee_pose}
            else:
                target_dict = {"left": target_ee_pose, "right": fixed_pose}

            action = env.target_eef_pose_to_action(
                target_eef_pose_dict=target_dict,
                gripper_action_dict={"left": idle[14:25], "right": idle[25:36]},  # keep fixed grippers truly idle
                action_noise_dict=None,
                env_id=0,
            )
            if action.ndim == 1:
                action = action.unsqueeze(0)
            action = action.to(device=env.device, dtype=torch.float32)

            if args_cli.visualize_goal:
                cur_eef_pose = env.get_robot_eef_pose(eef_name)[0].to(device=env.device, dtype=torch.float32)
                cur_pos = cur_eef_pose[:3, 3].detach()
                cur_quat = PoseUtils.quat_from_matrix(cur_eef_pose[:3, :3].unsqueeze(0))[0].detach()
                ee_pose_visualizer.visualize(translations=cur_pos.unsqueeze(0), orientations=cur_quat.unsqueeze(0))

            env.step(action)

            if (idx + 1) % 10 == 0 or idx == 0 or idx == len(planned_poses) - 1:
                print(f"[PlanHumanoid] Step {idx + 1}/{len(planned_poses)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    finally:
        simulation_app.close()
