# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# #!/usr/bin/env python3
import argparse
import traceback

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Plan and execute a humanoid arm lift with cuRobo.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--enable_pinocchio", action="store_true", default=False)

parser.add_argument("--arm", type=str, default="right", choices=["right", "left", "both"])
parser.add_argument("--goal", type=str, default="up", choices=["up", "lateral", "forward", "random"])
# dz, dx, dy semantics:
# - dz: upward offset in meters applied to target(s)
# - dy: forward offset in meters (positive moves away from the torso in +Y)
# - dx: for bimanual mode, half the lateral gap along X around the hand midpoint;
#       total hand-to-hand separation becomes 2*|dx|. Negative dx makes the arms cross sides.
parser.add_argument("--dz", type=float, default=0.05, help="Upward lift in meters.")
parser.add_argument(
    "--dx", type=float, default=0.05, help="Half lateral gap (bimanual) or lateral offset (single arm), in meters."
)
parser.add_argument("--dy", type=float, default=0.05, help="Forward offset in meters.")
# Retiming: --retime_deg > 0 enables linear resampling of the joint path with approximately
# uniform arc-length spacing of step_size = deg2rad(retime_deg). Use 0 to disable retiming.
parser.add_argument("--retime_deg", type=float, default=1.0, help="Joint retime step (deg); 0 disables retiming.")
parser.add_argument("--rest", type=int, default=10, help="Initial rest steps before planning.")
parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
parser.add_argument("--replay_trials", type=int, default=2, help="Number of trials to replay.")
parser.add_argument("--visualize_goal", action="store_true", help="Visualize target EE pose marker.")


# Append AppLauncher cli args and parse
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
from dataclasses import replace as dc_replace
from typing import Any, cast

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


def _build_temp_robot_yaml_from_usd(usd_path: str, arm: str, inactive_joints: list[str] | None = None) -> str:
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
    robot_config = create_robot_config(
        urdf_path,
        tool_links=tool_links,
        inactive_joints=inactive_joints or [],
        depth=2,
        verbose=False,
    )

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

    # Ensure ee_link points to the selected arm's tool link (guard against malformed YAML)
    if isinstance(robot_cfg_yaml, dict):
        kin = robot_cfg_yaml.get("kinematics")
        if isinstance(kin, dict):
            kin["ee_link"] = _tool_link_for_arm(arm)

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
    elif goal_type == "lateral":
        # Move laterally by dx
        target_pose[0, 3] = target_pose[0, 3] + float(args.dx)
        print(f"[PlanHumanoid] Goal: Move laterally by {args.dx}m")
    elif goal_type == "forward":
        # Move forward by dy
        target_pose[1, 3] = target_pose[1, 3] + float(args.dy)
        print(f"[PlanHumanoid] Goal: Move forward by {args.dy}m")
    elif goal_type == "random":
        # Random offset in x, y, z
        offset = torch.randn(3) * 0.05  # 5cm standard deviation
        target_pose[:3, 3] = target_pose[:3, 3] + offset.to(target_pose.device)
        print(f"[PlanHumanoid] Goal: Random offset {offset.cpu().numpy()}")

    return target_pose


def _build_env_and_planner(args_cli):
    """Create env, planner config, env, robot, and planner. Preserve logging."""
    # Load environment
    env_name = "Isaac-PickPlace-GR1T2-Abs-Mimic-v0"
    print(f"[PlanHumanoid] Env: {env_name}")

    print("[PlanHumanoid] Building env config...")
    env_cfg = PickPlaceGR1T2MimicEnvCfg()
    env_cfg.scene.num_envs = 1

    planner_cfg = CuroboPlannerCfg()
    planner_cfg.visualize_plan = True
    planner_cfg.visualize_spheres = False
    planner_cfg.debug_planner = args_cli.debug

    # Build robot YAML before the env is created to avoid PhysX invalidation
    usd_path = env_cfg.scene.robot.spawn.usd_path
    print(f"[PlanHumanoid] Detected robot USD: {usd_path}")
    inactive_joint_names = []
    try:
        inactive_joint_names = list(env_cfg.actions.pink_ik_cfg.ik_urdf_fixed_joint_names)
    except Exception as e:
        print(f"[PlanHumanoid] Error getting inactive joint names: {e}")
    robot_yaml = _build_temp_robot_yaml_from_usd(
        usd_path, args_cli.arm if args_cli.arm in ("left", "right") else "right", inactive_joints=inactive_joint_names
    )
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
    planner_cfg.approach_distance = 0.0
    planner_cfg.retreat_distance = 0.0
    planner_cfg.time_dilation_factor = 0.5
    planner_cfg.enable_finetune_trajopt = True
    planner_cfg.ee_link_name = _tool_link_for_arm(args_cli.arm if args_cli.arm in ("left", "right") else "right")
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
        active_joint_substrings = ("left_",) if args_cli.arm == "left" else ("right_",)
        hand_link_substrings = (
            ("GR1T2_fourier_hand_6dof_left_",) if args_cli.arm == "left" else ("GR1T2_fourier_hand_6dof_right_",)
        )

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
        traceback.print_exc()
        print(f"[PlanHumanoid] Error creating planner: {e}")
        raise e
    print("[PlanHumanoid] Planner ready.")

    return env, robot, planner


def _build_env_and_planners_both(args_cli):
    """Create env and two planners for right and left arms for sequential planning."""
    # Build common env and base config
    env_name = "Isaac-PickPlace-GR1T2-Abs-Mimic-v0"
    print(f"[PlanHumanoid] Env: {env_name}")

    print("[PlanHumanoid] Building env config (both arms)...")
    env_cfg = PickPlaceGR1T2MimicEnvCfg()
    env_cfg.scene.num_envs = 1

    # Build two robot YAMLs for each arm to ensure correct ee_link in kinematics
    usd_path = env_cfg.scene.robot.spawn.usd_path
    print(f"[PlanHumanoid] Detected robot USD: {usd_path}")
    inactive_joint_names = []
    try:
        inactive_joint_names = list(env_cfg.actions.pink_ik_cfg.ik_urdf_fixed_joint_names)
    except Exception:
        pass
    robot_yaml_right = _build_temp_robot_yaml_from_usd(usd_path, "right", inactive_joints=inactive_joint_names)
    robot_yaml_left = _build_temp_robot_yaml_from_usd(usd_path, "left", inactive_joints=inactive_joint_names)

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

    # Base planner config fields shared
    def _make_cfg(robot_yaml_path: str, ee_arm: str) -> CuroboPlannerCfg:
        cfg = CuroboPlannerCfg()
        cfg.visualize_plan = True
        cfg.visualize_spheres = False
        cfg.debug_planner = args_cli.debug
        cfg.robot_config_file = robot_yaml_path
        cfg.robot_name = "gr1"
        cfg.approach_distance = 0.0
        cfg.retreat_distance = 0.0
        cfg.time_dilation_factor = 0.5
        cfg.enable_finetune_trajopt = True
        cfg.ee_link_name = _tool_link_for_arm(ee_arm)
        cfg.enable_graph = True
        cfg.enable_graph_attempt = 4
        cfg.max_planning_attempts = 10
        cfg.gripper_open_positions = {}
        cfg.gripper_closed_positions = {}
        return cfg

    cfg_right = _make_cfg(robot_yaml_right, "right")
    cfg_left = _make_cfg(robot_yaml_left, "left")

    robot = env.scene["robot"]

    print("[PlanHumanoid] Creating planners for both arms...")
    try:
        # Note: collision_active_link_substrings keeps collision spheres enabled for both arms
        # during planning, so each arm's plan respects the other arm's geometry.
        planner_right = HumanoidArmCuroboPlanner(
            env=env,
            robot=robot,
            config=cfg_right,
            env_id=0,
            active_joint_substrings=("right_",),
            hand_link_substrings=("GR1T2_fourier_hand_6dof_right_",),
            collision_active_link_substrings=("left_", "right_"),
        )
        planner_left = HumanoidArmCuroboPlanner(
            env=env,
            robot=robot,
            config=cfg_left,
            env_id=0,
            active_joint_substrings=("left_",),
            hand_link_substrings=("GR1T2_fourier_hand_6dof_left_",),
            collision_active_link_substrings=("left_", "right_"),
        )
    except Exception as e:
        traceback.print_exc()
        print(f"[PlanHumanoid] Error creating planners: {e}")
        raise e
    print("[PlanHumanoid] Planners ready.")

    return env, robot, planner_right, planner_left


def _compute_world_and_site_frames(env, robot, planner, eef_name: str):
    """Compute ctrl site (world), env origin, T_W_B, T_W_T, and T_T_S mapping."""
    # Controller EEF pose (world/env-origin frame)
    ctrl_site_env = env.get_robot_eef_pose(eef_name)[0].to(device=env.device, dtype=torch.float32)

    # Env origin
    env_origin = env.scene.env_origins[0].to(device=env.device, dtype=torch.float32)
    print(f"[PlanHumanoid] Env origin: {env_origin}")

    # Current tool pose from cuRobo FK (base frame), then convert to world via robot root pose
    cu_js = planner._get_current_joint_state_for_curobo()
    ee_pose_cu = planner.get_ee_pose(cu_js)
    pos = planner._to_env_device(ee_pose_cu.position).reshape(-1, 3)[0]
    cu_quat = planner._to_env_device(getattr(ee_pose_cu, "quaternion", ee_pose_cu.get_rotation())).reshape(-1, 4)[0]
    # FK: tool pose in base frame (T_B_T)
    rot = PoseUtils.matrix_from_quat(cu_quat.unsqueeze(0))[0]
    T_base_tool_now = PoseUtils.make_pose(pos.unsqueeze(0), rot.unsqueeze(0))[0]

    # Base pose in world (env-origin) (T_W_B)
    base_pos_world = (robot.data.root_pos_w[0] - env_origin).to(device=env.device, dtype=torch.float32)
    base_rot_world = PoseUtils.matrix_from_quat(
        robot.data.root_quat_w[0].unsqueeze(0).to(device=env.device, dtype=torch.float32)
    )[0]
    T_world_base = PoseUtils.make_pose(base_pos_world.unsqueeze(0), base_rot_world.unsqueeze(0))[0]
    # Compose: tool in world (T_W_T = T_W_B @ T_B_T)
    T_world_tool_now = (T_world_base @ T_base_tool_now).clone()

    # Calibrate mapping tool->site (T_T_S = inv(T_W_T) @ T_W_S)
    site_from_curobo = torch.linalg.solve(T_world_tool_now, ctrl_site_env)

    # Mapping sanity check
    recon_env = (T_world_tool_now @ site_from_curobo).clone()
    pos_err = torch.linalg.vector_norm(recon_env[:3, 3] - ctrl_site_env[:3, 3]).item()
    rot_err_mat = recon_env[:3, :3].T @ ctrl_site_env[:3, :3]
    rot_err_trace = torch.clamp((torch.trace(rot_err_mat) - 1.0) / 2.0, -1.0, 1.0)
    rot_err_rad = torch.acos(rot_err_trace).item()
    print(f"[PlanHumanoid] Mapping check | pos_err={pos_err:.4e} m | rot_err={rot_err_rad:.4e} rad")

    print("[PlanHumanoid] Current EE pose:")
    print(f"Position: {T_world_tool_now[:3, 3].cpu().numpy()}")
    print(f"Rotation:\n{T_world_tool_now[:3, :3].cpu().numpy()}")

    return env_origin, ctrl_site_env, T_world_base, T_world_tool_now, site_from_curobo


def _build_site_goal(ctrl_site_env: torch.Tensor, args_cli, device) -> torch.Tensor:
    """Build goal in controller site (world/env-origin) frame (T_W_S_goal)."""
    target_pose_env_site = ctrl_site_env.clone()
    if args_cli.goal == "up":
        target_pose_env_site[2, 3] = target_pose_env_site[2, 3] + float(args_cli.dz)
        print(f"[PlanHumanoid] Goal: Move up by {args_cli.dz}m (site world)")
    elif args_cli.goal == "lateral":
        target_pose_env_site[0, 3] = target_pose_env_site[0, 3] + float(args_cli.dx)
        print(f"[PlanHumanoid] Goal: Move laterally by {args_cli.dx}m (site world)")
    elif args_cli.goal == "forward":
        target_pose_env_site[1, 3] = target_pose_env_site[1, 3] + float(args_cli.dy)
        print(f"[PlanHumanoid] Goal: Move forward by {args_cli.dy}m (site world)")
    elif args_cli.goal == "random":
        offset = torch.randn(3, device=device, dtype=torch.float32) * 0.05
        target_pose_env_site[:3, 3] = target_pose_env_site[:3, 3] + offset
        print(f"[PlanHumanoid] Goal: Random world offset {offset.cpu().numpy()}")
    return target_pose_env_site


def _plan_motion(planner, target_world_tool: torch.Tensor, step_size: float | None) -> bool:
    print(f"target_pose (world tool frame for cuRobo): {target_world_tool}")
    print("[PlanHumanoid] Planning...")
    try:
        ok = planner.update_world_and_plan_motion(
            target_pose=target_world_tool,
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
    return ok


def _diagnostics(planner, T_world_base, site_from_curobo, target_pose_env_site, env):
    try:
        planned_poses = planner.get_planned_poses()
        if len(planned_poses) > 0:
            # planned poses are in planner frame (base). Map to world (T_W_T = T_W_B @ T_B_T)
            last_world_tool = (T_world_base @ planned_poses[-1].to(device=env.device, dtype=torch.float32)).clone()
            # then to site/world (T_W_S = T_W_T @ T_T_S)
            last_world_site = (last_world_tool @ site_from_curobo).clone()
            goal_pos_err = torch.linalg.vector_norm(last_world_site[:3, 3] - target_pose_env_site[:3, 3]).item()
            goal_rot_err_mat = last_world_site[:3, :3].T @ target_pose_env_site[:3, :3]
            goal_rot_err_trace = torch.clamp((torch.trace(goal_rot_err_mat) - 1.0) / 2.0, -1.0, 1.0)
            goal_rot_err = torch.acos(goal_rot_err_trace).item()
            print(
                f"[PlanHumanoid] Planned final vs goal (world site) | pos_err={goal_pos_err:.4e} m |"
                f" rot_err={goal_rot_err:.4e} rad"
            )
    except Exception as e:
        print(f"[PlanHumanoid] Planned-goal diagnostics failed: {e}")


def _visualize_goal(args_cli, target_pose_env_site, env, eef_name):
    if not args_cli.visualize_goal:
        return None
    try:
        # Use a fresh instancer prim path each run to avoid stale Prototypes rel
        viz_path_goal = "/World/Visuals/goal_pose_marker"
        viz_path_ee = "/World/Visuals/ee_pose_marker"

        frame_cfg_goal = dc_replace(FRAME_MARKER_CFG, prim_path=viz_path_goal)
        frame_cfg_goal.markers["frame"].scale = (0.1, 0.1, 0.1)

        frame_cfg_ee = dc_replace(FRAME_MARKER_CFG, prim_path=viz_path_ee)
        frame_cfg_ee.markers["frame"].scale = (0.08, 0.08, 0.08)

        goal_pose_visualizer = VisualizationMarkers(frame_cfg_goal)
        ee_pose_visualizer = VisualizationMarkers(frame_cfg_ee)

        goal_pos = target_pose_env_site[:3, 3].detach().to(dtype=torch.float32)
        goal_quat = (
            PoseUtils.quat_from_matrix(target_pose_env_site[:3, :3].unsqueeze(0))[0].detach().to(dtype=torch.float32)
        )

        # Current EE pose marker at time of planning
        cur_eef_pose = env.get_robot_eef_pose(eef_name)[0].to(device=env.device, dtype=torch.float32)
        cur_pos = cur_eef_pose[:3, 3].detach()
        cur_quat = PoseUtils.quat_from_matrix(cur_eef_pose[:3, :3].unsqueeze(0))[0].detach()

        goal_pose_visualizer.visualize(translations=goal_pos.unsqueeze(0), orientations=goal_quat.unsqueeze(0))
        ee_pose_visualizer.visualize(translations=cur_pos.unsqueeze(0), orientations=cur_quat.unsqueeze(0))
        return ee_pose_visualizer
    except Exception as e:
        print(f"[PlanHumanoid] Goal visualization failed: {e}")
        return None


def _visualize_goals_bimanual(args_cli, target_pose_env_site_r, target_pose_env_site_l, env):
    if not args_cli.visualize_goal:
        return None, None
    try:
        # Distinct prim paths per arm
        viz_goal_r = "/World/Visuals/goal_pose_marker_right"
        viz_goal_l = "/World/Visuals/goal_pose_marker_left"
        viz_ee_r = "/World/Visuals/ee_pose_marker_right"
        viz_ee_l = "/World/Visuals/ee_pose_marker_left"

        # Goal markers
        frame_goal_r = dc_replace(FRAME_MARKER_CFG, prim_path=viz_goal_r)
        frame_goal_r.markers["frame"].scale = (0.1, 0.1, 0.1)
        frame_goal_l = dc_replace(FRAME_MARKER_CFG, prim_path=viz_goal_l)
        frame_goal_l.markers["frame"].scale = (0.1, 0.1, 0.1)

        # EE markers
        frame_ee_r = dc_replace(FRAME_MARKER_CFG, prim_path=viz_ee_r)
        frame_ee_r.markers["frame"].scale = (0.08, 0.08, 0.08)
        frame_ee_l = dc_replace(FRAME_MARKER_CFG, prim_path=viz_ee_l)
        frame_ee_l.markers["frame"].scale = (0.08, 0.08, 0.08)

        goal_vis_r = VisualizationMarkers(frame_goal_r)
        goal_vis_l = VisualizationMarkers(frame_goal_l)
        ee_vis_r = VisualizationMarkers(frame_ee_r)
        ee_vis_l = VisualizationMarkers(frame_ee_l)

        # Visualize goal frames
        goal_pos_r = target_pose_env_site_r[:3, 3].detach().to(dtype=torch.float32)
        goal_quat_r = (
            PoseUtils.quat_from_matrix(target_pose_env_site_r[:3, :3].unsqueeze(0))[0].detach().to(dtype=torch.float32)
        )
        goal_pos_l = target_pose_env_site_l[:3, 3].detach().to(dtype=torch.float32)
        goal_quat_l = (
            PoseUtils.quat_from_matrix(target_pose_env_site_l[:3, :3].unsqueeze(0))[0].detach().to(dtype=torch.float32)
        )
        goal_vis_r.visualize(translations=goal_pos_r.unsqueeze(0), orientations=goal_quat_r.unsqueeze(0))
        goal_vis_l.visualize(translations=goal_pos_l.unsqueeze(0), orientations=goal_quat_l.unsqueeze(0))

        # Current EE frames at planning time
        cur_eef_pose_r = env.get_robot_eef_pose("right")[0].to(device=env.device, dtype=torch.float32)
        cur_eef_pose_l = env.get_robot_eef_pose("left")[0].to(device=env.device, dtype=torch.float32)
        cur_pos_r = cur_eef_pose_r[:3, 3].detach()
        cur_quat_r = PoseUtils.quat_from_matrix(cur_eef_pose_r[:3, :3].unsqueeze(0))[0].detach()
        cur_pos_l = cur_eef_pose_l[:3, 3].detach()
        cur_quat_l = PoseUtils.quat_from_matrix(cur_eef_pose_l[:3, :3].unsqueeze(0))[0].detach()
        ee_vis_r.visualize(translations=cur_pos_r.unsqueeze(0), orientations=cur_quat_r.unsqueeze(0))
        ee_vis_l.visualize(translations=cur_pos_l.unsqueeze(0), orientations=cur_quat_l.unsqueeze(0))

        return ee_vis_r, ee_vis_l
    except Exception as e:
        print(f"[PlanHumanoid] Bimanual goal visualization failed: {e}")
        return None, None


def _execute_plan(
    env, robot, planner, env_origin, site_from_curobo, eef_name, args_cli, ee_pose_visualizer, active_arm=None
):
    # Get planned poses
    print(f"Current plan in joint space: {planner.current_plan}")
    planned_poses = planner.get_planned_poses()
    print(f"[PlanHumanoid] Generated {len(planned_poses)} waypoints")

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

    # Helper: build a 4x4 pose from idle slices for a given arm (T_W_S from pos, quat)
    def _pose_from_idle(_idle: torch.Tensor, arm: str) -> torch.Tensor:
        if arm == "left":
            pos = _idle[0:3]
            quat = _idle[3:7]
        else:
            pos = _idle[7:10]
            quat = _idle[10:14]
        rot = PoseUtils.matrix_from_quat(quat.unsqueeze(0))[0]
        return PoseUtils.make_pose(pos.unsqueeze(0), rot.unsqueeze(0))[0].to(env.device)

    arm_active = active_arm if active_arm is not None else args_cli.arm
    fixed_arm = "left" if arm_active == "right" else "right"
    fixed_pose = _pose_from_idle(idle, fixed_arm)

    # Execute waypoints
    for _ in range(args_cli.replay_trials):
        print(f"[PlanHumanoid] Replaying trial {_ + 1}/{args_cli.replay_trials}")
        env.reset()
        for idx, target_ee_pose in enumerate(planned_poses):
            # Sanitize target pose to 4x4 homogeneous, device/dtype
            if target_ee_pose.dim() == 3 and target_ee_pose.size(0) == 1:
                target_ee_pose = target_ee_pose[0]
            target_ee_pose = target_ee_pose.to(device=env.device, dtype=torch.float32)
            # Recompute base pose after reset (defensive for mobile variants): T_W_B_exec
            base_pos_world_exec = (robot.data.root_pos_w[0] - env_origin).to(device=env.device, dtype=torch.float32)
            base_rot_world_exec = PoseUtils.matrix_from_quat(
                robot.data.root_quat_w[0].unsqueeze(0).to(device=env.device, dtype=torch.float32)
            )[0]
            T_world_base_exec = PoseUtils.make_pose(base_pos_world_exec.unsqueeze(0), base_rot_world_exec.unsqueeze(0))[
                0
            ]
            # Map base->tool to world->tool, then to site/world (T_W_T = T_W_B_exec @ T_B_T; T_W_S = T_W_T @ T_T_S)
            target_world_tool = (T_world_base_exec @ target_ee_pose).clone()
            target_ee_pose_site = (target_world_tool @ site_from_curobo).clone()
            target_ee_pose_site[3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device, dtype=torch.float32)

            if arm_active == "right":
                target_dict = {"left": fixed_pose, "right": target_ee_pose_site}
            else:
                target_dict = {"left": target_ee_pose_site, "right": fixed_pose}

            action = env.target_eef_pose_to_action(
                target_eef_pose_dict=target_dict,
                gripper_action_dict={"left": idle[14:25], "right": idle[25:36]},  # keep fixed grippers truly idle
                action_noise_dict=None,
                env_id=0,
            )
            if action.ndim == 1:
                action = action.unsqueeze(0)
            action = action.to(device=env.device, dtype=torch.float32)

            # Verify action maps back to intended target (error in site/world)
            try:
                inferred_targets = env.action_to_target_eef_pose(action)
                inf_pose = inferred_targets["right" if arm_active == "right" else "left"][0].to(device=env.device)
                inf_pos_err = torch.linalg.vector_norm(inf_pose[:3, 3] - target_ee_pose_site[:3, 3]).item()
                inf_rot_err_mat = inf_pose[:3, :3].T @ target_ee_pose_site[:3, :3]
                inf_rot_err_trace = torch.clamp((torch.trace(inf_rot_err_mat) - 1.0) / 2.0, -1.0, 1.0)
                inf_rot_err = torch.acos(inf_rot_err_trace).item()
                if (idx == 0) or ((idx + 1) % 25 == 0) or (idx == len(planned_poses) - 1):
                    print(
                        f"[PlanHumanoid] Action inversion check | pos_err={inf_pos_err:.4e} m |"
                        f" rot_err={inf_rot_err:.4e} rad"
                    )
            except Exception as e:
                print(f"[PlanHumanoid] action_to_target_eef_pose check failed: {e}")

            if args_cli.visualize_goal and ee_pose_visualizer is not None:
                cur_eef_pose = env.get_robot_eef_pose(eef_name)[0].to(device=env.device, dtype=torch.float32)
                cur_pos = cur_eef_pose[:3, 3].detach()
                cur_quat = PoseUtils.quat_from_matrix(cur_eef_pose[:3, :3].unsqueeze(0))[0].detach()
                ee_pose_visualizer.visualize(translations=cur_pos.unsqueeze(0), orientations=cur_quat.unsqueeze(0))

            env.step(action)

            if (idx + 1) % 10 == 0 or idx == 0 or idx == len(planned_poses) - 1:
                print(f"[PlanHumanoid] Step {idx + 1}/{len(planned_poses)}")

    planner.plan_visualizer.close()
    planner.clear()


def _build_site_goal_bimanual(
    ctrl_site_env_r: torch.Tensor, ctrl_site_env_l: torch.Tensor, args_cli, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build paired goals for both arms in site/world frame so hands are close together in front.

    Logic:
    - Compute the current midpoint between hands; shift it by +dy (forward) and +dz (up).
    - Place right and left targets symmetrically at ±dx along X about that midpoint, so total
      hand-to-hand separation is 2*|dx|. If dx < 0, the arms cross to opposite sides.
    - Keep orientations unchanged (position-only offsets).
    """
    # Clone poses
    goal_r = ctrl_site_env_r.clone()
    goal_l = ctrl_site_env_l.clone()

    # Current mid-point between the two hands
    cur_mid = 0.5 * (ctrl_site_env_r[:3, 3] + ctrl_site_env_l[:3, 3])

    # Desired offsets
    half_gap = float(args_cli.dx) if hasattr(args_cli, "dx") else 0.05
    forward = float(args_cli.dy) if hasattr(args_cli, "dy") else 0.05
    upward = float(args_cli.dz) if hasattr(args_cli, "dz") else 0.05

    # Build a point in front of current midpoint
    target_mid = cur_mid.clone()
    target_mid[1] = target_mid[1] + forward
    target_mid[2] = target_mid[2] + upward

    # Place right/left around midpoint with small lateral gap along x
    goal_r[:3, 3] = target_mid
    goal_l[:3, 3] = target_mid
    goal_r[0, 3] = goal_r[0, 3] + half_gap
    goal_l[0, 3] = goal_l[0, 3] - half_gap

    print(
        f"[PlanHumanoid] Bimanual goals | mid={target_mid.cpu().numpy()} | gap={2*half_gap:.3f}m,"
        f" forward={forward:.3f}m, up={upward:.3f}m"
    )
    return goal_r, goal_l


def _execute_plans_together(
    env, robot, planner_r, planner_l, env_origin, site_from_r, site_from_l, args_cli, ee_vis_r=None, ee_vis_l=None
):
    print("[PlanHumanoid] Executing bimanual plans together...")
    planned_r = planner_r.get_planned_poses()
    planned_l = planner_l.get_planned_poses()
    if len(planned_r) == 0 or len(planned_l) == 0:
        print("[PlanHumanoid] One of the plans is empty; aborting execution.")
        return

    # Idle action (device/dtype aligned)
    idle = env.cfg.idle_action
    if not isinstance(idle, torch.Tensor):
        idle = torch.tensor(idle, dtype=torch.float32)
    else:
        idle = idle.to(dtype=torch.float32)
    idle = idle.to(device=env.device)

    total = max(len(planned_r), len(planned_l))
    for _ in range(args_cli.replay_trials):
        print(f"[PlanHumanoid] Replaying trial {_ + 1}/{args_cli.replay_trials}")
        env.reset()
        for idx in range(total):
            # Base pose in world (recompute each step)
            base_pos_world_exec = (robot.data.root_pos_w[0] - env_origin).to(device=env.device, dtype=torch.float32)
            base_rot_world_exec = PoseUtils.matrix_from_quat(
                robot.data.root_quat_w[0].unsqueeze(0).to(device=env.device, dtype=torch.float32)
            )[0]
            T_world_base_exec = PoseUtils.make_pose(base_pos_world_exec.unsqueeze(0), base_rot_world_exec.unsqueeze(0))[
                0
            ]

            # Get targets for this step (use last waypoint if shorter)
            pose_r_bt = planned_r[min(idx, len(planned_r) - 1)]
            pose_l_bt = planned_l[min(idx, len(planned_l) - 1)]
            pose_r_bt = pose_r_bt.to(device=env.device, dtype=torch.float32)
            pose_l_bt = pose_l_bt.to(device=env.device, dtype=torch.float32)

            # base->tool to world->tool then to site/world per arm
            target_world_tool_r = (T_world_base_exec @ pose_r_bt).clone()
            target_world_tool_l = (T_world_base_exec @ pose_l_bt).clone()
            target_site_r = (target_world_tool_r @ site_from_r).clone()
            target_site_l = (target_world_tool_l @ site_from_l).clone()
            target_site_r[3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device, dtype=torch.float32)
            target_site_l[3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device, dtype=torch.float32)

            target_dict = {"left": target_site_l, "right": target_site_r}
            action = env.target_eef_pose_to_action(
                target_eef_pose_dict=target_dict,
                gripper_action_dict={"left": idle[14:25], "right": idle[25:36]},
                action_noise_dict=None,
                env_id=0,
            )
            if action.ndim == 1:
                action = action.unsqueeze(0)
            action = action.to(device=env.device, dtype=torch.float32)

            # Update EE visualizers
            if args_cli.visualize_goal and ee_vis_r is not None and ee_vis_l is not None:
                cur_eef_pose_r = env.get_robot_eef_pose("right")[0].to(device=env.device, dtype=torch.float32)
                cur_pos_r = cur_eef_pose_r[:3, 3].detach()
                cur_quat_r = PoseUtils.quat_from_matrix(cur_eef_pose_r[:3, :3].unsqueeze(0))[0].detach()
                ee_vis_r.visualize(translations=cur_pos_r.unsqueeze(0), orientations=cur_quat_r.unsqueeze(0))
                cur_eef_pose_l = env.get_robot_eef_pose("left")[0].to(device=env.device, dtype=torch.float32)
                cur_pos_l = cur_eef_pose_l[:3, 3].detach()
                cur_quat_l = PoseUtils.quat_from_matrix(cur_eef_pose_l[:3, :3].unsqueeze(0))[0].detach()
                ee_vis_l.visualize(translations=cur_pos_l.unsqueeze(0), orientations=cur_quat_l.unsqueeze(0))

            env.step(action)
            if (idx + 1) % 10 == 0 or idx == 0 or idx == total - 1:
                print(f"[PlanHumanoid] Bimanual Step {idx + 1}/{total}")

    planner_r.plan_visualizer.close()
    planner_l.plan_visualizer.close()
    planner_r.clear()
    planner_l.clear()


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # Build env and planner(s)
    if args_cli.arm == "both":
        env, robot, planner_right, planner_left = _build_env_and_planners_both(args_cli)

        # Frames and calibration per arm
        env_origin_r, ctrl_site_env_r, T_world_base_r, T_world_tool_now_r, site_from_curobo_r = (
            _compute_world_and_site_frames(env, robot, planner_right, "right")
        )
        env_origin_l, ctrl_site_env_l, T_world_base_l, T_world_tool_now_l, site_from_curobo_l = (
            _compute_world_and_site_frames(env, robot, planner_left, "left")
        )

        # Build paired goals for arms
        env_device = cast(Any, env).device
        target_pose_env_site_r, target_pose_env_site_l = _build_site_goal_bimanual(
            ctrl_site_env_r, ctrl_site_env_l, args_cli, env_device
        )

        site_inv_r = torch.linalg.inv(site_from_curobo_r)
        site_inv_l = torch.linalg.inv(site_from_curobo_l)
        target_world_tool_r = (target_pose_env_site_r @ site_inv_r).clone()
        target_world_tool_l = (target_pose_env_site_l @ site_inv_l).clone()

        # Configure retiming (see flag description): deg->rad step size; None disables retiming
        step_size = np.deg2rad(args_cli.retime_deg) if args_cli.retime_deg > 0 else None

        # Plan sequentially (both arms active in collisions)
        ok_r = _plan_motion(planner_right, target_world_tool_r, step_size)
        if not ok_r:
            print("Planning failed for right arm.")
            return
        ok_l = _plan_motion(planner_left, target_world_tool_l, step_size)
        if not ok_l:
            print("Planning failed for left arm.")
            return

        # Visualize goals and current EE frames for both arms
        ee_vis_r, ee_vis_l = _visualize_goals_bimanual(args_cli, target_pose_env_site_r, target_pose_env_site_l, env)

        # Execute both together in one trial
        # Note: use env_origin_r for base frame; both share same env/robot
        _execute_plans_together(
            env,
            robot,
            planner_right,
            planner_left,
            env_origin_r,
            site_from_curobo_r,
            site_from_curobo_l,
            args_cli,
            ee_vis_r=ee_vis_r,
            ee_vis_l=ee_vis_l,
        )
        return

    # Single-arm path (unchanged)
    env, robot, planner = _build_env_and_planner(args_cli)

    # Frames and calibration
    eef_name = args_cli.arm  # "left" or "right"
    env_origin, ctrl_site_env, T_world_base, T_world_tool_now, site_from_curobo = _compute_world_and_site_frames(
        env, robot, planner, eef_name
    )

    # Build goal in site/world and convert to tool/world
    env_device = cast(Any, env).device
    target_pose_env_site = _build_site_goal(ctrl_site_env, args_cli, env_device)
    site_inv = torch.linalg.inv(site_from_curobo)
    target_world_tool = (target_pose_env_site @ site_inv).clone()

    # Configure retiming (see flag description): deg->rad step size; None disables retiming
    step_size = np.deg2rad(args_cli.retime_deg) if args_cli.retime_deg > 0 else None

    # Plan
    ok = _plan_motion(planner, target_world_tool, step_size)
    if not ok:
        print("Planning failed.")
        return

    # Diagnostics
    _diagnostics(planner, T_world_base, site_from_curobo, target_pose_env_site, env)

    # Visualize goal and current EE pose
    ee_pose_visualizer = _visualize_goal(args_cli, target_pose_env_site, env, eef_name)

    # Execute
    _execute_plan(env, robot, planner, env_origin, site_from_curobo, eef_name, args_cli, ee_pose_visualizer)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    finally:
        simulation_app.close()
