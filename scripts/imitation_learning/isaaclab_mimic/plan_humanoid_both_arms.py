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
parser.add_argument("--retime_deg", type=float, default=0.0, help="Joint retime step (deg); 0 disables retiming.")
parser.add_argument("--rest", type=int, default=10, help="Initial rest steps before planning.")
parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
parser.add_argument("--replay_trials", type=int, default=2, help="Number of trials to replay.")
parser.add_argument("--visualize_goal", action="store_true", help="Visualize target EE pose marker.")
parser.add_argument("--collision_aware", action="store_true", help="Enable collision-aware retiming for bimanual.")
parser.add_argument(
    "--collision_distance", type=float, default=0.05, help="EE proximity threshold (m) to mark a pair as colliding."
)
parser.add_argument(
    "--pair_densify", type=int, default=1, help="Subdivisions per segment for joint-level collision checks (>=1)."
)
parser.add_argument(
    "--joint_collision_margin",
    type=float,
    default=0.002,
    help="Self-collision activation distance (m) for joint-level pairing.",
)
parser.add_argument(
    "--pair_visualize", action="store_true", help="Visualize sampled colliding and collision-free joint pairs."
)
parser.add_argument("--pair_vis_samples", type=int, default=8, help="Number of samples per class to visualize.")
# Back-and-forth planning: if enabled, plan to goal and then back to home, per arm
parser.add_argument(
    "--back_forth",
    action="store_true",
    help="If set, plan to the goal and then back to the starting pose for each arm.",
)
# Optional per-arm overrides: when provided, independent goals are used instead of symmetric bimanual ones
parser.add_argument(
    "--right_dx", type=float, default=None, help="Right arm X offset (site/world). Overrides --dx if set."
)
parser.add_argument(
    "--right_dy", type=float, default=None, help="Right arm Y offset (site/world). Overrides --dy if set."
)
parser.add_argument(
    "--right_dz", type=float, default=None, help="Right arm Z offset (site/world). Overrides --dz if set."
)
parser.add_argument(
    "--left_dx", type=float, default=None, help="Left arm X offset (site/world). Overrides --dx if set."
)
parser.add_argument(
    "--left_dy", type=float, default=None, help="Left arm Y offset (site/world). Overrides --dy if set."
)
parser.add_argument(
    "--left_dz", type=float, default=None, help="Left arm Z offset (site/world). Overrides --dz if set."
)
# parser.add_argument("--headless", action="store_true", help="Headless mode.")

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

from curobo.types.state import JointState
from nvplan.applications.custream.config import create_robot_config
from nvplan.applications.custream.retime import retime_paths
from nvplan.applications.custream.spheres import load_spheres

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
        kin = robot_cfg_yaml.get("kinematics", None)
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
    except Exception:
        pass
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
    robot_yaml_both = _build_temp_robot_yaml_both_arms(usd_path)

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
    # cfg_both = _make_cfg(robot_yaml_both, "both")

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
            collision_active_link_substrings=("GR1T2_fourier_hand_6dof_left_", "GR1T2_fourier_hand_6dof_right_"),
        )
        planner_left = HumanoidArmCuroboPlanner(
            env=env,
            robot=robot,
            config=cfg_left,
            env_id=0,
            active_joint_substrings=("left_",),
            hand_link_substrings=("GR1T2_fourier_hand_6dof_left_",),
            collision_active_link_substrings=("GR1T2_fourier_hand_6dof_left_", "GR1T2_fourier_hand_6dof_right_"),
        )

    except Exception as e:
        traceback.print_exc()
        print(f"[PlanHumanoid] Error creating planners: {e}")
        raise e
    print("[PlanHumanoid] Planners ready.")

    return env, robot, planner_right, planner_left, robot_yaml_both


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


# TODO: Not used but keep for future reference
def _compute_colliding_pairs_ee(planned_r, planned_l, T_world_base, threshold: float, device) -> list[tuple[int, int]]:
    """Compute colliding pairs (i,j) using EE proximity in world frame.

    A pair is marked colliding if the Euclidean distance between EE positions is below `threshold`.
    """
    # Convert base->tool to world->tool using current T_world_base
    world_tools_r = [(T_world_base @ p.to(device=device, dtype=torch.float32)) for p in planned_r]
    world_tools_l = [(T_world_base @ p.to(device=device, dtype=torch.float32)) for p in planned_l]
    pos_r = torch.stack([w[:3, 3] for w in world_tools_r], dim=0)
    pos_l = torch.stack([w[:3, 3] for w in world_tools_l], dim=0)

    # Compute pairwise distances via broadcasting
    # pos_r: [Nr,3], pos_l: [Nl,3] -> distances [Nr, Nl]
    diff = pos_r.unsqueeze(1) - pos_l.unsqueeze(0)
    dists = torch.linalg.norm(diff, dim=-1)
    colliding = dists <= threshold
    pairs = torch.nonzero(colliding, as_tuple=False)
    return [(int(i.item()), int(j.item())) for i, j in pairs]


def _densify_plan_positions(pos: torch.Tensor, factor: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly densify a joint path by integer factor; return dense path and map to coarse indices.

    Returns (pos_dense, idx_map) where idx_map[k] gives the source coarse index for dense index k.
    """
    if factor <= 1 or pos.shape[0] <= 1:
        idx_map = torch.arange(pos.shape[0], device=pos.device, dtype=torch.long)
        return pos, idx_map
    segments = pos.shape[0] - 1
    steps_per_seg = factor
    # For each segment, sample steps_per_seg points excluding the last one to avoid duplication; add final point after loop
    out = []
    imap = []
    for i in range(segments):
        p0 = pos[i]
        p1 = pos[i + 1]
        for s in range(steps_per_seg):
            t = float(s) / float(steps_per_seg)
            out.append((1.0 - t) * p0 + t * p1)
            imap.append(i)
    out.append(pos[-1])
    imap.append(segments)
    return torch.stack(out, dim=0), torch.tensor(imap, device=pos.device, dtype=torch.long)


def _reverse_joint_state(js: JointState, drop_first: bool = True) -> JointState:
    """Create a reversed copy of the given JointState along waypoint dimension.

    If drop_first is True, the first element of the reversed sequence (which equals the
    forward last) is dropped to avoid duplicating the turning point when concatenating.
    """
    pos = cast(torch.Tensor, js.position)
    vel_t = cast(torch.Tensor, js.velocity) if getattr(js, "velocity", None) is not None else None
    acc_t = cast(torch.Tensor, js.acceleration) if getattr(js, "acceleration", None) is not None else None
    jerk_t = cast(torch.Tensor, js.jerk) if getattr(js, "jerk", None) is not None else None

    def maybe_flip(x: torch.Tensor | None):
        return None if x is None else torch.flip(x, dims=[0])

    pos_rev = maybe_flip(pos)
    vel_rev = maybe_flip(vel_t)
    acc_rev = maybe_flip(acc_t)
    jerk_rev = maybe_flip(jerk_t)

    if drop_first and pos_rev.shape[0] > 0:
        pos_rev = pos_rev[1:]
        vel_rev = vel_rev[1:] if vel_rev is not None else None
        acc_rev = acc_rev[1:] if acc_rev is not None else None
        jerk_rev = jerk_rev[1:] if jerk_rev is not None else None

    return JointState(
        position=pos_rev, velocity=vel_rev, acceleration=acc_rev, jerk=jerk_rev, joint_names=js.joint_names
    )


def _append_return_to_start(planner) -> None:
    """Append a reversed copy of the current plan to return to start, in-place on planner.

    No-op if there is no current plan or it has < 2 waypoints.
    """
    js = getattr(planner, "current_plan", None)
    if js is None or not hasattr(js, "position") or len(js.position) < 2:
        return
    back = _reverse_joint_state(js, drop_first=True)
    combined = js.stack(back)
    # Replace planner's current plan with the combined forward+back plan
    try:
        planner._current_plan = combined  # noqa: SLF001 (intentional internal state set)
        planner._plan_index = 0
    except Exception:
        pass


# def _compute_colliding_pairs_joint(planner_r, planner_l) -> list[tuple[int, int]]:
#     """Compute colliding waypoint pairs using self-collision on combined joint states.

#     Builds all pairwise combined joint configurations where the right-arm joints come
#     from the right plan at index i and the left-arm joints come from the left plan at
#     index j. Other joints (torso/base/grippers) are taken from the right plan baseline.
#     Uses cuRobo's RobotWorld self-collision cost to classify collisions.
#     """
#     import pdb; pdb.set_trace()
#     plan_r = getattr(planner_r, "current_plan", None)
#     plan_l = getattr(planner_l, "current_plan", None)
#     if plan_r is None or plan_l is None:
#         return []

#     pos_r = plan_r.position # pos_r.shape: [32, 21]
#     pos_l = plan_l.position # pos_l.shape: [32, 21]
#     if not isinstance(pos_r, torch.Tensor):
#         pos_r = torch.tensor(pos_r, dtype=torch.float32, device=planner_r.tensor_args.device)
#     if not isinstance(pos_l, torch.Tensor):
#         pos_l = torch.tensor(pos_l, dtype=torch.float32, device=planner_r.tensor_args.device)

#     # Densify per CLI factor
#     factor = max(1, int(getattr(args_cli, "pair_densify", 1)))
#     pos_r_dense, map_r = _densify_plan_positions(pos_r, factor)
#     pos_l_dense, map_l = _densify_plan_positions(pos_l, factor)
#     Nr = int(pos_r_dense.shape[0])
#     Nl = int(pos_l_dense.shape[0])
#     if Nr == 0 or Nl == 0:
#         return []

#     # Joint masks for each arm based on configured substrings
#     joint_names: list[str] = list(plan_r.joint_names)

#     def _mask_for(substrings: tuple[str, ...]) -> torch.Tensor:
#         return torch.tensor([any(s in name for s in substrings) for name in joint_names], dtype=torch.bool, device=planner_r.tensor_args.device)

#     substr_l = getattr(planner_l, "active_joint_substrings", ("left_",))
#     mask_l = _mask_for(tuple(substr_l))

#     # Build all pairwise combined configurations: baseline from right plan, override left-arm joints from left plan
#     q = pos_r_dense.repeat_interleave(Nl, dim=0).clone()  # [Nr*Nl, dof]
#     left_all = pos_l_dense.repeat(Nr, 1)                  # [Nr*Nl, dof]
#     q[:, mask_l] = left_all[:, mask_l]

#     # Evaluate self-collision on the combined configurations
#     from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

#     # Set self-collision activation distance from CLI margin to treat near-contacts as collisions
#     rw_cfg = RobotWorldConfig.load_from_config(
#         robot_config=planner_r.robot_cfg,
#         world_model=None,
#         tensor_args=planner_r.tensor_args,
#         n_envs=1,
#         self_collision_activation_distance=float(getattr(args_cli, "joint_collision_margin", 0.0)),
#     )
#     robot_world = RobotWorld(rw_cfg)

#     state = robot_world.get_kinematics(q)
#     spheres = state.link_spheres_tensor.view(q.shape[0], 1, -1, 4)
#     d_self = robot_world.get_self_collision(spheres).view(-1)

#     if getattr(args_cli, "pair_visualize", False):
#         # Visualize a few colliding and non-colliding samples in the viewer by setting joint targets
#         k_collide = torch.nonzero(d_self > 0.0, as_tuple=False).flatten().tolist()
#         k_free = torch.nonzero(d_self <= 0.0, as_tuple=False).flatten().tolist()
#         import random
#         random.shuffle(k_collide)
#         random.shuffle(k_free)
#         show_c = k_collide[: int(getattr(args_cli, "pair_vis_samples", 8))]
#         show_f = k_free[: int(getattr(args_cli, "pair_vis_samples", 8))]

#         # Use env robot to set joint position targets for brief frames
#         rob = planner_r.robot
#         device = rob.device
#         def _show_config(conf: torch.Tensor):
#             env_q = conf.to(device=planner_r.env.device, dtype=torch.float32).unsqueeze(0)
#             rob.set_joint_position_target(env_q, env_ids=[planner_r.env_id])
#             # step a few frames to let renderer update
#             for _ in range(3):
#                 planner_r.env.step(planner_r.env.cfg.idle_action if isinstance(planner_r.env.cfg.idle_action, torch.Tensor) else torch.tensor(planner_r.env.cfg.idle_action, device=planner_r.env.device).unsqueeze(0))

#         print(f"[PlanHumanoid] Visualizing {len(show_c)} colliding and {len(show_f)} free samples")
#         for k in show_c:
#             _show_config(q[k])
#         for k in show_f:
#             _show_config(q[k])

#     colliding_rows = torch.nonzero(d_self > 0.0, as_tuple=False).flatten()
#     if colliding_rows.numel() == 0:
#         return []

#     pairs: list[tuple[int, int]] = []
#     for idx in colliding_rows.tolist():
#         i_dense = idx // Nl
#         j_dense = idx % Nl
#         # Map back to original coarse waypoint indices
#         i_coarse = int(map_r[i_dense].item())
#         j_coarse = int(map_l[j_dense].item())
#         pairs.append((i_coarse, j_coarse))
#     return pairs


# Version 2: Using independent kinematics from both the planners
# def _compute_colliding_pairs_joint(planner_r, planner_l) -> list[tuple[int, int]]:
#     """Compute colliding waypoint pairs using self-collision on combined joint states.

#     Mirrors streams.py: build a full-DOF grid from the two joint paths by assigning
#     each path's joint subset via name->index arrays into a full baseline configuration.
#     """
#     import pdb; pdb.set_trace()
#     from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

#     plan_r = getattr(planner_r, "current_plan", None)
#     plan_l = getattr(planner_l, "current_plan", None)
#     if plan_r is None or plan_l is None:
#         return []

#     dev = planner_r.tensor_args.device
#     # Paths as tensors
#     pos_r = plan_r.position if isinstance(plan_r.position, torch.Tensor) else torch.tensor(plan_r.position, dtype=torch.float32, device=dev)
#     pos_l = plan_l.position if isinstance(plan_l.position, torch.Tensor) else torch.tensor(plan_l.position, dtype=torch.float32, device=dev)
#     pos_r = pos_r.to(device=dev, dtype=torch.float32)
#     pos_l = pos_l.to(device=dev, dtype=torch.float32)

#     # Optional densification
#     factor = max(1, int(getattr(args_cli, "pair_densify", 1)))
#     pos_r_dense, map_r = _densify_plan_positions(pos_r, factor)
#     pos_l_dense, map_l = _densify_plan_positions(pos_l, factor)
#     Nr, Nl = int(pos_r_dense.shape[0]), int(pos_l_dense.shape[0])
#     if Nr == 0 or Nl == 0:
#         return []

#     # Full robot joint ordering
#     full_joint_names_r: list[str] = list(planner_r.motion_gen.kinematics.joint_names)
#     full_joint_names_l: list[str] = list(planner_l.motion_gen.kinematics.joint_names)

#     # Map each plan’s joint names into the full ordering (streams.py uses get_joint_indices)
#     name_to_full_r = {n: i for i, n in enumerate(full_joint_names_r)}
#     name_to_full_l = {n: i for i, n in enumerate(full_joint_names_l)}
#     idx_r = torch.tensor([name_to_full_r[n] for n in plan_r.joint_names], dtype=torch.long, device=dev)
#     idx_l = torch.tensor([name_to_full_l[n] for n in plan_l.joint_names], dtype=torch.long, device=dev)

#     # Full-DOF baseline like world.retract_conf in streams.py
#     base_js = planner_r._get_current_joint_state_for_curobo()
#     base = base_js.position.squeeze(0).to(device=dev, dtype=torch.float32)  # [dof]

#     # Build cross product configurations on the full DOF vector
#     num = Nr * Nl
#     confs = base.repeat(num, 1)  # [Nr*Nl, dof]
#     # Right subset written for each i, repeated over Nl
#     confs[:, idx_r] = torch.repeat_interleave(pos_r_dense, repeats=Nl, dim=0)
#     # Left subset written for each j, tiled across Nr
#     confs[:, idx_l] = pos_l_dense.repeat(Nr, 1)

#     # Self-collision classification (like world.get_self_collisions)
#     rw_cfg = RobotWorldConfig.load_from_config(
#         robot_config=planner_r.robot_cfg,
#         world_model=None,
#         tensor_args=planner_r.tensor_args,
#         n_envs=1,
#         self_collision_activation_distance=float(getattr(args_cli, "joint_collision_margin", 0.0)),
#     )
#     robot_world = RobotWorld(rw_cfg)
#     state = robot_world.get_kinematics(confs)
#     spheres = state.link_spheres_tensor.view(num, 1, -1, 4)
#     d_self = robot_world.get_self_collision(spheres).view(-1)

#     rows = torch.nonzero(d_self > 0.0, as_tuple=False).flatten()
#     if rows.numel() == 0:
#         return []

#     pairs: list[tuple[int, int]] = []
#     for k in rows.tolist():
#         i_dense = k // Nl
#         j_dense = k % Nl
#         pairs.append((int(map_r[i_dense].item()), int(map_l[j_dense].item())))
#     return pairs


def _compute_colliding_pairs_joint(planner_r, planner_l) -> list[tuple[int, int]]:
    """Find colliding waypoint pairs by checking right-vs-left sphere overlaps per pair.

    Uses each planner's own kinematics to compute spheres and detects inter-arm overlaps.
    Avoids mapping left joint names into the right planner's joint list.
    """
    #

    plan_r = getattr(planner_r, "current_plan", None)
    plan_l = getattr(planner_l, "current_plan", None)
    if plan_r is None or plan_l is None:
        return []

    dev = planner_r.tensor_args.device
    # Paths as tensors
    pos_r = (
        plan_r.position
        if isinstance(plan_r.position, torch.Tensor)
        else torch.tensor(plan_r.position, dtype=torch.float32, device=dev)
    )
    pos_l = (
        plan_l.position
        if isinstance(plan_l.position, torch.Tensor)
        else torch.tensor(plan_l.position, dtype=torch.float32, device=dev)
    )
    pos_r = pos_r.to(device=dev, dtype=torch.float32)
    pos_l = pos_l.to(device=dev, dtype=torch.float32)

    # Optional densification
    factor = max(1, int(getattr(args_cli, "pair_densify", 1)))
    pos_r_dense, map_r = _densify_plan_positions(pos_r, factor)
    pos_l_dense, map_l = _densify_plan_positions(pos_l, factor)
    Nr, Nl = int(pos_r_dense.shape[0]), int(pos_l_dense.shape[0])
    if Nr == 0 or Nl == 0:
        return []

    # Cross-product joint sequences for each arm separately
    q_r = torch.repeat_interleave(pos_r_dense, repeats=Nl, dim=0)  # [Nr*Nl, dof_r]
    q_l = pos_l_dense.repeat(Nr, 1)  # [Nr*Nl, dof_l]
    num = q_r.shape[0]

    # Compute min proximity per combined config without allocating [num, nR, nL]
    kin_r = planner_r.motion_gen.kinematics
    kin_l = planner_l.motion_gen.kinematics

    B_NUM = int(getattr(args_cli, "pair_batch", 4096))  # batch over cross-product rows
    CR = int(getattr(args_cli, "pair_chunk_r", 64))  # chunk right spheres
    CL = int(getattr(args_cli, "pair_chunk_l", 64))  # chunk left spheres
    margin = float(getattr(args_cli, "joint_collision_margin", 0.0))

    colliding_rows = []

    num = q_r.shape[0]
    for bi in range(0, num, B_NUM):
        bj = min(num, bi + B_NUM)
        q_r_b = q_r[bi:bj]
        q_l_b = q_l[bi:bj]
        bsz = q_r_b.shape[0]

        state_r = kin_r.get_state(q_r_b)
        state_l = kin_l.get_state(q_l_b)
        sph_r = state_r.link_spheres_tensor.view(bsz, -1, 4)  # [bsz, nR, 4]
        sph_l = state_l.link_spheres_tensor.view(bsz, -1, 4)  # [bsz, nL, 4]
        nR = sph_r.shape[1]
        nL = sph_l.shape[1]

        c_r = sph_r[..., :3]
        r_r = sph_r[..., 3]
        c_l = sph_l[..., :3]
        r_l = sph_l[..., 3]

        min_prox_b = torch.full((bsz,), float("inf"), device=c_r.device, dtype=c_r.dtype)

        for r0 in range(0, nR, CR):
            r1 = min(nR, r0 + CR)
            c_r_ch = c_r[:, r0:r1, :]  # [bsz, cr, 3]
            r_r_ch = r_r[:, r0:r1]  # [bsz, cr]
            for l0 in range(0, nL, CL):
                l1 = min(nL, l0 + CL)
                c_l_ch = c_l[:, l0:l1, :]  # [bsz, cl, 3]
                r_l_ch = r_l[:, l0:l1]  # [bsz, cl]

                diff = c_r_ch.unsqueeze(2) - c_l_ch.unsqueeze(1)  # [bsz, cr, cl, 3]
                d = torch.linalg.norm(diff, dim=-1)  # [bsz, cr, cl]
                rs = r_r_ch.unsqueeze(2) + r_l_ch.unsqueeze(1)  # [bsz, cr, cl]
                prox = d - rs  # [bsz, cr, cl]
                local_min = prox.view(bsz, -1).min(dim=1).values
                min_prox_b = torch.minimum(min_prox_b, local_min)

        # rows that collide for this batch
        local_rows = torch.nonzero(min_prox_b <= margin, as_tuple=False).flatten()
        if local_rows.numel() > 0:
            colliding_rows.extend((bi + k.item()) for k in local_rows)

    if len(colliding_rows) == 0:
        return []

    # Map dense rows -> (i,j) -> coarse indices via map_r/map_l
    pairs = []
    for k in colliding_rows:
        i_dense = k // Nl
        j_dense = k % Nl
        pairs.append((int(map_r[i_dense].item()), int(map_l[j_dense].item())))
    return pairs


def _build_retimed_schedules(len_r: int, len_l: int, colliding_pairs: list[tuple[int, int]], min_dt: float = 0.05):
    """Build retimed schedules (times) for both paths using MILP-based retime.

    Returns two lists of times (seconds) aligned to each waypoint index for each path.
    """

    def _compress_pairs_min_j(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
        # Keep only the smallest j for each i (strict minimal collisions)
        min_j = {}
        for i, j in pairs:
            if (i not in min_j) or (j < min_j[i]):
                min_j[i] = j
        return [(i, j) for i, j in min_j.items()]

    def _cap_pairs(pairs: list[tuple[int, int]], limit: int = 1000) -> list[tuple[int, int]]:
        if len(pairs) <= limit:
            return pairs
        step = max(1, len(pairs) // limit)
        return pairs[::step]

    path1 = [None] * len_r
    path2 = [None] * len_l
    try:
        t1, t2 = retime_paths(
            path1, path2, colliding=colliding_pairs, linear=True, min_dt=min_dt, buffer=0.02, verbose=False
        )
        return t1, t2
    except AssertionError:
        # Retry with reduced constraints and larger dt
        reduced = _compress_pairs_min_j(colliding_pairs)
        reduced = _cap_pairs(reduced, limit=500)
        larger_dt = max(min_dt * 2.0, 0.1)
        try:
            t1, t2 = retime_paths(
                path1, path2, colliding=reduced, linear=True, min_dt=larger_dt, buffer=0.02, verbose=False
            )
            return t1, t2
        except AssertionError:
            return None


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


def _build_site_goal_independent(
    ctrl_site_env_r: torch.Tensor, ctrl_site_env_l: torch.Tensor, args_cli, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-arm goals in site/world with independent offsets (dx,dy,dz for each arm).

    If an arm-specific offset is None, fall back to the global dx/dy/dz for that component.
    """

    def pick(val_specific, val_global, default):
        return (
            float(val_specific)
            if val_specific is not None
            else (float(val_global) if val_global is not None else default)
        )

    # Resolve offsets
    rdx = pick(args_cli.right_dx, args_cli.dx, 0.0)
    rdy = pick(args_cli.right_dy, args_cli.dy, 0.0)
    rdz = pick(args_cli.right_dz, args_cli.dz, 0.0)
    ldx = pick(args_cli.left_dx, args_cli.dx, 0.0)
    ldy = pick(args_cli.left_dy, args_cli.dy, 0.0)
    ldz = pick(args_cli.left_dz, args_cli.dz, 0.0)

    # Clone base poses
    goal_r = ctrl_site_env_r.clone()
    goal_l = ctrl_site_env_l.clone()

    # Apply per-arm deltas directly in site/world
    goal_r[0, 3] = goal_r[0, 3] + rdx
    goal_r[1, 3] = goal_r[1, 3] + rdy
    goal_r[2, 3] = goal_r[2, 3] + rdz

    goal_l[0, 3] = goal_l[0, 3] + ldx
    goal_l[1, 3] = goal_l[1, 3] + ldy
    goal_l[2, 3] = goal_l[2, 3] + ldz

    print(
        f"[PlanHumanoid] Independent goals | R dpos=({rdx:.3f},{rdy:.3f},{rdz:.3f}) m |"
        f" L dpos=({ldx:.3f},{ldy:.3f},{ldz:.3f}) m"
    )
    return goal_r, goal_l


def _execute_plans_together(
    env,
    robot,
    planner_r,
    planner_l,
    env_origin,
    site_from_r,
    site_from_l,
    args_cli,
    ee_vis_r=None,
    ee_vis_l=None,
    *,
    retime_schedules: tuple[list[float], list[float]] | None = None,
) -> None:
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

    # Build discrete timeline if retimed schedules provided
    if retime_schedules is not None:
        times_r, times_l = retime_schedules
        # Discretize timeline at union of both time sequences
        all_times = sorted(set([float(t) for t in times_r] + [float(t) for t in times_l]))
        # Prepare joint-space paths for interpolation
        js_r = planner_r.current_plan.position
        js_l = planner_l.current_plan.position
        if not isinstance(js_r, torch.Tensor):
            js_r = torch.tensor(js_r, dtype=torch.float32, device=env.device)
        else:
            js_r = js_r.to(device=env.device, dtype=torch.float32)
        if not isinstance(js_l, torch.Tensor):
            js_l = torch.tensor(js_l, dtype=torch.float32, device=env.device)
        else:
            js_l = js_l.to(device=env.device, dtype=torch.float32)

        # For time-based interpolation, store as Python lists for bisect
        import bisect

        times_r_list = [float(t) for t in times_r]
        times_l_list = [float(t) for t in times_l]

        def interp(times_list, path_tensor, t):
            k = max(0, bisect.bisect_right(times_list, float(t)) - 1)
            if k >= len(times_list) - 1:
                return path_tensor[-1]
            t0 = times_list[k]
            t1 = times_list[k + 1]
            if t1 <= t0:
                return path_tensor[k]
            w = (float(t) - t0) / (t1 - t0)
            return (1.0 - w) * path_tensor[k] + w * path_tensor[k + 1]

        # Build a synthetic schedule of callable interpolants at each time
        schedule = []  # list of (t) to be used below for FK per time
        schedule = all_times
    else:
        schedule = list(zip(range(len(planned_r)), [min(i, len(planned_l) - 1) for i in range(len(planned_r))]))
        if len(planned_l) > len(planned_r):
            extra = [(len(planned_r) - 1, j) for j in range(len(planned_r), len(planned_l))]
            schedule.extend(extra)

    for trial in range(args_cli.replay_trials):
        print(f"[PlanHumanoid] Replaying trial {trial + 1}/{args_cli.replay_trials}")
        env.reset()
        for step in schedule:
            # Base pose in world (recompute each step)
            base_pos_world_exec = (robot.data.root_pos_w[0] - env_origin).to(device=env.device, dtype=torch.float32)
            base_rot_world_exec = PoseUtils.matrix_from_quat(
                robot.data.root_quat_w[0].unsqueeze(0).to(device=env.device, dtype=torch.float32)
            )[0]
            T_world_base_exec = PoseUtils.make_pose(base_pos_world_exec.unsqueeze(0), base_rot_world_exec.unsqueeze(0))[
                0
            ]

            if retime_schedules is not None:
                t = step
                # Interpolate joint positions per arm
                q_r_t = interp(times_r_list, js_r, t)
                q_l_t = interp(times_l_list, js_l, t)
                # FK to base->tool
                state_r = planner_r.motion_gen.kinematics.get_state(q_r_t.unsqueeze(0))
                state_l = planner_l.motion_gen.kinematics.get_state(q_l_t.unsqueeze(0))
                pose_r_bt = getattr(state_r, "ee_pose", None)
                pose_l_bt = getattr(state_l, "ee_pose", None)

                # Convert cuRobo Pose objects to 4x4 tensors (fallback to position/quaternion if needed)
                def _pose_to_mat(pose_obj, state):
                    if pose_obj is not None and hasattr(pose_obj, "position"):
                        pos = pose_obj.position
                        if hasattr(pose_obj, "quaternion"):
                            quat = pose_obj.quaternion.view(1, 4)
                            rot_m = PoseUtils.matrix_from_quat(quat)[0]
                        else:
                            rot_m = pose_obj.get_rotation()
                            if rot_m.dim() == 3:
                                rot_m = rot_m[0]
                        return PoseUtils.make_pose(pos.view(1, 3), rot_m.unsqueeze(0))[0]
                    ee_pos = getattr(state, "ee_position").view(1, 3)
                    ee_rot = getattr(state, "ee_quaternion").view(1, 4)
                    rot_m = PoseUtils.matrix_from_quat(ee_rot)[0]
                    return PoseUtils.make_pose(ee_pos, rot_m.unsqueeze(0))[0]

                pose_r_bt = _pose_to_mat(pose_r_bt, state_r).to(device=env.device, dtype=torch.float32)
                pose_l_bt = _pose_to_mat(pose_l_bt, state_l).to(device=env.device, dtype=torch.float32)
            else:
                # Index-based default
                idx_r, idx_l = step
                pose_r_bt = planned_r[idx_r].to(device=env.device, dtype=torch.float32)
                pose_l_bt = planned_l[idx_l].to(device=env.device, dtype=torch.float32)

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
        print("[PlanHumanoid] Trial complete")

    planner_r.plan_visualizer.close()
    planner_l.plan_visualizer.close()
    planner_r.clear()
    planner_l.clear()


def _build_temp_robot_yaml_both_arms(usd_path: str) -> str:
    # import pdb; pdb.set_trace()
    tmp_dir = tempfile.mkdtemp(prefix="gr1_curobo_full_")
    urdf_path, _ = ControllerUtils.convert_usd_to_urdf(usd_path, tmp_dir, force_conversion=True)

    tool_links = [
        "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
        "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
    ]
    robot_config = create_robot_config(
        urdf_path,
        tool_links=tool_links,
        inactive_joints=[],
        depth=2,
        verbose=False,
    )
    max_spheres = 225 - len(tool_links) * 50
    load_spheres(robot_config, max_spheres=max_spheres, max_link_spheres=int(1e9))

    robot_cfg_yaml = robot_config["robot_cfg"]
    robot_cfg_yaml = to_python(robot_cfg_yaml)

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
    # Ensure ee_link exists; pick right tool by default (doesn’t affect collisions)
    if isinstance(robot_cfg_yaml, dict):
        kin = robot_cfg_yaml.get("kinematics", None)
        if isinstance(kin, dict):
            kin["ee_link"] = "GR1T2_fourier_hand_6dof_right_hand_pitch_link"

    out_dir = tempfile.mkdtemp(prefix="curobo_robot_cfg_full_")
    out_path = os.path.join(out_dir, "gr1_full.yml")
    with open(out_path, "w") as f:
        yaml.safe_dump({"robot_cfg": robot_cfg_yaml}, f, sort_keys=False)
    return out_path


def _compute_colliding_pairs_joint_singlecall(planner_r, planner_l, full_robot_yaml: str) -> list[tuple[int, int]]:
    from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

    plan_r = getattr(planner_r, "current_plan", None)
    plan_l = getattr(planner_l, "current_plan", None)
    if plan_r is None or plan_l is None:
        return []

    dev = planner_r.tensor_args.device

    pos_r = (
        plan_r.position
        if isinstance(plan_r.position, torch.Tensor)
        else torch.tensor(plan_r.position, dtype=torch.float32, device=dev)
    )
    pos_l = (
        plan_l.position
        if isinstance(plan_l.position, torch.Tensor)
        else torch.tensor(plan_l.position, dtype=torch.float32, device=dev)
    )
    pos_r = pos_r.to(device=dev, dtype=torch.float32)
    pos_l = pos_l.to(device=dev, dtype=torch.float32)

    factor = max(1, int(getattr(args_cli, "pair_densify", 1)))
    pos_r_dense, map_r = _densify_plan_positions(pos_r, factor)
    pos_l_dense, map_l = _densify_plan_positions(pos_l, factor)
    Nr, Nl = int(pos_r_dense.shape[0]), int(pos_l_dense.shape[0])
    if Nr == 0 or Nl == 0:
        return []

    # Build full-robot RobotWorld (both arms)
    rw_cfg = RobotWorldConfig.load_from_config(
        robot_config=full_robot_yaml,
        world_model=None,
        tensor_args=planner_r.tensor_args,
        n_envs=1,
        self_collision_activation_distance=float(getattr(args_cli, "joint_collision_margin", 0.0)),
    )
    robot_world = RobotWorld(rw_cfg)
    full_joint_names = list(robot_world.kinematics.joint_names)
    name_to_full = {n: i for i, n in enumerate(full_joint_names)}

    # Map plan joints into full model indices
    idx_r_full = torch.tensor([name_to_full[n] for n in plan_r.joint_names], dtype=torch.long, device=dev)
    idx_l_full = torch.tensor([name_to_full[n] for n in plan_l.joint_names], dtype=torch.long, device=dev)

    # Base full-DOF conf from env’s current joint state
    env_joint_names = list(planner_r.robot.data.joint_names)
    env_name_to_idx = {n: i for i, n in enumerate(env_joint_names)}
    env_q = planner_r.robot.data.joint_pos[planner_r.env_id, :].to(device=dev, dtype=torch.float32)
    base = torch.zeros(len(full_joint_names), device=dev, dtype=torch.float32)
    for n, j in name_to_full.items():
        if n in env_name_to_idx:
            base[j] = env_q[env_name_to_idx[n]]

    # Cross-product in the full model
    num = Nr * Nl
    confs = base.repeat(num, 1)
    confs[:, idx_r_full] = torch.repeat_interleave(pos_r_dense, repeats=Nl, dim=0)
    confs[:, idx_l_full] = pos_l_dense.repeat(Nr, 1)

    # Single call: self-collision on full robot (captures inter-arm collisions)
    B_NUM = int(getattr(args_cli, "pair_batch", 4096))
    rows = []
    for b0 in range(0, num, B_NUM):
        b1 = min(num, b0 + B_NUM)
        state = robot_world.get_kinematics(confs[b0:b1])
        spheres = state.link_spheres_tensor.view(b1 - b0, 1, -1, 4)
        d_self = robot_world.get_self_collision(spheres).view(-1)  # >0 => collision
        local = torch.nonzero(d_self > 0.0, as_tuple=False).flatten()
        if local.numel() > 0:
            rows.extend((b0 + k.item()) for k in local)

    if not rows:
        return []

    pairs = []
    for k in rows:
        i_dense = k // Nl
        j_dense = k % Nl
        pairs.append((int(map_r[i_dense].item()), int(map_l[j_dense].item())))
    return pairs


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # Build env and planner(s)
    if args_cli.arm == "both":
        env, robot, planner_right, planner_left, robot_yaml_both = _build_env_and_planners_both(args_cli)

        # Frames and calibration per arm
        env_origin_r, ctrl_site_env_r, T_world_base_r, T_world_tool_now_r, site_from_curobo_r = (
            _compute_world_and_site_frames(env, robot, planner_right, "right")
        )
        env_origin_l, ctrl_site_env_l, T_world_base_l, T_world_tool_now_l, site_from_curobo_l = (
            _compute_world_and_site_frames(env, robot, planner_left, "left")
        )

        # Build paired goals for arms
        env_device = cast(Any, env).device
        use_independent = any(
            x is not None
            for x in (
                args_cli.right_dx,
                args_cli.right_dy,
                args_cli.right_dz,
                args_cli.left_dx,
                args_cli.left_dy,
                args_cli.left_dz,
            )
        )
        if use_independent:
            target_pose_env_site_r, target_pose_env_site_l = _build_site_goal_independent(
                ctrl_site_env_r, ctrl_site_env_l, args_cli, env_device
            )
        else:
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
        print(f"[PlanHumanoid] RIGHT: inactive collision links = {len(planner_right._inactive_collision_links())}")
        print("[PlanHumanoid] Planning RIGHT arm...")
        ok_r = _plan_motion(planner_right, target_world_tool_r, step_size)
        if not ok_r:
            print("Planning failed for right arm.")
            return
        print(f"[PlanHumanoid] RIGHT planned waypoints: {len(planner_right.get_planned_poses())}")

        print(f"[PlanHumanoid] LEFT: inactive collision links = {len(planner_left._inactive_collision_links())}")
        print("[PlanHumanoid] Planning LEFT arm...")
        ok_l = _plan_motion(planner_left, target_world_tool_l, step_size)
        if not ok_l:
            print("Planning failed for left arm.")
            return
        print(f"[PlanHumanoid] LEFT planned waypoints: {len(planner_left.get_planned_poses())}")

        # If requested, append return-to-start for both arms by reversing their plans
        if args_cli.back_forth:
            print("[PlanHumanoid] Appending return-to-start segment for both arms (back_forth enabled)")
            _append_return_to_start(planner_right)
            _append_return_to_start(planner_left)
            print(f"[PlanHumanoid] RIGHT total waypoints (forward+back): {len(planner_right.current_plan.position)}")
            print(f"[PlanHumanoid] LEFT total waypoints (forward+back): {len(planner_left.current_plan.position)}")

        # Visualize goals and current EE frames for both arms
        ee_vis_r, ee_vis_l = _visualize_goals_bimanual(args_cli, target_pose_env_site_r, target_pose_env_site_l, env)

        # Collision-aware retiming (optional)
        retime_sched = None
        if args_cli.collision_aware:
            # Prefer joint-level collision detection across both arms
            colliding_pairs = _compute_colliding_pairs_joint(
                planner_right, planner_left
            )  # _compute_colliding_pairs_joint_singlecall(planner_right, planner_left, robot_yaml_both)
            # if not colliding_pairs:
            #     # Fallback to EE proximity if joint-level yields none
            #     print("[PlanHumanoid] No colliding joint waypoint pairs detected")
            #     colliding_pairs = _compute_colliding_pairs_ee(
            #         planner_right.get_planned_poses(),
            #         planner_left.get_planned_poses(),
            #         T_world_base_r,
            #         threshold=float(args_cli.collision_distance),
            #         device=env_device,
            #     )
            if colliding_pairs:
                print(f"[PlanHumanoid] Colliding pairs (joint-level preferred): {len(colliding_pairs)}")
                retime_sched = _build_retimed_schedules(
                    len(planner_right.get_planned_poses()),
                    len(planner_left.get_planned_poses()),
                    colliding_pairs,
                    min_dt=0.01,
                )
                if retime_sched is None:
                    print(
                        "[PlanHumanoid] Retimer infeasible after relaxations; proceeding without collision-aware"
                        " retime."
                    )
            else:
                print("[PlanHumanoid] No colliding waypoint pairs detected")

        # Execute both together in one trial (with optional retime)
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
            retime_schedules=retime_sched,
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

    if args_cli.back_forth:
        print("[PlanHumanoid] Appending return-to-start segment (back_forth enabled)")
        _append_return_to_start(planner)

    # Diagnostics only for forward goal when not back-forth
    if not args_cli.back_forth:
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
