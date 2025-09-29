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

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
from isaaclab_mimic.envs.pinocchio_envs.nutpour_gr1t2_mimic_env_cfg import NutPourGR1T2MimicEnvCfg
from isaaclab_mimic.envs.pinocchio_envs.pickplace_gr1t2_mimic_env_cfg import PickPlaceGR1T2MimicEnvCfg
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
from isaaclab_mimic.motion_planners.curobo.curobo_planner_humanoid import HumanoidArmCuroboPlanner

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg import PickPlaceGR1T2EnvCfg


def _detect_robot_usd_path(env):
    """Detect robot USD path from environment configuration."""
    try:
        usd_path = env.cfg.scene.robot.spawn.usd_path
        if isinstance(usd_path, str) and len(usd_path) > 0:
            return usd_path
    except Exception:
        # Default GR1T2 USD path
        usd_path = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Robots/FourierIntelligence/GR-1/GR1T2_fourier_hand_6dof/GR1T2_fourier_hand_6dof.usd"
    return usd_path


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


def rest_robot(env, robot, steps=10):
    """Let robot settle for a few steps."""
    print(f"[PlanHumanoid] Resting robot for {steps} steps...")
    for i in range(steps):
        env.step(torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device))
        if (i + 1) % 5 == 0:
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
    env_cfg.num_envs = args_cli.num_envs

    print("[PlanHumanoid] Creating env...")
    try:
        env = gym.make(env_name, cfg=env_cfg).unwrapped
        env.reset()
    except Exception as e:
        print(f"[PlanHumanoid] Error creating env: {e}")
        raise e
    print("[PlanHumanoid] Env ready.")

    # Let robot settle
    if args_cli.rest > 0:
        rest_robot(env, env.scene["robot"], args_cli.rest)

    # Build planner config
    planner_cfg = CuroboPlannerCfg()
    planner_cfg.visualize_plan = True
    planner_cfg.visualize_spheres = False
    planner_cfg.debug_planner = args_cli.debug

    # Configure for GR1 robot
    usd_path = _detect_robot_usd_path(env)
    print(f"[PlanHumanoid] Detected robot USD: {usd_path}")
    robot_yaml = _build_temp_robot_yaml_from_usd(usd_path, args_cli.arm)
    print(f"[PlanHumanoid] Generated cuRobo robot YAML: {robot_yaml}")

    planner_cfg.robot_config_file = robot_yaml
    planner_cfg.robot_name = "gr1"
    planner_cfg.approach_distance = 0.01  # Small approach for safety
    planner_cfg.retreat_distance = 0.01  # Small retreat for safety
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

    # Get current end-effector pose
    cu_js = planner._get_current_joint_state_for_curobo()
    ee_pose_cu = planner.get_ee_pose(cu_js)
    pos = planner._to_env_device(ee_pose_cu.position)
    rot = planner._to_env_device(ee_pose_cu.get_rotation())
    current_pose: torch.Tensor = PoseUtils.make_pose(pos, rot)[0]

    print("[PlanHumanoid] Current EE pose:")
    print("Position: {current_pose[:3, 3].cpu().numpy()}")
    print("Rotation:\n{current_pose[:3, :3].cpu().numpy()}")

    # Generate target pose based on goal type
    target_pose = generate_goal_pose(current_pose, args_cli.goal, args_cli)

    print("[PlanHumanoid] Target EE pose:")
    print(f"Position: {target_pose[:3, 3].cpu().numpy()}")

    # Configure retiming
    step_size = np.deg2rad(args_cli.retime_deg) if args_cli.retime_deg > 0 else None

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

    # Get planned poses
    planned_poses = planner.get_planned_poses()
    print(f"[PlanHumanoid] Generated {len(planned_poses)} waypoints")

    if len(planned_poses) == 0:
        print("[PlanHumanoid] No waypoints generated!")
        return

    # Execute the plan
    print(f"[PlanHumanoid] Executing {len(planned_poses)} waypoints...")

    # Get the other arm's current pose to keep it fixed
    other_arm = "left" if args_cli.arm == "right" else "right"
    other_link = _tool_link_for_arm(other_arm)
    other_eef = planner.get_attached_pose(other_link, cu_js)
    other_pos = planner._to_env_device(other_eef.position)
    other_rot = planner._to_env_device(other_eef.get_rotation())
    other_fixed = PoseUtils.make_pose(other_pos, other_rot)[0]

    # Get current hand joint states
    hand_state = env.obs_buf["policy"]["hand_joint_state"][0].to(env.device, dtype=torch.float32)
    left_hand = hand_state[:11]
    right_hand = hand_state[11:22]

    # Execute waypoints
    for idx, target_ee_pose in enumerate(planned_poses):
        try:
            # Set up target poses for both arms
            if args_cli.arm == "right":
                target_dict = {
                    "left": other_fixed,
                    "right": target_ee_pose,
                }
            else:
                target_dict = {
                    "left": target_ee_pose,
                    "right": other_fixed,
                }

            # Convert to action
            action = env.target_eef_pose_to_action(
                target_eef_pose_dict=target_dict,
                gripper_action_dict={
                    "left": left_hand,
                    "right": right_hand,
                },
                action_noise_dict=None,
                env_id=0,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[PlanHumanoid] Error converting waypoint {idx + 1}: {e}")
            raise e

        # Ensure action has correct shape
        if action.ndim == 1:
            action = action.unsqueeze(0)
        action = action.to(device=env.device, dtype=torch.float32)

        # Step environment
        try:
            env.step(action)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"[PlanHumanoid] Error executing step {idx + 1}: {e}")
            raise e

        # Progress logging
        if (idx + 1) % 10 == 0 or idx == 0 or idx == len(planned_poses) - 1:
            print(f"[PlanHumanoid] Step {idx + 1}/{len(planned_poses)}")

    print("[PlanHumanoid] Execution complete!")

    # Final rest to observe result
    print("[PlanHumanoid] Final rest...")
    rest_robot(env, robot, 20)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    finally:
        simulation_app.close()
