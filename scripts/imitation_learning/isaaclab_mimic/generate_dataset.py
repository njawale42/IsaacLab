# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Main data generation script.
"""


"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Generate demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--generation_num_trials", type=int, help="Number of demos to be generated.", default=None)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to instantiate for generating datasets."
)
parser.add_argument("--input_file", type=str, default=None, required=True, help="File path to the source dataset file.")
parser.add_argument(
    "--output_file",
    type=str,
    default="./datasets/output_dataset.hdf5",
    help="File path to export recorded and generated episodes.",
)
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="pause after every subtask during generation for debugging - only useful with render flag",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--use_skillgen",
    action="store_true",
    default=False,
    help="use skillgen to generate motion trajectories",
)
parser.add_argument(
    "--skillgen_type",
    type=str,
    choices=["single_arm", "bimanual"],
    default="single_arm",
    help="SkillGen mode: single_arm (default, Franka-style) or bimanual (humanoid).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import asyncio
import gymnasium as gym
import inspect
import numpy as np
import random
import torch

import omni

from isaaclab.envs import ManagerBasedRLMimicEnv

import isaaclab_mimic.envs  # noqa: F401

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation, setup_env_config
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths

import isaaclab_tasks  # noqa: F401


def main():
    num_envs = args_cli.num_envs

    # Setup output paths and get env name
    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    task_name = args_cli.task
    if task_name:
        task_name = args_cli.task.split(":")[-1]
    env_name = task_name or get_env_name_from_dataset(args_cli.input_file)

    # Configure environment
    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
    )

    # Ensure env cfg reflects CLI skillgen toggle so DataGenerator takes the skillgen path
    try:
        env_cfg.datagen_config.use_skillgen = bool(args_cli.use_skillgen)
    except Exception:
        pass

    # Precompute humanoid planner configs BEFORE creating the env (ordering matters)
    prebuilt_humanoid_cfgs = None
    if args_cli.use_skillgen and args_cli.skillgen_type == "bimanual" and "nutpour-gr1t2" in env_name.lower():
        from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
        from isaaclab_mimic.motion_planners.curobo.humanoid_robot_yaml import build_humanoid_yaml_from_usd

        # USD path and inactive joints from env_cfg
        usd_path = env_cfg.scene.robot.spawn.usd_path
        inactive_joint_names = list(env_cfg.actions.gr1_action.ik_urdf_fixed_joint_names)
        
        # TODO: (Neel) This is not working as expected (spheres are still being generated for hand links)
        # add hand joints to inactive joints
        inactive_joint_names_left = inactive_joint_names + [j for j in env_cfg.actions.gr1_action.hand_joint_names if "L_" in j]
        inactive_joint_names_right = inactive_joint_names + [j for j in env_cfg.actions.gr1_action.hand_joint_names if "R_" in j]
        print(f"Inactive joints left: {inactive_joint_names_left}")
        print(f"Inactive joints right: {inactive_joint_names_right}")

        # Build per-arm YAML once
        yaml_right = build_humanoid_yaml_from_usd(usd_path, arm="right", inactive_joints=inactive_joint_names_right)
        yaml_left = build_humanoid_yaml_from_usd(usd_path, arm="left", inactive_joints=inactive_joint_names_left)

        def _mk_cfg(yaml_path: str, arm: str) -> CuroboPlannerCfg:
            # Set arm-specific attachment link (the hand pitch link for each arm)
            attached_link = f"GR1T2_fourier_hand_6dof_{arm}_hand_pitch_link"
            return CuroboPlannerCfg(
                robot_config_file=yaml_path,
                robot_name="gr1",
                hand_link_names=[
                    "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
                    "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
                ],
                # Use actual EE link for object attachment (not virtual "attached_object" link)
                attached_object_link_name=attached_link,
                ee_link_name=attached_link,
                static_objects=["table", "scale", "bin"],
                # Ignore specific USD prims; remove items one-by-one to debug collisions
                world_ignore_substrings=[
                    "/World/envs/env_0/Table",
                    # "/World/envs/env_0/SortingScale",
                    # "/World/envs/env_0/SortingBowl",
                    # "/World/envs/env_0/SortingBeaker",
                    "/World/envs/env_0/FactoryNut",
                    # "/World/envs/env_0/BlackSortingBin",
                    "/World/envs/env_0/RobotPOVCam",
                    "/World/envs/env_0/Robot",
                    "/World/GroundPlane",
                ],
                approach_distance=0.0,
                retreat_distance=0.0,
                approach_retreat_frame="world",  # "eef" = direction in EE frame; "world" = Z is world up/down
                time_dilation_factor=0.5,
                maximum_trajectory_dt=None, #0.50,  # Increase (e.g. 0.25) if MotionGenStatus.DT_EXCEPTION
                enable_finetune_trajopt=True,
                collision_activation_distance=0.04,
                motion_step_size=None,
                visualize_spheres=False,
                visualize_plan=True,
                debug_planner=True,
                approach_direction=(0.0, 0.0, -1.0),
                surface_sphere_radius=0.005,
                extra_collision_spheres={"attached_object": 100},
                # palm_offset_from_ee=(0.0, -0.07, -0.10),
                # Dexterous hand grasp detection using finger joints + XY distance
                # Works for cylindrical objects (beakers) grasped at any height
                grasp_detection_mode="dexterous",
                grasp_xy_distance_threshold=0.18,  # XY distance from EE (wrist) to object center
                dexterous_finger_closed_threshold=0.2,  # Finger joint threshold (radians)
            )

        cfg_right = _mk_cfg(yaml_right, "right")
        cfg_left = _mk_cfg(yaml_left, "left")
        prebuilt_humanoid_cfgs = (cfg_left, cfg_right)

    # Create environment AFTER prebuilding robot YAML/configs
    env = gym.make(env_name, cfg=env_cfg).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    # Check if the mimic API from this environment contains decprecated signatures
    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        omni.log.warn(
            f'The "noise" parameter in the "{env_name}" environment\'s mimic API "target_eef_pose_to_action", '
            "is deprecated. Please update the API to take action_noise_dict instead."
        )

    # Set seed for generation
    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)

    # Reset before starting (ensures scene is instantiated before planner queries)
    env.reset()

    motion_planners = None
    if args_cli.use_skillgen:
        from isaaclab_mimic.motion_planners.curobo.bimanual_humanoid_planner import BimanualHumanoidPlanner
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

        motion_planners = {}
        for env_id in range(num_envs):
            print(f"Initializing motion planner for environment {env_id}")
            if (
                args_cli.skillgen_type == "bimanual"
                and "nutpour-gr1t2" in env_name.lower()
                and prebuilt_humanoid_cfgs is not None
            ):
                cfg_left, cfg_right = prebuilt_humanoid_cfgs
                motion_planners[env_id] = BimanualHumanoidPlanner(
                    env=env,
                    robot=env.scene["robot"],
                    cfg_right=cfg_right,
                    cfg_left=cfg_left,
                    env_id=env_id,
                )
            else:
                planner_config = CuroboPlannerCfg.from_task_name(env_name)
                # No planner visualization during dataset generation
                # planner_config.visualize_spheres = False
                # planner_config.visualize_plan = False
                motion_planners[env_id] = CuroboPlanner(
                    env=env,
                    robot=env.scene["robot"],
                    config=planner_config,
                    env_id=env_id,
                )

    # Setup and run async data generation
    async_components = setup_async_generation(
        env=env,
        num_envs=args_cli.num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        pause_subtask=args_cli.pause_subtask,
        motion_planners=motion_planners,  # Pass the motion planners dictionary
        skillgen_type=args_cli.skillgen_type,
    )

    try:
        data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        env_loop(
            env,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["info_pool"],
            async_components["event_loop"],
        )
    except asyncio.CancelledError:
        print("Tasks were cancelled.")
    finally:
        # Cancel all async tasks when env_loop finishes
        data_gen_tasks.cancel()
        try:
            # Wait for tasks to be cancelled
            async_components["event_loop"].run_until_complete(data_gen_tasks)
        except asyncio.CancelledError:
            print("Remaining async tasks cancelled and cleaned up.")
        except Exception as e:
            print(f"Error cancelling remaining async tasks: {e}")
        # Cleanup of motion planners and their visualizers
        if motion_planners is not None:
            for env_id, planner in motion_planners.items():
                if getattr(planner, "plan_visualizer", None) is not None:
                    print(f"Closing plan visualizer for environment {env_id}")
                    planner.plan_visualizer.close()
                    planner.plan_visualizer = None
            motion_planners.clear()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    # Close sim app
    simulation_app.close()
