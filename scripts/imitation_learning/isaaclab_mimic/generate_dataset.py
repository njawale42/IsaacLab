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
parser.add_argument(
    "--schedule_strategy",
    type=str,
    choices=["retiming", "dag"],
    default=None,
    help="Scheduling strategy for bimanual collision avoidance: "
         "'retiming' (waypoint-level MILP, default) or 'dag' (segment-level MILP).",
)
schedule_offline_group = parser.add_mutually_exclusive_group()
schedule_offline_group.add_argument(
    "--schedule_offline",
    dest="schedule_offline",
    action="store_true",
    help="Enable offline scheduling before execution. Intended for bimanual SkillGen workflows.",
)
schedule_offline_group.add_argument(
    "--no_schedule_offline",
    dest="schedule_offline",
    action="store_false",
    help="Disable offline scheduling and use normal online SkillGen execution.",
)
parser.set_defaults(schedule_offline=None)
parser.add_argument(
    "--planner_max_batch",
    type=int,
    default=8,
    help="Maximum cuRobo micro-batch size for shared cube-stack batch planning.",
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
    async_components = None
    data_gen_tasks = None
    shared_motion_planner_backend = None

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

    if args_cli.schedule_strategy is not None:
        try:
            env_cfg.datagen_config.schedule_strategy = args_cli.schedule_strategy
        except Exception:
            pass

    # Precompute humanoid planner configs BEFORE creating the env (ordering matters)
    prebuilt_humanoid_cfgs = None
    if args_cli.use_skillgen and args_cli.skillgen_type == "bimanual" and "nutpour-gr1t2" in env_name.lower():
        from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

        usd_path = env_cfg.scene.robot.spawn.usd_path
        inactive_joint_names = list(env_cfg.actions.gr1_action.ik_urdf_fixed_joint_names)
        prebuilt_humanoid_cfgs = CuroboPlannerCfg.gr1t2_nutpour_bimanual_configs(usd_path, inactive_joint_names)

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
        from isaaclab_mimic.motion_planners.curobo.batched_cube_stack_planner import (
            create_batched_cube_stack_motion_planners,
        )
        from isaaclab_mimic.motion_planners.curobo.bimanual_humanoid_planner import BimanualHumanoidPlanner
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

        planner_config = CuroboPlannerCfg.from_task_name(env_name)
        if args_cli.skillgen_type != "bimanual" and "stack-cube" in env_name.lower():
            print("Initializing shared batched cube-stack planner backend")
            shared_motion_planner_backend, motion_planners = create_batched_cube_stack_motion_planners(
                env=env,
                robot=env.scene["robot"],
                config=planner_config,
                num_envs=num_envs,
                max_batch=args_cli.planner_max_batch,
            )
        else:
            motion_planners = {}
            if (
                args_cli.skillgen_type == "bimanual"
                and "nutpour-gr1t2" in env_name.lower()
                and prebuilt_humanoid_cfgs is not None
            ):
                for env_id in range(num_envs):
                    cfg_left, cfg_right = prebuilt_humanoid_cfgs
                    motion_planners[env_id] = BimanualHumanoidPlanner(
                        env=env,
                        robot=env.scene["robot"],
                        cfg_right=cfg_right,
                        cfg_left=cfg_left,
                        env_id=env_id,
                    )
            else:
                print(f"Initializing shared motion planner for {num_envs} environments...")
                planners = CuroboPlanner.create_shared_planners(
                    env=env,
                    robot=env.scene["robot"],
                    config=planner_config,
                    num_envs=num_envs,
                )
                for env_id, planner in enumerate(planners):
                    motion_planners[env_id] = planner
                print(f"Shared planner ready: 1 MotionGen serving {num_envs} environments")

    # Setup and run async data generation
    async_components = setup_async_generation(
        env=env,
        num_envs=args_cli.num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        pause_subtask=args_cli.pause_subtask,
        motion_planners=motion_planners,  # Pass the motion planners dictionary
        skillgen_type=args_cli.skillgen_type,
        schedule_offline=args_cli.schedule_offline,
    )

    try:
        data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        env_loop(
            env,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["info_pool"],
            async_components["event_loop"],
            data_gen_tasks=data_gen_tasks,
        )
    except asyncio.CancelledError:
        print("Tasks were cancelled.")
    finally:
        # Cancel all async tasks when env_loop finishes
        if data_gen_tasks is not None:
            data_gen_tasks.cancel()
        try:
            # Wait for tasks to be cancelled
            if async_components is not None and data_gen_tasks is not None:
                async_components["event_loop"].run_until_complete(data_gen_tasks)
        except asyncio.CancelledError:
            print("Remaining async tasks cancelled and cleaned up.")
        except Exception as e:
            print(f"Error cancelling remaining async tasks: {e}")
        if shared_motion_planner_backend is not None and async_components is not None:
            async_components["event_loop"].run_until_complete(shared_motion_planner_backend.aclose())
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
