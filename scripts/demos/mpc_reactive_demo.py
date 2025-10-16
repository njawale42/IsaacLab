#!/usr/bin/env python3

"""
Modern Isaac Lab reactive MPC demo using CuroboMPCPlanner.

Usage:
  python scripts/demos/mpc_reactive_demo.py --task Isaac-Stack-Cube-Franka-v0 --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, help="Gym task name (e.g., Isaac-Stack-Cube-Franka-v0)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
import gymnasium as gym
import torch

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv

from isaaclab_mimic.motion_planners.curobo.curobo_mpc_planner import CuroboMPCPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers


def main():
    env: ManagerBasedRLMimicEnv = gym.make(args.task, headless=args.headless).unwrapped  # type: ignore[assignment]
    env.reset()

    planner_cfg = CuroboPlannerCfg.from_task_name(args.task)
    planner = CuroboMPCPlanner(env=env, robot=env.scene["robot"], config=planner_cfg, env_id=0)  # type: ignore[abstract]

    # Use the env's ee_frame as a visual target; move it slightly to define a goal
    ee_frame = env.scene["ee_frame"]
    origin = env.scene.env_origins[0]
    target_pos = ee_frame.data.target_pos_w[0, 0, :] - origin
    target_quat = ee_frame.data.target_quat_w[0, 0, :]
    # Offset target a bit in x-y to observe motion
    target_pos = target_pos + torch.tensor([0.10, 0.05, 0.00], device=target_pos.device, dtype=target_pos.dtype)
    target_pose = PoseUtils.make_pose(target_pos, PoseUtils.matrix_from_quat(target_quat.unsqueeze(0))[0])[0]

    planner.update_world_and_plan_motion(target_pose=target_pose, env_id=0)

    # Visualize goal in the correct env EE world frame provided by the planner
    if not args.headless:
        goal_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/goal_poses_mpc_demo")
        goal_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        goal_viz = VisualizationMarkers(goal_marker_cfg)
        T_goal_env_world = planner.get_goal_env_world_pose()
        if T_goal_env_world is not None:
            g_pos, g_rot = PoseUtils.unmake_pose(T_goal_env_world)
            g_quat = PoseUtils.quat_from_matrix(g_rot.unsqueeze(0) if g_rot.dim() == 2 else g_rot)
            goal_viz.visualize(
                translations=g_pos.unsqueeze(0) if g_pos.dim() == 1 else g_pos,
                orientations=g_quat,
            )

    try:
        # Run for a fixed number of sim steps
        steps = 300
        for _ in range(steps):
            if planner.has_next_waypoint():
                ee_pose = planner.get_next_waypoint_ee_pose()
            else:
                ee_pose = target_pose

            # Prefer joint command from planner if available
            cmd_q = planner.get_last_joint_positions()
            if cmd_q is not None:
                # Write directly to sim for quick demo
                if cmd_q.dim() == 1:
                    cmd_q = cmd_q.unsqueeze(0)
                env.scene["robot"].write_joint_position_to_sim(cmd_q)
                env.sim.step()
            else:
                # Fallback: convert pose to action using env helper
                play_action = env.target_eef_pose_to_action(  # type: ignore[attr-defined]
                    target_eef_pose_dict={"eef": ee_pose},
                    gripper_action_dict={"eef": torch.zeros(2, device=env.device)},
                    action_noise_dict={"eef": 0.0},
                    env_id=0,
                )
                if play_action.dim() == 1:
                    play_action = play_action.unsqueeze(0)
                env.step(play_action)

        print("Demo finished")
    finally:
        # Ensure env closes before the app to avoid _is_closed destructor warning
        try:
            # Mark closed guard if attribute is missing to avoid __del__ AttributeError
            if not hasattr(env, "_is_closed"):
                try:
                    setattr(env, "_is_closed", True)
                except Exception:
                    pass
            else:
                try:
                    env._is_closed = True  # type: ignore[attr-defined]
                except Exception:
                    pass
            env.close()
        except Exception:
            pass
        # Drop reference and collect to run __del__ while attributes still exist
        try:
            import gc  # local import to avoid top-level dependency
            del env
            gc.collect()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

