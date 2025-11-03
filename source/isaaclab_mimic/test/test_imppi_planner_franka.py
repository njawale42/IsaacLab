# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from collections.abc import Generator
from typing import Any

import pytest

SEED: int = 42
random.seed(SEED)

from isaaclab.app import AppLauncher

headless = False
app_launcher = AppLauncher(headless=headless)
simulation_app: Any = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.utils.assets as _al_assets
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from isaaclab_mimic.motion_planners.curobo.curobo_mpc_planner import CuroboMPCPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_joint_pos_env_cfg import (
    FrankaCubeStackEnvCfg,
)


# Predefined EE goals for the test (env frame, behind the wall)
predefined_ee_goals_and_ids = [
    ({"pos": [0.70, -0.25, 0.25], "quat": [0.0, 0.707, 0.0, 0.707]}, "Behind wall, left"),
    ({"pos": [0.70, 0.25, 0.25], "quat": [0.0, 0.707, 0.0, 0.707]}, "Behind wall, right"),
    ({"pos": [0.65, 0.0, 0.45], "quat": [0.0, 1.0, 0.0, 0.0]}, "Behind wall, center, high"),
    ({"pos": [0.80, -0.15, 0.35], "quat": [0.0, 0.5, 0.0, 0.866]}, "Behind wall, far left"),
    ({"pos": [0.80, 0.15, 0.35], "quat": [0.0, 0.5, 0.0, 0.866]}, "Behind wall, far right"),
]


@pytest.fixture(scope="class")
def imppi_test_env() -> Generator[dict[str, Any], None, None]:
    env_cfg = FrankaCubeStackEnvCfg()
    env_cfg.scene.num_envs = 1
    # Add a static wall obstacle similar to the MPPI test
    ISAAC_NUCLEUS_DIR: str = getattr(_al_assets, "ISAAC_NUCLEUS_DIR", "/Isaac")
    wall_props = RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
    wall_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_0/moving_wall",
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd",
            scale=(0.5, 4.5, 7.0),
            rigid_props=wall_props,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.80)),
    )
    setattr(env_cfg.scene, "moving_wall", wall_cfg)

    env: ManagerBasedRLMimicEnv = gym.make(  # type: ignore[assignment]
        "Isaac-Stack-Cube-Franka-v0", cfg=env_cfg, headless=headless
    ).unwrapped
    env.reset()

    # Planner config using IMPPI optimizer
    planner_cfg = CuroboPlannerCfg.franka_config()
    planner_cfg.mpc_use_imppi = True
    # Optional: fine-tune IMPPI params if desired
    planner_cfg.imppi_target_kl = 0.03
    planner_cfg.imppi_reuse_prev_iter = True
    planner_cfg.imppi_max_backtracks = 2
    planner_cfg.imppi_backtrack_coeff = 0.5

    planner = CuroboMPCPlanner(env=env, robot=env.scene["robot"], config=planner_cfg)  # type: ignore[abstract]
    if not headless:
        planner.enable_visual_rollouts(True)
    goal_pose_visualizer = None
    if not headless:
        goal_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/goal_poses_imppi")  # type: ignore[attr-defined]
        goal_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        goal_pose_visualizer = VisualizationMarkers(goal_marker_cfg)
    yield {"env": env, "planner": planner, "goal_pose_visualizer": goal_pose_visualizer}
    env.close()


class TestImppiPlannerFranka:
    @pytest.fixture(autouse=True)
    def setup(self, imppi_test_env) -> None:
        self.env: ManagerBasedRLMimicEnv = imppi_test_env["env"]
        self.planner: CuroboMPCPlanner = imppi_test_env["planner"]
        self.goal_pose_visualizer: VisualizationMarkers | None = imppi_test_env["goal_pose_visualizer"]

    def test_reactive_reach_imppi(self) -> None:
        for goal_spec, goal_id in predefined_ee_goals_and_ids:
            pos = torch.tensor(goal_spec["pos"], device=self.env.device, dtype=torch.float32)
            quat = torch.tensor(goal_spec["quat"], device=self.env.device, dtype=torch.float32)

            assert pos[0] > 0.55, f"Goal '{goal_id}' is not behind the wall (x={pos[0].item():.3f})"

            rot_matrix = math_utils.matrix_from_quat(quat.unsqueeze(0))[0]
            ee_goal = math_utils.make_pose(pos, rot_matrix)

            planned = self.planner.update_world_and_plan_motion(ee_goal, env_id=0)
            assert planned, f"IMPPI planner failed to find a plan for goal: {goal_id}"

            if not headless and self.goal_pose_visualizer is not None:
                T_goal_env_world = self.planner.get_goal_env_world_pose()
                if T_goal_env_world is not None:
                    g_pos, g_rot = math_utils.unmake_pose(T_goal_env_world)
                    g_quat = math_utils.quat_from_matrix(g_rot.unsqueeze(0) if g_rot.dim() == 2 else g_rot)
                    self.goal_pose_visualizer.visualize(
                        translations=g_pos.unsqueeze(0) if g_pos.dim() == 1 else g_pos,
                        orientations=g_quat,
                    )

            max_steps = int(1e10)  # effectively infinite loop until goal reached
            for _ in range(max_steps):
                if self.planner.has_next_waypoint():
                    next_pose = self.planner.get_next_waypoint_ee_pose()
                else:
                    break
                cmd_q = self.planner.get_last_joint_positions()
                if cmd_q is not None:
                    if cmd_q.dim() == 1:
                        cmd_q = cmd_q.unsqueeze(0)
                    if cmd_q.shape[-1] == 7:
                        fingers = torch.tensor([0.04, 0.04], device=cmd_q.device, dtype=cmd_q.dtype).unsqueeze(0)
                        cmd_q = torch.cat([cmd_q, fingers], dim=-1)
                    self.env.scene["robot"].write_joint_position_to_sim(cmd_q)
                    self.env.sim.step()
                else:
                    play_action = self.env.target_eef_pose_to_action(  # type: ignore[attr-defined]
                        target_eef_pose_dict={"eef": next_pose},
                        gripper_action_dict={"eef": torch.zeros(2, device=self.env.device)},
                        action_noise_dict={"eef": 0.0},
                        env_id=0,
                    )
                    if play_action.dim() == 1:
                        play_action = play_action.unsqueeze(0)
                    for _ in range(10):
                        self.env.step(play_action)

            pos_err, _ = self.planner._current_pose_error()  # type: ignore[attr-defined]
            assert pos_err < 0.05, f"Final position error too high for goal '{goal_id}': {pos_err:.3f}"


