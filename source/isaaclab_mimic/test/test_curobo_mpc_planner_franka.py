import random
from typing import Any, Generator

import pytest

SEED: int = 42
random.seed(SEED)

from isaaclab.app import AppLauncher

headless = False
app_launcher = AppLauncher(headless=headless)
simulation_app: Any = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLMimicEnv

from isaaclab_mimic.motion_planners.curobo.curobo_mpc_planner import CuroboMPCPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_joint_pos_env_cfg import (
    FrankaCubeStackEnvCfg,
)
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
import isaaclab.utils.assets as _al_assets
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg


@pytest.fixture(scope="class")
def mpc_test_env() -> Generator[dict[str, Any], None, None]:
    env_cfg = FrankaCubeStackEnvCfg()
    env_cfg.scene.num_envs = 1
    # Add a static wall obstacle similar to the trajectory planner test
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
    planner = CuroboMPCPlanner(env=env, robot=env.scene["robot"], config=CuroboPlannerCfg.franka_config())  # type: ignore[abstract]
    goal_pose_visualizer = None
    if not headless:
        goal_marker_cfg = FRAME_MARKER_CFG.replace(prim_path="/World/Visuals/goal_poses_mpc")
        goal_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        goal_pose_visualizer = VisualizationMarkers(goal_marker_cfg)
    yield {"env": env, "planner": planner, "goal_pose_visualizer": goal_pose_visualizer}
    env.close()


class TestCuroboMPCPlanner:
    @pytest.fixture(autouse=True)
    def setup(self, mpc_test_env) -> None:
        self.env: ManagerBasedRLMimicEnv = mpc_test_env["env"]
        self.planner: CuroboMPCPlanner = mpc_test_env["planner"]
        self.goal_pose_visualizer: VisualizationMarkers | None = mpc_test_env["goal_pose_visualizer"]

    def test_reactive_reach(self) -> None:
        # Define a reachable offset goal from current ee pose
        ee_frame = self.env.scene["ee_frame"]
        origin = self.env.scene.env_origins[0]
        pos = ee_frame.data.target_pos_w[0, 0, :] - origin
        quat = ee_frame.data.target_quat_w[0, 0, :]
        pos = pos + torch.tensor([0.10, 0.05, 0.00], device=pos.device, dtype=pos.dtype)
        goal = math_utils.make_pose(pos, math_utils.matrix_from_quat(quat.unsqueeze(0))[0])[0]

        # Plan first so the planner caches the goal transform for viz
        self.planner.update_world_and_plan_motion(goal, env_id=0)

        # Visualize goal in the correct env EE world frame provided by the planner
        if not headless and self.goal_pose_visualizer is not None:
            T_goal_env_world = self.planner.get_goal_env_world_pose()
            if T_goal_env_world is not None:
                g_pos, g_rot = math_utils.unmake_pose(T_goal_env_world)
                g_quat = math_utils.quat_from_matrix(g_rot.unsqueeze(0) if g_rot.dim() == 2 else g_rot)
                self.goal_pose_visualizer.visualize(
                    translations=g_pos.unsqueeze(0) if g_pos.dim() == 1 else g_pos,
                    orientations=g_quat,
                )

        # Run for a bounded number of steps and check progress
        max_steps = int(1e10)
        for _ in range(max_steps):
            if self.planner.has_next_waypoint():
                next_pose = self.planner.get_next_waypoint_ee_pose()
            else:
                break
            cmd_q = self.planner.get_last_joint_positions()
            if cmd_q is not None:
                if cmd_q.dim() == 1:
                    cmd_q = cmd_q.unsqueeze(0)
                # Append default finger positions if only arm joints are present
                if cmd_q.shape[-1] == 7:
                    fingers = torch.tensor([0.04, 0.04], device=cmd_q.device, dtype=cmd_q.dtype).unsqueeze(0)
                    cmd_q = torch.cat([cmd_q, fingers], dim=-1)
                # Use target controller for smoother tracking rather than teleporting
                # self.env.scene["robot"].set_joint_position_target(cmd_q, env_ids=[0])
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
                self.env.step(play_action)

        # Validate we moved closer to goal position using actual EE pose via the planner
        pos_err, _ = self.planner._current_pose_error()  # type: ignore[attr-defined]
        # pos_err = torch.linalg.vector_norm((ee_frame.data.target_pos_w[0, 0, :] - origin) - pos).item()
        assert pos_err < 0.05
