# Copyright (c) 2025-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import random
from collections.abc import Generator
from typing import Any

import pytest

SEED = 42
random.seed(SEED)

from isaaclab.app import AppLauncher

def _get_test_headless() -> bool:
    raw_value = os.getenv("ISAACLAB_BATCHED_CUBE_STACK_TEST_HEADLESS", "0").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


headless = _get_test_headless()
app_launcher = AppLauncher(headless=headless)
simulation_app: Any = app_launcher.app

import gymnasium as gym
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_env import ManagerBasedEnv

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401

from isaaclab_mimic.envs.franka_stack_ik_rel_skillgen_env_cfg import FrankaCubeStackIKRelSkillgenEnvCfg
from isaaclab_mimic.motion_planners.curobo.batched_cube_stack_planner import (
    BatchedCubeStackPlannerBackend,
    BatchedCubeStackPlannerHandle,
    create_batched_cube_stack_motion_planners,
)
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

GRIPPER_OPEN_CMD = 1.0
GRIPPER_CLOSE_CMD = -1.0
DOWN_FACING_QUAT = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32)


def _get_test_num_envs() -> int:
    raw_value = os.getenv("ISAACLAB_BATCHED_CUBE_STACK_TEST_NUM_ENVS", "8").strip()
    num_envs = int(raw_value)
    if num_envs <= 0:
        raise ValueError("ISAACLAB_BATCHED_CUBE_STACK_TEST_NUM_ENVS must be positive")
    return num_envs


def _get_test_max_batch() -> int | None:
    raw_value = os.getenv("ISAACLAB_BATCHED_CUBE_STACK_TEST_MAX_BATCH", "8").strip()
    max_batch = int(raw_value)
    if max_batch <= 0:
        raise ValueError("ISAACLAB_BATCHED_CUBE_STACK_TEST_MAX_BATCH must be positive when set")
    return max_batch


def _assert_bucket_history(backend: BatchedCubeStackPlannerBackend, mode_name: str, num_envs: int) -> None:
    bucket_sizes = [size for mode, size in backend.batch_history if mode == mode_name]
    assert bucket_sizes, f"No recorded batches for mode '{mode_name}'"
    assert sum(bucket_sizes) == num_envs
    assert all(1 <= size <= backend.max_batch for size in bucket_sizes)


def _eef_name(env: ManagerBasedEnv) -> str:
    return list(env.cfg.subtask_configs.keys())[0]


def _action_from_pose(
    env: ManagerBasedEnv,
    target_pose: torch.Tensor,
    gripper_binary_action: float,
    env_id: int,
) -> torch.Tensor:
    eef = _eef_name(env)
    play_action = env.target_eef_pose_to_action(
        target_eef_pose_dict={eef: target_pose},
        gripper_action_dict={eef: torch.tensor([gripper_binary_action], device=env.device, dtype=torch.float32)},
        env_id=env_id,
    )
    if play_action.dim() == 1:
        play_action = play_action.unsqueeze(0)
    return play_action


def _build_actions_for_poses(
    env: ManagerBasedEnv,
    target_poses: dict[int, torch.Tensor],
    gripper_binary_action: float,
) -> torch.Tensor:
    actions = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
    for env_id, target_pose in target_poses.items():
        action = _action_from_pose(env, target_pose, gripper_binary_action, env_id=env_id)
        actions[env_id] = action[0]
    return actions


def _execute_batched_plans(
    env: ManagerBasedEnv,
    planners: dict[int, BatchedCubeStackPlannerHandle],
    gripper_binary_action: float,
) -> None:
    planned_poses = {env_id: planner.get_planned_poses() for env_id, planner in planners.items()}
    if not any(planned_poses.values()):
        return

    max_steps = max(len(poses) for poses in planned_poses.values())
    for step_idx in range(max_steps):
        target_poses: dict[int, torch.Tensor] = {}
        for env_id, poses in planned_poses.items():
            if not poses:
                continue
            pose_idx = min(step_idx, len(poses) - 1)
            target_poses[env_id] = poses[pose_idx]
        env.step(_build_actions_for_poses(env, target_poses, gripper_binary_action))


def _execute_gripper_action_batch(
    env: ManagerBasedEnv,
    gripper_binary_action: float,
    steps: int = 12,
) -> None:
    eef = _eef_name(env)
    for _ in range(steps):
        current_poses = env.get_robot_eef_pose(eef_name=eef, env_ids=None)
        target_poses = {env_id: current_poses[env_id] for env_id in range(env.num_envs)}
        env.step(_build_actions_for_poses(env, target_poses, gripper_binary_action))


def _pose_from_xy_quat(xy: torch.Tensor, z: float, quat: torch.Tensor) -> torch.Tensor:
    pos = torch.cat([xy, torch.tensor([z], dtype=xy.dtype, device=xy.device)])
    rot = math_utils.matrix_from_quat(quat.to(xy.device).unsqueeze(0))[0]
    return math_utils.make_pose(pos, rot)


def _get_cube_pos(env: ManagerBasedEnv, cube_name: str, env_id: int) -> torch.Tensor:
    object_pose = env.get_object_poses(env_ids=[env_id])[cube_name][0]
    return object_pose[:3, 3].clone().detach()


def _build_pre_grasp_pose(env: ManagerBasedEnv, cube_name: str, env_id: int, height: float = 0.1) -> torch.Tensor:
    cube_pos = _get_cube_pos(env, cube_name, env_id)
    return _pose_from_xy_quat(cube_pos[:2], height, DOWN_FACING_QUAT)


def _build_place_pose(env: ManagerBasedEnv, cube_name: str, env_id: int, height_offset: float = 0.15) -> torch.Tensor:
    cube_pos = _get_cube_pos(env, cube_name, env_id)
    return _pose_from_xy_quat(cube_pos[:2], cube_pos[2].item() + height_offset, DOWN_FACING_QUAT)


async def _plan_batch_async(
    planners: dict[int, BatchedCubeStackPlannerHandle],
    target_poses: dict[int, torch.Tensor],
    expected_objects: dict[int, str | None],
) -> list[bool]:
    coroutines = [
        planners[env_id].update_world_and_plan_motion_async(
            target_pose=target_poses[env_id],
            expected_attached_object=expected_objects[env_id],
            env_id=env_id,
        )
        for env_id in sorted(planners.keys())
    ]
    return list(await asyncio.gather(*coroutines))


@pytest.fixture(scope="class")
def batched_cube_stack_test_env() -> Generator[dict[str, Any], None, None]:
    random.seed(SEED)
    torch.manual_seed(SEED)

    env_cfg = FrankaCubeStackIKRelSkillgenEnvCfg()
    env_cfg.scene.num_envs = _get_test_num_envs()
    for frame in env_cfg.scene.ee_frame.target_frames:
        if frame.name == "end_effector":
            frame.offset.pos = (0.0, 0.0, 0.0)

    env: ManagerBasedEnv = gym.make(
        "Isaac-Stack-Cube-Franka-IK-Rel-Skillgen-v0",
        cfg=env_cfg,
        headless=headless,
    ).unwrapped
    env.reset()

    robot: Articulation = env.scene["robot"]
    planner_cfg = CuroboPlannerCfg.franka_stack_cube_config()
    planner_cfg.visualize_plan = False
    planner_cfg.visualize_spheres = False
    planner_cfg.debug_planner = False
    planner_cfg.time_dilation_factor = 1.0

    backend, planners = create_batched_cube_stack_motion_planners(
        env=env,
        robot=robot,
        config=planner_cfg,
        num_envs=env.num_envs,
        max_batch=_get_test_max_batch(),
    )
    loop = asyncio.new_event_loop()

    yield {
        "env": env,
        "robot": robot,
        "backend": backend,
        "planners": planners,
        "loop": loop,
    }

    loop.run_until_complete(backend.aclose())
    loop.close()
    env.close()


class TestBatchedCubeStackPlanner:
    @pytest.fixture(autouse=True)
    def setup(self, batched_cube_stack_test_env) -> None:
        self.env: ManagerBasedEnv = batched_cube_stack_test_env["env"]
        self.robot: Articulation = batched_cube_stack_test_env["robot"]
        self.backend: BatchedCubeStackPlannerBackend = batched_cube_stack_test_env["backend"]
        self.planners: dict[int, BatchedCubeStackPlannerHandle] = batched_cube_stack_test_env["planners"]
        self.loop: asyncio.AbstractEventLoop = batched_cube_stack_test_env["loop"]

    def test_detached_batch_plans_pre_grasp(self) -> None:
        self.env.reset()
        self.backend.batch_history.clear()

        target_poses = {
            env_id: _build_pre_grasp_pose(self.env, "cube_1", env_id)
            for env_id in range(self.env.num_envs)
        }
        expected_objects = {env_id: None for env_id in range(self.env.num_envs)}
        results = self.loop.run_until_complete(_plan_batch_async(self.planners, target_poses, expected_objects))

        assert results == [True] * self.env.num_envs
        _assert_bucket_history(self.backend, "open_detached", self.env.num_envs)
        for planner in self.planners.values():
            assert planner.current_plan is not None

    def test_attached_batch_plans_with_canonical_cube(self) -> None:
        self.env.reset()
        self.backend.batch_history.clear()

        pre_grasp_poses = {
            env_id: _build_pre_grasp_pose(self.env, "cube_1", env_id)
            for env_id in range(self.env.num_envs)
        }
        detached_objects = {env_id: None for env_id in range(self.env.num_envs)}
        pre_grasp_results = self.loop.run_until_complete(
            _plan_batch_async(self.planners, pre_grasp_poses, detached_objects)
        )
        assert pre_grasp_results == [True] * self.env.num_envs
        _execute_batched_plans(self.env, self.planners, gripper_binary_action=GRIPPER_OPEN_CMD)
        _execute_gripper_action_batch(self.env, gripper_binary_action=GRIPPER_CLOSE_CMD, steps=16)

        self.backend.batch_history.clear()
        place_poses = {
            env_id: _build_place_pose(self.env, "cube_2", env_id)
            for env_id in range(self.env.num_envs)
        }
        attached_objects = {env_id: "cube_1" for env_id in range(self.env.num_envs)}
        results = self.loop.run_until_complete(_plan_batch_async(self.planners, place_poses, attached_objects))

        assert results == [True] * self.env.num_envs
        _assert_bucket_history(self.backend, "closed_attached", self.env.num_envs)
        assert self.backend._canonical_attachment_pose is not None
        for planner in self.planners.values():
            assert planner.current_plan is not None
