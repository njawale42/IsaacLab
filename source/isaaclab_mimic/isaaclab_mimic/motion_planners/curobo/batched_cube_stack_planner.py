#!/usr/bin/env python3
# Copyright (c) 2025-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, MotionGenResult

import isaaclab.utils.math as PoseUtils
from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_env import ManagerBasedEnv

from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
from isaaclab_mimic.motion_planners.motion_planner_base import MotionPlannerBase


@dataclass(slots=True)
class _PlanRequest:
    handle: "BatchedCubeStackPlannerHandle"
    target_pose: torch.Tensor
    expected_attached_object: str | None
    step_size: float | None
    enable_retiming: bool | None
    future: asyncio.Future


class BatchedCubeStackPlannerBackend:
    """Shared cuRobo backend for batched cube-stack planning."""

    def __init__(
        self,
        env: ManagerBasedEnv,
        robot: Articulation,
        config: CuroboPlannerCfg,
        num_envs: int,
        max_batch: int | None = None,
    ) -> None:
        self.env = env
        self.robot = robot
        self.config = config
        self.num_envs = max(1, int(num_envs))
        requested_max_batch = 8 if max_batch is None else int(max_batch)
        self.max_batch = max(1, min(self.num_envs, requested_max_batch))
        self.logger = logging.getLogger("BatchedCubeStackPlannerBackend")

        setup_curobo_logger("warn")

        if torch.cuda.is_available():
            idx = self.config.cuda_device if self.config.cuda_device is not None else torch.cuda.current_device()
            self.tensor_args = TensorDeviceType(device=torch.device(f"cuda:{idx}"), dtype=torch.float32)
        else:
            self.tensor_args = TensorDeviceType()

        if self.config.robot_config_file is None:
            raise ValueError("robot_config_file is required")

        self.usd_helper = UsdHelper()
        self.usd_helper.load_stage(env.scene.stage)

        self.robot_cfg = load_yaml(self.config.robot_config_file)["robot_cfg"]
        if self.config.collision_spheres_file:
            self.robot_cfg["kinematics"]["collision_spheres"] = self.config.collision_spheres_file
        if self.config.extra_collision_spheres:
            self.robot_cfg["kinematics"]["extra_collision_spheres"] = self.config.extra_collision_spheres

        template_world = self._extract_template_world_config()
        world_cfgs = [deepcopy(template_world) for _ in range(self.max_batch)]

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            world_cfgs,
            tensor_args=self.tensor_args,
            collision_checker_type=self.config.collision_checker_type,
            num_trajopt_seeds=self.config.num_trajopt_seeds,
            num_graph_seeds=self.config.num_graph_seeds,
            interpolation_dt=self.config.interpolation_dt,
            collision_cache=self.config.collision_cache_size,
            trajopt_tsteps=self.config.trajopt_tsteps,
            maximum_trajectory_dt=self.config.maximum_trajectory_dt,
            collision_activation_distance=self.config.collision_activation_distance,
            position_threshold=self.config.position_threshold,
            rotation_threshold=self.config.rotation_threshold,
            n_collision_envs=self.max_batch,
            use_cuda_graph=False,
        )
        self.motion_gen = MotionGen(motion_gen_config)

        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=None,
            max_attempts=self.config.max_planning_attempts,
            enable_finetune_trajopt=self.config.enable_finetune_trajopt,
            time_dilation_factor=self.config.time_dilation_factor,
        )

        self._sync_joint_limits_from_isaac_lab()
        self.motion_gen.warmup(enable_graph=False, warmup_js_trajopt=False, batch_env_mode=True)

        world_model = self.motion_gen.world_coll_checker.world_model
        template_world_model = world_model[0] if isinstance(world_model, list) else world_model
        self._object_mappings = self._discover_object_mappings(template_world_model)
        self._dynamic_object_names = [
            name
            for name in self.env.scene.rigid_objects.keys()
            if not any(static_name in name.lower() for static_name in getattr(self.config, "static_objects", []))
        ]
        self._gripper_joint_name_to_idx = {
            joint_name: idx for idx, joint_name in enumerate(self.robot.data.joint_names)
        }

        self._canonical_attachment_pose: Pose | None = None
        self._canonical_attachment_template: Any | None = None
        self._canonical_attachment_active = False
        self.batch_history: list[tuple[str, int]] = []

        self._queue: asyncio.Queue[_PlanRequest | None] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    async def submit(
        self,
        handle: "BatchedCubeStackPlannerHandle",
        target_pose: torch.Tensor,
        expected_attached_object: str | None,
        step_size: float | None,
        enable_retiming: bool | None,
    ) -> bool:
        self._ensure_worker()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put(
            _PlanRequest(
                handle=handle,
                target_pose=target_pose.clone(),
                expected_attached_object=expected_attached_object,
                step_size=step_size,
                enable_retiming=enable_retiming,
                future=future,
            )
        )
        return await future

    async def aclose(self) -> None:
        if self._worker_task is None:
            return
        await self._queue.put(None)
        await self._worker_task
        self._worker_task = None

    def _ensure_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def _worker_loop(self) -> None:
        while True:
            request = await self._queue.get()
            if request is None:
                self._queue.task_done()
                break

            batch = [request]
            stop_after_batch = False
            await asyncio.sleep(0)
            while not self._queue.empty() and len(batch) < self.max_batch:
                next_request = self._queue.get_nowait()
                if next_request is None:
                    stop_after_batch = True
                    break
                batch.append(next_request)

            try:
                await self._process_requests(batch)
            except Exception as exc:
                self.logger.exception("Batched cube-stack planning failed: %s", exc)
                for req in batch:
                    if not req.future.done():
                        req.future.set_result(False)
            finally:
                for _ in batch:
                    self._queue.task_done()

            if stop_after_batch:
                self._queue.task_done()
                return

    async def _process_requests(self, requests: list[_PlanRequest]) -> None:
        open_detached: list[_PlanRequest] = []
        closed_detached: list[_PlanRequest] = []
        closed_attached: list[_PlanRequest] = []

        for request in requests:
            if request.expected_attached_object is None:
                open_detached.append(request)
            elif self._is_object_grasped(request.handle.env_id):
                closed_attached.append(request)
            else:
                closed_detached.append(request)

        for bucket, gripper_closed, attach_canonical in (
            (open_detached, False, False),
            (closed_detached, True, False),
            (closed_attached, True, True),
        ):
            if bucket:
                self._solve_bucket(bucket, gripper_closed=gripper_closed, attach_canonical=attach_canonical)

    def _solve_bucket(
        self,
        requests: list[_PlanRequest],
        *,
        gripper_closed: bool,
        attach_canonical: bool,
    ) -> None:
        for start_idx in range(0, len(requests), self.max_batch):
            chunk = requests[start_idx : start_idx + self.max_batch]
            pending_chunk = [request for request in chunk if not request.future.done()]
            if not pending_chunk:
                continue
            try:
                self._solve_chunk(pending_chunk, gripper_closed=gripper_closed, attach_canonical=attach_canonical)
            except torch.OutOfMemoryError as exc:
                self._fail_chunk_oom(pending_chunk, exc)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                self._fail_chunk_oom(pending_chunk, exc)

    def _fail_chunk_oom(self, chunk: list[_PlanRequest], exc: Exception) -> None:
        self.logger.warning(
            "Fixed-size batch chunk of %d requests failed with OOM at max_batch=%d.",
            len(chunk),
            self.max_batch,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for request in chunk:
            request.handle.reset_plan()
            if not request.future.done():
                request.future.set_result(False)
        self.logger.warning("Chunk OOM details: %s", exc)

    def _solve_chunk(
        self,
        chunk: list[_PlanRequest],
        *,
        gripper_closed: bool,
        attach_canonical: bool,
    ) -> None:
        if not chunk:
            return

        if attach_canonical:
            mode_name = "closed_attached"
        elif gripper_closed:
            mode_name = "closed_detached"
        else:
            mode_name = "open_detached"

        start_states = [self.get_current_joint_state(req.handle.env_id) for req in chunk]
        self._set_gripper_state(gripper_closed)
        self._detach_canonical_attachment()
        if attach_canonical:
            self._ensure_canonical_attachment(chunk[0], start_states[0])
            if not self._attach_canonical_attachment(start_states[0]):
                for request in chunk:
                    request.handle.reset_plan()
                    if not request.future.done():
                        request.future.set_result(False)
                return

        for slot, request in enumerate(chunk):
            carried_object = request.expected_attached_object if attach_canonical else None
            self._populate_batch_slot(slot=slot, env_id=request.handle.env_id, carried_object=carried_object)

        phase_targets: dict[BatchedCubeStackPlannerHandle, list[Pose]] = {}
        phase_contacts: list[bool] | None = None
        for request, start_state in zip(chunk, start_states):
            targets, contacts = self._build_phase_targets(start_state=start_state, target_pose=request.target_pose)
            phase_targets[request.handle] = targets
            if phase_contacts is None:
                phase_contacts = contacts

        if phase_contacts is None:
            return

        full_plans: dict[BatchedCubeStackPlannerHandle, JointState | None] = {
            request.handle: None for request in chunk
        }
        active_requests = list(chunk)
        active_states = list(start_states)

        for phase_idx, contact_flag in enumerate(phase_contacts):
            if not active_requests:
                break

            restore_attached_spheres = None
            if contact_flag:
                restore_attached_spheres = self._disable_contact_links(include_attachment=attach_canonical)

            batch_start_state = self._stack_joint_states(active_states)
            batch_goal_pose = self._stack_goal_poses(
                [phase_targets[request.handle][phase_idx] for request in active_requests]
            )
            batch_plan_config = self.plan_config.clone()
            result = self.motion_gen.plan_batch_env(batch_start_state, batch_goal_pose, batch_plan_config)

            if restore_attached_spheres is not None:
                self._restore_contact_links(restore_attached_spheres)

            next_active_requests: list[_PlanRequest] = []
            next_active_states: list[JointState] = []

            for row_idx, request in enumerate(active_requests):
                success = bool(result.success[row_idx].item()) if result.success is not None else False
                if not success:
                    request.handle.reset_plan()
                    continue

                row_plan = self._extract_row_plan(result, row_idx)
                row_plan = self.motion_gen.get_full_js(row_plan)
                common_js_names = [name for name in self.robot.data.joint_names if name in row_plan.joint_names]
                row_plan = row_plan.get_ordered_joint_state(common_js_names)
                row_plan = self._ensure_trajectory_joint_state(row_plan)

                full_plan = full_plans[request.handle]
                full_plans[request.handle] = row_plan if full_plan is None else full_plan.stack(row_plan)

                last_waypoint = row_plan.position[-1].unsqueeze(0)
                current_state = JointState(
                    position=last_waypoint,
                    velocity=torch.zeros_like(last_waypoint),
                    acceleration=torch.zeros_like(last_waypoint),
                    joint_names=row_plan.joint_names,
                    tensor_args=self.tensor_args,
                )
                next_active_requests.append(request)
                next_active_states.append(current_state.get_ordered_joint_state(self.motion_gen.kinematics.joint_names))

            active_requests = next_active_requests
            active_states = next_active_states

        for request in chunk:
            plan = full_plans[request.handle]
            if plan is None:
                plan = self._retry_single_request(
                    request=request,
                    gripper_closed=gripper_closed,
                    attach_canonical=attach_canonical,
                )
            if plan is None:
                request.handle.reset_plan()
                if not request.future.done():
                    request.future.set_result(False)
                continue

            enable_retiming = request.enable_retiming
            if enable_retiming is None:
                enable_retiming = request.step_size is not None
            if enable_retiming and request.step_size is not None:
                plan = self._linearly_retime_plan(step_size=request.step_size, plan=plan)

            request.handle._current_plan = plan
            request.handle._plan_index = 0
            if not request.future.done():
                request.future.set_result(True)

        self.batch_history.append((mode_name, len(chunk)))

    def get_current_joint_state(self, env_id: int) -> JointState:
        joint_pos = self.robot.data.joint_pos[env_id, :].unsqueeze(0)
        joint_vel = torch.zeros_like(joint_pos)
        joint_acc = torch.zeros_like(joint_pos)
        state = JointState(
            position=self._to_curobo_device(joint_pos),
            velocity=self._to_curobo_device(joint_vel),
            acceleration=self._to_curobo_device(joint_acc),
            joint_names=self.robot.data.joint_names,
            tensor_args=self.tensor_args,
        )
        return state.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

    def get_ee_pose(self, joint_state: JointState) -> Pose:
        return self.motion_gen.compute_kinematics(joint_state).ee_pose

    def _extract_template_world_config(self) -> WorldConfig:
        from isaaclab.utils.math import quat_inv

        env_prim_path = "/World/envs/env_0"
        robot_prim_path = self.config.robot_prim_path or f"{env_prim_path}/Robot"
        ignore_list = list(
            self.config.world_ignore_substrings
            or [f"{env_prim_path}/target", "/World/defaultGroundPlane", "/curobo"]
        )
        if robot_prim_path not in ignore_list:
            ignore_list.append(robot_prim_path)

        world_cfg = self.usd_helper.get_obstacles_from_stage(
            only_paths=[env_prim_path],
            reference_prim_path=None,
            ignore_substring=ignore_list,
        )
        robot_pos_w = self.robot.data.root_pos_w[0]
        robot_quat_w = self.robot.data.root_quat_w[0]
        self._transform_world_config_to_robot_frame(world_cfg, robot_pos_w, quat_inv(robot_quat_w))
        return world_cfg.get_collision_check_world()

    def _transform_world_config_to_robot_frame(
        self,
        world_cfg: WorldConfig,
        robot_pos_w: torch.Tensor,
        robot_quat_inv: torch.Tensor,
    ) -> None:
        from isaaclab.utils.math import quat_apply, quat_mul

        def _transform_obstacle(obj: Any) -> None:
            if obj is None or not hasattr(obj, "pose") or obj.pose is None:
                return
            pos = torch.tensor(obj.pose[:3], device=robot_pos_w.device, dtype=torch.float32)
            quat = torch.tensor(obj.pose[3:], device=robot_pos_w.device, dtype=torch.float32)
            rel_pos = pos - robot_pos_w
            new_pos = quat_apply(robot_quat_inv, rel_pos)
            new_quat = quat_mul(robot_quat_inv, quat)
            obj.pose = [
                float(new_pos[0].item()),
                float(new_pos[1].item()),
                float(new_pos[2].item()),
                float(new_quat[0].item()),
                float(new_quat[1].item()),
                float(new_quat[2].item()),
                float(new_quat[3].item()),
            ]

        for primitive_type in ("cuboid", "mesh", "cylinder", "capsule", "sphere"):
            primitive_list = getattr(world_cfg, primitive_type, None)
            if not primitive_list:
                continue
            for primitive in primitive_list:
                _transform_obstacle(primitive)

    def _discover_object_mappings(self, world_model: Any) -> dict[str, str]:
        mappings: dict[str, str] = {}
        env_prefix = "/World/envs/env_0/"
        world_object_paths: list[str] = []

        for primitive_type in ("mesh", "cuboid", "sphere", "capsule", "cylinder", "voxel", "blox"):
            primitive_list = getattr(world_model, primitive_type, None)
            if not primitive_list:
                continue
            for primitive in primitive_list:
                if primitive.name and env_prefix in str(primitive.name):
                    world_object_paths.append(str(primitive.name))

        for object_name in self.env.scene.rigid_objects.keys():
            for object_path in world_object_paths:
                if object_name.lower().replace("_", "") in object_path.lower().replace("_", ""):
                    mappings[object_name] = object_path
                    break

        return mappings

    def _sync_joint_limits_from_isaac_lab(self) -> None:
        joint_limits = self.motion_gen.kinematics.get_joint_limits()
        isaac_limits = self.robot.data.joint_pos_limits[0]
        isaac_name_to_idx = {name: idx for idx, name in enumerate(self.robot.data.joint_names)}

        for curobo_idx, joint_name in enumerate(joint_limits.joint_names):
            if joint_name not in isaac_name_to_idx:
                continue
            isaac_idx = isaac_name_to_idx[joint_name]
            joint_limits.position[0, curobo_idx] = isaac_limits[isaac_idx, 0].item()
            joint_limits.position[1, curobo_idx] = isaac_limits[isaac_idx, 1].item()

    def _to_curobo_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.tensor_args.device, dtype=self.tensor_args.dtype)

    def _to_env_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.env.device, dtype=tensor.dtype)

    def _make_pose(
        self,
        *,
        position: torch.Tensor | list[float] | None = None,
        quaternion: torch.Tensor | list[float] | None = None,
    ) -> Pose:
        if position is None:
            position = torch.zeros((1, 3), device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        if quaternion is None:
            quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        if not isinstance(quaternion, torch.Tensor):
            quaternion = torch.tensor(quaternion, device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        if position.dim() == 1:
            position = position.unsqueeze(0)
        if quaternion.dim() == 1:
            quaternion = quaternion.unsqueeze(0)
        return Pose(
            position=self._to_curobo_device(position),
            quaternion=self._to_curobo_device(quaternion),
            normalize_rotation=False,
        )

    def _stack_joint_states(self, joint_states: list[JointState]) -> JointState:
        position = torch.cat([state.position for state in joint_states], dim=0)
        velocity = torch.cat([state.velocity for state in joint_states], dim=0)
        acceleration = torch.cat([state.acceleration for state in joint_states], dim=0)
        return JointState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            joint_names=joint_states[0].joint_names,
            tensor_args=self.tensor_args,
        )

    def _stack_goal_poses(self, goal_poses: list[Pose]) -> Pose:
        position = torch.cat([pose.position for pose in goal_poses], dim=0)
        quaternion = torch.cat([pose.quaternion for pose in goal_poses], dim=0)
        return Pose(position=position, quaternion=quaternion, normalize_rotation=False)

    def _is_object_grasped(self, env_id: int) -> bool:
        if not self.config.gripper_joint_names:
            return False
        joint_name = self.config.gripper_joint_names[0]
        joint_idx = self._gripper_joint_name_to_idx[joint_name]
        gripper_val = self.robot.data.joint_pos[env_id, joint_idx].item()
        return gripper_val < self.config.grasp_gripper_open_val

    def _set_gripper_state(self, gripper_closed: bool) -> None:
        if gripper_closed:
            locked_joints = self.config.gripper_closed_positions
        else:
            locked_joints = self.config.gripper_open_positions
        self.motion_gen.update_locked_joints(locked_joints, self.robot_cfg)

    def _populate_batch_slot(self, slot: int, env_id: int, carried_object: str | None) -> None:
        from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

        robot_pos_w = self.robot.data.root_pos_w[env_id]
        robot_quat_w = self.robot.data.root_quat_w[env_id]
        robot_quat_inv = quat_inv(robot_quat_w)

        for object_name in self._dynamic_object_names:
            if object_name not in self._object_mappings:
                continue

            obj = self.env.scene.rigid_objects[object_name]
            obj_pos_w = obj.data.root_pos_w[env_id]
            obj_quat_w = obj.data.root_quat_w[env_id]
            rel_pos = obj_pos_w - robot_pos_w
            pos_base = quat_apply(robot_quat_inv, rel_pos)
            quat_base = quat_mul(robot_quat_inv, obj_quat_w)
            curobo_pose = self._make_pose(position=pos_base, quaternion=quat_base)
            object_path = self._object_mappings[object_name]
            self.motion_gen.world_coll_checker.update_obstacle_pose(object_path, curobo_pose, env_idx=slot)
            self.motion_gen.world_coll_checker.enable_obstacle(object_path, enable=True, env_idx=slot)

        if carried_object is not None and carried_object in self._object_mappings:
            self.motion_gen.world_coll_checker.enable_obstacle(
                self._object_mappings[carried_object],
                enable=False,
                env_idx=slot,
            )

    def _get_object_pose_in_base_frame(self, env_id: int, object_name: str) -> Pose:
        from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

        obj = self.env.scene.rigid_objects[object_name]
        robot_pos_w = self.robot.data.root_pos_w[env_id]
        robot_quat_w = self.robot.data.root_quat_w[env_id]
        obj_pos_w = obj.data.root_pos_w[env_id]
        obj_quat_w = obj.data.root_quat_w[env_id]
        robot_quat_inv = quat_inv(robot_quat_w)
        rel_pos = quat_apply(robot_quat_inv, obj_pos_w - robot_pos_w)
        rel_quat = quat_mul(robot_quat_inv, obj_quat_w)
        return self._make_pose(position=rel_pos, quaternion=rel_quat)

    def _ensure_canonical_attachment(self, request: _PlanRequest, start_state: JointState) -> None:
        if self._canonical_attachment_pose is not None and self._canonical_attachment_template is not None:
            return
        if request.expected_attached_object is None:
            raise ValueError("attached request must include an expected object")

        object_name = request.expected_attached_object
        if object_name not in self._object_mappings:
            raise KeyError(f"Object '{object_name}' not found in cuRobo world mappings")

        actual_object_pose = self._get_object_pose_in_base_frame(request.handle.env_id, object_name)
        ee_pose = self.get_ee_pose(start_state)
        self._canonical_attachment_pose = ee_pose.inverse().multiply(actual_object_pose)

        world_model = self.motion_gen.world_coll_checker.world_model
        template_world_model = world_model[0] if isinstance(world_model, list) else world_model
        template_obstacle = template_world_model.get_obstacle(self._object_mappings[object_name])
        self._canonical_attachment_template = deepcopy(template_obstacle)

    def _attach_canonical_attachment(self, representative_state: JointState) -> bool:
        if self._canonical_attachment_pose is None or self._canonical_attachment_template is None:
            return False

        ee_pose = self.get_ee_pose(representative_state)
        canonical_object_pose = ee_pose.multiply(self._canonical_attachment_pose)
        canonical_obstacle = deepcopy(self._canonical_attachment_template)
        canonical_obstacle.pose = self._pose_to_list(canonical_object_pose)
        success = self.motion_gen.attach_external_objects_to_robot(
            joint_state=representative_state,
            external_objects=[canonical_obstacle],
            link_name=self.config.attached_object_link_name,
            surface_sphere_radius=self.config.surface_sphere_radius,
            sphere_fit_type=SphereFitType.SAMPLE_SURFACE,
        )
        self._canonical_attachment_active = bool(success)
        return bool(success)

    def _detach_canonical_attachment(self) -> None:
        if not self._canonical_attachment_active:
            return
        self.motion_gen.detach_object_from_robot(link_name=self.config.attached_object_link_name)
        self._canonical_attachment_active = False

    def _pose_to_list(self, pose: Pose) -> list[float]:
        position = pose.position.reshape(-1, 3)[0]
        quaternion = pose.quaternion.reshape(-1, 4)[0]
        return [
            float(position[0].item()),
            float(position[1].item()),
            float(position[2].item()),
            float(quaternion[0].item()),
            float(quaternion[1].item()),
            float(quaternion[2].item()),
            float(quaternion[3].item()),
        ]

    def _build_phase_targets(self, start_state: JointState, target_pose: torch.Tensor) -> tuple[list[Pose], list[bool]]:
        target_pose_cuda = self._to_curobo_device(target_pose)
        target_pos, target_rot = PoseUtils.unmake_pose(target_pose_cuda)
        goal_pose = self._make_pose(position=target_pos, quaternion=PoseUtils.quat_from_matrix(target_rot))

        targets: list[Pose] = []
        contacts: list[bool] = []
        in_world_frame = self.config.approach_retreat_frame == "world"
        world_sign = -1.0 if in_world_frame else 1.0
        direction = self.config.approach_direction

        if self.config.retreat_distance is not None and self.config.retreat_distance > 0:
            ee_pose = self.get_ee_pose(start_state)
            if in_world_frame:
                pos = ee_pose.position.reshape(-1, 3)[0]
                offset = torch.tensor(
                    [world_sign * val * self.config.retreat_distance for val in direction],
                    device=pos.device,
                    dtype=pos.dtype,
                )
                retreat_pose = self._make_pose(position=pos + offset, quaternion=ee_pose.quaternion)
            else:
                retreat_pose = ee_pose.multiply(
                    self._make_pose(position=[val * self.config.retreat_distance for val in direction])
                )
            targets.append(retreat_pose)
            contacts.append(True)

        contacts.append(False)
        if self.config.approach_distance is not None and self.config.approach_distance > 0:
            if in_world_frame:
                pos = goal_pose.position.reshape(-1, 3)[0]
                offset = torch.tensor(
                    [world_sign * val * self.config.approach_distance for val in direction],
                    device=pos.device,
                    dtype=pos.dtype,
                )
                approach_pose = self._make_pose(position=pos + offset, quaternion=goal_pose.quaternion)
            else:
                approach_pose = goal_pose.multiply(
                    self._make_pose(position=[val * self.config.approach_distance for val in direction])
                )
            targets.append(approach_pose)
            contacts.append(True)

        targets.append(goal_pose)
        return targets, contacts[: len(targets)]

    def _disable_contact_links(self, include_attachment: bool) -> torch.Tensor | None:
        for link_name in self.config.hand_link_names:
            self.motion_gen.kinematics.kinematics_config.disable_link_spheres(link_name)

        if not include_attachment:
            return None

        attached_link = self.config.attached_object_link_name
        spheres = self.motion_gen.kinematics.kinematics_config.get_link_spheres(attached_link).clone()
        self.motion_gen.kinematics.kinematics_config.disable_link_spheres(attached_link)
        return spheres

    def _restore_contact_links(self, restore_attached_spheres: torch.Tensor | None) -> None:
        for link_name in self.config.hand_link_names:
            self.motion_gen.kinematics.kinematics_config.enable_link_spheres(link_name)

        if restore_attached_spheres is not None:
            self.motion_gen.kinematics.kinematics_config.update_link_spheres(
                self.config.attached_object_link_name,
                restore_attached_spheres,
            )

    def _extract_single_result(self, batched_result: MotionGenResult, idx: int) -> MotionGenResult:
        return MotionGenResult(
            success=batched_result.success[idx : idx + 1] if batched_result.success is not None else None,
            optimized_plan=batched_result.optimized_plan[idx] if batched_result.optimized_plan is not None else None,
            optimized_dt=(
                batched_result.optimized_dt[idx : idx + 1]
                if isinstance(batched_result.optimized_dt, torch.Tensor) and batched_result.optimized_dt.ndim > 0
                else batched_result.optimized_dt
            ),
            interpolated_plan=(
                batched_result.interpolated_plan[idx] if batched_result.interpolated_plan is not None else None
            ),
            path_buffer_last_tstep=(
                [batched_result.path_buffer_last_tstep[idx]]
                if batched_result.path_buffer_last_tstep is not None
                else None
            ),
            status=batched_result.status[idx] if isinstance(batched_result.status, list) else batched_result.status,
            interpolation_dt=batched_result.interpolation_dt,
        )

    def _extract_row_plan(self, batched_result: MotionGenResult, idx: int) -> JointState:
        single_result = self._extract_single_result(batched_result, idx)
        if single_result.optimized_plan is not None and single_result.optimized_plan.position.numel() != 0:
            return single_result.optimized_plan
        return single_result.get_interpolated_plan()

    def _linearly_retime_plan(self, step_size: float, plan: JointState) -> JointState:
        if len(plan.position) <= 1:
            return plan

        path = plan.position
        deltas = path[1:] - path[:-1]
        distances = torch.norm(deltas, dim=-1)
        waypoints = [path[0]]
        for distance, waypoint in zip(distances, path[1:]):
            if distance > 1e-6:
                waypoints.append(waypoint)
        if len(waypoints) <= 1:
            return plan

        waypoints_tensor = torch.stack(waypoints)
        deltas = waypoints_tensor[1:] - waypoints_tensor[:-1]
        distances = torch.norm(deltas, dim=-1)
        cumulative = torch.cat([torch.zeros(1, device=distances.device), torch.cumsum(distances, dim=0)])
        total_distance = cumulative[-1].item()
        if total_distance < 1e-6:
            return plan

        num_samples = max(2, int(total_distance / step_size) + 1)
        target_distances = torch.linspace(0.0, total_distance, num_samples, device=waypoints_tensor.device)

        retimed = [waypoints_tensor[0]]
        segment_idx = 0
        for target_distance in target_distances[1:-1]:
            while segment_idx < len(distances) - 1 and target_distance > cumulative[segment_idx + 1]:
                segment_idx += 1
            segment_start = waypoints_tensor[segment_idx]
            segment_end = waypoints_tensor[segment_idx + 1]
            segment_distance = max(float(distances[segment_idx].item()), 1e-6)
            alpha = float((target_distance - cumulative[segment_idx]).item()) / segment_distance
            retimed.append((1.0 - alpha) * segment_start + alpha * segment_end)
        retimed.append(waypoints_tensor[-1])
        retimed_tensor = torch.stack(retimed)
        return JointState(
            position=retimed_tensor,
            velocity=torch.zeros_like(retimed_tensor),
            acceleration=torch.zeros_like(retimed_tensor),
            joint_names=plan.joint_names,
            tensor_args=plan.tensor_args,
        )

    def _ensure_trajectory_joint_state(self, plan: JointState) -> JointState:
        if not isinstance(plan.position, torch.Tensor) or plan.position.ndim > 1:
            return plan

        velocity = plan.velocity.unsqueeze(0) if isinstance(plan.velocity, torch.Tensor) and plan.velocity.ndim == 1 else plan.velocity
        acceleration = (
            plan.acceleration.unsqueeze(0)
            if isinstance(plan.acceleration, torch.Tensor) and plan.acceleration.ndim == 1
            else plan.acceleration
        )
        jerk = plan.jerk.unsqueeze(0) if isinstance(plan.jerk, torch.Tensor) and plan.jerk.ndim == 1 else plan.jerk
        return JointState(
            position=plan.position.unsqueeze(0),
            velocity=velocity,
            acceleration=acceleration,
            joint_names=plan.joint_names,
            jerk=jerk,
            tensor_args=plan.tensor_args,
        )

    def _retry_single_request(
        self,
        request: _PlanRequest,
        *,
        gripper_closed: bool,
        attach_canonical: bool,
    ) -> JointState | None:
        self.logger.warning(
            "Retrying failed batched plan individually for env %d (attach=%s)",
            request.handle.env_id,
            attach_canonical,
        )

        start_state = self.get_current_joint_state(request.handle.env_id)
        self._set_gripper_state(gripper_closed)
        self._detach_canonical_attachment()
        if attach_canonical:
            self._ensure_canonical_attachment(request, start_state)
            if not self._attach_canonical_attachment(start_state):
                return None

        carried_object = request.expected_attached_object if attach_canonical else None
        self._populate_batch_slot(slot=0, env_id=request.handle.env_id, carried_object=carried_object)
        targets, contacts = self._build_phase_targets(start_state=start_state, target_pose=request.target_pose)

        current_state = start_state
        full_plan: JointState | None = None
        for phase_idx, contact_flag in enumerate(contacts):
            restore_attached_spheres = None
            if contact_flag:
                restore_attached_spheres = self._disable_contact_links(include_attachment=attach_canonical)

            batch_start_state = self._stack_joint_states([current_state])
            batch_goal_pose = self._stack_goal_poses([targets[phase_idx]])
            batch_plan_config = self.plan_config.clone()
            result = self.motion_gen.plan_batch_env(batch_start_state, batch_goal_pose, batch_plan_config)

            if restore_attached_spheres is not None:
                self._restore_contact_links(restore_attached_spheres)

            success = bool(result.success[0].item()) if result.success is not None else False
            if not success:
                return None

            row_plan = self._extract_row_plan(result, 0)
            row_plan = self.motion_gen.get_full_js(row_plan)
            common_js_names = [name for name in self.robot.data.joint_names if name in row_plan.joint_names]
            row_plan = row_plan.get_ordered_joint_state(common_js_names)
            row_plan = self._ensure_trajectory_joint_state(row_plan)
            full_plan = row_plan if full_plan is None else full_plan.stack(row_plan)

            last_waypoint = row_plan.position[-1].unsqueeze(0)
            current_state = JointState(
                position=last_waypoint,
                velocity=torch.zeros_like(last_waypoint),
                acceleration=torch.zeros_like(last_waypoint),
                joint_names=row_plan.joint_names,
                tensor_args=self.tensor_args,
            )
            current_state = current_state.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        return full_plan


class BatchedCubeStackPlannerHandle(MotionPlannerBase):
    """Per-environment handle backed by a shared batched cuRobo solver."""

    def __init__(
        self,
        backend: BatchedCubeStackPlannerBackend,
        env: ManagerBasedEnv,
        robot: Articulation,
        config: CuroboPlannerCfg,
        env_id: int,
    ) -> None:
        super().__init__(env=env, robot=robot, env_id=env_id, debug=config.debug_planner)
        self.backend = backend
        self.config = config
        self.motion_gen = backend.motion_gen
        self.n_repeat = config.n_repeat
        self.step_size = config.motion_step_size
        self.visualize_spheres = False
        self.visualize_plan = False
        self.plan_visualizer = None
        self._current_plan: JointState | None = None
        self._plan_index = 0

    @property
    def current_plan(self) -> JointState | None:
        return self._current_plan

    async def update_world_and_plan_motion_async(
        self,
        target_pose: torch.Tensor,
        expected_attached_object: str | None = None,
        env_id: int | None = None,
        step_size: float | None = None,
        enable_retiming: bool | None = None,
        **_: Any,
    ) -> bool:
        if env_id is not None and env_id != self.env_id:
            raise ValueError(f"Planner handle is bound to env {self.env_id}, received env_id={env_id}")
        return await self.backend.submit(
            handle=self,
            target_pose=target_pose,
            expected_attached_object=expected_attached_object,
            step_size=step_size,
            enable_retiming=enable_retiming,
        )

    def update_world_and_plan_motion(self, target_pose: torch.Tensor, **kwargs: Any) -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.update_world_and_plan_motion_async(target_pose, **kwargs))
        raise RuntimeError("Use update_world_and_plan_motion_async() from an active asyncio event loop")

    def has_next_waypoint(self) -> bool:
        return self._current_plan is not None and self._plan_index < len(self._current_plan.position)

    def get_next_waypoint_ee_pose(self) -> torch.Tensor:
        if self._current_plan is None:
            raise RuntimeError("No active plan available")
        next_joint_state = self._current_plan[self._plan_index]
        self._plan_index += 1
        eef_pose = self.motion_gen.compute_kinematics(next_joint_state).ee_pose
        position = self.backend._to_env_device(eef_pose.position)
        rotation = self.backend._to_env_device(eef_pose.get_rotation())
        return PoseUtils.make_pose(position, rotation)[0]

    def reset_plan(self) -> None:
        self._plan_index = 0
        self._current_plan = None

    def get_planned_poses(self) -> list[torch.Tensor]:
        if self._current_plan is None:
            return []

        planned_poses: list[torch.Tensor] = []
        original_plan_index = self._plan_index
        self._plan_index = 0

        while self.has_next_waypoint():
            next_joint_state = self._current_plan[self._plan_index]
            self._plan_index += 1
            eef_pose = self.motion_gen.compute_kinematics(next_joint_state).ee_pose
            position = self.backend._to_env_device(eef_pose.position)
            rotation = self.backend._to_env_device(eef_pose.get_rotation())
            planned_poses.append(PoseUtils.make_pose(position, rotation)[0])

        self._plan_index = original_plan_index
        if self.n_repeat is not None and self.n_repeat > 0 and planned_poses:
            planned_poses.extend([planned_poses[-1]] * self.n_repeat)
        return planned_poses

    def _get_current_joint_state_for_curobo(self) -> JointState:
        return self.backend.get_current_joint_state(self.env_id)

    def get_ee_pose(self, joint_state: JointState) -> Pose:
        return self.backend.get_ee_pose(joint_state)

    def _update_visualization_at_joint_positions(self, joint_positions: torch.Tensor) -> None:
        del joint_positions
        return


def create_batched_cube_stack_motion_planners(
    env: ManagerBasedEnv,
    robot: Articulation,
    config: CuroboPlannerCfg,
    num_envs: int,
    max_batch: int | None = None,
) -> tuple[BatchedCubeStackPlannerBackend, dict[int, BatchedCubeStackPlannerHandle]]:
    backend = BatchedCubeStackPlannerBackend(
        env=env,
        robot=robot,
        config=config,
        num_envs=num_envs,
        max_batch=max_batch,
    )
    handles = {
        env_id: BatchedCubeStackPlannerHandle(
            backend=backend,
            env=env,
            robot=robot,
            config=config,
            env_id=env_id,
        )
        for env_id in range(num_envs)
    }
    return backend, handles
