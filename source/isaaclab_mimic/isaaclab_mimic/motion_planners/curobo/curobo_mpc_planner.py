# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Reactive MPC-based motion planner for Isaac Lab using cuRobo MPPI.

This planner implements the MotionPlannerBase interface but, unlike the
trajectory planner, it plans reactively: on every call to
get_next_waypoint_ee_pose() it runs one MPC step given the current joint
state and target pose, returning the next end-effector pose to track.
"""

from __future__ import annotations

import logging
import torch
from typing import Any
from collections import deque
import os
import re

from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig, DiffusionMpcSolver

import isaaclab.utils.math as PoseUtils
from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_env import ManagerBasedEnv

from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
from isaaclab_mimic.motion_planners.motion_planner_base import MotionPlannerBase


class CuroboMPCPlanner(MotionPlannerBase):
    """Reactive MPC planner powered by cuRobo MPPI.

    The planner updates the collision world and runs an MPC step each time
    the caller requests the next waypoint. This provides reactivity to
    environment changes and moving obstacles.
    """

    reactive: bool = True

    def __init__(
        self,
        env: ManagerBasedEnv,
        robot: Articulation,
        config: CuroboPlannerCfg,
        *,
        env_id: int = 0,
        debug: bool | None = None,
    ) -> None:
        super().__init__(
            env=env, robot=robot, env_id=env_id, debug=bool(config.debug_planner if debug is None else debug)
        )

        self.logger = logging.getLogger(f"CuroboMPCPlanner_{env_id}")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG if (config.debug_planner if debug is None else debug) else logging.INFO)

        self.config: CuroboPlannerCfg = config
        self.visualize_spheres: bool = bool(self.config.visualize_spheres)

        setup_curobo_logger("warn")

        # Force cuRobo to always use CUDA device when available
        if torch.cuda.is_available():
            idx = self.config.cuda_device if self.config.cuda_device is not None else torch.cuda.current_device()
            self.tensor_args = TensorDeviceType(device=torch.device(f"cuda:{idx}"), dtype=torch.float32)
            self.logger.debug(f"MPC will run on CUDA device {idx}")
        else:
            self.tensor_args = TensorDeviceType()
            self.logger.warning("CUDA not available, MPC running on CPU")

        # Load robot configuration file (required)
        if self.config.robot_config_file is None:
            raise ValueError("robot_config_file is required for MPC planner")

        # World configuration (static baseline). We'll extract static world from the USD stage
        # and initialize the solver with that; dynamic objects are updated each step.
        # Build MPC solver once, using world config from planner cfg (will be replaced with USD scene next)
        initial_world_cfg = self.config.get_world_config()
        if bool(getattr(self.config, "mpc_use_diffusion", False)):
            self.logger.info("MPC: Using diffusion-guided MPPI")
            self.mpc = DiffusionMpcSolver.create_from_robot_config(
                robot_cfg=self.config.robot_config_file,
                world_model=initial_world_cfg,
                tensor_args=self.tensor_args,
                compute_metrics=True,
                self_collision_check=True,
                collision_checker_type=self.config.collision_checker_type,
                store_rollouts=True,
                collision_cache=self.config.collision_cache_size,
                collision_activation_distance=self.config.collision_activation_distance,
                step_dt=self.config.interpolation_dt,
                diffusion_ckpt_path=getattr(self.config, "diffusion_ckpt_path", None),
                diffusion_alpha=float(getattr(self.config, "diffusion_alpha", 1.0)),
            )
        else:
            mpc_config: MpcSolverConfig = MpcSolverConfig.load_from_robot_config(
                robot_cfg=self.config.robot_config_file,
                world_model=initial_world_cfg,
                tensor_args=self.tensor_args,
                compute_metrics=True,
                use_cuda_graph=True,
                self_collision_check=True,
                collision_checker_type=self.config.collision_checker_type,
                store_rollouts=True,
                collision_cache=self.config.collision_cache_size,
                collision_activation_distance=self.config.collision_activation_distance,
                step_dt=self.config.interpolation_dt,
                use_mppi=not bool(self.config.mpc_use_imppi),
                use_imppi=bool(self.config.mpc_use_imppi),
                imppi_target_kl=float(self.config.imppi_target_kl),
                imppi_reuse_prev_iter=bool(self.config.imppi_reuse_prev_iter),
                imppi_max_backtracks=int(self.config.imppi_max_backtracks),
                imppi_backtrack_coeff=float(self.config.imppi_backtrack_coeff),
            )
            self.mpc = MpcSolver(mpc_config)

        # Initialize diffusion prior (if enabled) before attaching to solver
        self._diff_prior = None
        if bool(getattr(self.config, "mpc_use_diffusion", False)):
            try:
                self._init_diffusion_prior()
            except Exception:
                self.logger.warning("Diffusion prior initialization failed; proceeding without it")
            # Report optimizer type
            try:
                opt_cls = type(self.mpc.solver.optimizers[0]).__name__
                self.logger.info(f"MPC optimizer: {opt_cls}")
            except Exception:
                pass

        # If diffusion MPPI, connect a prior sampler
        if isinstance(self.mpc, DiffusionMpcSolver) and self._diff_prior is not None:
            try:
                self.mpc.set_prior_sampler(self._diff_prior.sample_prior_batched)
            except Exception:
                self.logger.warning("Failed to attach diffusion prior sampler; using zero prior")

        # Prepare USD helper and initialize static world from the current stage
        self.usd_helper: UsdHelper = UsdHelper()
        self.usd_helper.load_stage(env.scene.stage)
        self._initialize_static_world()

        # Planning state
        self._target_pose_env: torch.Tensor | None = None  # 4x4 target pose on env.device
        self._target_pose_cu: Pose | None = None  # cuRobo Pose on cuda/cpu
        self._target_pos_env: torch.Tensor | None = None  # (3,)
        self._target_quat_env: torch.Tensor | None = None  # (4,) wxyz
        self._goal_buffer: Goal | None = None
        self._max_reactive_steps: int = 200
        self._step_count: int = 0
        from typing import Any as _Any

        self._last_cmd_state: _Any | None = None
        # Calibration between env ee frame and cuRobo ee link (world frame transform)
        self._envEE_to_solverEE: torch.Tensor | None = None  # 4x4
        # For debugging/visualization: goal in env ee frame, world coordinates (4x4)
        self._goal_env_world_pose: torch.Tensor | None = None

        # Cached object mappings for world sync
        self._cached_object_mappings: dict[str, str] | None = None
        self._expected_objects: set[str] | None = None
        # Track currently attached objects (by Isaac Lab short name)
        self._attached_objects: set[str] = set()

        # Debug draw state for visualizing MPPI rollouts
        self._draw_rollouts_enabled: bool = False
        self._debug_draw_iface = None
        self._draw_log_counter: int = 0

        # (moved diffusion prior initialization above)

    # =============================
    # Device utilities
    # =============================
    def _to_curobo_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.tensor_args.device, dtype=self.tensor_args.dtype)

    def _to_env_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.env.device, dtype=tensor.dtype)

    # =============================
    # World initialization and sync
    # =============================
    def _initialize_static_world(self) -> None:
        env_prim_path = f"/World/envs/env_{self.env_id}"
        robot_prim_path = self.config.robot_prim_path or f"{env_prim_path}/Robot"

        ignore_list = self.config.world_ignore_substrings or [
            f"{env_prim_path}/Robot",
            f"{env_prim_path}/target",
            "/World/defaultGroundPlane",
            "/curobo",
        ]

        world_cfg = self.usd_helper.get_obstacles_from_stage(
            only_paths=[env_prim_path],
            reference_prim_path=robot_prim_path,
            ignore_substring=ignore_list,
        )
        self._static_world_config = world_cfg.get_collision_check_world()
        # Initialize solver world
        # (Note: this loads the collision model the first time; we will incrementally update poses later.)
        _ = self.mpc.update_world(self._static_world_config)

    # =============================
    # Diffusion prior integration
    # =============================
    def _init_diffusion_prior(self) -> None:
        try:
            import robomimic.utils.file_utils as _RM_FileUtils
            import robomimic.utils.tensor_utils as _RM_TensorUtils
        except Exception as e:
            self.logger.warning(f"Robomimic not available for diffusion prior: {e}")
            return

        device = torch.device(self.tensor_args.device)
        ckpt_path = getattr(self.config, "diffusion_ckpt_path", None)
        if ckpt_path is None or not os.path.isfile(str(ckpt_path)):
            # Auto-discover a checkpoint from logs/robomimic
            base = os.path.join(os.getcwd(), "logs", "robomimic")
            best_path = None
            best_epoch = -1
            for root, _dirs, files in os.walk(base):
                for f in files:
                    if f.startswith("model_epoch_") and f.endswith(".pth"):
                        m = re.search(r"model_epoch_(\\d+)\\.pth", f)
                        if m:
                            ep = int(m.group(1))
                            if ep > best_epoch:
                                best_epoch = ep
                                best_path = os.path.join(root, f)
            ckpt_path = best_path
        if ckpt_path is None:
            self.logger.warning("No diffusion checkpoint found; using zero prior")
            return

        policy, _ = _RM_FileUtils.policy_from_checkpoint(ckpt_path=str(ckpt_path), device=device)

        class _DiffusionPolicyPrior:
            def __init__(self, policy, device, logger, mppi_h, d_action):
                self.policy = policy
                self.device = device
                self.logger = logger
                self.mppi_h = int(mppi_h)
                self.d_action = int(d_action)
                # horizons
                try:
                    self.To = int(policy.policy.global_config.algo.horizon.observation_horizon)
                    self.Ta = int(policy.policy.global_config.algo.horizon.action_horizon)
                    self.Tp = int(policy.policy.global_config.algo.horizon.prediction_horizon)
                except Exception:
                    self.To = 2
                    self.Ta = self.mppi_h
                    self.Tp = max(self.Ta, self.mppi_h)
                # obs shapes and keys
                try:
                    self.obs_shapes = dict(policy.policy.obs_shapes)
                    self.obs_keys = list(self.obs_shapes.keys())
                except Exception:
                    self.obs_shapes = {"eef_pos": (3,), "eef_quat": (4,), "gripper_pos": (1,), "object": (4,)}
                    self.obs_keys = list(self.obs_shapes.keys())
                self.buffers = {k: deque(maxlen=self.To) for k in self.obs_keys}

            def prime(self, obs_policy_dict):
                for k in self.buffers:
                    if k in obs_policy_dict:
                        v = torch.as_tensor(obs_policy_dict[k]).detach().to(self.device)
                    else:
                        shape = self.obs_shapes.get(k, None)
                        if shape is None:
                            continue
                        v = torch.zeros(shape, device=self.device, dtype=torch.float32)
                    self.buffers[k].clear()
                    for _ in range(self.To):
                        self.buffers[k].append(v)

            def update_from_env_obs(self, obs_policy_dict):
                for k in self.buffers:
                    if k in obs_policy_dict:
                        v = torch.as_tensor(obs_policy_dict[k]).detach().to(self.device)
                    else:
                        # If missing, reuse last value if available, otherwise zeros
                        if len(self.buffers[k]) > 0:
                            v = self.buffers[k][-1]
                        else:
                            shape = self.obs_shapes.get(k, None)
                            if shape is None:
                                continue
                            v = torch.zeros(shape, device=self.device, dtype=torch.float32)
                    self.buffers[k].append(v)

            def _build_obs_bt(self):
                obs_bt = {}
                for k, dq in self.buffers.items():
                    if len(dq) == 0:
                        # backfill zeros if empty
                        shape = self.obs_shapes.get(k, None)
                        if shape is None:
                            continue
                        zeros = torch.zeros(shape, device=self.device, dtype=torch.float32)
                        for _ in range(self.To):
                            dq.append(zeros)
                    frames = [torch.as_tensor(x).detach().to(self.device).squeeze(0) for x in list(dq)]
                    stacked = torch.stack(frames, dim=0)
                    obs_bt[k] = stacked.unsqueeze(0)
                return obs_bt

            def _sample_sequence_once(self):
                obs_bt = self._build_obs_bt()
                m = self.policy.policy
                nets = m.ema.averaged_model if m.ema is not None else m.nets
                inputs = {"obs": self.policy._prepare_observation(obs_bt, batched_ob=True), "goal": None}
                for k in m.obs_shapes:
                    if inputs["obs"][k].ndim - 1 == len(m.obs_shapes[k]):
                        inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
                obs_features = _RM_TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
                obs_cond = obs_features.flatten(start_dim=1)
                naction = torch.randn((1, self.Tp, m.ac_dim), device=self.device)
                # Prefer DDIM with 10 steps for speed
                try:
                    m.algo_config.ddpm.enabled = False
                    m.algo_config.ddim.enabled = True
                    m.algo_config.ddim.num_inference_timesteps = 30
                except Exception:
                    pass
                m.noise_scheduler.set_timesteps(10)
                for k in m.noise_scheduler.timesteps:
                    noise_pred = nets["policy"]["noise_pred_net"](sample=naction, timestep=k, global_cond=obs_cond)
                    naction = m.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
                start = max(0, self.To - 1)
                end = start + self.Ta
                action = naction[:, start:end]
                return action[0]  # [Ta, ac_dim]

            def sample_prior_batched(self, batch_size: int) -> torch.Tensor:
                try:
                    seq = self._sample_sequence_once()  # [Ta, ac_dim]
                except Exception as e:
                    self.logger.warning(f"Diffusion prior inference failed: {e}")
                    seq = torch.zeros((self.Ta, self.d_action), device=self.device, dtype=torch.float32)
                # Match d_action
                ac_dim = seq.shape[-1]
                if ac_dim > self.d_action:
                    seq = seq[..., : self.d_action]
                elif ac_dim < self.d_action:
                    pad = torch.zeros((seq.shape[0], self.d_action - ac_dim), device=self.device, dtype=seq.dtype)
                    seq = torch.cat([seq, pad], dim=-1)
                # Match horizon to MPPI
                if seq.shape[0] > self.mppi_h:
                    seq = seq[: self.mppi_h, :]
                elif seq.shape[0] < self.mppi_h:
                    pad_t = torch.zeros((self.mppi_h - seq.shape[0], seq.shape[1]), device=self.device, dtype=seq.dtype)
                    seq = torch.cat([seq, pad_t], dim=0)
                seq = seq.unsqueeze(0).expand(int(batch_size), -1, -1).contiguous()
                return seq.to(dtype=self.policy.policy.tensor_args.dtype if hasattr(self.policy.policy, "tensor_args") else torch.float32)

        self._diff_prior = _DiffusionPolicyPrior(policy, device, self.logger, self.mpc.rollout_fn.action_horizon, self.mpc.rollout_fn.d_action)

    def update_world(self) -> None:
        # Establish validation baseline on first call, validate on subsequent calls
        if self._expected_objects is None:
            self._expected_objects = set(self._get_world_object_names())
        else:
            current_objects = set[str](self._get_world_object_names())
            if current_objects != self._expected_objects:
                self._cached_object_mappings = None
                added = current_objects - self._expected_objects
                removed = self._expected_objects - current_objects
                raise RuntimeError(
                    f"World objects changed at runtime! Added: {added}, Removed: {removed}. Reinitialize MPC world."
                )

        self._sync_object_poses_with_isaaclab()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _get_world_object_names(self) -> list[str]:
        try:
            if self.mpc.world_coll_checker is None:  # type: ignore[truthy-bool]
                return []
            world_model = self.mpc.world_coll_checker.world_model
            if isinstance(world_model, list):
                if len(world_model) <= self.env_id:
                    return []
                world_model = world_model[self.env_id]

            names: list[str] = []
            for primitive_type in ["mesh", "cuboid", "sphere", "capsule", "cylinder", "voxel", "blox"]:
                if hasattr(world_model, primitive_type) and getattr(world_model, primitive_type):
                    primitive_list = getattr(world_model, primitive_type)
                    for primitive in primitive_list:
                        if primitive.name:
                            names.append(str(primitive.name))
            return names
        except Exception:
            return []

    def _get_object_mappings(self) -> dict[str, str]:
        if self._cached_object_mappings is None:
            if self.mpc.world_coll_checker is None:  # type: ignore[truthy-bool]
                return {}
            world_model = self.mpc.world_coll_checker.world_model
            rigid_objects = self.env.scene.rigid_objects
            mappings: dict[str, str] = {}
            env_prefix = f"/World/envs/env_{self.env_id}/"
            world_object_paths: list[str] = []
            for primitive_type in ["mesh", "cuboid", "sphere", "capsule", "cylinder", "voxel", "blox"]:
                primitive_list = getattr(world_model, primitive_type)
                for primitive in primitive_list:
                    if primitive.name and env_prefix in str(primitive.name):
                        world_object_paths.append(str(primitive.name))
            for object_name in rigid_objects.keys():
                for path in world_object_paths:
                    if object_name.lower().replace("_", "") in path.lower().replace("_", ""):
                        mappings[object_name] = path
                        break
            self._cached_object_mappings = mappings
        return self._cached_object_mappings

    def _sync_object_poses_with_isaaclab(self) -> None:
        object_mappings = self._get_object_mappings()
        if self.mpc.world_coll_checker is None:  # type: ignore[truthy-bool]
            return
        world_model = self.mpc.world_coll_checker.world_model
        rigid_objects = self.env.scene.rigid_objects

        for object_name, object_path in object_mappings.items():
            if object_name not in rigid_objects:
                continue
            # Skip static objects if configured
            static_objects = getattr(self.config, "static_objects", [])
            if any(static_name in object_name.lower() for static_name in static_objects):
                continue

            obj = rigid_objects[object_name]
            env_origin = self.env.scene.env_origins[self.env_id]
            current_pos_raw = obj.data.root_pos_w[self.env_id] - env_origin
            current_quat_raw = obj.data.root_quat_w[self.env_id]

            current_pos = self._to_curobo_device(current_pos_raw)
            current_quat = self._to_curobo_device(current_quat_raw)

            pose_list = [
                float(current_pos[0].item()),
                float(current_pos[1].item()),
                float(current_pos[2].item()),
                float(current_quat[0].item()),
                float(current_quat[1].item()),
                float(current_quat[2].item()),
                float(current_quat[3].item()),
            ]

            self._update_object_in_world_model(world_model, object_path, pose_list)

            # Also update in collision checker (keeps static model intact)
        curobo_pose = self._make_pose(position=current_pos, quaternion=current_quat)
        if self.mpc.world_coll_checker is not None:  # type: ignore[truthy-bool]
            self.mpc.world_coll_checker.update_obstacle_pose(object_path, curobo_pose, update_cpu_reference=True)  # type: ignore

    def _update_object_in_world_model(self, world_model, object_path: str, pose_list: list[float]) -> bool:
        if isinstance(world_model, list):
            if len(world_model) > self.env_id:
                world_model = world_model[self.env_id]
            else:
                return False
        for primitive_type in ["mesh", "cuboid", "sphere", "capsule", "cylinder", "voxel", "blox"]:
            primitive_list = getattr(world_model, primitive_type)
            for primitive in primitive_list:
                if primitive.name:
                    primitive_name = str(primitive.name)
                    if object_path == primitive_name or object_path in primitive_name or primitive_name in object_path:
                        primitive.pose = pose_list
                        return True
        return False

    # =============================
    # MPC planning interface
    # =============================
    def update_world_and_plan_motion(
        self,
        target_pose: torch.Tensor,
        expected_attached_object: str | None = None,
        env_id: int = 0,
        step_dt: float | None = None,
        **_: Any,
    ) -> bool:
        # One-time debug: log joint name alignment env vs solver
        try:
            env_joint_names = list(self.robot.data.joint_names)
            solver_joint_names = list(self.mpc.rollout_fn.joint_names)
            self.logger.info(f"Env joint names ({len(env_joint_names)}): {env_joint_names}")
            self.logger.info(f"Solver joint names ({len(solver_joint_names)}): {solver_joint_names}")
            missing_in_solver = [n for n in env_joint_names if n not in solver_joint_names]
            missing_in_env = [n for n in solver_joint_names if n not in env_joint_names]
            self.logger.info(f"Missing in solver: {missing_in_solver}")
            self.logger.info(f"Missing in env: {missing_in_env}")
        except Exception:
            pass
        # Sync world first
        self.update_world()

        # Handle attachment state: attach object geometry to robot when grasped, re-enable otherwise
        try:
            gripper_pos = self.robot.data.joint_pos[env_id, -2:]

            def _check_object_grasped() -> bool:
                try:
                    open_val = float(self.config.grasp_gripper_open_val)
                except Exception:
                    open_val = 0.02
                return bool(gripper_pos[0].item() < open_val)

            # Re-enable any previously attached objects if none expected now
            if expected_attached_object is None and self._attached_objects:
                # Detach spheres from robot and re-enable obstacles in MPC world
                self.mpc.detach_object_from_robot(link_name=self.config.attached_object_link_name)
                # Re-enable hand link collisions (if previously disabled)
                self._toggle_hand_link_collisions(enable=True)
                self._detach_objects()
            elif expected_attached_object is not None:
                # If grasped and not already attached, attach spheres to robot and disable obstacle in world model
                if _check_object_grasped() and expected_attached_object not in self._attached_objects:
                    # Build current joint state for cuRobo
                    cu_js_for_attach = self._get_current_joint_state_for_curobo()
                    # Attach object geometry to robot link in MPC (and disable obstacle in world)
                    attached_ok = self.mpc.attach_objects_to_robot(
                        joint_state=cu_js_for_attach,
                        object_names=[expected_attached_object],
                        surface_sphere_radius=getattr(self.config, "surface_sphere_radius", 0.001),
                        link_name=getattr(self.config, "attached_object_link_name", "attached_object"),
                        world_objects_pose_offset=None,
                        remove_obstacles_from_world_config=False,
                    )
                    if attached_ok:
                        # track and log
                        self._attached_objects.add(expected_attached_object)
                        self.logger.info(
                            f"MPC: attached '{expected_attached_object}' with spheres on link "
                            f"{getattr(self.config, 'attached_object_link_name', 'attached_object')}"
                        )
                        # Allow in-contact motion: disable hand link collisions during grasp
                        self._toggle_hand_link_collisions(enable=False)
                    else:
                        self.logger.warning(
                            f"MPC: failed to attach '{expected_attached_object}' (no spheres/obstacle missing)"
                        )
                    # Also disable obstacle in MPC world mapping for safety
                    self._attach_object(expected_attached_object, env_id)
                # If not grasped but currently marked attached, re-enable obstacle
                elif not _check_object_grasped() and expected_attached_object in self._attached_objects:
                    # Detach from robot link
                    self.mpc.detach_object_from_robot(link_name=self.config.attached_object_link_name)
                    # Re-enable hand link collisions
                    self._toggle_hand_link_collisions(enable=True)
                    # Re-enable in MPC world
                    self._detach_objects(names={expected_attached_object})
        except Exception:
            # Attachment handling is best-effort; continue even if it fails
            import traceback
            traceback.print_exc()
            pass

        # Ensure target_pose is a 4x4 matrix on the env device first
        if not isinstance(target_pose, torch.Tensor):
            target_pose = torch.as_tensor(target_pose, device=self.env.device, dtype=torch.float32)
        else:
            target_pose = target_pose.to(device=self.env.device, dtype=torch.float32)

        # Handle batched pose (1, 4, 4) by extracting first element
        if target_pose.dim() == 3 and target_pose.shape[0] == 1:
            target_pose = target_pose[0]

        # Validate shape
        if target_pose.dim() != 2 or target_pose.shape != (4, 4):
            raise ValueError(f"Expected 4x4 pose matrix, got shape {target_pose.shape}")

        # Convert target pose to cuRobo device (matching trajectory planner approach)
        target_pose_cuda = self._to_curobo_device(target_pose)

        # Extract position and rotation from 4x4 matrix
        tgt_pos_cuda, tgt_rot_cuda = PoseUtils.unmake_pose(target_pose_cuda)
        tgt_quat_cuda = PoseUtils.quat_from_matrix(tgt_rot_cuda)

        # Log for debugging
        self.logger.info(f"Goal position (cuRobo device): {tgt_pos_cuda}")

        # Store goal for visualization (already on env device from above)
        self._target_pose_env = target_pose
        self._goal_env_world_pose = target_pose

        # Cache position and quaternion for error checking (convert to env device)
        self._target_pos_env = tgt_pos_cuda.to(device=self.env.device)
        self._target_quat_env = tgt_quat_cuda.to(device=self.env.device)

        # Create cuRobo Pose directly (matching trajectory planner)
        self._target_pose_cu = self._make_pose(
            position=tgt_pos_cuda,
            quaternion=tgt_quat_cuda,
        )

        # Build current joint state on cuRobo device
        cu_js = self._get_current_joint_state_for_curobo()

        # Create MPC goal and goal buffer
        goal = Goal(current_state=cu_js, goal_state=cu_js, goal_pose=self._target_pose_cu)
        self._goal_buffer = self.mpc.setup_solve_single(goal, num_seeds=1)
        self.mpc.update_goal(self._goal_buffer)

        self._step_count = 0
        # step_dt currently informs internal MPC horizon. Env cadence remains external.
        return True

    # =============================
    # Attachment helpers
    # =============================
    def _attach_object(self, object_name: str, env_id: int) -> None:
        mappings = self._get_object_mappings()
        object_path = mappings.get(object_name)
        if object_path and self.mpc.world_coll_checker is not None:
            # try:
            # Disable obstacle in collision checker while attached
            self.mpc.world_coll_checker.enable_obstacle(object_path, enable=False)  # type: ignore
            if object_name not in self._attached_objects:
                self._attached_objects.add(object_name)
                # Visibility: info-level so it shows up in console
                self.logger.info(f"Attached object '{object_name}' (disabled in MPC world): {object_path}")
                print(f"MPC: attached '{object_name}' -> disabled obstacle {object_path}")
            # except Exception:
            #     pass

    def _detach_objects(self, names: set[str] | None = None) -> None:
        if not self._attached_objects:
            return
        mappings = self._get_object_mappings()
        to_detach = self._attached_objects if names is None else (self._attached_objects & names)
        for name in list(to_detach):
            object_path = mappings.get(name)
            if object_path and self.mpc.world_coll_checker is not None:
                # try:
                self.mpc.world_coll_checker.enable_obstacle(object_path, enable=True)  # type: ignore
                # except Exception:
                #     pass
            if name in self._attached_objects:
                self._attached_objects.discard(name)
                self.logger.info(f"Detached object '{name}' (re-enabled in MPC world): {object_path}")
                print(f"MPC: detached '{name}' -> re-enabled obstacle {object_path}")

    # -------------------------------------------------------------------------------------
    # DEBUG / VISUALIZATION HELPERS
    # -------------------------------------------------------------------------------------
    def get_goal_env_world_pose(self) -> torch.Tensor | None:
        """Return the goal transform in the environment EE frame, world coordinates (4x4)."""
        return self._goal_env_world_pose

    def enable_visual_rollouts(self, enable: bool = True) -> None:
        """Enable/disable drawing MPPI rollouts using the Isaac Sim debug-draw interface."""
        self._draw_rollouts_enabled = bool(enable)
        self.logger.info(f"Visual rollouts {'ENABLED' if self._draw_rollouts_enabled else 'DISABLED'}")

    def _acquire_debug_draw_interface(self):
        if self._debug_draw_iface is not None:
            return self._debug_draw_iface
        import isaacsim.util.debug_draw._debug_draw as _debug_draw_mod  # type: ignore

        self._debug_draw_iface = _debug_draw_mod.acquire_debug_draw_interface()
        self.logger.info("Acquired debug-draw interface (isaacsim.util.debug_draw._debug_draw)")
        return self._debug_draw_iface

    def _draw_mpc_rollouts(self) -> None:
        if not self._draw_rollouts_enabled:
            return
        draw = self._acquire_debug_draw_interface()
        if draw is None:
            self.logger.debug("Debug draw interface not available")
            return
        rollouts_tensor = self.mpc.get_visual_rollouts() if hasattr(self.mpc, "get_visual_rollouts") else None
        if not isinstance(rollouts_tensor, torch.Tensor):
            self.logger.debug("No rollouts tensor available")
            return
        cpu_rollouts = rollouts_tensor.detach().to("cpu").numpy()
        if cpu_rollouts.ndim != 3 or cpu_rollouts.shape[-1] < 3:
            return
        b, h, _ = cpu_rollouts.shape
        points: list[tuple[float, float, float]] = []
        colors: list[tuple[float, float, float, float]] = []
        sizes: list[float] = []
        denom = float(max(b, 1))
        for i in range(b):
            t = (i + 1.0) / denom
            color = (1.0 - t, 0.3 * t, 0.0, 0.5)
            for j in range(h):
                x, y, z = cpu_rollouts[i, j, 0], cpu_rollouts[i, j, 1], cpu_rollouts[i, j, 2]
                points.append((float(x), float(y), float(z)))
                colors.append(color)
                sizes.append(12.0)
        try:
            draw.clear_points()
        except Exception:
            pass
        if points:
            draw.draw_points(points, colors, sizes)
            if self._draw_log_counter % 10 == 0:  # Log every 10th draw to avoid spam
                self.logger.debug(f"Drew {len(points)} rollout points from {b} trajectories")
            self._draw_log_counter += 1

    def _get_current_ee_pose_matrix(self) -> torch.Tensor:
        cu_js = self._get_current_joint_state_for_curobo()
        kin = self.mpc.compute_kinematics(cu_js)
        pos = getattr(kin, "ee_pos_seq", None)
        quat = getattr(kin, "ee_quat_seq", None)
        # Retrieve env origin to convert solver env-local -> world coordinates
        try:
            env_origins = getattr(self.env.scene, "env_origins", None)
            if isinstance(env_origins, torch.Tensor):
                env_origin = env_origins[self.env_id, :3]
            else:
                env_origin = torch.tensor([0.0, 0.0, 0.0], device=self.env.device, dtype=torch.float32)
        except Exception:
            env_origin = torch.tensor([0.0, 0.0, 0.0], device=self.env.device, dtype=torch.float32)
        if pos is not None and quat is not None:
            pos_env = self._to_env_device(pos) if isinstance(pos, torch.Tensor) else torch.as_tensor(pos)
            quat_env = self._to_env_device(quat) if isinstance(quat, torch.Tensor) else torch.as_tensor(quat)
            # ensure batch
            if pos_env.dim() == 1:
                pos_env_b = pos_env.unsqueeze(0)
            else:
                pos_env_b = pos_env
            # convert to world coordinates
            pos_world_b = pos_env_b + env_origin.unsqueeze(0)
            rot_b = PoseUtils.matrix_from_quat(quat_env.unsqueeze(0) if quat_env.dim() == 1 else quat_env)
            return PoseUtils.make_pose(pos_world_b, rot_b)[0]
        ee = kin.ee_pose
        if ee is None:
            # Fallback to identity
            eye = torch.eye(4, device=self.env.device, dtype=torch.float32)
            return eye
        pos_e = ee.position
        rot_e = ee.get_rotation()
        if pos_e is None or rot_e is None:  # type: ignore[truthy-bool]
            eye = torch.eye(4, device=self.env.device, dtype=torch.float32)
            return eye
        pos_env = self._to_env_device(pos_e)
        rot_env = self._to_env_device(rot_e)
        pos_env_b = pos_env.unsqueeze(0) if pos_env.dim() == 1 else pos_env
        rot_env_b = rot_env.unsqueeze(0) if rot_env.dim() == 2 else rot_env
        # Convert from env-local to world coordinates
        pos_world_b = pos_env_b + env_origin.unsqueeze(0)
        return PoseUtils.make_pose(pos_world_b, rot_env_b)[0]

    def _normalize_target_pose_to_matrix(self, pose_in: torch.Tensor) -> torch.Tensor:
        # Accept 4x4, (3,) position only, (4,) quaternion only, or (7,) [pos(3), quat(4)]
        if isinstance(pose_in, torch.Tensor):
            t = pose_in
        else:
            t = torch.as_tensor(pose_in)
        t = t.to(device=self.env.device, dtype=torch.float32)
        if t.dim() == 2 and t.shape[-2:] == (4, 4):
            return t
        current = self._get_current_ee_pose_matrix()
        cur_pos, cur_rot = PoseUtils.unmake_pose(current)
        if t.dim() == 1 and t.shape[0] == 7:
            pos = t[:3]
            quat = t[3:7]
            rot_b = PoseUtils.matrix_from_quat(quat.unsqueeze(0))
            pos_b = pos.unsqueeze(0)
            return PoseUtils.make_pose(pos_b, rot_b)[0]
        if t.dim() == 1 and t.shape[0] == 4:
            quat = t
            rot_b = PoseUtils.matrix_from_quat(quat.unsqueeze(0))
            pos_b = cur_pos.unsqueeze(0) if cur_pos.dim() == 1 else cur_pos
            return PoseUtils.make_pose(pos_b, rot_b)[0]
        if t.dim() == 1 and t.shape[0] == 3:
            pos = t
            pos_b = pos.unsqueeze(0)
            rot_b = cur_rot.unsqueeze(0) if cur_rot.dim() == 2 else cur_rot
            return PoseUtils.make_pose(pos_b, rot_b)[0]
        # Unknown shape: return current pose as safe fallback
        return current

    def _get_current_joint_state_for_curobo(self) -> JointState:
        joint_pos_raw: torch.Tensor = self.robot.data.joint_pos[self.env_id, :].unsqueeze(0)
        joint_vel_raw: torch.Tensor = torch.zeros_like(joint_pos_raw)
        joint_acc_raw: torch.Tensor = torch.zeros_like(joint_pos_raw)
        joint_jerk_raw: torch.Tensor = torch.zeros_like(joint_pos_raw)
        joint_pos = self._to_curobo_device(joint_pos_raw)
        joint_vel = self._to_curobo_device(joint_vel_raw)
        joint_acc = self._to_curobo_device(joint_acc_raw)
        joint_jerk = self._to_curobo_device(joint_jerk_raw)
        cu_js = JointState(
            position=joint_pos,
            velocity=joint_vel,
            acceleration=joint_acc,
            jerk=joint_jerk,
            joint_names=self.robot.data.joint_names,
            tensor_args=self.tensor_args,
        )
        return cu_js.get_ordered_joint_state(self.mpc.rollout_fn.joint_names)

    def _current_pose_error(self) -> tuple[float, float]:
        if self._target_pose_env is None:
            return 1e9, 1e9
        # Compute EE pose for current joint state
        cu_js = self._get_current_joint_state_for_curobo()
        kin_state = self.mpc.compute_kinematics(cu_js)
        # Prefer explicit pos/quat sequences if available
        pos_seq = getattr(kin_state, "ee_pos_seq", None)
        quat_seq = getattr(kin_state, "ee_quat_seq", None)
        if pos_seq is not None and quat_seq is not None:
            cur_pos = self._to_env_device(pos_seq) if isinstance(pos_seq, torch.Tensor) else torch.as_tensor(pos_seq)
            cur_quat = (
                self._to_env_device(quat_seq) if isinstance(quat_seq, torch.Tensor) else torch.as_tensor(quat_seq)
            )
        else:
            ee_pose = kin_state.ee_pose
            if ee_pose is None:
                return 1e9, 1e9
            pos_e = ee_pose.position
            rot_e = ee_pose.get_rotation()
            if pos_e is None or rot_e is None:  # type: ignore[truthy-bool]
                return 1e9, 1e9
            cur_pos = self._to_env_device(pos_e)
            cur_rot = self._to_env_device(rot_e)
            cur_quat = PoseUtils.quat_from_matrix(cur_rot)
        # Distances using cached target pos/quat
        tgt_pos = self._target_pos_env
        tgt_quat = self._target_quat_env
        if tgt_pos is None or tgt_quat is None:
            return 1e9, 1e9
        pos_err = torch.linalg.vector_norm(tgt_pos - cur_pos).item()
        # Normalize
        tgt_quat = tgt_quat / torch.linalg.vector_norm(tgt_quat)
        cur_quat = cur_quat / torch.linalg.vector_norm(cur_quat)
        # angle = 2*acos(|dot|)
        dot = torch.clamp(torch.sum(tgt_quat * cur_quat), -1.0, 1.0)
        rot_err = float(2.0 * torch.arccos(torch.abs(dot)).item())
        return pos_err, rot_err

    def has_next_waypoint(self) -> bool:
        if self._target_pose_env is None:
            return False
        pos_thr = float(self.config.position_threshold)
        rot_thr = float(self.config.rotation_threshold)
        pos_err, rot_err = self._current_pose_error()
        if pos_err <= pos_thr and rot_err <= rot_thr:
            return False
        if self._step_count >= self._max_reactive_steps:
            self.logger.info("Reached max reactive steps; stopping.")
            return False
        return True

    def get_next_waypoint_ee_pose(self) -> tuple[torch.Tensor, Any]:
        if not self.has_next_waypoint():
            raise IndexError("No more MPC waypoints; target reached or limit hit.")

        # World sync and one MPC step
        self.update_world()

        # Update diffusion prior observation buffers
        try:
            if self._diff_prior is not None:
                env_obs = self.env.get_observations()
                if isinstance(env_obs, dict) and "policy" in env_obs:
                    obs_policy = env_obs["policy"]
                else:
                    obs_policy = env_obs if isinstance(env_obs, dict) else {}
                if len(self._diff_prior.buffers) and all(len(v) == 0 for v in self._diff_prior.buffers.values()):
                    self._diff_prior.prime(obs_policy)
                else:
                    self._diff_prior.update_from_env_obs(obs_policy)
        except Exception:
            pass

        current_js = self._get_current_joint_state_for_curobo()
        self.logger.debug("Running MPC step ...")
        result = self.mpc.step(current_js, max_attempts=2)
        # Diffusion prior debug: print once per step
        try:
            if isinstance(self.mpc, DiffusionMpcSolver):
                used, shape, norm = self.mpc.get_last_prior_stats()
                self.logger.debug(
                    f"Diff prior used={used} shape={shape} norm={norm:.4f}"
                )
        except Exception:
            pass
        self.logger.debug("MPC step complete; attempting rollout draw")
        # Draw MPPI rollouts if enabled
        self._draw_mpc_rollouts()

        # Convert resulting action to EE pose for the waypoint
        from typing import cast

        try:
            cmd_state_full = cast(JointState, result.js_action)
        except AttributeError:
            # Fallback to action if js_action is unavailable
            cmd_state_full = cast(JointState, result.action)  # type: ignore

        # Output filtering: EMA + rate limit on joint positions
        try:
            alpha = float(self.config.ema_alpha)
        except Exception:
            alpha = 0.8
        try:
            rate_limit = float(self.config.rate_limit)
        except Exception:
            rate_limit = 0.015

        prev_pos = None
        last_pos_attr = getattr(self._last_cmd_state, "position", None) if self._last_cmd_state is not None else None
        if isinstance(last_pos_attr, torch.Tensor):
            prev_pos = last_pos_attr
        curr_pos_attr = getattr(cmd_state_full, "position", None)
        if isinstance(prev_pos, torch.Tensor) and isinstance(curr_pos_attr, torch.Tensor):
            # match shapes (B, D) or (D)
            if curr_pos_attr.dim() == 2 and curr_pos_attr.shape[0] == 1:
                cpos = curr_pos_attr[0]
            else:
                cpos = curr_pos_attr
            if prev_pos.dim() == 2 and prev_pos.shape[0] == 1:
                ppos = prev_pos[0]
            else:
                ppos = prev_pos
            if ppos.shape == cpos.shape:
                # EMA smoothing
                smoothed = alpha * ppos + (1.0 - alpha) * cpos
                # Rate limit
                delta = torch.clamp(smoothed - ppos, min=-rate_limit, max=rate_limit)
                new_pos = ppos + delta
                if curr_pos_attr.dim() == 2 and curr_pos_attr.shape[0] == 1:
                    cmd_state_full.position = torch.stack([new_pos], dim=0)
                else:
                    cmd_state_full.position = new_pos

        # Save last commanded joint state for consumers that want joint control
        self._last_cmd_state = cmd_state_full

        kin_state = self.mpc.compute_kinematics(cmd_state_full)
        # Build solver ee transform in WORLD coordinates
        pos_val = getattr(kin_state, "ee_pos_seq", None)
        rot_val = getattr(kin_state, "ee_quat_seq", None)
        if pos_val is not None and rot_val is not None:
            pos_env = self._to_env_device(pos_val) if isinstance(pos_val, torch.Tensor) else torch.as_tensor(pos_val)
            rot_env = self._to_env_device(rot_val) if isinstance(rot_val, torch.Tensor) else torch.as_tensor(rot_val)
            # env-local -> world
            try:
                env_origins = getattr(self.env.scene, "env_origins", None)
                if isinstance(env_origins, torch.Tensor):
                    env_origin = env_origins[self.env_id, :3]
                else:
                    env_origin = torch.tensor([0.0, 0.0, 0.0], device=self.env.device, dtype=torch.float32)
            except Exception:
                env_origin = torch.tensor([0.0, 0.0, 0.0], device=self.env.device, dtype=torch.float32)
            pos_world = pos_env + env_origin
            T_solver_world = PoseUtils.make_pose(pos_world, PoseUtils.matrix_from_quat(rot_env))[0]
        else:
            ee_pose = kin_state.ee_pose
            if ee_pose is None:
                cu_js = self._get_current_joint_state_for_curobo()
                kin_state2 = self.mpc.compute_kinematics(cu_js)
                ee_pose = kin_state2.ee_pose
            pos_e = ee_pose.position
            rot_e = ee_pose.get_rotation()
            if pos_e is None or rot_e is None:  # type: ignore[truthy-bool]
                T_solver_world = torch.eye(4, device=self.env.device, dtype=torch.float32)
            else:
                pos_env = self._to_env_device(pos_e)
                rot_env = self._to_env_device(rot_e)
                try:
                    env_origins = getattr(self.env.scene, "env_origins", None)
                    if isinstance(env_origins, torch.Tensor):
                        env_origin = env_origins[self.env_id, :3]
                    else:
                        env_origin = torch.tensor([0.0, 0.0, 0.0], device=self.env.device, dtype=torch.float32)
                except Exception:
                    env_origin = torch.tensor([0.0, 0.0, 0.0], device=self.env.device, dtype=torch.float32)
                pos_world = pos_env + env_origin
                T_solver_world = PoseUtils.make_pose(pos_world, rot_env)[0]

        # Use solver transform directly without EE calibration (matches trajectory planner behavior)
        ee_tf = T_solver_world

        self._step_count += 1
        return ee_tf, self._last_cmd_state

    def get_last_joint_positions(self) -> torch.Tensor | None:
        """Return the last commanded joint positions on env device (1D tensor)."""
        if self._last_cmd_state is None or getattr(self._last_cmd_state, "position", None) is None:
            return None
        # Reorder to match env robot joint order
        env_joint_names: list[str] = list(self.robot.data.joint_names)
        cmd_joint_names: list[str] = list(self._last_cmd_state.joint_names)
        cmd_pos_tensor: torch.Tensor = (
            self._last_cmd_state.position
            if isinstance(self._last_cmd_state.position, torch.Tensor)
            else torch.as_tensor(self._last_cmd_state.position)
        )
        if cmd_pos_tensor.dim() == 2 and cmd_pos_tensor.shape[0] == 1:
            cmd_pos_tensor = cmd_pos_tensor[0]
        # Build mapping
        name_to_index = {name: i for i, name in enumerate(cmd_joint_names)}
        out = torch.zeros(len(env_joint_names), dtype=self.tensor_args.dtype, device=self.env.device)
        for i, name in enumerate(env_joint_names):
            if name in name_to_index:
                out[i] = self._to_env_device(cmd_pos_tensor[name_to_index[name]])
            else:
                # if missing (e.g., gripper), use current sim joint or configured open position
                try:
                    out[i] = self.robot.data.joint_pos[self.env_id, i]
                except Exception:
                    out[i] = torch.tensor(0.0, device=self.env.device, dtype=self.tensor_args.dtype)
        return out

    # Internal helper to create cuRobo Pose on correct device
    def _make_pose(
        self,
        position: torch.Tensor | list[float],
        quaternion: torch.Tensor | list[float],
        *,
        name: str | None = None,
        normalize_rotation: bool = False,
    ) -> Pose:
        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, dtype=self.tensor_args.dtype, device=self.tensor_args.device)
        else:
            position = self._to_curobo_device(position)
        if not isinstance(quaternion, torch.Tensor):
            quaternion = torch.tensor(quaternion, dtype=self.tensor_args.dtype, device=self.tensor_args.device)
        else:
            quaternion = self._to_curobo_device(quaternion)
        if name is None:
            return Pose(position=position, quaternion=quaternion, normalize_rotation=normalize_rotation)
        return Pose(position=position, quaternion=quaternion, name=name, normalize_rotation=normalize_rotation)

    def reset_plan(self) -> None:
        """Reset any internal planner state (compatibility with MotionPlannerBase)."""
        self._step_count = 0
        self._last_cmd_state = None

    # Optional hooks used by data generator visualization; safe no-ops here
    def _update_visualization_at_joint_positions(self, joint_positions: torch.Tensor) -> None:  # noqa: D401
        return  # Intentionally left blank; sphere visualization is handled by the trajectory planner

    # For compatibility with SkillGen conversion util; not used in reactive path
    def get_planned_poses(self) -> list[torch.Tensor]:
        return []
