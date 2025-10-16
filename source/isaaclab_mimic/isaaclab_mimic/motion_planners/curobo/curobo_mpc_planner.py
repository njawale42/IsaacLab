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

from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

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
            use_mppi=True,
        )
        self.mpc: MpcSolver = MpcSolver(mpc_config)

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

        # Debug draw state for visualizing MPPI rollouts
        self._draw_rollouts_enabled: bool = False
        self._debug_draw_iface = None
        self._draw_log_counter: int = 0

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

    def update_world(self) -> None:
        # Establish validation baseline on first call, validate on subsequent calls
        if self._expected_objects is None:
            self._expected_objects = set(self._get_world_object_names())
        else:
            current_objects = set(self._get_world_object_names())
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

        # Normalize and save target as 4x4 matrix
        self._target_pose_env = self._normalize_target_pose_to_matrix(target_pose)

        # Convert target to cuRobo pose on MPC device
        # Extra guard in case a non-matrix sneaks in
        if not (
            isinstance(self._target_pose_env, torch.Tensor)
            and self._target_pose_env.dim() == 2
            and self._target_pose_env.shape[-2:] == (4, 4)
        ):
            self._target_pose_env = self._normalize_target_pose_to_matrix(self._target_pose_env)  # type: ignore[arg-type]

        # Build world-frame target matrix for env ee frame
        if isinstance(self._target_pose_env, torch.Tensor):
            T_env_goal_world = self._target_pose_env
        else:
            T_env_goal_world = torch.as_tensor(self._target_pose_env, device=self.env.device, dtype=torch.float32)

        # Guard: accept quaternion/position vectors and convert to matrix here if encountered
        if isinstance(T_env_goal_world, torch.Tensor) and T_env_goal_world.dim() == 1:
            cur_T = self._get_current_ee_pose_matrix()
            cur_pos, cur_rot = PoseUtils.unmake_pose(cur_T)
            if T_env_goal_world.shape[0] == 4:
                rot_b = PoseUtils.matrix_from_quat(T_env_goal_world.unsqueeze(0))
                pos_b = cur_pos.unsqueeze(0) if cur_pos.dim() == 1 else cur_pos
                T_env_goal_world = PoseUtils.make_pose(pos_b, rot_b)[0]
            elif T_env_goal_world.shape[0] == 3:
                pos_b = T_env_goal_world.unsqueeze(0)
                rot_b = cur_rot.unsqueeze(0) if cur_rot.dim() == 2 else cur_rot
                T_env_goal_world = PoseUtils.make_pose(pos_b, rot_b)[0]
            else:
                self.logger.info(f"Unexpected goal shape {T_env_goal_world.shape}, using current pose")
                T_env_goal_world = cur_T

        # One-time calibration of envEE->solverEE transform in world frame
        if self._envEE_to_solverEE is None:
            try:
                # Current env ee frame world transform
                ee_frame = self.env.scene["ee_frame"]
                # Guard on array shape access; fall back to zeros if indexing fails
                try:
                    pos_w = ee_frame.data.target_pos_w[self.env_id, 0, :]
                    quat_w = ee_frame.data.target_quat_w[self.env_id, 0, :]
                except Exception:
                    pos_w = torch.zeros(3, device=self.env.device, dtype=torch.float32)
                    quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.env.device, dtype=torch.float32)
                # Build batched pose → unbatch
                pos_b = pos_w.unsqueeze(0) if pos_w.dim() == 1 else pos_w
                rot_b = PoseUtils.matrix_from_quat(quat_w.unsqueeze(0) if quat_w.dim() == 1 else quat_w)
                T_env_current_world = PoseUtils.make_pose(pos_b, rot_b)[0]
                # Current solver ee world transform
                T_solver_current_world = self._get_current_ee_pose_matrix()
                self._envEE_to_solverEE = torch.linalg.inv(T_env_current_world) @ T_solver_current_world
                if self._envEE_to_solverEE is not None:
                    self.logger.info(
                        f"Calibrated envEE->solverEE (pos,deg): {self._envEE_to_solverEE[:3,3]}, "
                        f"{torch.rad2deg(torch.tensor([0.0]))}"
                    )
            except Exception as e:
                self.logger.info(f"EE calibration failed (continuing w/o): {e}")
                self._envEE_to_solverEE = None

        # Store goal in env ee frame (world) for visualization/debug
        self._goal_env_world_pose = T_env_goal_world

        # Convert env ee goal to solver ee goal in world frame using calibration (if available)
        if self._envEE_to_solverEE is not None:
            T_solver_goal_world = T_env_goal_world @ self._envEE_to_solverEE
        else:
            T_solver_goal_world = T_env_goal_world

        tgt_pos, tgt_rot = PoseUtils.unmake_pose(T_solver_goal_world)
        tgt_quat = PoseUtils.quat_from_matrix(tgt_rot)

        # Convert target position from world frame to env-local frame expected by cuRobo (subtract env origin)
        try:
            env_origins = getattr(self.env.scene, "env_origins", None)
            if isinstance(env_origins, torch.Tensor):
                env_origin = env_origins[self.env_id, :3]
            else:
                env_origin = torch.tensor([0.0, 0.0, 0.0], device=self.env.device, dtype=torch.float32)
            env_origin = env_origin.to(device=self.env.device, dtype=torch.float32)
            # Explicit INFO logs so they are visible even when debug_planner is False
            self.logger.info(f"Env origin: {env_origin}")
            self.logger.info(f"Target world pos (pre-local): {tgt_pos}")
            tgt_pos = tgt_pos - env_origin
            self.logger.info(f"Target env-local pos: {tgt_pos}")
        except Exception as e:
            self.logger.info(f"Env origin/target logging failed: {e}")
        # Cache env-local pos/quat for later error checks without unmake_pose
        self._target_pos_env = (
            tgt_pos if isinstance(tgt_pos, torch.Tensor) else torch.as_tensor(tgt_pos, device=self.env.device)
        )
        self._target_quat_env = (
            tgt_quat if isinstance(tgt_quat, torch.Tensor) else torch.as_tensor(tgt_quat, device=self.env.device)
        )

        self._target_pose_cu = self._make_pose(
            position=self._to_curobo_device(self._target_pos_env),
            quaternion=self._to_curobo_device(self._target_quat_env),
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
        try:
            # Preferred in Kit builds
            from omni.isaac.debug_draw import _debug_draw  # type: ignore

            self._debug_draw_iface = _debug_draw.acquire_debug_draw_interface()
            self.logger.info("Acquired debug-draw interface from omni.isaac.debug_draw")
            return self._debug_draw_iface
        except Exception as e:
            self.logger.debug(f"omni.isaac.debug_draw unavailable: {e}")
        try:
            # Fallback commonly available in Isaac Sim python
            from isaacsim.util import debug_draw as _debug_draw  # type: ignore

            self._debug_draw_iface = _debug_draw.acquire_debug_draw_interface()
            self.logger.info("Acquired debug-draw interface from isaacsim.util.debug_draw")
            return self._debug_draw_iface
        except Exception as e:
            self.logger.debug(f"isaacsim.util.debug_draw unavailable: {e}")
        try:
            # Isaac Lab tutorial import path
            import isaacsim.util.debug_draw._debug_draw as _debug_draw_mod  # type: ignore

            self._debug_draw_iface = _debug_draw_mod.acquire_debug_draw_interface()
            self.logger.info("Acquired debug-draw interface from isaacsim.util.debug_draw._debug_draw")
            return self._debug_draw_iface
        except Exception as e:
            self.logger.debug(f"isaacsim.util.debug_draw._debug_draw unavailable: {e}")
        try:
            # As requested: support module name "debug_drawing" if present
            from isaacsim.util import debug_drawing as _debug_drawing  # type: ignore

            # Some builds expose the same API
            self._debug_draw_iface = _debug_drawing.acquire_debug_draw_interface()  # type: ignore[attr-defined]
            self.logger.info("Acquired debug-draw interface from isaacsim.util.debug_drawing")
            return self._debug_draw_iface
        except Exception as e:
            self.logger.debug(f"isaacsim.util.debug_drawing unavailable: {e}")
            self._debug_draw_iface = None
            return None

    def _draw_mpc_rollouts(self) -> None:
        if not self._draw_rollouts_enabled:
            # First few calls print the disabled state
            if self._draw_log_counter < 5:
                self.logger.info("Visual rollouts disabled; skipping draw")
                self._draw_log_counter += 1
            return
        try:
            draw = self._acquire_debug_draw_interface()
            if draw is None:
                if self._draw_log_counter < 10 or self._draw_log_counter % 50 == 0:
                    self.logger.info("No debug-draw interface available; cannot draw rollouts")
                    self._draw_log_counter += 1
                return
            # cuRobo exposes rollouts for visualization when store_rollouts=True
            rollouts = getattr(self.mpc, "get_visual_rollouts", None)
            if rollouts is None:
                if self._draw_log_counter < 10 or self._draw_log_counter % 50 == 0:
                    self.logger.info("mpc.get_visual_rollouts() not available; ensure store_rollouts=True")
                    self._draw_log_counter += 1
                return
            rollouts_tensor = self.mpc.get_visual_rollouts()
            if rollouts_tensor is None:
                if self._draw_log_counter < 10 or self._draw_log_counter % 50 == 0:
                    self.logger.info("No rollout samples returned; waiting for MPC to populate")
                    self._draw_log_counter += 1
                return
            if isinstance(rollouts_tensor, torch.Tensor):
                cpu_rollouts = rollouts_tensor.detach().to("cpu").numpy()
            else:
                # Unexpected type; nothing to draw
                if self._draw_log_counter < 10 or self._draw_log_counter % 50 == 0:
                    self.logger.info(f"Unexpected rollouts type: {type(rollouts_tensor)}")
                    self._draw_log_counter += 1
                return
            if cpu_rollouts.ndim != 3 or cpu_rollouts.shape[-1] < 3:
                if self._draw_log_counter < 10 or self._draw_log_counter % 50 == 0:
                    self.logger.info(f"Rollouts have unexpected shape: {cpu_rollouts.shape}")
                    self._draw_log_counter += 1
                return
            b, h, _ = cpu_rollouts.shape
            if self._draw_log_counter < 10 or self._draw_log_counter % 50 == 0:
                self.logger.info(f"Drawing rollouts: batches={b}, horizon={h}")
                self._draw_log_counter += 1
            points: list[tuple[float, float, float]] = []
            colors: list[tuple[float, float, float, float]] = []
            sizes: list[float] = []
            denom = float(max(b, 1))
            for i in range(b):
                for j in range(h):
                    x, y, z = cpu_rollouts[i, j, 0], cpu_rollouts[i, j, 1], cpu_rollouts[i, j, 2]
                    points.append((float(x), float(y), float(z)))
                    t = (i + 1.0) / denom
                    # Increase opacity and size for better visibility
                    colors.append((1.0 - t, 0.3 * t, 0.0, 0.5))
                    sizes.append(12.0)
            try:
                draw.clear_points()
            except Exception:
                # Some interfaces auto-clear; ignore
                pass
            if points:
                draw.draw_points(points, colors, sizes)
                if self._draw_log_counter < 10 or self._draw_log_counter % 50 == 0:
                    self.logger.info(f"Drew {len(points)} rollout points")
                    self._draw_log_counter += 1
        except Exception as e:
            # Keep drawing best-effort; avoid breaking planning loop
            self.logger.debug(f"Rollout drawing skipped due to error: {e}")

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
        return PoseUtils.make_pose(pos_env_b, rot_env_b)[0]

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

    def get_next_waypoint_ee_pose(self) -> torch.Tensor:
        if not self.has_next_waypoint():
            raise IndexError("No more MPC waypoints; target reached or limit hit.")

        # World sync and one MPC step
        self.update_world()

        current_js = self._get_current_joint_state_for_curobo()
        self.logger.debug("Running MPC step ...")
        result = self.mpc.step(current_js, max_attempts=2)
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

        # Convert solver ee world transform to env ee world transform using calibration
        if self._envEE_to_solverEE is not None:
            T_env_world = T_solver_world @ torch.linalg.inv(self._envEE_to_solverEE)
        else:
            T_env_world = T_solver_world

        ee_tf = T_env_world

        self._step_count += 1
        return ee_tf

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
