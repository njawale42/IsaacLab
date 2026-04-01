# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import logging
import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, List, cast

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModelState
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

import isaaclab.utils.math as PoseUtils
from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.sim.spawners.meshes import MeshSphereCfg, spawn_mesh_sphere

from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
from isaaclab_mimic.motion_planners.motion_planner_base import MotionPlannerBase


class PlannerLogger:
    """Logger class for motion planner debugging and monitoring.

    This class provides standard logging functionality while maintaining isolation from
    the main application's logging configuration. The logger supports configurable verbosity
    levels and formats messages consistently for debugging motion planning operations,
    collision checking, and object manipulation.
    """

    def __init__(self, name: str, level: int = logging.INFO):
        """Initialize the logger with specified name and level.

        Args:
            name: Logger name for identification in log messages
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self._name = name
        self._level = level
        self._logger = None

    @property
    def logger(self):
        """Get the underlying logger instance, initializing it if needed.

        Returns:
            Configured Python logger instance with stream handler and formatter
        """
        if self._logger is None:
            self._logger = logging.getLogger(self._name)
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)
                self._logger.setLevel(self._level)
        return self._logger

    def debug(self, msg, *args, **kwargs):
        """Log debug-level message for detailed internal state information.

        Args:
            msg: Message string or format string
            *args: Positional arguments for message formatting
            **kwargs: Keyword arguments passed to underlying logger
        """
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """Log info-level message for important operational events.

        Args:
            msg: Message string or format string
            *args: Positional arguments for message formatting
            **kwargs: Keyword arguments passed to underlying logger
        """
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """Log warning-level message for potentially problematic conditions.

        Args:
            msg: Message string or format string
            *args: Positional arguments for message formatting
            **kwargs: Keyword arguments passed to underlying logger
        """
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Log error-level message for serious problems and failures.

        Args:
            msg: Message string or format string
            *args: Positional arguments for message formatting
            **kwargs: Keyword arguments passed to underlying logger
        """
        self.logger.error(msg, *args, **kwargs)


@dataclass
class Attachment:
    """Stores object attachment information for robot manipulation.

    This dataclass tracks the relative pose between an attached object and its parent link,
    enabling the robot to maintain consistent object positioning during motion planning.
    """

    pose: Pose  # Relative pose from parent link to object
    parent: str  # Parent link name


class CuroboPlanner(MotionPlannerBase):
    """Motion planner for robot manipulation using cuRobo.

    This planner provides collision-aware motion planning capabilities for robotic manipulation tasks.
    It integrates with Isaac Lab environments to:

    - Update collision world from current stage state
    - Plan collision-free paths to target poses
    - Handle object attachment and detachment during manipulation
    - Execute planned motions with proper collision checking

    The planner uses cuRobo for fast motion generation and supports
    multi-phase planning for contact scenarios like grasping and placing objects.
    """

    def __init__(
        self,
        env: ManagerBasedEnv,
        robot: Articulation,
        config: CuroboPlannerCfg,
        task_name: str | None = None,
        env_id: int = 0,
        collision_checker: CollisionCheckerType = CollisionCheckerType.MESH,
        num_trajopt_seeds: int = 12,
        num_graph_seeds: int = 12,
        interpolation_dt: float = 0.05,
        shared_motion_gen: MotionGen | None = None,
    ) -> None:
        """Initialize the motion planner for a specific environment.

        Sets up the cuRobo motion generator with collision checking, configures the robot model,
        and prepares visualization components if enabled. The planner is isolated to CUDA device
        regardless of Isaac Lab's device configuration.

        When ``shared_motion_gen`` is provided, this planner reuses an existing MotionGen instance
        instead of creating its own. This enables multiple environments to share a single GPU-heavy
        planner, dramatically reducing CUDA memory usage. The shared MotionGen's collision world
        is swapped to the current env's state before each planning call.

        Args:
            env: The Isaac Lab environment instance containing the robot and scene
            robot: Robot articulation to plan motions for
            config: Configuration object containing planner parameters and settings
            task_name: Task name for auto-configuration
            env_id: Environment ID for multi-environment setups (0 to num_envs-1)
            collision_checker: Type of collision checker
            num_trajopt_seeds: Number of seeds for trajectory optimization
            num_graph_seeds: Number of seeds for graph search
            interpolation_dt: Time step for interpolating waypoints
            shared_motion_gen: Pre-created MotionGen instance to share across environments.
                When provided, skips MotionGen creation and warmup.

        Raises:
            ValueError: If ``robot_config_file`` is not provided
        """
        # Initialize base class
        super().__init__(env=env, robot=robot, env_id=env_id, debug=config.debug_planner)

        # Initialize planner logger with debug level based on config
        log_level = logging.DEBUG if config.debug_planner else logging.INFO
        self.logger = PlannerLogger(f"CuroboPlanner_{env_id}", log_level)

        # Store instance variables
        self.config: CuroboPlannerCfg = config
        self.n_repeat: int | None = self.config.n_repeat
        self.step_size: float | None = self.config.motion_step_size
        self.visualize_plan: bool = self.config.visualize_plan
        self.visualize_spheres: bool = self.config.visualize_spheres
        self._is_shared: bool = shared_motion_gen is not None

        # Log the config parameter values
        self.logger.info(f"Config parameter values: {self.config}")

        # Initialize plan visualizer if enabled
        if self.visualize_plan:
            from isaaclab_mimic.motion_planners.curobo.plan_visualizer import PlanVisualizer

            # Use env-local base translation for multi-env rendering consistency
            env_origin = self.env.scene.env_origins[env_id, :3]
            base_translation = (self.robot.data.root_pos_w[env_id, :3] - env_origin).detach().cpu().numpy()
            self.plan_visualizer = PlanVisualizer(
                robot_name=self.config.robot_name,
                recording_id=f"curobo_plan_{env_id}",
                debug=config.debug_planner,
                base_translation=base_translation,
            )

        # Store attached objects as Attachment objects
        self.attached_objects: dict[str, Attachment] = {}  # object_name -> Attachment

        # Initialize cuRobo components - FORCE CUDA DEVICE FOR ISOLATION
        setup_curobo_logger("warn")

        # Force cuRobo to always use CUDA device regardless of Isaac Lab device
        self.tensor_args: TensorDeviceType
        if torch.cuda.is_available():
            idx = self.config.cuda_device if self.config.cuda_device is not None else torch.cuda.current_device()
            self.tensor_args = TensorDeviceType(device=torch.device(f"cuda:{idx}"), dtype=torch.float32)
            self.logger.debug(f"cuRobo motion planner initialized on CUDA device {idx}")
        else:
            self.tensor_args = TensorDeviceType()
            self.logger.warning("CUDA not available, cuRobo using CPU - this may cause device compatibility issues")

        # Load robot configuration
        if self.config.robot_config_file is None:
            raise ValueError("robot_config_file is required")
        robot_cfg_file = self.config.robot_config_file
        robot_cfg: dict[str, Any] = load_yaml(robot_cfg_file)["robot_cfg"]

        if self.config.collision_spheres_file:
            robot_cfg["kinematics"]["collision_spheres"] = self.config.collision_spheres_file
        if self.config.extra_collision_spheres:
            robot_cfg["kinematics"]["extra_collision_spheres"] = self.config.extra_collision_spheres

        self.robot_cfg: dict[str, Any] = robot_cfg

        if shared_motion_gen is not None:
            # Reuse existing MotionGen — skip creation, warmup, and static world init.
            # The primary planner (env_id=0) already set up the world.
            self.motion_gen: MotionGen = shared_motion_gen
        else:
            # Full initialisation path: build a new MotionGen from scratch.
            world_cfg: WorldConfig = self.config.get_world_config()
            motion_gen_config: MotionGenConfig = MotionGenConfig.load_from_robot_config(
                robot_cfg,
                world_cfg,
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
            )
            self.motion_gen: MotionGen = MotionGen(motion_gen_config)

        # Set motion generator reference for plan visualizer if enabled
        if self.visualize_plan:
            self.plan_visualizer.set_motion_generator_reference(self.motion_gen)

        # Create plan config with parameters from configuration
        self.plan_config: MotionGenPlanConfig = MotionGenPlanConfig(
            enable_graph=self.config.enable_graph,
            enable_graph_attempt=self.config.enable_graph_attempt,
            max_attempts=self.config.max_planning_attempts,
            enable_finetune_trajopt=self.config.enable_finetune_trajopt,
            time_dilation_factor=self.config.time_dilation_factor,
        )

        # Create USD helper
        self.usd_helper: UsdHelper = UsdHelper()
        self.usd_helper.load_stage(env.scene.stage)

        # Initialize planning state
        self._current_plan: JointState | None = None
        self._plan_index: int = 0

        # Initialize visualization state
        self.frame_counter: int = 0
        self.spheres: list[tuple[str, float]] | None = None
        self.sphere_update_freq: int = self.config.sphere_update_freq

        if shared_motion_gen is None:
            # Primary planner path: full init
            self._sync_joint_limits_from_isaac_lab()
            self.logger.info("Warming up motion planner...")
            self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
            self._initialize_static_world()

        # Defer object validation baseline until first update_world() call when scene is fully loaded
        self._expected_objects: set[str] | None = None

        # Define supported cuRobo primitive types for object discovery and pose synchronization
        self.primitive_types: list[str] = ["mesh", "cuboid", "sphere", "capsule", "cylinder", "voxel", "blox"]

        # Cache object mappings
        # Only recompute when objects are added/removed, not when poses change
        self._cached_object_mappings: dict[str, str] | None = None

    # =====================================================================================
    # SHARED PLANNER FACTORY
    # =====================================================================================

    @classmethod
    def create_shared_planners(
        cls,
        env: ManagerBasedEnv,
        robot: Articulation,
        config: CuroboPlannerCfg,
        num_envs: int,
    ) -> list["CuroboPlanner"]:
        """Create N planner instances sharing a single MotionGen.

        The first planner (env_id=0) creates the MotionGen, warms it up, and initialises
        the static world. Subsequent planners reuse that MotionGen, saving significant
        CUDA memory (one set of IK/trajopt/graph-planner GPU buffers instead of N).

        All planners share the same collision world model. Before each planning call the
        per-env planner updates object poses for its own env_id, so the collision world
        always reflects the correct state for the env being planned.

        Args:
            env: The Isaac Lab environment.
            robot: Robot articulation shared across envs.
            config: Planner configuration (same for every env).
            num_envs: Number of environments.

        Returns:
            List of CuroboPlanner instances, one per env, sharing one MotionGen.
        """
        primary = cls(env=env, robot=robot, config=config, env_id=0)
        planners: list[CuroboPlanner] = [primary]
        for i in range(1, num_envs):
            planner = cls(
                env=env,
                robot=robot,
                config=config,
                env_id=i,
                shared_motion_gen=primary.motion_gen,
            )
            planners.append(planner)
        return planners

    # =====================================================================================
    # DEVICE CONVERSION UTILITIES
    # =====================================================================================

    def _to_curobo_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to cuRobo device for isolated device management.

        Ensures all tensors used by cuRobo are on CUDA device, providing device isolation
        from Isaac Lab's potentially different device configuration. This prevents device
        mismatch errors and optimizes cuRobo performance.

        Args:
            tensor: Input tensor (may be on any device)

        Returns:
            Tensor converted to cuRobo's CUDA device with appropriate dtype
        """
        return tensor.to(device=self.tensor_args.device, dtype=self.tensor_args.dtype)

    def _to_env_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor back to environment device for Isaac Lab compatibility.

        Converts cuRobo tensors back to the environment's device to ensure compatibility
        with Isaac Lab operations that expect tensors on the environment's configured device.

        Args:
            tensor: Input tensor from cuRobo operations (typically on CUDA)

        Returns:
            Tensor converted to environment's device while preserving dtype
        """
        return tensor.to(device=self.env.device, dtype=tensor.dtype)

    # =====================================================================================
    # INITIALIZATION AND CONFIGURATION
    # =====================================================================================

    def _sync_joint_limits_from_isaac_lab(self) -> None:
        """Synchronize joint limits from Isaac Lab articulation to cuRobo kinematics.

        This ensures cuRobo plans within the same joint limits that the IK controller
        and physics simulation use. Without this sync, cuRobo might plan trajectories
        that exceed the controller's limits, causing execution failures.

        The method maps Isaac Lab joint names to cuRobo joint names and updates
        cuRobo's position limits to match Isaac Lab's articulation limits.
        """
        # Get cuRobo's joint limits and names
        curobo_limits = self.motion_gen.kinematics.get_joint_limits()
        curobo_joint_names = list(curobo_limits.joint_names)

        # Get Isaac Lab's joint limits (shape: [num_instances, num_joints, 2])
        isaac_limits = self.robot.data.joint_pos_limits[self.env_id]  # [num_joints, 2]
        isaac_joint_names = self.robot.data.joint_names

        # Build mapping from Isaac Lab joint name to index
        isaac_name_to_idx = {name: idx for idx, name in enumerate(isaac_joint_names)}

        # Track updates for logging
        updated_joints = []
        mismatches = []

        for curobo_idx, joint_name in enumerate(curobo_joint_names):
            if joint_name in isaac_name_to_idx:
                isaac_idx = isaac_name_to_idx[joint_name]
                isaac_lower = isaac_limits[isaac_idx, 0].item()
                isaac_upper = isaac_limits[isaac_idx, 1].item()

                curobo_lower = curobo_limits.position[0, curobo_idx].item()
                curobo_upper = curobo_limits.position[1, curobo_idx].item()

                # Check for significant mismatch (more than 0.01 rad difference)
                if abs(isaac_lower - curobo_lower) > 0.01 or abs(isaac_upper - curobo_upper) > 0.01:
                    mismatches.append(
                        f"{joint_name}: cuRobo=[{curobo_lower:.3f}, {curobo_upper:.3f}] "
                        f"-> Isaac=[{isaac_lower:.3f}, {isaac_upper:.3f}]"
                    )

                # Update cuRobo limits to match Isaac Lab
                curobo_limits.position[0, curobo_idx] = isaac_lower
                curobo_limits.position[1, curobo_idx] = isaac_upper
                updated_joints.append(joint_name)

        if mismatches:
            self.logger.info(f"Synced {len(updated_joints)} joint limits from Isaac Lab to cuRobo")
            for mismatch in mismatches[:5]:  # Log first 5 mismatches
                self.logger.debug(f"  Joint limit updated: {mismatch}")
            if len(mismatches) > 5:
                self.logger.debug(f"  ... and {len(mismatches) - 5} more")
        else:
            self.logger.debug(f"Joint limits already in sync ({len(updated_joints)} joints checked)")

    def _initialize_static_world(self) -> None:
        """Initialize static world geometry from USD stage.

        Reads static environment geometry once during planner initialization to establish
        the base collision world. This includes walls, tables, bins, and other fixed obstacles
        that don't change during the simulation. Dynamic objects are synchronized separately
        in update_world() to maintain performance.

        Obstacles are extracted in WORLD FRAME, then manually transformed to robot base frame
        using the physics root pose (robot.data.root_pos_w/root_quat_w). This ensures consistency
        with dynamic object transforms which also use physics pose.

        Note: We don't use robot_prim_path as reference because the USD prim and physics root
        can have different transforms (e.g., internal offset in USD asset). Using physics pose
        for both static and dynamic objects ensures they align correctly.
        """
        from isaaclab.utils.math import quat_inv

        env_prim_path = f"/World/envs/env_{self.env_id}"
        robot_prim_path = self.config.robot_prim_path or f"{env_prim_path}/Robot"

        base_ignore = self.config.world_ignore_substrings or [
            f"{env_prim_path}/target",
            "/World/defaultGroundPlane",
            "/curobo",
        ]
        ignore_list = self._expand_ignore_list_for_env(base_ignore, self.env_id)
        # Always ignore the current robot prim for static obstacle extraction
        if robot_prim_path not in ignore_list:
            ignore_list.append(robot_prim_path)

        # Extract obstacles in WORLD FRAME (no reference prim).
        self._static_world_config = self.usd_helper.get_obstacles_from_stage(
            only_paths=[env_prim_path],
            reference_prim_path=None,
            ignore_substring=ignore_list,
        )

        # Transform obstacles from world frame to robot base frame using PHYSICS pose.
        # This ensures consistency with dynamic object transforms in _sync_object_poses_with_isaaclab.
        robot_pos_w = self.robot.data.root_pos_w[self.env_id]
        robot_quat_w = self.robot.data.root_quat_w[self.env_id]  # (w, x, y, z)
        robot_quat_inv = quat_inv(robot_quat_w)

        self._transform_world_config_to_robot_frame(
            self._static_world_config, robot_pos_w, robot_quat_inv
        )

        self._static_world_config = self._static_world_config.get_collision_check_world()

        # Initialize cuRobo world with static geometry
        self.motion_gen.update_world(self._static_world_config)

    def _transform_world_config_to_robot_frame(
        self, world_cfg, robot_pos_w: torch.Tensor, robot_quat_inv: torch.Tensor
    ) -> None:
        """Transform world config obstacles from world frame to robot base frame.

        Args:
            world_cfg: WorldConfig with obstacles in world frame
            robot_pos_w: Robot position in world frame
            robot_quat_inv: Inverse of robot quaternion (for rotation)
        """
        from isaaclab.utils.math import quat_apply, quat_mul

        def _transform_obstacle(obj):
            if obj is None or not hasattr(obj, "pose") or obj.pose is None:
                return
            # obj.pose is [x, y, z, qw, qx, qy, qz]
            pos = torch.tensor(obj.pose[:3], device=robot_pos_w.device, dtype=torch.float32)
            quat = torch.tensor(obj.pose[3:], device=robot_pos_w.device, dtype=torch.float32)

            # Transform position: p_robot = R_robot^-1 * (p_obj - p_robot)
            rel_pos = pos - robot_pos_w
            new_pos = quat_apply(robot_quat_inv, rel_pos)

            # Transform orientation: q_robot = q_robot^-1 * q_obj
            new_quat = quat_mul(robot_quat_inv, quat)

            # Update pose
            obj.pose = [
                float(new_pos[0].item()),
                float(new_pos[1].item()),
                float(new_pos[2].item()),
                float(new_quat[0].item()),
                float(new_quat[1].item()),
                float(new_quat[2].item()),
                float(new_quat[3].item()),
            ]

        def _transform_list(obj_list):
            if not obj_list:
                return
            for obj in obj_list:
                _transform_obstacle(obj)

        # Handle both single config and list of configs
        cfgs = world_cfg if isinstance(world_cfg, list) else [world_cfg]
        for cfg in cfgs:
            if cfg is None:
                continue
            _transform_list(getattr(cfg, "cuboid", None))
            _transform_list(getattr(cfg, "mesh", None))
            _transform_list(getattr(cfg, "cylinder", None))
            _transform_list(getattr(cfg, "capsule", None))
            _transform_list(getattr(cfg, "sphere", None))

    def _expand_ignore_list_for_env(self, items: list[str], env_id: int) -> list[str]:
        """Clone ignore substrings for the current env id (replace /env_0/ -> /env_{env_id}/)."""
        expanded: list[str] = []
        token = f"/env_{env_id}/"
        for s in items:
            if "/env_0/" in s:
                expanded.append(s.replace("/env_0/", token))
            else:
                expanded.append(s)
        return expanded

    # =====================================================================================
    # PROPERTIES AND BASIC GETTERS
    # =====================================================================================

    @property
    def attached_link(self) -> str:
        """Default link name for object attachment operations."""
        return self.config.attached_object_link_name

    @property
    def attachment_links(self) -> set[str]:
        """Set of parent link names that currently have attached objects."""
        return {attachment.parent for attachment in self.attached_objects.values()}

    @property
    def current_plan(self) -> JointState | None:
        """Current plan from cuRobo motion generator."""
        return self._current_plan

    # =====================================================================================
    # WORLD AND OBJECT MANAGEMENT, ATTACHMENT, AND DETACHMENT
    # =====================================================================================

    def get_object_pose(self, object_name: str) -> Pose | None:
        """Retrieve object pose from cuRobo's collision world model.

        Searches the collision world model for the specified object and returns its current
        pose. This is useful for attachment calculations and debugging collision world state.
        The method handles both mesh and cuboid object types automatically.

        Args:
            object_name: Short object name used in Isaac Lab scene (e.g., "cube_1")

        Returns:
            Object pose in cuRobo coordinate frame, or None if object not found
        """
        # Get cached object mappings
        object_mappings = self._get_object_mappings()
        world_model = self.motion_gen.world_coll_checker.world_model

        object_path = object_mappings.get(object_name)
        if not object_path:
            self.logger.debug(f"Object {object_name} not found in world model")
            return None

        # Search for object in world model
        for obj_list, _ in [
            (world_model.mesh, "mesh"),
            (world_model.cuboid, "cuboid"),
        ]:
            if not obj_list:
                continue

            for obj in obj_list:
                if obj.name and object_path in str(obj.name):
                    if obj.pose is not None:
                        return Pose.from_list(obj.pose, tensor_args=self.tensor_args)

        self.logger.debug(f"Object {object_name} found in mappings but pose not available")
        return None

    def get_attached_pose(self, link_name: str, joint_state: JointState | None = None) -> Pose:
        """Calculate pose of specified link using forward kinematics.

        Computes the world pose of any robot link at the given joint configuration.
        This is essential for attachment calculations where we need to know the exact
        pose of the parent link to compute relative object positions.

        Args:
            link_name: Name of the robot link to get pose for
            joint_state: Joint configuration to use for calculation, uses current state if None

        Returns:
            World pose of the specified link in cuRobo coordinate frame

        Raises:
            KeyError: If link_name is not found in the computed link poses
        """
        if joint_state is None:
            joint_state = self._get_current_joint_state_for_curobo()

        # Get all link states using the robot model
        link_state = self.motion_gen.kinematics.get_state(
            q=joint_state.position.detach().clone().to(device=self.tensor_args.device, dtype=self.tensor_args.dtype),
            calculate_jacobian=False,
        )

        # Extract all link poses
        link_poses = {}
        if link_state.links_position is not None and link_state.links_quaternion is not None:
            for i, link in enumerate(link_state.link_names):
                link_poses[link] = self._make_pose(
                    position=link_state.links_position[..., i, :],
                    quaternion=link_state.links_quaternion[..., i, :],
                    name=link,
                )

        # For attached object link, use ee_link from robot config as parent
        if link_name == self.config.attached_object_link_name:
            ee_link = self.config.ee_link_name or self.robot_cfg["kinematics"]["ee_link"]
            if ee_link in link_poses:
                self.logger.debug(f"Using {ee_link} for {link_name}")
                return link_poses[ee_link]

        # Return directly for other links
        if link_name in link_poses:
            return link_poses[link_name]
        raise KeyError(f"Link {link_name} not found in computed link poses")

    def create_attachment(
        self, object_name: str, link_name: str | None = None, joint_state: JointState | None = None
    ) -> Attachment:
        """Create attachment relationship between object and robot link.

        Computes the relative pose between an object and a robot link to enable the robot
        to carry the object consistently during motion planning. The attachment stores the transform
        from the parent link frame to the object frame, which remains constant while grasped.

        Args:
            object_name: Name of the object to attach
            link_name: Parent link for attachment, uses default attached_object_link if None
            joint_state: Robot configuration for calculation, uses current state if None

        Returns:
            Attachment object containing relative pose and parent link information
        """
        if link_name is None:
            link_name = self.attached_link
        if joint_state is None:
            joint_state = self._get_current_joint_state_for_curobo()

        # Get current link pose
        link_pose = self.get_attached_pose(link_name, joint_state)
        self.logger.info(f"Getting object pose for {object_name}")
        obj_pose = self.get_object_pose(object_name)

        # Compute relative pose (object in wrist local frame)
        attach_pose = link_pose.inverse().multiply(obj_pose)

        self.logger.debug(f"Creating attachment for {object_name} to {link_name}")
        self.logger.debug(f"Link pose: {link_pose.position}")
        self.logger.debug(f"Link quat: {link_pose.quaternion}")
        self.logger.debug(f"Object pose (ACTUAL): {obj_pose.position}")
        self.logger.debug(f"Computed relative pose (LOCAL FRAME): {attach_pose.position}")

        # Debug: compute offset in world frame for comparison
        offset_world = obj_pose.position.squeeze() - link_pose.position.squeeze()
        self.logger.info(f"[PALM OFFSET DEBUG] Offset in WORLD frame: {offset_world.cpu().tolist()}")
        self.logger.info(f"[PALM OFFSET DEBUG] Offset in LOCAL frame: {attach_pose.position.squeeze().cpu().tolist()}")
        palm_cfg = getattr(self.config, 'palm_offset_from_ee', None)
        self.logger.info(f"[PALM OFFSET DEBUG] Config palm_offset_from_ee: {palm_cfg}")

        return Attachment(attach_pose, link_name)

    def update_world(self) -> None:
        """Synchronize collision world with current Isaac Lab scene state.

        Updates all dynamic object poses in cuRobo's collision world to match their current
        positions in Isaac Lab. This ensures collision checking uses accurate object positions
        after simulation steps, resets, or manual object movements. Static world geometry
        is loaded once during initialization and not updated here for performance.

        The method validates that the set of objects hasn't changed at runtime, as cuRobo
        requires world model reinitialization when objects are added or removed.

        Raises:
            RuntimeError: If the set of objects has changed at runtime
        """

        # Establish validation baseline on first call, validate on subsequent calls
        if self._expected_objects is None:
            self._expected_objects = set(self._get_world_object_names())
            self.logger.debug(f"Established object validation baseline: {len(self._expected_objects)} objects")
        else:
            # Subsequent calls: validate no changes
            current_objects = set(self._get_world_object_names())
            if current_objects != self._expected_objects:
                added = current_objects - self._expected_objects
                removed = self._expected_objects - current_objects

                error_msg = "World objects changed at runtime!\n"
                if added:
                    error_msg += f"Added: {added}\n"
                if removed:
                    error_msg += f"Removed: {removed}\n"
                error_msg += "cuRobo world model must be reinitialized."

                # Invalidate cached mappings since object set changed
                self._cached_object_mappings = None

                raise RuntimeError(error_msg)

        # Sync object poses with Isaac Lab
        self._sync_object_poses_with_isaaclab()

        if self.visualize_spheres:
            self._update_sphere_visualization(force_update=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _get_world_object_names(self) -> list[str]:
        """Extract all object names from cuRobo's collision world model.

        Iterates through all supported primitive types (mesh, cuboid, sphere, etc.) in the
        collision world and collects their names. This is used for world validation to detect
        when objects are added or removed at runtime.

        Returns:
            List of all object names currently in the collision world model
        """
        try:
            world_model = self.motion_gen.world_coll_checker.world_model

            # Handle case where world_model might be a list
            if isinstance(world_model, list):
                if len(world_model) <= self.env_id:
                    return []
                world_model = world_model[self.env_id]

            object_names = []

            # Get all primitive object names using the defined primitive types
            for primitive_type in self.primitive_types:
                if hasattr(world_model, primitive_type) and getattr(world_model, primitive_type):
                    primitive_list = getattr(world_model, primitive_type)
                    for primitive in primitive_list:
                        if primitive.name:
                            object_names.append(str(primitive.name))

            return object_names

        except Exception as e:
            self.logger.debug(f"ERROR getting world object names: {e}")
            return []

    def _sync_object_poses_with_isaaclab(self) -> None:
        """Synchronize cuRobo collision world with Isaac Lab object positions.

        Updates all dynamic object poses in cuRobo's world model to match their current
        positions in Isaac Lab. This ensures accurate collision checking after simulation
        steps or manual object movements. Static objects (bins, tables, walls) are skipped
        for performance as they shouldn't move during simulation.

        Objects are transformed to ROBOT BASE FRAME to match static obstacles extracted
        with robot prim as reference. This ensures correct collision checking for robots
        with non-identity orientation (like GR1T2 which is rotated 90 degrees).

        The method updates both the world model and the collision checker to ensure
        consistency across all cuRobo components.
        """
        from isaaclab.utils.math import quat_inv, quat_apply, quat_mul

        # Get cached object mappings and world model
        object_mappings = self._get_object_mappings()
        world_model = self.motion_gen.world_coll_checker.world_model
        rigid_objects = self.env.scene.rigid_objects

        # Get robot world pose for transforming objects to robot base frame
        robot_pos_w = self.robot.data.root_pos_w[self.env_id]
        robot_quat_w = self.robot.data.root_quat_w[self.env_id]  # (w, x, y, z)
        robot_quat_inv = quat_inv(robot_quat_w)

        updated_count = 0

        for object_name, object_path in object_mappings.items():
            if object_name not in rigid_objects:
                continue

            # Skip static mesh objects - they should not be dynamically updated
            static_objects = getattr(self.config, "static_objects", [])
            if any(static_name in object_name.lower() for static_name in static_objects):
                self.logger.debug(f"SYNC: Skipping static object {object_name}")
                continue

            # Get current pose from Lab in world frame
            obj = rigid_objects[object_name]
            obj_pos_w = obj.data.root_pos_w[self.env_id]
            obj_quat_w = obj.data.root_quat_w[self.env_id]  # (w, x, y, z)

            # DEBUG: Print object position in world frame
            print(f"[SYNC DEBUG] {object_name}: Isaac Lab pos (world) = {obj_pos_w.cpu().tolist()}")

            # Transform to robot base frame: p_robot = R_robot^-1 * (p_obj - p_robot)
            rel_pos = obj_pos_w - robot_pos_w
            current_pos_raw = quat_apply(robot_quat_inv, rel_pos)

            # Transform orientation: q_robot = q_robot^-1 * q_obj
            current_quat_raw = quat_mul(robot_quat_inv, obj_quat_w)

            # Convert to cuRobo device and extract float values for pose list
            current_pos = self._to_curobo_device(current_pos_raw)
            current_quat = self._to_curobo_device(current_quat_raw)

            # Convert to cuRobo pose format [x, y, z, w, x, y, z]
            pose_list = [
                float(current_pos[0].item()),
                float(current_pos[1].item()),
                float(current_pos[2].item()),
                float(current_quat[0].item()),
                float(current_quat[1].item()),
                float(current_quat[2].item()),
                float(current_quat[3].item()),
            ]

            # DEBUG: Print transformed position
            print(f"[SYNC DEBUG] {object_name}: CuRobo pos (base) = {pose_list[:3]}")

            # Update object pose in cuRobo's world model
            if self._update_object_in_world_model(world_model, object_name, object_path, pose_list):
                updated_count += 1
                print(f"[SYNC DEBUG] {object_name}: Updated in world model")
            else:
                print(f"[SYNC DEBUG] {object_name}: NOT found in world model!")

        self.logger.debug(f"SYNC: Updated {updated_count} object poses in cuRobo world model")

        # Sync object poses with collision checker
        if updated_count > 0:
            # Update individual obstacle poses in collision checker
            # This preserves static mesh objects unlike load_collision_model which rebuilds everything
            for object_name, object_path in object_mappings.items():
                if object_name not in rigid_objects:
                    continue

                # Skip static mesh objects - they should not be dynamically updated
                static_objects = getattr(self.config, "static_objects", [])
                if any(static_name in object_name.lower() for static_name in static_objects):
                    continue

                # Get current pose and transform to robot base frame
                obj = rigid_objects[object_name]
                obj_pos_w = obj.data.root_pos_w[self.env_id]
                obj_quat_w = obj.data.root_quat_w[self.env_id]

                # Transform to robot base frame
                rel_pos = obj_pos_w - robot_pos_w
                current_pos_raw = quat_apply(robot_quat_inv, rel_pos)
                current_quat_raw = quat_mul(robot_quat_inv, obj_quat_w)

                current_pos = self._to_curobo_device(current_pos_raw)
                current_quat = self._to_curobo_device(current_quat_raw)

                # Create cuRobo pose and update collision checker directly
                curobo_pose = self._make_pose(position=current_pos, quaternion=current_quat)
                self.motion_gen.world_coll_checker.update_obstacle_pose(  # type: ignore
                    object_path, curobo_pose, update_cpu_reference=True
                )

            self.logger.debug(f"Updated {updated_count} object poses in collision checker")

    def _get_object_mappings(self) -> dict[str, str]:
        """Get object mappings with caching for performance optimization.

        Returns cached mappings if available, otherwise computes and caches them.
        Cache is invalidated when the object set changes.

        Returns:
            Dictionary mapping Isaac Lab object names to their corresponding USD paths
        """
        if self._cached_object_mappings is None:
            world_model = self.motion_gen.world_coll_checker.world_model
            rigid_objects = self.env.scene.rigid_objects
            self._cached_object_mappings = self._discover_object_mappings(world_model, rigid_objects)
            self.logger.debug(f"Computed and cached object mappings: {len(self._cached_object_mappings)} objects")

        return self._cached_object_mappings

    @property
    def _world_model_env_id(self) -> int:
        """Env id whose USD paths are stored in the shared collision world model.

        For shared MotionGen the static world was built from env_0, so all obstacle
        paths use the ``/World/envs/env_0/`` prefix regardless of which env this
        planner instance is serving.
        """
        return 0 if self._is_shared else self.env_id

    def _discover_object_mappings(self, world_model, rigid_objects) -> dict[str, str]:
        """Build mapping between Isaac Lab object names and cuRobo world paths.

        Automatically discovers the correspondence between Isaac Lab's rigid object names
        and their full USD paths in cuRobo's world model. This mapping is essential for
        pose synchronization and attachment operations, as cuRobo uses full USD paths
        while Isaac Lab uses short object names.

        For shared MotionGen, the world model always uses env_0 paths. Planners for
        other envs will discover objects using the env_0 prefix, then update those
        same paths with their own env_id's poses.

        Args:
            world_model: cuRobo's collision world model containing primitive objects
            rigid_objects: Isaac Lab's rigid objects dictionary

        Returns:
            Dictionary mapping Isaac Lab object names to their corresponding USD paths
        """
        mappings = {}
        env_prefix = f"/World/envs/env_{self._world_model_env_id}/"
        world_object_paths = []

        # Collect all primitive objects from cuRobo world model
        for primitive_type in self.primitive_types:
            primitive_list = getattr(world_model, primitive_type)
            for primitive in primitive_list:
                if primitive.name and env_prefix in str(primitive.name):
                    world_object_paths.append(str(primitive.name))

        # Match Isaac Lab object names to world paths
        for object_name in rigid_objects.keys():
            for path in world_object_paths:
                if object_name.lower().replace("_", "") in path.lower().replace("_", ""):
                    mappings[object_name] = path
                    self.logger.debug(f"MAPPING: {object_name} -> {path}")
                    break
            else:
                self.logger.debug(f"WARNING: Could not find world path for {object_name}")

        return mappings

    def _update_object_in_world_model(
        self, world_model, object_name: str, object_path: str, pose_list: list[float]
    ) -> bool:
        """Update a single object's pose in cuRobo's collision world model.

        Searches through all primitive types in the world model to find the specified object
        and updates its pose. Uses flexible matching to handle variations in path naming
        between Isaac Lab and cuRobo representations.

        Args:
            world_model: cuRobo's collision world model
            object_name: Short object name from Isaac Lab (e.g., "cube_1")
            object_path: Full USD path for the object in cuRobo world
            pose_list: New pose as [x, y, z, w, x, y, z] list in cuRobo format

        Returns:
            True if object was found and successfully updated, False otherwise
        """
        # Handle case where world_model might be a list
        if isinstance(world_model, list):
            if len(world_model) > self.env_id:
                world_model = world_model[self.env_id]
            else:
                return False

        # Update all primitive types
        for primitive_type in self.primitive_types:
            primitive_list = getattr(world_model, primitive_type)
            for primitive in primitive_list:
                if primitive.name:
                    primitive_name = str(primitive.name)
                    # Use bidirectional matching for robust path matching
                    if object_path == primitive_name or object_path in primitive_name or primitive_name in object_path:
                        primitive.pose = pose_list
                        self.logger.debug(f"Updated {primitive_type} {object_name} pose")
                        return True

        self.logger.debug(f"WARNING: Object {object_name} not found in world model")
        return False

    def _attach_object(self, object_name: str, object_path: str, env_id: int) -> bool:
        """Attach an object to the robot for manipulation planning.

        Establishes an attachment between the specified object and the robot's end-effector
        or configured attachment link. This enables the robot to carry the object during
        motion planning while maintaining proper collision checking. The object's collision
        geometry is disabled in the world model since it's now part of the robot.

        Args:
            object_name: Short name of the object to attach (e.g., "cube_2")
            object_path: Full USD path for the object in cuRobo world model
            env_id: Environment ID for multi-environment support

        Returns:
            True if attachment succeeded, False if attachment failed
        """
        current_joint_state = self._get_current_joint_state_for_curobo()

        self.logger.debug(f"Attaching {object_name} at path {object_path}")

        # Create attachment record (relative pose object-frame to parent link)
        attachment = self.create_attachment(
            object_name,
            self.config.attached_object_link_name,
            current_joint_state,
        )
        self.attached_objects[object_name] = attachment
        success = self.motion_gen.attach_objects_to_robot(
            joint_state=current_joint_state,
            object_names=[object_path],
            link_name=self.config.attached_object_link_name,
            surface_sphere_radius=self.config.surface_sphere_radius,
            sphere_fit_type=SphereFitType.SAMPLE_SURFACE,
            world_objects_pose_offset=None,
        )

        if success:
            self.logger.debug(f"Successfully attached {object_name}")
            self.logger.debug(f"Current attached objects: {list(self.attached_objects.keys())}")

            # Force sphere visualization update
            if self.visualize_spheres:
                self._update_sphere_visualization(force_update=True)

            self.logger.info(f"Sphere count after attach is successful: {self._count_active_spheres()}")

            # Deactivate the original obstacle as it's now carried by the robot
            self.motion_gen.world_coll_checker.enable_obstacle(object_path, enable=False)

            return True
        else:
            self.logger.error(f"cuRobo attach_objects_to_robot failed for {object_name}")
            # Clean up on failure
            if object_name in self.attached_objects:
                del self.attached_objects[object_name]
            return False

    def _detach_objects(self, link_names: set[str] | None = None) -> bool:
        """Detach objects from robot and restore collision checking.

        Removes object attachments from specified links and re-enables collision checking
        for both the objects and the parent links. This is necessary when placing objects
        or changing grasps. All attached objects are detached if no specific links are provided.

        Args:
            link_names: Set of parent link names to detach objects from, detaches all if None

        Returns:
            True if detachment operations completed successfully, False otherwise
        """
        if link_names is None:
            link_names = self.attachment_links

        self.logger.debug(f"Detaching objects from links: {link_names}")
        self.logger.debug(f"Current attached objects: {list(self.attached_objects.keys())}")

        # Get cached object mappings to find the USD path for re-enabling
        object_mappings = self._get_object_mappings()

        detached_info = []
        detached_links = set()
        for object_name, attachment in list(self.attached_objects.items()):
            if attachment.parent not in link_names:
                continue

            # Find object path and re-enable it in the world
            object_path = object_mappings.get(object_name)
            if object_path:
                self.motion_gen.world_coll_checker.enable_obstacle(object_path, enable=True)  # type: ignore
                self.logger.debug(f"Re-enabled obstacle {object_path}")

            # Collect the link that will need re-enabling
            detached_links.add(attachment.parent)

            # Remove from attached objects and log info
            del self.attached_objects[object_name]
            detached_info.append((object_name, attachment.parent))

        if detached_info:
            for obj_name, parent_link in detached_info:
                self.logger.debug(f"Detached {obj_name} from {parent_link}")

        # Re-enable collision checking for the attachment links (following the planning pattern)
        if detached_links:
            self._set_active_links(list(detached_links), active=True)
            self.logger.debug(f"Re-enabled collision for attachment links: {detached_links}")

        # Call cuRobo's detach for tracked links
        for link_name in link_names:
            try:
                self.motion_gen.detach_object_from_robot(link_name=link_name)
                self.logger.debug(f"Called cuRobo detach for link {link_name}")
            except ValueError:
                # No spheres attached to this link - continue to next
                self.logger.debug(f"No spheres attached to {link_name}, skipping detach")

        # Detach from configured attachment link to clean up any orphaned spheres
        # Only do this if the link actually exists in the kinematics model
        configured_link = self.config.attached_object_link_name
        if configured_link and configured_link not in link_names:
            # Check if the link exists in the kinematics model before calling detach
            # This prevents cuRobo from logging errors for non-existent links (e.g., "attached_object" on humanoids)
            link_idx_map = getattr(
                self.motion_gen.kinematics.kinematics_config, "link_name_to_idx_map", {}
            )
            if configured_link in link_idx_map:
                try:
                    self.motion_gen.detach_object_from_robot(link_name=configured_link)
                    self.logger.debug(f"Called cuRobo detach for configured attachment link: {configured_link}")
                except ValueError:
                    pass
            else:
                self.logger.debug(f"Skipping detach for non-existent link: {configured_link}")

        return True

    def get_attached_objects(self) -> list[str]:
        """Get list of currently attached object names.

        Returns the short names of all objects currently attached to the robot.
        These names correspond to Isaac Lab scene object names, not full USD paths.

        Returns:
            List of attached object names (e.g., ["cube_1", "cube_2"])"""
        return list(self.attached_objects.keys())

    def has_attached_objects(self) -> bool:
        """Check if any objects are currently attached to the robot.

        Useful for determining gripper state and collision checking configuration
        before planning motions.

        Returns:
            True if one or more objects are attached, False if no attachments exist
        """
        return len(self.attached_objects) != 0

    def detach_all_objects(self) -> bool:
        """Detach all objects from the robot.

        Public wrapper for _detach_objects that detaches from all attachment links.

        Returns:
            True if detachment succeeded
        """
        return self._detach_objects()

    # =====================================================================================
    # JOINT STATE AND KINEMATICS
    # =====================================================================================

    def _get_current_joint_state_for_curobo(self) -> JointState:
        """
        Construct the current joint state for cuRobo with zero velocity and acceleration.

        This helper reads the robot's joint positions from Isaac Lab for the current environment
        and pairs them with zero velocities and accelerations as required by cuRobo planning.
        All tensors are moved to the cuRobo device and reordered to match the kinematic chain
        used by the cuRobo motion generator.

        Returns:
            JointState on the cuRobo device, ordered according to
            `self.motion_gen.kinematics.joint_names`, with position from the robot
            and zero velocity/acceleration.
        """
        # Fetch joint position (shape: [1, num_joints])
        joint_pos_raw: torch.Tensor = self.robot.data.joint_pos[self.env_id, :].unsqueeze(0)
        joint_vel_raw: torch.Tensor = torch.zeros_like(joint_pos_raw)
        joint_acc_raw: torch.Tensor = torch.zeros_like(joint_pos_raw)

        # Move to cuRobo device
        joint_pos: torch.Tensor = self._to_curobo_device(joint_pos_raw)
        joint_vel: torch.Tensor = self._to_curobo_device(joint_vel_raw)
        joint_acc: torch.Tensor = self._to_curobo_device(joint_acc_raw)

        cu_js: JointState = JointState(
            position=joint_pos,
            velocity=joint_vel,
            acceleration=joint_acc,
            joint_names=self.robot.data.joint_names,
            tensor_args=self.tensor_args,
        )
        return cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

    def get_ee_pose(self, joint_state: JointState) -> Pose:
        """Compute end-effector pose from joint configuration.

        Uses cuRobo's forward kinematics to calculate the end-effector pose
        at the specified joint configuration. Handles device conversion to ensure
        compatibility with cuRobo's CUDA-based computations.

        Args:
            joint_state: Robot joint configuration to compute end-effector pose from

        Returns:
            End-effector pose in world coordinates
        """
        # Ensure joint state is on CUDA device for cuRobo
        if isinstance(joint_state.position, torch.Tensor):
            cuda_position = self._to_curobo_device(joint_state.position)
        else:
            cuda_position = self._to_curobo_device(torch.tensor(joint_state.position))

        # Create new joint state with CUDA tensors
        cuda_joint_state = JointState(
            position=cuda_position,
            velocity=(
                self._to_curobo_device(joint_state.velocity.detach().clone())
                if joint_state.velocity is not None
                else torch.zeros_like(cuda_position)
            ),
            acceleration=(
                self._to_curobo_device(joint_state.acceleration.detach().clone())
                if joint_state.acceleration is not None
                else torch.zeros_like(cuda_position)
            ),
            joint_names=joint_state.joint_names,
            tensor_args=self.tensor_args,
        )

        kin_state: Any = self.motion_gen.rollout_fn.compute_kinematics(cuda_joint_state)
        return kin_state.ee_pose

    # =====================================================================================
    # PLANNING CORE METHODS
    # =====================================================================================

    def _make_pose(
        self,
        position: torch.Tensor | np.ndarray | list[float] | None = None,
        quaternion: torch.Tensor | np.ndarray | list[float] | None = None,
        *,
        name: str | None = None,
        normalize_rotation: bool = False,
    ) -> Pose:
        """Create a cuRobo Pose with sensible defaults and device/dtype alignment.

        Auto-populates missing fields with identity values and ensures tensors are
        on the cuRobo device with the correct dtype.

        Args:
            position: Optional position as Tensor/ndarray/list. Defaults to [0, 0, 0].
            quaternion: Optional quaternion as Tensor/ndarray/list (w, x, y, z). Defaults to [1, 0, 0, 0].
            name: Optional name of the link that this pose represents.
            normalize_rotation: Whether to normalize the quaternion inside Pose.

        Returns:
            Pose: A cuRobo Pose on the configured cuRobo device and dtype.
        """
        # Defaults
        if position is None:
            position = torch.tensor([0.0, 0.0, 0.0], dtype=self.tensor_args.dtype, device=self.tensor_args.device)
        if quaternion is None:
            quaternion = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=self.tensor_args.dtype, device=self.tensor_args.device
            )

        # Convert to tensors if needed
        if not isinstance(position, torch.Tensor):
            position = torch.tensor(position, dtype=self.tensor_args.dtype, device=self.tensor_args.device)
        else:
            position = self._to_curobo_device(position)

        if not isinstance(quaternion, torch.Tensor):
            quaternion = torch.tensor(quaternion, dtype=self.tensor_args.dtype, device=self.tensor_args.device)
        else:
            quaternion = self._to_curobo_device(quaternion)

        return Pose(position=position, quaternion=quaternion, name=name, normalize_rotation=normalize_rotation)

    def _set_active_links(self, links: list[str], active: bool) -> None:
        """Configure collision checking for specific robot links.

        Enables or disables collision sphere checking for the specified links.
        This is essential for contact scenarios where certain links (like fingers
        or attachment points) need collision checking disabled to allow contact
        with objects being grasped.

        Args:
            links: List of link names to enable or disable collision checking for
            active: True to enable collision checking, False to disable
        """
        for link in links:
            try:
                if active:
                    self.motion_gen.kinematics.kinematics_config.enable_link_spheres(link)
                else:
                    self.motion_gen.kinematics.kinematics_config.disable_link_spheres(link)
            except ValueError:
                # Link not found in sphere configuration - skip
                self.logger.debug(f"Link {link} not found in sphere config, skipping")

    def plan_motion(
        self,
        target_pose: torch.Tensor,
        step_size: float | None = None,
        enable_retiming: bool | None = None,
        *,
        link_target_poses_base: dict[str, torch.Tensor] | None = None,
        start_joint_state: torch.Tensor | None = None,
    ) -> bool:
        """Plan collision-free motion to target pose.

        Plans a trajectory from the current robot configuration to the specified target pose.
        The method assumes that world updates and locked joint configurations have already
        been handled. Supports optional linear retiming for consistent execution speeds.

        Args:
            target_pose: Target end-effector pose as 4x4 transformation matrix
            step_size: Step size for linear retiming, enables retiming if provided
            enable_retiming: Whether to enable linear retiming, auto-detected from step_size if None
            start_joint_state: Optional start joint state tensor in cuRobo planner ordering.
                If provided, bypasses reading from articulation buffer (for offline planning).

        Returns:
            True if planning succeeded and a valid trajectory was found, False otherwise
        """
        if enable_retiming is None:
            enable_retiming = step_size is not None

        # Ensure target pose is on cuRobo device (CUDA) for device isolation
        target_pose_cuda = self._to_curobo_device(target_pose)

        target_pos: torch.Tensor
        target_rot: torch.Tensor
        target_pos, target_rot = PoseUtils.unmake_pose(target_pose_cuda)
        target_curobo_pose: Pose = self._make_pose(
            position=target_pos,
            quaternion=PoseUtils.quat_from_matrix(target_rot),
        )

        start_state: JointState
        if start_joint_state is not None:
            # Use provided start state (for offline planning)
            pos = start_joint_state.to(device=self.tensor_args.device, dtype=self.tensor_args.dtype)
            if pos.dim() == 1:
                pos = pos.unsqueeze(0)
            # Clamp to joint limits
            limits = self.motion_gen.kinematics.get_joint_limits().position
            low, high = limits[0], limits[1]
            margin = 1e-4
            pos = torch.clamp(pos, low + margin, high - margin)
            self.logger.debug("Using provided start_joint_state for planning")
            start_state = JointState(
                position=pos,
                velocity=torch.zeros_like(pos),
                acceleration=torch.zeros_like(pos),
                joint_names=self.motion_gen.kinematics.joint_names,
                tensor_args=self.tensor_args,
            )
        else:
            start_state = self._get_current_joint_state_for_curobo()

        self.logger.debug(f"Retiming enabled: {enable_retiming}, Step size: {step_size}")

        # If multi-link targets provided, do a direct single attempt with link poses
        if link_target_poses_base:
            # Build a dict[name->Pose] for only the constrained links (robust to cuRobo variants)
            link_pose_dict: dict[str, Pose] = {}
            for name, T_base_link in link_target_poses_base.items():
                pos_l, rot_l = PoseUtils.unmake_pose(
                    T_base_link.to(device=self.tensor_args.device, dtype=self.tensor_args.dtype)
                )
                link_pose_dict[name] = self._make_pose(position=pos_l, quaternion=PoseUtils.quat_from_matrix(rot_l))
            # Prefer start-state-as-retract to bias IK/trajopt to current configuration
            prev_use_start = getattr(self.plan_config, "use_start_state_as_retract", None)
            if prev_use_start is not None:
                self.plan_config.use_start_state_as_retract = True
            result: Any = self.motion_gen.plan_single(
                start_state,
                target_curobo_pose,
                self.plan_config,
                link_poses=link_pose_dict,
            )
            if prev_use_start is not None:
                self.plan_config.use_start_state_as_retract = prev_use_start

            success = bool(result.success.item()) if hasattr(result, "success") else False
            if success:
                if result.optimized_plan is not None and len(result.optimized_plan.position) != 0:
                    self._current_plan = result.optimized_plan
                else:
                    self._current_plan = result.get_interpolated_plan()

                self._current_plan = self.motion_gen.get_full_js(self._current_plan)
                common_js_names: list[str] = [
                    x for x in self.robot.data.joint_names if x in self._current_plan.joint_names
                ]
                self._current_plan = self._current_plan.get_ordered_joint_state(common_js_names)
            else:
                self._current_plan = None
                print(f"Plan failed: {result.status}")
        else:
            success: bool = self._plan_to_contact(
                start_state=start_state,
                goal_pose=target_curobo_pose,
                retreat_distance=self.config.retreat_distance,
                approach_distance=self.config.approach_distance,
                retime_plan=enable_retiming,
                step_size=step_size,
                contact=False,
            )

        # Visualize plan if enabled (ensure cuRobo kinematic joint ordering for FK)
        if success and self.visualize_plan and self._current_plan is not None:
            try:
                viz_plan = self._current_plan.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
            except Exception:
                viz_plan = self._current_plan
            # Get current spheres for visualization
            self._sync_object_poses_with_isaaclab()
            cu_js = self._get_current_joint_state_for_curobo()
            sphere_list = self.motion_gen.kinematics.get_robot_as_spheres(cu_js.position)[0]

            # Split spheres into robot and attached object spheres
            robot_spheres = []
            attached_spheres = []
            robot_link_count = 0

            # Count robot link spheres
            collision_link_names = self.robot_cfg["kinematics"]["collision_link_names"]
            attached_link = self.config.attached_object_link_name
            is_dedicated_attachment_link = attached_link and (
                attached_link.startswith("attached_object") or attached_link not in collision_link_names
            )

            if is_dedicated_attachment_link:
                robot_links = [link for link in collision_link_names if link != attached_link]
            else:
                robot_links = list(collision_link_names)

            for link_name in robot_links:
                link_spheres = self.motion_gen.kinematics.kinematics_config.get_link_spheres(link_name)
                if link_spheres is not None:
                    robot_link_count += int(torch.sum(link_spheres[:, 3] > 0).item())

            # Split spheres
            for i, sphere in enumerate(sphere_list):
                if i < robot_link_count:
                    robot_spheres.append(sphere)
                else:
                    attached_spheres.append(sphere)

            # Compute end-effector positions for visualization
            ee_positions_list = []
            try:
                for i in range(len(viz_plan.position)):
                    js: JointState = viz_plan[i]
                    kin = self.motion_gen.compute_kinematics(js)
                    ee_pos = kin.ee_position if hasattr(kin, "ee_position") else kin.ee_pose.position
                    ee_positions_list.append(ee_pos.cpu().numpy().squeeze())

                self.logger.debug(
                    f"Link names from kinematics: {kin.link_names if len(ee_positions_list) > 0 else 'No EE positions'}"
                )

            except Exception as e:
                self.logger.debug(f"Failed to compute EE positions for visualization: {e}")
                ee_positions_list = None

            try:
                world_scene = WorldConfig.get_scene_graph(self.motion_gen.world_coll_checker.world_model)
            except Exception:
                world_scene = None

            # Visualize plan
            self.plan_visualizer.visualize_plan(
                plan=viz_plan,
                target_pose=target_pose,
                robot_spheres=robot_spheres,
                attached_spheres=attached_spheres,
                ee_positions=np.array(ee_positions_list) if ee_positions_list else None,
                world_scene=world_scene,
            )

            # Animate EE positions over the timeline for playback
            if ee_positions_list:
                self.plan_visualizer.animate_plan(np.array(ee_positions_list))

            # Animate spheres along the path for collision visualization
            self.plan_visualizer.animate_spheres_along_path(
                plan=viz_plan,
                robot_spheres_at_start=robot_spheres,
                attached_spheres_at_start=attached_spheres,
                timeline="sphere_animation",
                interpolation_steps=15,  # More steps for smoother animation
            )

        return success

    def _plan_to_contact_pose(
        self,
        start_state: JointState,
        goal_pose: Pose,
        contact: bool = True,
        *,
        link_poses: dict[str, Pose] | None = None,
    ) -> bool:
        """Plan motion with configurable collision checking for contact scenarios.

        Plans a trajectory while optionally disabling collision checking for hand links and
        attached objects. This is crucial for grasping and placing operations where contact
        is expected and collision checking would prevent successful planning.

        Args:
            start_state: Starting joint configuration for planning
            goal_pose: Target pose to reach in cuRobo coordinate frame
            contact: True to disable hand/attached object collisions for contact planning
            retime_plan: Whether to apply linear retiming to the resulting trajectory
            step_size: Step size for retiming if retime_plan is True

        Returns:
            True if planning succeeded, False if no valid trajectory found
        """
        # Use configured hand link names instead of hardcoded ones
        disable_link_names: list[str] = self.config.hand_link_names.copy()
        link_spheres: dict[str, torch.Tensor] = {}

        # Count spheres before planning
        sphere_counts_before = self._count_active_spheres()
        self.logger.debug(
            f"Planning phase contact={contact}: Spheres before - Total: {sphere_counts_before['total']}, Robot:"
            f" {sphere_counts_before['robot_links']}, Attached: {sphere_counts_before['attached_objects']}"
        )

        if contact:
            # Store current spheres for the attached link so we can restore later
            attached_links: list[str] = list(self.attachment_links)
            for attached_link in attached_links:
                link_spheres[attached_link] = self.motion_gen.kinematics.kinematics_config.get_link_spheres(
                    attached_link
                ).clone()

            self.logger.debug(f"Attached link: {attached_links}")
            # Disable all specified links for contact planning
            self.logger.debug(f"Disable link names: {disable_link_names}")
            self._set_active_links(disable_link_names + attached_links, active=False)
        else:
            self.logger.debug(f"Disable link names: {disable_link_names}")

        # Count spheres after link disabling
        sphere_counts_after_disable = self._count_active_spheres()
        self.logger.debug(
            f"Planning phase contact={contact}: Spheres after disable - Total:"
            f" {sphere_counts_after_disable['total']}, Robot: {sphere_counts_after_disable['robot_links']},"
            f" Attached: {sphere_counts_after_disable['attached_objects']}"
        )

        planning_success = False
        try:
            # Convert dict link poses to list aligned with kinematics link order if provided
            link_pose_list: list[Pose] | None = None
            if link_poses is not None and len(link_poses) > 0:
                names = list(self.motion_gen.kinematics.link_names)
                link_pose_list = []
                for name in names:
                    if name in link_poses:
                        link_pose_list.append(link_poses[name])
                    else:
                        link_pose_list.append(None)
            result: Any = self.motion_gen.plan_single(
                start_state,
                goal_pose,
                self.plan_config,
                link_poses=cast(list[Pose], link_pose_list) if link_pose_list is not None else None,
            )

            if result.success.item():
                if result.optimized_plan is not None and len(result.optimized_plan.position) != 0:
                    self._current_plan = result.optimized_plan
                    self.logger.debug(f"Using optimized plan with {len(self._current_plan.position)} waypoints")
                else:
                    self._current_plan = result.get_interpolated_plan()
                    self.logger.debug(f"Using interpolated plan with {len(self._current_plan.position)} waypoints")

                self._current_plan = self.motion_gen.get_full_js(self._current_plan)
                common_js_names: list[str] = [
                    x for x in self.robot.data.joint_names if x in self._current_plan.joint_names
                ]
                self._current_plan = self._current_plan.get_ordered_joint_state(common_js_names)
                self._plan_index = 0

                planning_success = True
                self.logger.debug(f"Contact planning succeeded with {len(self._current_plan.position)} waypoints")
            else:
                self.logger.debug(f"Contact planning failed: {result.status}")

        except Exception as e:
            self.logger.debug(f"Error during planning: {e}")

        # Always restore sphere state after planning, regardless of success
        if contact:
            self._set_active_links(disable_link_names, active=True)
            for attached_link, spheres in link_spheres.items():
                self.motion_gen.kinematics.kinematics_config.update_link_spheres(attached_link, spheres)
        return planning_success

    def _plan_to_contact(
        self,
        start_state: JointState,
        goal_pose: Pose,
        retreat_distance: float,
        approach_distance: float,
        contact: bool = False,
        retime_plan: bool = False,
        step_size: float | None = None,
        *,
        link_poses: dict[str, Pose] | None = None,
    ) -> bool:
        """Execute multi-phase contact planning with approach and retreat phases.

        Implements a planning strategy for manipulation tasks that require approach and contact handling.
        Plans multiple trajectory segments with different collision checking configurations.

        Args:
            start_state: Starting joint state for planning
            goal_pose: Target pose to reach
            retreat_distance: Distance to retreat before transition to contact
            approach_distance: Distance to approach before final pose
            contact: Whether to enable contact planning mode
            retime_plan: Whether to retime the resulting plan
            step_size: Step size for retiming (only used if retime_plan is True)

        Returns:
            True if all planning phases succeeded, False if any phase failed
        """
        self.logger.debug(f"Multi-phase planning: retreat={retreat_distance}, approach={approach_distance}")

        target_poses: list[Pose] = []
        contacts: list[bool] = []
        approach_dir = self.config.approach_direction
        in_world_frame = self.config.approach_retreat_frame == "world"

        # In EEF frame: pose * translate(dir * d) moves pose along EE's dir. For Franka (0,0,-1) = up.
        # In world frame we negate so the same config (0,0,-1) gives retreat up, approach from above.
        world_sign = -1.0 if in_world_frame else 1.0

        if retreat_distance is not None and retreat_distance > 0:
            ee_pose: Pose = self.get_ee_pose(start_state)
            if in_world_frame:
                pos = ee_pose.position.squeeze()
                offset = torch.tensor(
                    [world_sign * d * retreat_distance for d in approach_dir],
                    dtype=pos.dtype,
                    device=pos.device,
                )
                retreat_pose = self._make_pose(
                    position=pos + offset,
                    quaternion=ee_pose.quaternion,
                )
            else:
                retreat_offset = [d * retreat_distance for d in approach_dir]
                retreat_pose = ee_pose.multiply(
                    self._make_pose(position=retreat_offset)
                )
            target_poses.append(retreat_pose)
            contacts.append(True)
        contacts.append(contact)
        if approach_distance is not None and approach_distance > 0:
            if in_world_frame:
                pos = goal_pose.position.squeeze()
                offset = torch.tensor(
                    [world_sign * d * approach_distance for d in approach_dir],
                    dtype=pos.dtype,
                    device=pos.device,
                )
                approach_pose = self._make_pose(
                    position=pos + offset,
                    quaternion=goal_pose.quaternion,
                )
            else:
                approach_offset = [d * approach_distance for d in approach_dir]
                approach_pose = goal_pose.multiply(
                    self._make_pose(position=approach_offset)
                )
            target_poses.append(approach_pose)
            contacts.append(True)

        target_poses.append(goal_pose)

        current_state: JointState = start_state
        full_plan: JointState | None = None

        for i, (target_pose, contact_flag) in enumerate(zip(target_poses, contacts)):
            self.logger.debug(
                f"Planning phase {i + 1} of {len(target_poses)}: contact={contact_flag} (collision"
                f" {'disabled' if contact_flag else 'enabled'})"
            )

            success: bool = self._plan_to_contact_pose(
                start_state=current_state,
                goal_pose=target_pose,
                contact=contact_flag,
                link_poses=link_poses,
            )

            if not success:
                self.logger.debug(f"Phase {i + 1} planning failed")
                return False

            if full_plan is None:
                full_plan = self._current_plan
            else:
                full_plan = full_plan.stack(self._current_plan)

            last_waypoint: torch.Tensor = self._current_plan.position[-1]
            current_state = JointState(
                position=last_waypoint.unsqueeze(0),
                velocity=torch.zeros_like(last_waypoint.unsqueeze(0)),
                acceleration=torch.zeros_like(last_waypoint.unsqueeze(0)),
                joint_names=self._current_plan.joint_names,
            )
            current_state = current_state.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        self._current_plan = full_plan
        self._plan_index = 0

        if retime_plan and step_size is not None:
            original_length: int = len(self._current_plan.position)
            self._current_plan = self._linearly_retime_plan(step_size=step_size, plan=self._current_plan)
            self.logger.debug(
                f"Retimed complete plan from {original_length} to {len(self._current_plan.position)} waypoints"
            )

        self.logger.debug(f"Multi-phase planning succeeded with {len(self._current_plan.position)} total waypoints")

        return True

    def _linearly_retime_plan(
        self,
        step_size: float = 0.01,
        plan: JointState | None = None,
    ) -> JointState | None:
        """Apply linear retiming to trajectory for consistent execution speed.

        Resamples the trajectory with uniform spacing between waypoints to ensure
        consistent motion speed during execution.

        Args:
            step_size: Desired spacing between waypoints in joint space
            plan: Trajectory to retime, uses current plan if None

        Returns:
            Retimed trajectory with uniform waypoint spacing, or None if plan is invalid
        """
        if plan is None:
            plan = self._current_plan

        if plan is None or len(plan.position) == 0:
            return plan

        path = plan.position

        if len(path) <= 1:
            return plan

        deltas = path[1:] - path[:-1]
        distances = torch.norm(deltas, dim=-1)

        waypoints = [path[0]]
        for distance, waypoint in zip(distances, path[1:]):
            if distance > 1e-6:
                waypoints.append(waypoint)

        if len(waypoints) <= 1:
            return plan

        waypoints = torch.stack(waypoints)

        if len(waypoints) > 1:
            deltas = waypoints[1:] - waypoints[:-1]
            distances = torch.norm(deltas, dim=-1)
            cum_distances = torch.cat([torch.zeros(1, device=distances.device), torch.cumsum(distances, dim=0)])

        if len(waypoints) < 2 or cum_distances[-1] < 1e-6:
            return plan

        total_distance = cum_distances[-1]
        num_steps = int(torch.ceil(total_distance / step_size).item()) + 1

        # Create linearly spaced distances
        sampled_distances = torch.linspace(cum_distances[0], cum_distances[-1], num_steps, device=cum_distances.device)

        # Linear interpolation
        indices = torch.searchsorted(cum_distances, sampled_distances)
        indices = torch.clamp(indices, 1, len(cum_distances) - 1)

        # Get interpolation weights
        weights = (sampled_distances - cum_distances[indices - 1]) / (
            cum_distances[indices] - cum_distances[indices - 1]
        )
        weights = weights.unsqueeze(-1)

        # Interpolate waypoints
        sampled_waypoints = (1 - weights) * waypoints[indices - 1] + weights * waypoints[indices]

        self.logger.debug(
            f"Retiming: {len(path)} to {len(sampled_waypoints)} waypoints, "
            f"Distance: {total_distance:.3f}, Step size: {step_size}"
        )

        retimed_plan = JointState(
            position=sampled_waypoints,
            velocity=torch.zeros(
                (len(sampled_waypoints), plan.velocity.shape[-1]),
                device=plan.velocity.device,
                dtype=plan.velocity.dtype,
            ),
            acceleration=torch.zeros(
                (len(sampled_waypoints), plan.acceleration.shape[-1]),
                device=plan.acceleration.device,
                dtype=plan.acceleration.dtype,
            ),
            joint_names=plan.joint_names,
        )

        return retimed_plan

    def has_next_waypoint(self) -> bool:
        """Check if more waypoints remain in the current trajectory.

        Returns:
            True if there are unprocessed waypoints, False if trajectory is complete or empty
        """
        return self._current_plan is not None and self._plan_index < len(self._current_plan.position)

    def get_next_waypoint_ee_pose(self) -> Pose:
        """Get end-effector pose for the next waypoint in the trajectory.

        Advances the trajectory execution index and computes the end-effector pose
        for the next waypoint using forward kinematics.

        Returns:
            End-effector pose for the next waypoint in world coordinates

        Raises:
            IndexError: If no more waypoints remain in the trajectory
        """
        if not self.has_next_waypoint():
            raise IndexError("No more waypoints in the plan.")
        next_joint_state: JointState = self._current_plan[self._plan_index]
        self._plan_index += 1
        eef_state: CudaRobotModelState = self.motion_gen.compute_kinematics(next_joint_state)
        return eef_state.ee_pose

    def reset_plan(self) -> None:
        """Reset trajectory execution state.

        Clears the current trajectory and resets the execution index to zero.
        This prepares the planner for a new planning operation.
        """
        self._plan_index = 0
        self._current_plan = None
        if self.visualize_plan and hasattr(self, "plan_visualizer"):
            self.plan_visualizer.clear_visualization()
            self.plan_visualizer.mark_idle()

    def get_planned_poses(self) -> list[torch.Tensor]:
        """Extract all end-effector poses from current trajectory.

        Computes end-effector poses for all waypoints in the current trajectory without
        affecting the execution state. Optionally repeats the final pose multiple times
        if configured for stable goal reaching.

        Returns:
            List of end-effector poses as 4x4 transformation matrices, with optional repetition
        """
        if self._current_plan is None:
            return []

        # Save current execution state
        original_plan_index = self._plan_index

        # Iterate through the plan to get all poses
        planned_poses: list[torch.Tensor] = []
        self._plan_index = 0
        while self.has_next_waypoint():
            # Directly use the joint state from the plan to compute pose
            # without advancing the main plan index in get_next_waypoint_ee_pose
            next_joint_state: JointState = self._current_plan[self._plan_index]
            self._plan_index += 1  # Manually advance index for this loop
            eef_state: Any = self.motion_gen.compute_kinematics(next_joint_state)
            planned_pose: Pose | None = eef_state.ee_pose

            if planned_pose is not None:
                # Convert pose to environment device for compatibility
                position = (
                    self._to_env_device(planned_pose.position)
                    if isinstance(planned_pose.position, torch.Tensor)
                    else planned_pose.position
                )
                rotation = (
                    self._to_env_device(planned_pose.get_rotation())
                    if isinstance(planned_pose.get_rotation(), torch.Tensor)
                    else planned_pose.get_rotation()
                )
                planned_poses.append(PoseUtils.make_pose(position, rotation)[0])

        # Restore the original execution state
        self._plan_index = original_plan_index

        if self.n_repeat is not None and self.n_repeat > 0 and len(planned_poses) > 0:
            self.logger.info(f"Repeating final pose {self.n_repeat} times")
            final_pose: torch.Tensor = planned_poses[-1]
            planned_poses.extend([final_pose] * self.n_repeat)

        return planned_poses

    # =====================================================================================
    # VISUALIZATION METHODS
    # =====================================================================================

    def _update_visualization_at_joint_positions(self, joint_positions: torch.Tensor) -> None:
        """Update sphere visualization for the robot at specific joint positions.

        Args:
            joint_positions: Joint configuration to visualize collision spheres at
        """
        if not self.visualize_spheres:
            return

        self.frame_counter += 1
        if self.frame_counter % self.sphere_update_freq != 0:
            return

        original_joints: torch.Tensor = self.robot.data.joint_pos[self.env_id].clone()

        try:
            # Ensure joint positions are on environment device for robot commands
            env_joint_positions = (
                self._to_env_device(joint_positions) if joint_positions.device != self.env.device else joint_positions
            )
            self.robot.set_joint_position_target(env_joint_positions.view(1, -1), env_ids=[self.env_id])
            self._update_sphere_visualization(force_update=False)
        finally:
            self.robot.set_joint_position_target(original_joints.unsqueeze(0), env_ids=[self.env_id])

    def _update_sphere_visualization(self, force_update: bool = True) -> None:
        """Update visual representation of robot collision spheres in USD stage.

        Creates or updates sphere primitives in the USD stage to show the robot's
        collision model. Different colors are used for robot links (green) and
        attached objects (orange) to help distinguish collision boundaries.

        Args:
            force_update: True to recreate all spheres, False to update existing positions only
        """
        # Get current sphere data
        cu_js = self._get_current_joint_state_for_curobo()
        sphere_position = self._to_curobo_device(
            cu_js.position if isinstance(cu_js.position, torch.Tensor) else torch.tensor(cu_js.position)
        )
        sphere_list = self.motion_gen.kinematics.get_robot_as_spheres(sphere_position)[0]
        robot_link_count = self._get_robot_link_sphere_count()

        # Remove existing spheres if force update or first time
        if (self.spheres is None or force_update) and self.spheres is not None:
            self._remove_existing_spheres()

        # Initialize sphere list if needed
        if self.spheres is None or force_update:
            self.spheres = []

        # Create or update all spheres
        for sphere_idx, sphere in enumerate(sphere_list):
            if not self._is_valid_sphere(sphere):
                continue

            sphere_config = self._create_sphere_config(sphere_idx, sphere, robot_link_count)
            prim_path = f"/curobo/robot_sphere_{sphere_idx}"

            # Remove old sphere if updating
            if not (self.spheres is None or force_update):
                if sphere_idx < len(self.spheres) and self.usd_helper.stage.GetPrimAtPath(prim_path).IsValid():
                    self.usd_helper.stage.RemovePrim(prim_path)

            # Spawn sphere
            spawn_mesh_sphere(prim_path=prim_path, translation=sphere_config["position"], cfg=sphere_config["cfg"])

            # Store reference if creating new
            if self.spheres is None or force_update or sphere_idx >= len(self.spheres):
                self.spheres.append((prim_path, float(sphere.radius)))

    def _get_robot_link_sphere_count(self) -> int:
        """Calculate total number of collision spheres for robot links.

        Counts active collision spheres for all robot links. Only excludes the attachment
        link if it's a DEDICATED attachment link (not in collision_link_names). For humanoid
        robots where the EE link is used for attachment, all spheres are counted as robot
        spheres since the EE link is a real kinematic link.

        Returns:
            Total number of active collision spheres for robot links
        """
        sphere_config = self.motion_gen.kinematics.kinematics_config
        collision_link_names = self.robot_cfg["kinematics"]["collision_link_names"]
        attached_link = self.config.attached_object_link_name

        # Dedicated attachment link (Franka-style) should be excluded even if present in the list.
        # Humanoid uses a real EE link for attachment, so we keep it in the robot links.
        is_dedicated_attachment_link = attached_link and (
            attached_link.startswith("attached_object") or attached_link not in collision_link_names
        )

        if is_dedicated_attachment_link:
            robot_links = [link for link in collision_link_names if link != attached_link]
        else:
            robot_links = list(collision_link_names)

        return sum(
            int(torch.sum(sphere_config.get_link_spheres(link_name)[:, 3] > 0).item()) for link_name in robot_links
        )

    def _remove_existing_spheres(self) -> None:
        """Remove all existing sphere visualization primitives from the USD stage.

        Iterates through all stored sphere references and removes their corresponding
        USD primitives from the stage. This is used during force updates or when
        recreating the sphere visualization from scratch.
        """
        stage = self.usd_helper.stage
        for prim_path, _ in self.spheres:
            if stage.GetPrimAtPath(prim_path).IsValid():
                stage.RemovePrim(prim_path)

    def _is_valid_sphere(self, sphere) -> bool:
        """Validate sphere data for visualization rendering.

        Checks if a sphere has valid position coordinates (no NaN values) and a positive
        radius. Invalid spheres are skipped during visualization to prevent rendering errors.

        Args:
            sphere: Sphere object containing position and radius data

        Returns:
            True if sphere has valid position and positive radius, False otherwise
        """
        pos_tensor = torch.tensor(sphere.position, dtype=torch.float32)
        return not torch.isnan(pos_tensor).any() and sphere.radius > 0

    def _create_sphere_config(self, sphere_idx: int, sphere, robot_link_count: int) -> dict:
        """Create sphere configuration with position and visual properties for USD rendering.

        Determines sphere type (robot link vs attached object), calculates world position,
        and creates the appropriate visual configuration including colors and materials.
        Robot link spheres are green with lower opacity, while attached object spheres
        are orange with higher opacity for better distinction.

        Args:
            sphere_idx: Index of the sphere in the sphere list
            sphere: Sphere object containing position and radius data
            robot_link_count: Total number of robot link spheres (for type determination)

        Returns:
            Dictionary containing 'position' (world coordinates) and 'cfg' (MeshSphereCfg)
        """

        is_attached = sphere_idx >= robot_link_count
        color = (1.0, 0.5, 0.0) if is_attached else (0.0, 1.0, 0.0)
        opacity = 0.9 if is_attached else 0.5

        # Calculate position in world frame (do not use env_origin)
        root_translation = (self.robot.data.root_pos_w[self.env_id, :3]).detach().cpu().numpy()
        position = sphere.position.cpu().numpy() if hasattr(sphere.position, "cpu") else sphere.position
        if not is_attached:
            position = position + root_translation

        return {
            "position": position,
            "cfg": MeshSphereCfg(
                radius=float(sphere.radius),
                visual_material=PreviewSurfaceCfg(diffuse_color=color, opacity=opacity, emissive_color=color),
            ),
        }

    def _is_sphere_attached_object(self, sphere_index: int, sphere_config: Any) -> bool:
        """Check if a sphere belongs to attached_object link.

        Args:
            sphere_index: Index of the sphere to check
            sphere_config: Sphere configuration object

        Returns:
            True if sphere belongs to an attached object, False if it's a robot link sphere
        """
        # Get total number of robot link spheres
        collision_link_names = self.robot_cfg["kinematics"]["collision_link_names"]
        attached_link = self.config.attached_object_link_name
        # Dedicated attachment link (Franka-style) should be excluded even if present in the list.
        # Humanoid uses a real EE link for attachment, so we keep it in the robot links.
        is_dedicated_attachment_link = attached_link and (
            attached_link.startswith("attached_object") or attached_link not in collision_link_names
        )

        if is_dedicated_attachment_link:
            robot_links = [link for link in collision_link_names if link != attached_link]
        else:
            robot_links = list(collision_link_names)

        total_robot_spheres = 0
        for link_name in robot_links:
            try:
                link_spheres = sphere_config.get_link_spheres(link_name)
                active_spheres = torch.sum(link_spheres[:, 3] > 0).item()
                total_robot_spheres += int(active_spheres)
            except Exception:
                continue

        # If sphere_index >= total_robot_spheres, it's an attached object sphere
        is_attached = sphere_index >= total_robot_spheres

        if sphere_index < 5:  # Debug first few spheres
            self.logger.debug(
                f"SPHERE {sphere_index}: total_robot_spheres={total_robot_spheres}, is_attached={is_attached}"
            )

        return is_attached

    # =====================================================================================
    # HIGH-LEVEL PLANNING INTERFACE
    # =====================================================================================

    def _reset_shared_attachment_state(self) -> None:
        """Ensure the shared MotionGen has a clean attachment state before planning.

        When multiple envs share one MotionGen, the previous env's planning may have
        left attached-object collision spheres on the kinematics model. This method
        unconditionally detaches everything so the current env can set up its own
        attachment state from scratch.

        Only called when ``self._is_shared`` is True.
        """
        configured_link = self.config.attached_object_link_name
        link_idx_map = getattr(
            self.motion_gen.kinematics.kinematics_config, "link_name_to_idx_map", {}
        )
        if configured_link in link_idx_map:
            try:
                self.motion_gen.detach_object_from_robot(link_name=configured_link)
            except ValueError:
                pass

        # Re-enable all obstacles that may have been disabled by attachment
        object_mappings = self._get_object_mappings()
        for _, object_path in object_mappings.items():
            try:
                self.motion_gen.world_coll_checker.enable_obstacle(object_path, enable=True)
            except Exception:
                pass

    def update_world_and_plan_motion(
        self,
        target_pose: torch.Tensor,
        expected_attached_object: str | None = None,
        env_id: int = 0,
        step_size: float | None = None,
        enable_retiming: bool | None = None,
        link_target_poses_base: dict[str, torch.Tensor] | None = None,
        skip_world_update: bool = False,
        start_joint_state: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> bool:
        """Complete planning pipeline with world updates and object attachment handling.

        Provides a high-level interface that handles the complete planning workflow:
        world synchronization, object attachment/detachment, gripper configuration,
        and motion planning.

        Args:
            target_pose: Target end-effector pose as 4x4 transformation matrix
            expected_attached_object: Name of object that should be attached, None for no attachment
            env_id: Environment ID for multi-environment setups
            step_size: Step size for linear retiming if retiming is enabled
            enable_retiming: Whether to enable linear retiming of trajectory
            skip_world_update: If True, skip world synchronization (for offline planning where
                object poses are pre-set in the collision world)
            start_joint_state: Optional start joint state tensor in cuRobo planner ordering.
                If provided, bypasses reading from articulation buffer (for offline planning).

        Returns:
            True if complete planning pipeline succeeded, False if any step failed
        """
        # Always reset the plan before starting a new one to ensure a clean state
        self.reset_plan()

        # For shared MotionGen, clean up attachment state left by previous env
        if self._is_shared:
            self._reset_shared_attachment_state()

        self.logger.debug("=== MOTION PLANNING DEBUG ===")
        self.logger.debug(f"Expected attached object: {expected_attached_object}")

        if not skip_world_update:
            self.update_world()
        else:
            self.logger.debug("Skipping world update (offline mode)")
        gripper_closed = expected_attached_object is not None
        self._set_gripper_state(gripper_closed)
        current_attached = self.get_attached_objects()
        gripper_pos = self.robot.data.joint_pos[env_id, -2:]

        self.logger.debug(f"Current attached objects: {current_attached}")

        # Attach object if expected but not currently attached
        if expected_attached_object and expected_attached_object not in current_attached:
            self.logger.debug(f"Need to attach {expected_attached_object}")

            object_mappings = self._get_object_mappings()

            self.logger.debug(f"Object mappings found: {list(object_mappings.keys())}")

            if expected_attached_object in object_mappings:
                expected_path = object_mappings[expected_attached_object]

                self.logger.debug(f"Object path: {expected_path}")

                # Debug object poses
                rigid_objects = self.env.scene.rigid_objects
                if expected_attached_object in rigid_objects:
                    obj = rigid_objects[expected_attached_object]
                    origin = self.env.scene.env_origins[env_id]
                    obj_pos = obj.data.root_pos_w[env_id] - origin
                    self.logger.debug(f"Isaac Lab object position: {obj_pos}")

                    # Debug end-effector position (handle missing ee_frame gracefully)
                    # Note: InteractiveScene doesn't implement __contains__, use keys() instead
                    try:
                        scene_keys = set(self.env.scene.keys()) if hasattr(self.env.scene, "keys") else set()
                        if "ee_frame" in scene_keys:
                            ee_frame = self.env.scene["ee_frame"]
                            ee_pos = ee_frame.data.target_pos_w[env_id, 0, :] - origin
                            self.logger.debug(f"End-effector position: {ee_pos}")
                            distance = torch.linalg.vector_norm(obj_pos - ee_pos).item()
                            self.logger.debug(f"Distance EE to object: {distance:.4f}")
                    except (KeyError, AttributeError):
                        self.logger.debug("ee_frame not available in scene (bimanual robot)")

                    # Debug gripper state
                    gripper_open_val = self.config.grasp_gripper_open_val
                    self.logger.debug(f"Gripper positions: {gripper_pos}")
                    self.logger.debug(f"Gripper open val: {gripper_open_val}")

                is_grasped = self._check_object_grasped(gripper_pos, expected_attached_object)

                self.logger.debug(f"Is grasped check result: {is_grasped}")

                if is_grasped:
                    self._attach_object(expected_attached_object, expected_path, env_id)
                    self.logger.debug(f"Attached {expected_attached_object}")
                else:
                    self.logger.debug(
                        "Object not detected as grasped - attachment skipped"
                    )  # This will cause collision with ghost object!
            else:
                self.logger.debug(f"Object {expected_attached_object} not found in world mappings")

        # Detach objects if no object should be attached (i.e., placing/releasing)
        # But skip if in offline mode (skip_world_update=True) - attachments are managed externally
        if expected_attached_object is None and current_attached and not skip_world_update:
            self.logger.debug("Detaching all objects as no object expected to be attached")
            self._detach_objects()
        elif expected_attached_object is None and current_attached and skip_world_update:
            self.logger.debug("Preserving attachment in offline mode (skip_world_update=True)")

        self.logger.debug(f"Planning motion with attached objects: {self.get_attached_objects()}")

        plan_success = self.plan_motion(
            target_pose,
            step_size,
            enable_retiming,
            link_target_poses_base=link_target_poses_base,
            start_joint_state=start_joint_state,
        )

        self.logger.debug(f"Planning result: {plan_success}")
        self.logger.debug("=== END POST-GRASP DEBUG ===")

        # Only detach in online mode (not skip_world_update)
        # In offline mode (skip_world_update=True), attachments are managed by the data generator
        # and should persist across multiple planning calls
        if not skip_world_update and self.has_attached_objects():
            self._detach_objects()

        return plan_success

    # =====================================================================================
    # UTILITY METHODS
    # =====================================================================================

    def _check_object_grasped(self, gripper_pos: torch.Tensor, object_name: str) -> bool:
        """Check if a specific object is currently grasped by the robot.

        Supports multiple detection modes configured via config.grasp_detection_mode:
        - 'gripper': Uses gripper joint position (for parallel jaw grippers like Franka)
        - 'distance': Uses 3D object-to-EE distance (simple dexterous check)
        - 'dexterous': Uses finger joint positions + XY distance (for cylindrical objects)
        - 'callback': Uses environment's check_object_grasped method if available

        Args:
            gripper_pos: Gripper position tensor (used in 'gripper' mode)
            object_name: Name of object to check (e.g., "cube_1", "FactoryNut")

        Returns:
            True if object is detected as grasped
        """
        detection_mode = getattr(self.config, "grasp_detection_mode", "gripper")
        object_grasped = False

        if detection_mode == "callback":
            # Use environment callback if available
            object_grasped = self._check_object_grasped_callback(object_name)

        elif detection_mode == "distance":
            # Simple 3D distance-based detection
            object_grasped = self._check_object_grasped_by_distance(object_name)

        elif detection_mode == "dexterous":
            # Dexterous hand: finger joints + XY distance (for cylindrical objects)
            object_grasped = self._check_object_grasped_dexterous(object_name)

        else:
            # Default: gripper-based detection for parallel jaw grippers
            object_grasped = self._check_object_grasped_by_gripper(gripper_pos, object_name)

        self.logger.info(
            f"Object {object_name} grasp check ({detection_mode}): {object_grasped}"
        )
        return object_grasped

    def _check_object_grasped_by_gripper(self, gripper_pos: torch.Tensor, object_name: str) -> bool:
        """Check grasp using parallel jaw gripper position.

        Args:
            gripper_pos: Gripper joint positions tensor
            object_name: Name of object to check

        Returns:
            True if gripper is closed (below open threshold)
        """
        gripper_open_val = self.config.grasp_gripper_open_val
        return gripper_pos[0].item() < gripper_open_val

    def _check_object_grasped_by_distance(self, object_name: str) -> bool:
        """Check grasp using 3D distance between object and end-effector.

        Simple distance check - suitable for small objects where center distance is meaningful.

        Args:
            object_name: Name of object to check

        Returns:
            True if object is within grasp_distance_threshold of EE
        """
        threshold = getattr(self.config, "grasp_distance_threshold", 0.08)
        ee_frame_name = getattr(self.config, "grasp_ee_frame_name", None) or "ee_frame"

        # Get object position
        rigid_objects = self.env.scene.rigid_objects
        if object_name not in rigid_objects:
            self.logger.warning(f"Object {object_name} not found in scene rigid objects")
            return False

        obj = rigid_objects[object_name]
        origin = self.env.scene.env_origins[self.env_id]
        obj_pos = obj.data.root_pos_w[self.env_id] - origin

        # Get EE position - try multiple approaches
        ee_pos = self._get_ee_position_for_grasp_check(ee_frame_name)
        if ee_pos is None:
            self.logger.warning("Could not get EE position for grasp check")
            return False

        # Compute 3D distance
        distance = torch.linalg.vector_norm(obj_pos - ee_pos).item()
        self.logger.debug(f"Object {object_name} distance to EE: {distance:.4f}m (threshold: {threshold})")

        return distance < threshold

    def _check_object_grasped_dexterous(self, object_name: str) -> bool:
        """Check grasp for dexterous hands using finger joints + XY distance.

        For cylindrical objects like beakers that can be grasped at any height,
        this method checks:
        1. Finger joints are closed (bent past threshold)
        2. Object is within XY distance of EE (ignoring Z for height variance)

        The XY threshold is intentionally generous because the EE frame is typically
        at the wrist/palm, while actual grasp contact is at the fingertips (10-15cm away).

        Args:
            object_name: Name of object to check

        Returns:
            True if fingers are closed AND object is within XY distance threshold
        """
        xy_threshold = getattr(self.config, "grasp_xy_distance_threshold", 0.15)  # Default 15cm for palm-to-fingertip
        finger_threshold = getattr(self.config, "dexterous_finger_closed_threshold", 0.3)
        ee_frame_name = getattr(self.config, "grasp_ee_frame_name", None) or "ee_frame"

        print(f"\n=== DEXTEROUS GRASP CHECK: {object_name} ===")
        print(f"  Thresholds: XY={xy_threshold:.3f}m, finger={finger_threshold:.3f}rad")

        # Check finger joint positions - returns (is_closed, closed_ratio)
        fingers_closed, closed_ratio = self._check_dexterous_fingers_closed(finger_threshold)
        if not fingers_closed:
            print("  RESULT: NOT GRASPED (fingers not closed)")
            return False

        # Get object position
        rigid_objects = self.env.scene.rigid_objects
        if object_name not in rigid_objects:
            self.logger.warning(f"Object {object_name} not found in scene rigid objects")
            return False

        obj = rigid_objects[object_name]
        origin = self.env.scene.env_origins[self.env_id]
        obj_pos = obj.data.root_pos_w[self.env_id] - origin

        # Get EE position
        ee_pos = self._get_ee_position_for_grasp_check(ee_frame_name)
        if ee_pos is None:
            self.logger.warning("Could not get EE position for grasp check")
            return False

        # Compute distances
        xy_diff = obj_pos[:2] - ee_pos[:2]
        xy_distance = torch.linalg.vector_norm(xy_diff).item()
        z_diff = abs(obj_pos[2].item() - ee_pos[2].item())
        full_3d_distance = torch.linalg.vector_norm(obj_pos - ee_pos).item()

        print(f"  Object pos (env frame): [{obj_pos[0].item():.3f}, {obj_pos[1].item():.3f}, {obj_pos[2].item():.3f}]")
        print(f"  EE pos (env frame):     [{ee_pos[0].item():.3f}, {ee_pos[1].item():.3f}, {ee_pos[2].item():.3f}]")
        print(f"  XY distance: {xy_distance:.4f}m (threshold: {xy_threshold}m) {'PASS' if xy_distance < xy_threshold else 'FAIL'}")
        print(f"  Z difference: {z_diff:.4f}m (ignored for cylindrical objects)")
        print(f"  Full 3D distance: {full_3d_distance:.4f}m")

        # If all/most fingers are strongly closed, be more lenient with distance
        # This handles cases where EE frame is at wrist but fingers are clearly wrapped around object
        effective_threshold = xy_threshold
        if closed_ratio >= 0.8:  # 80%+ fingers closed
            effective_threshold = xy_threshold * 1.5  # 50% more lenient
            print(f"  [Strong finger closure ({closed_ratio:.0%})] Using relaxed threshold: {effective_threshold:.3f}m")

        is_grasped = xy_distance < effective_threshold
        print(f"  RESULT: {'GRASPED' if is_grasped else 'NOT GRASPED'}")
        print("=" * 50)

        return is_grasped

    def _check_dexterous_fingers_closed(self, threshold: float) -> tuple[bool, float]:
        """Check if dexterous hand fingers are in closed/grasping position.

        Checks if key finger joints are bent past the threshold, indicating
        the hand is in a grasping configuration. Uses ABSOLUTE VALUE of joint
        position since some robots (like GR1T2) use negative values for closed.

        Args:
            threshold: Joint position threshold (radians) for considering closed

        Returns:
            Tuple of (is_closed, closed_ratio) where:
            - is_closed: True if sufficient fingers are closed
            - closed_ratio: Fraction of fingers that are closed (0.0 to 1.0)
        """
        # Get configured finger joint names or use defaults
        finger_joint_names = getattr(self.config, "dexterous_finger_joint_names", None)

        if finger_joint_names is None:
            # Default: use common finger joints for the robot
            finger_joint_names = self._get_default_finger_joint_names()

        if not finger_joint_names:
            print("  [Finger Check] WARNING: No finger joint names configured, assuming grasped")
            return True, 1.0  # Assume grasped if we can't check

        # Get joint positions from robot
        try:
            joint_ids, found_names = self.robot.find_joints(finger_joint_names)
            if len(joint_ids) == 0:
                print(f"  [Finger Check] WARNING: Could not find finger joints: {finger_joint_names}")
                return True, 1.0  # Assume grasped if joints not found

            joint_positions = self.robot.data.joint_pos[self.env_id, joint_ids]

            # Print each joint's position and status
            # Use ABSOLUTE VALUE since GR1T2 and some robots use negative values for closed
            print(f"  [Finger Check] Checking {len(joint_ids)} finger joints (threshold: {threshold:.3f} rad, using |pos|):")
            closed_joints = []
            open_joints = []
            for i, (jid, jname) in enumerate(zip(joint_ids, found_names)):
                pos = joint_positions[i].item()
                abs_pos = abs(pos)
                is_closed = abs_pos > threshold
                status = "CLOSED" if is_closed else "open"
                print(f"    - {jname}: {pos:.4f} rad (|{abs_pos:.4f}|) [{status}]")
                if is_closed:
                    closed_joints.append(jname)
                else:
                    open_joints.append(jname)

            # Check how many fingers are closed (|joint position| > threshold)
            closed_count = len(closed_joints)
            total_joints = len(joint_ids)
            closed_ratio = closed_count / total_joints if total_joints > 0 else 0.0

            # Consider grasped if majority of finger joints are closed
            min_closed = max(1, total_joints // 2)
            fingers_closed = closed_count >= min_closed

            print(f"  [Finger Check] Summary: {closed_count}/{total_joints} closed (min required: {min_closed})")
            print(f"  [Finger Check] Result: {'FINGERS CLOSED' if fingers_closed else 'FINGERS OPEN'}")

            return fingers_closed, closed_ratio

        except Exception as e:
            print(f"  [Finger Check] ERROR: {e}")
            return True, 1.0  # Assume grasped on error

    def _get_default_finger_joint_names(self) -> list[str]:
        """Get default finger joint names based on robot configuration.

        Override in subclasses for robot-specific finger joints.

        Returns:
            List of finger joint names to check for grasp detection
        """
        # Check if config has gripper joint names
        if self.config.gripper_joint_names:
            return list(self.config.gripper_joint_names)

        # No default finger joints for base class
        return []

    def _get_ee_position_for_grasp_check(self, ee_frame_name: str) -> torch.Tensor | None:
        """Get end-effector position for grasp distance checking.

        Args:
            ee_frame_name: Name of the EE frame in the scene

        Returns:
            EE position tensor (3,) relative to env origin, or None if not found
        """
        origin = self.env.scene.env_origins[self.env_id]

        # Try to get EE frame from scene
        # Note: InteractiveScene doesn't implement __contains__, use keys() instead
        scene_keys = set(self.env.scene.keys()) if hasattr(self.env.scene, "keys") else set()
        if ee_frame_name in scene_keys:
            ee_frame = self.env.scene[ee_frame_name]
            if hasattr(ee_frame, "data") and hasattr(ee_frame.data, "target_pos_w"):
                return ee_frame.data.target_pos_w[self.env_id, 0, :] - origin

        # Fallback: Use robot root position + FK from cuRobo
        try:
            cu_js = self._get_current_joint_state_for_curobo()
            ee_pose = self.get_ee_pose(cu_js)
            if ee_pose is not None and hasattr(ee_pose, "position"):
                # EE pose from cuRobo is in robot base frame, need to convert
                ee_pos_base = self._to_env_device(ee_pose.position).reshape(-1)
                # Add robot base position (relative to env origin)
                robot_pos = self.robot.data.root_pos_w[self.env_id] - origin
                return robot_pos + ee_pos_base
        except Exception as e:
            self.logger.debug(f"FK fallback failed: {e}")

        return None

    def _check_object_grasped_callback(self, object_name: str) -> bool:
        """Check grasp using environment callback.

        Allows the environment to define custom grasp detection logic.

        Args:
            object_name: Name of object to check

        Returns:
            True if environment reports object as grasped
        """
        # Check if environment has custom grasp detection
        if hasattr(self.env, "check_object_grasped"):
            return bool(self.env.check_object_grasped(object_name, self.env_id))

        # Fallback to distance-based if callback not available
        self.logger.warning(
            "Environment has no check_object_grasped method, falling back to distance mode"
        )
        return self._check_object_grasped_by_distance(object_name)

    def _set_gripper_state(self, has_attached_objects: bool) -> None:
        """Configure gripper joint positions based on object attachment status.

        Sets the gripper to closed position when objects are attached and open position
        when no objects are attached. This ensures proper collision checking and planning
        with the correct gripper configuration.

        Args:
            has_attached_objects: True if robot currently has attached objects requiring closed gripper
        """
        if has_attached_objects:
            # Closed gripper for grasping
            locked_joints = self.config.gripper_closed_positions
        else:
            # Open gripper for manipulation
            locked_joints = self.config.gripper_open_positions

        self.motion_gen.update_locked_joints(locked_joints, self.robot_cfg)

    def _count_active_spheres(self) -> dict[str, int]:
        """Count active collision spheres by category for debugging.

        Analyzes the current collision sphere configuration to provide detailed
        statistics about robot links vs attached object spheres. This is helpful
        for debugging collision checking issues and attachment problems.

        Returns:
            Dictionary containing sphere counts by category (total, robot_links, attached_objects)
        """
        cu_js = self._get_current_joint_state_for_curobo()

        # Ensure position tensor is on CUDA for cuRobo
        if isinstance(cu_js.position, torch.Tensor):
            sphere_position = self._to_curobo_device(cu_js.position)
        else:
            # Convert list to tensor and move to CUDA
            sphere_position = self._to_curobo_device(torch.tensor(cu_js.position))

        sphere_list = self.motion_gen.kinematics.get_robot_as_spheres(sphere_position)[0]

        # Get sphere configuration
        sphere_config = self.motion_gen.kinematics.kinematics_config
        collision_link_names = self.robot_cfg["kinematics"]["collision_link_names"]
        attached_link = self.config.attached_object_link_name

        # Dedicated attachment link (Franka-style) should be excluded even if present in the list.
        # Humanoid uses a real EE link for attachment, so we keep it in the robot links.
        is_dedicated_attachment_link = attached_link and (
            attached_link.startswith("attached_object") or attached_link not in collision_link_names
        )

        if is_dedicated_attachment_link:
            robot_links = [link for link in collision_link_names if link != attached_link]
        else:
            robot_links = list(collision_link_names)

        robot_sphere_count = 0
        for link_name in robot_links:
            if hasattr(sphere_config, "get_link_spheres"):
                link_spheres = sphere_config.get_link_spheres(link_name)
                if link_spheres is not None:
                    active_spheres = torch.sum(link_spheres[:, 3] > 0).item()
                    robot_sphere_count += int(active_spheres)

        # Count attached object spheres by checking actual sphere list
        attached_sphere_count = 0

        # Handle sphere_list as either a list or single Sphere object
        total_spheres = len(list(sphere_list))

        # Any spheres beyond robot_sphere_count are attached object spheres
        attached_sphere_count = max(0, total_spheres - robot_sphere_count)

        self.logger.debug(
            f"SPHERE COUNT: Total={total_spheres}, Robot={robot_sphere_count},Attached={attached_sphere_count}"
        )

        return {
            "total": total_spheres,
            "robot_links": robot_sphere_count,
            "attached_objects": attached_sphere_count,
        }
