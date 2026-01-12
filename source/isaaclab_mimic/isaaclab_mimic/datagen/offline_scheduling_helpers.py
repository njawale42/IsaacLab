# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper classes for offline trajectory scheduling in bimanual manipulation.

These utilities consolidate common patterns for joint state management,
object pose manipulation during offline planning, and grasp check patching.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

import isaaclab.utils.math as PoseUtils


@dataclass
class JointMapper:
    """
    Caches joint name mappings between a motion planner and the robot articulation.

    This class eliminates redundant joint name lookups by building index mappings
    once and reusing them for projection, expansion, and merging operations.
    """

    robot_articulation: Any
    planner: Any
    device: torch.device
    _env_joint_names: list[str] = field(default_factory=list, init=False)
    _planner_joint_names: list[str] = field(default_factory=list, init=False)
    _active_substrings: list[str] | None = field(default=None, init=False)
    _env_to_planner_indices: list[int] = field(default_factory=list, init=False)
    _planner_to_env_indices: list[tuple[int, int]] = field(default_factory=list, init=False)
    _planner_dof: int = field(default=7, init=False)

    def __post_init__(self):
        self._build_mappings()

    def _build_mappings(self) -> None:
        """Build index mappings between articulation and planner joint spaces."""
        self._env_joint_names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in self.robot_articulation.data.joint_names
        ]

        motion_gen = getattr(self.planner, "motion_gen", None)
        if motion_gen is not None and hasattr(motion_gen, "kinematics"):
            self._planner_joint_names = list(motion_gen.kinematics.joint_names)
            self._planner_dof = len(self._planner_joint_names)
        else:
            self._planner_joint_names = []
            self._planner_dof = 7

        self._active_substrings = getattr(self.planner, "active_joint_substrings", None)

        # Build env_idx -> planner_idx mapping (for projection)
        self._env_to_planner_indices = []
        for name in self._planner_joint_names:
            if name in self._env_joint_names:
                self._env_to_planner_indices.append(self._env_joint_names.index(name))

        # Build planner_idx -> env_idx mapping with active filtering (for expansion/merge)
        self._planner_to_env_indices = []
        for planner_idx, name in enumerate(self._planner_joint_names):
            if name not in self._env_joint_names:
                continue
            if self._active_substrings:
                if not any(sub in name for sub in self._active_substrings):
                    continue
            env_idx = self._env_joint_names.index(name)
            self._planner_to_env_indices.append((planner_idx, env_idx))

    @property
    def planner_dof(self) -> int:
        """Number of DOFs in the planner's joint space."""
        return self._planner_dof

    def project_to_planner(self, full_joints: torch.Tensor) -> torch.Tensor:
        """
        Project a full articulation joint vector to the planner's joint ordering.

        Args:
            full_joints: Joint positions in articulation ordering.

        Returns:
            Joint positions in planner ordering.
        """
        if full_joints.shape[-1] == self._planner_dof:
            return full_joints.to(device=self.device)

        if not self._env_to_planner_indices:
            return torch.zeros(self._planner_dof, device=self.device)

        return full_joints[self._env_to_planner_indices].to(device=self.device)

    def expand_to_full(self, planner_joints: torch.Tensor, env_id: int) -> torch.Tensor:
        """
        Expand planner joint vector to full articulation ordering.

        Only updates joints that are ACTIVE for the arm planner (matching
        active_joint_substrings). Preserves other joints from current articulation state.

        Args:
            planner_joints: Joint positions in planner ordering.
            env_id: Environment ID for reading current articulation state.

        Returns:
            Joint positions in full articulation ordering.
        """
        full_size = self.robot_articulation.data.joint_pos.shape[1]
        if planner_joints.shape[0] == full_size:
            return planner_joints.to(device=self.device)

        full = self.robot_articulation.data.joint_pos[env_id].clone()
        vec = planner_joints.to(device=full.device)

        for planner_idx, env_idx in self._planner_to_env_indices:
            if planner_idx < len(vec):
                full[env_idx] = vec[planner_idx]

        return full

    def merge_to_global(
        self, global_state: torch.Tensor, arm_joints: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge arm joint positions into a global robot state.

        Updates only the joints controlled by this planner while preserving
        other joints (e.g., the other arm's positions for bimanual robots).

        Args:
            global_state: Current global robot joint state.
            arm_joints: New joint positions from the arm planner.

        Returns:
            Updated global robot joint state.
        """
        updated = global_state.clone()
        arm_vec = arm_joints.to(device=updated.device)

        for planner_idx, env_idx in self._planner_to_env_indices:
            if planner_idx < len(arm_vec):
                updated[env_idx] = arm_vec[planner_idx]

        return updated


class JointMapperCache:
    """
    Caches JointMapper instances per (articulation, planner) pair.

    Avoids rebuilding joint mappings on every method call.
    """

    def __init__(self, robot_articulation: Any, device: torch.device):
        self._robot_articulation = robot_articulation
        self._device = device
        self._cache: dict[int, JointMapper] = {}

    def get(self, planner: Any) -> JointMapper:
        """Get or create a JointMapper for the given planner."""
        planner_id = id(planner)
        if planner_id not in self._cache:
            self._cache[planner_id] = JointMapper(
                robot_articulation=self._robot_articulation,
                planner=planner,
                device=self._device,
            )
        return self._cache[planner_id]

    def clear(self) -> None:
        """Clear the cache (e.g., on environment reset)."""
        self._cache.clear()


class OfflineObjectHelper:
    """
    Helper for manipulating object poses during offline motion planning.

    During offline planning, the robot hasn't physically interacted with objects,
    so we need to temporarily move objects to expected positions for correct
    attachment/collision computations.
    """

    def __init__(self, env: Any):
        self._env = env

    def find_object(self, object_name: str) -> Any | None:
        """Find a rigid object in the scene by name (substring match)."""
        rigid_objects = self._env.scene.rigid_objects
        for name, obj in rigid_objects.items():
            if object_name in name or name in object_name:
                return obj
        return None

    def get_object_pose(self, env_id: int, object_name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get the current pose (pos, quat) of an object."""
        obj = self.find_object(object_name)
        if obj is None:
            return None
        return (
            obj.data.root_pos_w[env_id].clone(),
            obj.data.root_quat_w[env_id].clone(),
        )

    def set_object_position(self, env_id: int, object_name: str, position: torch.Tensor) -> bool:
        """Set an object's position (keeps current orientation)."""
        obj = self.find_object(object_name)
        if obj is None:
            return False
        device = obj.data.root_pos_w.device
        obj.data.root_pos_w[env_id] = position.to(device=device)
        return True

    def set_object_pose(
        self, env_id: int, object_name: str, position: torch.Tensor, quaternion: torch.Tensor
    ) -> bool:
        """Set an object's full pose (position and orientation)."""
        obj = self.find_object(object_name)
        if obj is None:
            return False
        device = obj.data.root_pos_w.device
        obj.data.root_pos_w[env_id] = position.to(device=device)
        obj.data.root_quat_w[env_id] = quaternion.to(device=device)
        return True

    def move_to_ee_position(
        self, env_id: int, object_name: str, ee_position: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Move an object to the EE position and return its original pose.

        Args:
            env_id: Environment ID.
            object_name: Name of the object to move.
            ee_position: EE position to move object to.

        Returns:
            Tuple of (original_pos, original_quat) to restore later, or None if failed.
        """
        original_pose = self.get_object_pose(env_id, object_name)
        if original_pose is None:
            return None

        self.set_object_position(env_id, object_name, ee_position)
        return original_pose

    def restore_pose(
        self, env_id: int, object_name: str, original_pose: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Restore an object to its original pose."""
        original_pos, original_quat = original_pose
        self.set_object_pose(env_id, object_name, original_pos, original_quat)


@contextmanager
def force_grasp_check(planner: Any, expected_object: str):
    """
    Context manager to temporarily force grasp checks to pass for an expected object.

    This allows normal attachment flow during offline planning where the robot
    isn't at the actual grasping position.

    Args:
        planner: The arm-specific planner instance.
        expected_object: The object name that should pass the grasp check.

    Yields:
        None
    """
    if planner is None or not hasattr(planner, "_check_object_grasped"):
        yield
        return

    original_method = planner._check_object_grasped
    expected_lower = expected_object.lower()

    def forced_grasp_check(gripper_pos: Any, object_name: str) -> bool:
        object_lower = object_name.lower() if object_name else ""
        if expected_lower in object_lower or object_lower in expected_lower:
            return True
        return original_method(gripper_pos, object_name)

    planner._check_object_grasped = forced_grasp_check
    try:
        yield
    finally:
        planner._check_object_grasped = original_method


def get_ee_position_from_fk(
    robot_articulation: Any,
    arm_planner: Any,
    env_id: int,
) -> torch.Tensor | None:
    """
    Get the EE position using FK from the arm planner's current joint state.

    Args:
        robot_articulation: Robot articulation for reading joint state.
        arm_planner: Arm-specific planner with FK capability.
        env_id: Environment ID.

    Returns:
        EE position in world frame, or None if computation fails.
    """
    if arm_planner is None:
        return None

    joint_state = arm_planner._get_current_joint_state_for_curobo()
    if joint_state is None:
        return None

    link_name = getattr(arm_planner.config, "attached_object_link_name", None)
    if link_name is None:
        return None

    if not hasattr(arm_planner, "get_attached_pose"):
        return None

    link_pose = arm_planner.get_attached_pose(link_name, joint_state)
    if link_pose is None:
        return None

    # Convert from robot base frame to world frame
    base_pos = robot_articulation.data.root_pos_w[env_id]
    base_quat = robot_articulation.data.root_quat_w[env_id]
    link_pos_base = link_pose.position.squeeze().to(device=base_pos.device)

    link_pos_rotated = PoseUtils.quat_apply(base_quat, link_pos_base)
    return base_pos + link_pos_rotated


def interpolate_pose(p0: torch.Tensor, p1: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Interpolate between two 4x4 pose matrices.

    Uses linear interpolation for translation and orthogonalized blend for rotation.

    Args:
        p0: Start pose (4x4 matrix).
        p1: End pose (4x4 matrix).
        alpha: Interpolation factor in [0, 1].

    Returns:
        Interpolated pose (4x4 matrix).
    """
    t_interp = p0[:3, 3] * (1.0 - alpha) + p1[:3, 3] * alpha
    R_blend = p0[:3, :3] * (1.0 - alpha) + p1[:3, :3] * alpha

    # Orthogonalize via SVD for valid rotation
    U, _, Vh = torch.linalg.svd(R_blend)
    R_interp = U @ Vh

    result = torch.eye(4, device=p0.device, dtype=p0.dtype)
    result[:3, :3] = R_interp
    result[:3, 3] = t_interp
    return result


def compute_interpolation_steps(
    joints_prev: torch.Tensor | None,
    joints_curr: torch.Tensor | None,
    max_joint_step: float,
) -> int:
    """
    Compute number of interpolation steps needed to smooth large joint jumps.

    Args:
        joints_prev: Previous joint positions.
        joints_curr: Current joint positions.
        max_joint_step: Maximum allowed joint change per step (radians).

    Returns:
        Number of interpolation steps needed (0 if no interpolation required).
    """
    import math

    if joints_prev is None or joints_curr is None:
        return 0

    if joints_prev.numel() == 0 or joints_curr.numel() == 0:
        return 0

    max_delta = (joints_curr - joints_prev).abs().max().item()

    if max_delta <= max_joint_step:
        return 0

    return int(math.ceil(max_delta / max_joint_step)) - 1


# ==================== Transform Utilities ====================


def get_world_to_base_transform(
    robot_articulation: Any,
    env_origins: torch.Tensor,
    env_id: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the world-to-base transform for the robot.

    This is a common operation needed for frame conversions between
    world frame and the robot's base frame.

    Args:
        robot_articulation: Robot articulation for reading pose.
        env_origins: Environment origins tensor.
        env_id: Environment ID.
        device: Target device for tensors.

    Returns:
        4x4 homogeneous transform matrix (T_world_base).
    """
    base_pos = (robot_articulation.data.root_pos_w[env_id] - env_origins[env_id]).to(
        device=device, dtype=torch.float32
    )
    base_rot = PoseUtils.matrix_from_quat(
        robot_articulation.data.root_quat_w[env_id].unsqueeze(0).to(device=device, dtype=torch.float32)
    )[0]
    return PoseUtils.make_pose(base_pos.unsqueeze(0), base_rot.unsqueeze(0))[0]


def resolve_eef_name_from_planner(motion_planner: Any, default: str = "right") -> str:
    """
    Resolve the end-effector name from a motion planner instance.

    Motion planners may store the arm name in different attributes depending
    on the planner type. This utility checks common attribute patterns.

    Args:
        motion_planner: Motion planner instance.
        default: Default EEF name if resolution fails.

    Returns:
        Resolved EEF name string.
    """
    if hasattr(motion_planner, "_last_arm") and motion_planner._last_arm is not None:
        return motion_planner._last_arm
    if hasattr(motion_planner, "_arm_side"):
        try:
            result = motion_planner._arm_side()
            if result is not None:
                return result
        except Exception:
            pass
    return default


def transform_base_to_world(
    pose_base: torch.Tensor,
    T_world_base: torch.Tensor,
    T_tool_site: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Transform a pose from base frame to world frame, optionally applying tool-to-site correction.

    Args:
        pose_base: 4x4 pose in base (robot) frame.
        T_world_base: 4x4 world-to-base transform.
        T_tool_site: Optional 4x4 tool-to-site transform for bimanual robots.

    Returns:
        4x4 pose in world (or site) frame.
    """
    T_world_tool = (T_world_base @ pose_base).clone()
    if T_tool_site is not None:
        return (T_world_tool @ T_tool_site).clone()
    return T_world_tool


def is_grasp_subtask(subtask_cfg: Any) -> bool:
    """Check if a subtask configuration represents a grasp operation."""
    signal = str(getattr(subtask_cfg, "subtask_term_signal", "")).lower()
    return "grasp" in signal


def is_place_subtask(subtask_cfg: Any) -> bool:
    """Check if a subtask configuration represents a place/release operation."""
    signal = str(getattr(subtask_cfg, "subtask_term_signal", "")).lower()
    return "place" in signal or "release" in signal


def get_subtask_object_ref(subtask_cfg: Any) -> str | None:
    """Get the object reference from a subtask configuration."""
    return getattr(subtask_cfg, "object_ref", None)
