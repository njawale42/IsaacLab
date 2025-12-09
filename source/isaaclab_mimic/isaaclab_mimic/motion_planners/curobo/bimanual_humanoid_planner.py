"""
Wrapper planner that selects the appropriate humanoid arm planner (left/right)
per call, while keeping collision spheres active for both arms.

Interface matches MotionPlannerBase for compatibility with DataGenerator.
Provides attachment/detachment support for bimanual manipulation.
"""

from __future__ import annotations

import traceback
import torch
from typing import Any

from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_env import ManagerBasedEnv

from isaaclab_mimic.motion_planners.motion_planner_base import MotionPlannerBase
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
from isaaclab_mimic.motion_planners.curobo.curobo_planner_humanoid import HumanoidArmCuroboPlanner


class BimanualHumanoidPlanner(MotionPlannerBase):
    """
    Delegates to two HumanoidArmCuroboPlanner instances (right and left).

    Arm selection per update is decided by comparing the target pose to the
    current EEF poses and picking the closest arm.

    Object Attachment:
        Each arm maintains its own attachment state. When an object is attached
        during planning (via expected_attached_object), it is attached to the
        arm that successfully planned the motion. Use get_attached_objects() to
        see what each arm is holding, or get_attached_objects_for_arm() for
        arm-specific queries.
    """

    def __init__(
        self,
        env: ManagerBasedEnv,
        robot: Articulation,
        *,
        cfg_right: CuroboPlannerCfg,
        cfg_left: CuroboPlannerCfg,
        env_id: int = 0,
    ) -> None:
        super().__init__(env=env, robot=robot, env_id=env_id, debug=bool(getattr(cfg_right, "debug_planner", False)))

        # Create right/left arm planners. Keep both arms' spheres active for collision with the other arm.
        self._planner_right = HumanoidArmCuroboPlanner(
            env=env,
            robot=robot,
            config=cfg_right,
            env_id=env_id,
            active_joint_substrings=("right_",),
            hand_link_substrings=("GR1T2_fourier_hand_6dof_right_",),
            collision_active_link_substrings=("GR1T2_fourier_hand_6dof_left_", "GR1T2_fourier_hand_6dof_right_"),
        )
        self._planner_left = HumanoidArmCuroboPlanner(
            env=env,
            robot=robot,
            config=cfg_left,
            env_id=env_id,
            active_joint_substrings=("left_",),
            hand_link_substrings=("GR1T2_fourier_hand_6dof_left_",),
            collision_active_link_substrings=("GR1T2_fourier_hand_6dof_left_", "GR1T2_fourier_hand_6dof_right_"),
        )

        # Default to right for config accessors
        self.config = cfg_right
        self.visualize_spheres = bool(getattr(cfg_right, "visualize_spheres", False))

        self._last_arm: str | None = None

        # Track which arm last attached an object (for coordinated detachment)
        self._arm_with_attachment: str | None = None

    # =========================================================================
    # ARM SELECTION
    # =========================================================================

    def _choose_arm(self, target_pose: torch.Tensor) -> str:
        """
        Choose the arm whose current EEF is closest to the target pose.
        """
        # Current EEF poses from env are in controller frame (world/site). Use positions only.
        left_pose = self.env.get_robot_eef_pose("left", env_ids=[self.env_id])[0]
        right_pose = self.env.get_robot_eef_pose("right", env_ids=[self.env_id])[0]

        tpos = target_pose[:3, 3]
        ldist = torch.linalg.vector_norm(left_pose[:3, 3] - tpos)
        rdist = torch.linalg.vector_norm(right_pose[:3, 3] - tpos)
        try:
            print(
                f"[BimanualHumanoidPlanner] Distances -> left: {float(ldist):.4f} m, right: {float(rdist):.4f} m"
            )
        except Exception:
            pass
        return "left" if ldist <= rdist else "right"

    def _get_planner_for_arm(self, arm: str) -> HumanoidArmCuroboPlanner:
        """Get the planner instance for the specified arm."""
        return self._planner_left if arm == "left" else self._planner_right

    def _get_active_planner(self) -> HumanoidArmCuroboPlanner:
        """Get the planner for the last used arm (defaults to right)."""
        return self._planner_left if self._last_arm == "left" else self._planner_right

    # =========================================================================
    # OBJECT ATTACHMENT INTERFACE
    # =========================================================================

    @property
    def attached_objects(self) -> dict[str, Any]:
        """Combined attached objects from both arms.

        Returns dict mapping object_name -> Attachment for all attached objects.
        """
        combined = {}
        combined.update(self._planner_right.attached_objects)
        combined.update(self._planner_left.attached_objects)
        return combined

    @property
    def attached_link(self) -> str:
        """Default attachment link name from the active arm's config."""
        return self._get_active_planner().attached_link

    @property
    def attachment_links(self) -> set[str]:
        """Set of all parent links with attachments across both arms."""
        return self._planner_right.attachment_links | self._planner_left.attachment_links

    def get_attached_objects(self) -> list[str]:
        """Get list of all currently attached object names from both arms.

        Returns:
            List of attached object names (e.g., ["cube_1", "beaker"])
        """
        return list(self.attached_objects.keys())

    def get_attached_objects_for_arm(self, arm: str) -> list[str]:
        """Get attached objects for a specific arm.

        Args:
            arm: "left" or "right"

        Returns:
            List of object names attached to the specified arm
        """
        planner = self._get_planner_for_arm(arm)
        return planner.get_attached_objects()

    def has_attached_objects(self) -> bool:
        """Check if any objects are attached to either arm.

        Returns:
            True if one or more objects are attached, False otherwise
        """
        return self._planner_right.has_attached_objects() or self._planner_left.has_attached_objects()

    def has_attached_objects_for_arm(self, arm: str) -> bool:
        """Check if the specified arm has attached objects.

        Args:
            arm: "left" or "right"

        Returns:
            True if the arm has attachments
        """
        return self._get_planner_for_arm(arm).has_attached_objects()

    def detach_objects(self, arm: str | None = None) -> bool:
        """Detach objects from specified arm or all arms.

        Args:
            arm: "left", "right", or None to detach from both

        Returns:
            True if detachment succeeded
        """
        if arm == "left":
            return self._planner_left.detach_all_objects()
        elif arm == "right":
            return self._planner_right.detach_all_objects()
        else:
            # Detach from both arms
            left_ok = self._planner_left.detach_all_objects()
            right_ok = self._planner_right.detach_all_objects()
            self._arm_with_attachment = None
            return left_ok and right_ok

    def get_arm_with_object(self, object_name: str) -> str | None:
        """Get which arm is holding a specific object.

        Args:
            object_name: Name of the object to look for

        Returns:
            "left", "right", or None if not attached
        """
        if object_name in self._planner_right.attached_objects:
            return "right"
        if object_name in self._planner_left.attached_objects:
            return "left"
        return None

    # =========================================================================
    # MOTION PLANNING
    # =========================================================================

    def update_world_and_plan_motion(
        self,
        target_pose: torch.Tensor,
        expected_attached_object: str | None = None,
        env_id: int = 0,
        step_size: float | None = None,
        enable_retiming: bool | None = None,
        link_target_poses_base: dict[str, torch.Tensor] | None = None,
        arm: str | None = None,
        **kwargs: Any,
    ) -> bool:
        """Plan motion to target pose with optional object attachment.

        Args:
            target_pose: Target end-effector pose (4x4 matrix)
            expected_attached_object: Object name to attach during planning, or None
            env_id: Environment ID
            step_size: Optional step size for retiming
            enable_retiming: Whether to enable trajectory retiming
            link_target_poses_base: Optional link target poses in base frame
            arm: Force specific arm ("left" or "right"), or None to auto-select
            **kwargs: Additional arguments (e.g., input_is_site_frame)

        Returns:
            True if planning succeeded
        """
        # Auto-select arm or use specified
        arm_primary = arm if arm in ("left", "right") else self._choose_arm(target_pose)
        arm_fallback = "left" if arm_primary == "right" else "right"
        print(f"[BimanualHumanoidPlanner] Selected arm: {arm_primary}")

        def _attempt(arm_name: str) -> bool:
            planner_local = self._get_planner_for_arm(arm_name)
            try:
                ok = planner_local.update_world_and_plan_motion(
                    target_pose=target_pose,
                    expected_attached_object=expected_attached_object,
                    env_id=env_id,
                    step_size=step_size,
                    enable_retiming=enable_retiming,
                    link_target_poses_base=link_target_poses_base,
                    **kwargs,
                )
                if ok:
                    self._last_arm = arm_name
                    # Track which arm got the attachment if object was specified
                    if expected_attached_object is not None and planner_local.has_attached_objects():
                        self._arm_with_attachment = arm_name
                        print(f"[BimanualHumanoidPlanner] Object '{expected_attached_object}' attached to {arm_name}")
                return bool(ok)
            except Exception:
                print("[BimanualHumanoidPlanner] ERROR during update_world_and_plan_motion:")
                print(
                    f"  arm={arm_name} env_id={env_id} step_size={step_size} enable_retiming={enable_retiming}"
                )
                print(f"  expected_attached_object={expected_attached_object}")
                if "input_is_site_frame" in kwargs:
                    print(f"  input_is_site_frame={kwargs['input_is_site_frame']}")
                try:
                    print(f"  target_pose=\n{target_pose}")
                except Exception:
                    pass
                traceback.print_exc()
                return False

        # Try primary arm; if it fails, try the other arm once (unless arm was forced)
        if _attempt(arm_primary):
            return True
        if arm is None:
            print(f"[BimanualHumanoidPlanner] Primary arm {arm_primary} failed; trying fallback arm {arm_fallback}")
            return _attempt(arm_fallback)
        return False

    # =========================================================================
    # PLAN EXECUTION
    # =========================================================================

    def has_next_waypoint(self) -> bool:
        """Check if current plan has more waypoints."""
        return self._get_active_planner().has_next_waypoint()

    def get_next_waypoint_ee_pose(self) -> Any:
        """Get next waypoint end-effector pose from active arm's plan."""
        return self._get_active_planner().get_next_waypoint_ee_pose()

    def get_planned_poses(self) -> list[Any]:
        """Get all planned poses from active arm's plan."""
        return self._get_active_planner().get_planned_poses()

    def reset_plan(self) -> None:
        """Reset plans for both arms."""
        self._planner_right.reset_plan()
        self._planner_left.reset_plan()
        self._last_arm = None

    # =========================================================================
    # ACCESSORS
    # =========================================================================

    @property
    def last_arm(self) -> str | None:
        """The arm that was last used for planning."""
        return self._last_arm

    @property
    def motion_gen(self) -> Any:
        """Motion generator from the active arm's planner.

        Note: Use with caution - for bimanual, prefer arm-specific access.
        """
        return self._get_active_planner().motion_gen

    @property
    def plan_visualizer(self) -> Any:
        """Plan visualizer from the active arm's planner."""
        return self._get_active_planner().plan_visualizer

    def get_planner_for_arm(self, arm: str) -> HumanoidArmCuroboPlanner:
        """Public access to arm-specific planner.

        Args:
            arm: "left" or "right"

        Returns:
            The HumanoidArmCuroboPlanner for that arm
        """
        return self._get_planner_for_arm(arm)

    def update_world(self) -> None:
        """Update collision world for both arm planners."""
        self._planner_right.update_world()
        self._planner_left.update_world()
