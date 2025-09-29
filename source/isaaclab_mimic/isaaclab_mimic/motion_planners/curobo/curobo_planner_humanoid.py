# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import Any
from collections.abc import Iterable

from curobo.types.state import JointState

from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg


class HumanoidArmCuroboPlanner(CuroboPlanner):
    """
    cuRobo planner restricted to one humanoid arm.

    - Disables collision spheres for non-active-arm links
    - Optionally narrows hand/link contact filtering to the active arm only
    """

    def __init__(
        self,
        env,
        robot,
        config: CuroboPlannerCfg,
        *,
        env_id: int = 0,
        active_joint_substrings: Iterable[str] = ("right_",),
        hand_link_substrings: Iterable[str] | None = None,
    ) -> None:
        """Initialize humanoid arm planner with arm-specific configuration.

        Args:
            env: Isaac Lab environment
            robot: Robot articulation
            config: Planner configuration
            env_id: Environment ID
            active_joint_substrings: Substrings to identify active arm joints (e.g., ("right_",))
            hand_link_substrings: Substrings to identify hand links for collision filtering
        """
        super().__init__(env=env, robot=robot, config=config, env_id=env_id)

        self.env_id = env_id
        self.active_joint_substrings = tuple(active_joint_substrings)
        self.hand_link_substrings = tuple(hand_link_substrings) if hand_link_substrings else None

        # Populate hand_link_names from substrings if provided
        if self.hand_link_substrings:
            all_links = list(self.robot_cfg["kinematics"]["collision_link_names"])
            inferred = [l for l in all_links if any(s in l for s in self.hand_link_substrings)]
            if inferred:
                self.config.hand_link_names = inferred
                self.logger.info(f"Using hand links for contact planning: {self.config.hand_link_names}")

    # ---- Joint and collision helpers ----
    def _is_active_joint(self, joint_name: str) -> bool:
        """Check if a joint belongs to the active arm."""
        if not self.active_joint_substrings:
            return True
        return any(s in joint_name for s in self.active_joint_substrings)

    def _inactive_collision_links(self) -> list[str]:
        """Get collision links to disable for the inactive arm."""
        all_links = list(self.robot_cfg["kinematics"]["collision_link_names"])

        # Keep active arm links and essential torso/base links
        keep_tokens = tuple(self.active_joint_substrings) + ("waist_", "base_link", "torso_", "pelvis_")
        active_links = [l for l in all_links if any(t in l for t in keep_tokens)]

        # Inactive links are everything else
        inactive = [l for l in all_links if l not in set(active_links)]

        # Don't disable attached object link even if it's on the inactive side
        if self.config.attached_object_link_name in inactive:
            inactive.remove(self.config.attached_object_link_name)

        return inactive

    # ---- Main planning entry ----
    def update_world_and_plan_motion(
        self,
        target_pose: torch.Tensor,
        expected_attached_object: str | None = None,
        env_id: int = 0,
        step_size: float | None = None,
        enable_retiming: bool | None = None,
    ) -> bool:
        """Plan motion for single humanoid arm with collision management.

        This method:
        1. Disables collision spheres for inactive links
        2. Plans motion using the base planner
        3. Restores collision configuration

        Note: We don't lock joints as this causes dimension mismatches in cuRobo.
        Instead, we rely on collision sphere disabling to prevent interference.
        """
        # Disable spheres for inactive links during planning
        inactive_links = self._inactive_collision_links()

        try:
            # Temporarily disable collision checking for inactive links
            if inactive_links:
                self.logger.debug(f"Disabling collision for {len(inactive_links)} inactive links")
                self._set_active_links(inactive_links, active=False)

            # Call parent planning method
            result = super().update_world_and_plan_motion(
                target_pose=target_pose,
                expected_attached_object=expected_attached_object,
                env_id=env_id,
                step_size=step_size,
                enable_retiming=enable_retiming,
            )

            return result

        finally:
            # Always restore collision checking
            if inactive_links:
                self._set_active_links(inactive_links, active=True)
