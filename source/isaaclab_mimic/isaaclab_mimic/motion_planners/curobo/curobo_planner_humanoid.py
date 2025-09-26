from __future__ import annotations

import torch
from typing import Iterable

from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg


class HumanoidArmCuroboPlanner(CuroboPlanner):
    """
    cuRobo planner restricted to one humanoid arm.

    - Locks all non-active-arm joints at their current values before planning.
    - Optionally narrows hand/link contact filtering to the active arm only.
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
        super().__init__(env=env, robot=robot, config=config, env_id=env_id)

        self.env_id = env_id
        self.active_joint_substrings = tuple(active_joint_substrings)
        self.hand_link_substrings = tuple(hand_link_substrings) if hand_link_substrings else None

        # Optionally restrict contact-disabled hand links to the chosen arm
        # if self.hand_link_substrings:
        #     filtered = []
        #     for link in list(self.config.hand_link_names):
        #         if any(s in link for s in self.hand_link_substrings):
        #             filtered.append(link)
        #     if filtered:
        #         self.config.hand_link_names = filtered
        #         self.logger.info(f"Using hand links for contact planning: {self.config.hand_link_names}")
        # Populate hand_link_names from substrings if empty
        if self.hand_link_substrings:
            # If config already has names, filter to active side; else, auto-populate
            all_links = list(self.robot_cfg["kinematics"]["collision_link_names"])
            inferred = [l for l in all_links if any(s in l for s in self.hand_link_substrings)]
            if inferred:
                self.config.hand_link_names = inferred
                self.logger.info(f"Using hand links for contact planning: {self.config.hand_link_names}")

    # ---- Joint locking helpers ----
    def _current_joint_map(self) -> dict[str, float]:
        js = self._get_current_joint_state_for_curobo()
        names = list(js.joint_names)
        pos = js.position.detach().cpu().numpy().reshape(-1)
        return {n: float(p) for n, p in zip(names, pos)}

    def _is_active_joint(self, joint_name: str) -> bool:
        if not self.active_joint_substrings:
            return True
        return any(s in joint_name for s in self.active_joint_substrings)

    def _locked_joints_for_inactive(self) -> dict[str, float]:
        joint_map = self._current_joint_map()
        locked = {n: v for n, v in joint_map.items() if not self._is_active_joint(n)}
        return locked

    # def _inactive_collision_links(self) -> list[str]:
    #     # Heuristic: keep links for the active arm and torso/waist/base; disable everything else
    #     all_links = list(self.robot_cfg["kinematics"]["collision_link_names"])
    #     keep_tokens = tuple(self.active_joint_substrings)# + ("waist_", "torso_", "base_")
    #     active_links = [l for l in all_links if any(t in l for t in keep_tokens)]
    #     return [l for l in all_links if l not in set(active_links)]

    def _inactive_collision_links(self) -> list[str]:
        # Keep only the selected arm and minimal torso/base; disable everything else
        all_links = list(self.robot_cfg["kinematics"]["collision_link_names"])
        keep_tokens = tuple(self.active_joint_substrings) + ("waist_", "base_link")
        active_links = [l for l in all_links if any(t in l for t in keep_tokens)]
        return [l for l in all_links if l not in set(active_links)]

    # ---- Main planning entry (same signature as base) ----
    def update_world_and_plan_motion(
        self,
        target_pose: torch.Tensor,
        expected_attached_object: str | None = None,
        env_id: int = 0,
        step_size: float | None = None,
        enable_retiming: bool | None = None,
    ) -> bool:
        # # Lock all non-active-arm joints at their current values
        # locked = self._locked_joints_for_inactive()
        # if locked:
        #     self.motion_gen.update_locked_joints(locked, self.robot_cfg)  #, self.robot_cfg)

        # return super().update_world_and_plan_motion(
        #     target_pose=target_pose,
        #     expected_attached_object=expected_attached_object,
        #     env_id=env_id,
        #     step_size=step_size,
        #     enable_retiming=enable_retiming,
        # )
        # 1) Lock non-active-arm joints
        # locked = self._locked_joints_for_inactive()
        # if locked:
        #     self.motion_gen.update_locked_joints(locked)

        # 2) Disable spheres for inactive links during planning
        inactive_links = self._inactive_collision_links()
        try:
            if inactive_links:
                self._set_active_links(inactive_links, active=False)
            return super().update_world_and_plan_motion(
                target_pose=target_pose,
                expected_attached_object=expected_attached_object,
                env_id=env_id,
                step_size=step_size,
                enable_retiming=enable_retiming,
            )
        finally:
            if inactive_links:
                self._set_active_links(inactive_links, active=True)

    def _plan_to_contact_pose(
        self,
        start_state,
        goal_pose,
        contact: bool = True,
    ):
        # Build link_poses for all links at the current (start) state to hold them fixed
        # during planning, except the EE link (which gets the goal).
        kin = self.motion_gen.kinematics.get_state(
            q=start_state.position.detach().clone().to(device=self.tensor_args.device, dtype=self.tensor_args.dtype),
            calculate_jacobian=False,
        )

        # Assemble link_poses dict (link -> Pose)
        link_poses = {}
        for i, link in enumerate(kin.link_names):
            pos = kin.links_position[..., i, :]
            quat = kin.links_quaternion[..., i, :]
            link_poses[link] = self._make_pose(position=pos, quaternion=quat)

        # Remove EE from constraints so it can reach the goal
        ee_link = self.config.ee_link_name or self.robot_cfg["kinematics"]["ee_link"]
        if ee_link in link_poses:
            link_poses.pop(ee_link, None)

        # Prepare disabling for contact (hand links + any attached objects)
        disable_link_names = self.config.hand_link_names.copy()
        saved_spheres = {}

        # Count spheres before
        _ = self._count_active_spheres()

        if contact:
            # Save current spheres for any attached links and disable collision on hand links
            attached_links = list(self.attachment_links)
            for attached_link in attached_links:
                saved_spheres[attached_link] = self.motion_gen.kinematics.kinematics_config.get_link_spheres(attached_link).clone()
            self._set_active_links(disable_link_names + attached_links, active=False)

        # Do the motion generation with link_poses constraints (key difference)
        planning_success = False
        try:
            result = self.motion_gen.plan_single(start_state, goal_pose, self.plan_config, link_poses)
            if result.success.item():
                if result.optimized_plan is not None and len(result.optimized_plan.position) != 0:
                    self._current_plan = result.optimized_plan
                else:
                    self._current_plan = result.get_interpolated_plan()

                # Normalize to full JS, then to env joint ordering
                self._current_plan = self.motion_gen.get_full_js(self._current_plan)
                common_js_names = [x for x in self.robot.data.joint_names if x in self._current_plan.joint_names]
                self._current_plan = self._current_plan.get_ordered_joint_state(common_js_names)
                self._plan_index = 0
                planning_success = True
            else:
                self.logger.debug(f"Contact planning failed: {result.status}")
        except Exception as e:
            self.logger.debug(f"Error during planning: {e}")
        finally:
            # Restore spheres after contact planning
            if contact:
                self._set_active_links(disable_link_names, active=True)
                for attached_link, spheres in saved_spheres.items():
                    self.motion_gen.kinematics.kinematics_config.update_link_spheres(attached_link, spheres)

        return planning_success