# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import tempfile
import torch
import yaml
from collections.abc import Iterable

from curobo.types.state import JointState

import isaaclab.utils.math as PoseUtils

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
        collision_active_link_substrings: Iterable[str] | None = None,
    ) -> None:
        """Initialize humanoid arm planner with arm-specific configuration.

        Args:
            env: Isaac Lab environment
            robot: Robot articulation
            config: Planner configuration
            env_id: Environment ID
            active_joint_substrings: Substrings to identify active arm joints (e.g., ("right_",))
            hand_link_substrings: Substrings to identify hand links for collision filtering
            collision_active_link_substrings: Optional substrings to force collision spheres to stay
                active for links matching any of the substrings (e.g., ("left_", "right_"))
        """
        # Pre-apply collision_sphere_buffer into the robot YAML so cuRobo kinematics picks it up
        if isinstance(config.robot_config_file, str) and os.path.isfile(config.robot_config_file):
            with open(config.robot_config_file) as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "robot_cfg" in data and "kinematics" in data["robot_cfg"]:
                kin = data["robot_cfg"]["kinematics"]
                # Mirror behavior of extra_collision_spheres: set buffer as a kinematics key
                if getattr(config, "collision_sphere_buffer", None) is not None:
                    kin["collision_sphere_buffer"] = float(config.collision_sphere_buffer)
            tmp_dir = tempfile.mkdtemp(prefix="curobo_robot_cfg_")
            out_path = os.path.join(tmp_dir, os.path.basename(config.robot_config_file))
            with open(out_path, "w") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            config.robot_config_file = out_path

        super().__init__(env=env, robot=robot, config=config, env_id=env_id)

        self.env_id = env_id
        self.active_joint_substrings = tuple(active_joint_substrings)
        self.hand_link_substrings = tuple(hand_link_substrings) if hand_link_substrings else None
        self.collision_active_link_substrings = (
            tuple(collision_active_link_substrings) if collision_active_link_substrings else None
        )

        # Populate hand_link_names from substrings if provided
        if self.hand_link_substrings:
            all_links = list(self.robot_cfg["kinematics"]["collision_link_names"])
            inferred = [link for link in all_links if any(s in link for s in self.hand_link_substrings)]
            if inferred:
                self.config.hand_link_names = inferred
                self.logger.info(f"Using hand links for contact planning: {self.config.hand_link_names}")

    # ---- Joint and collision helpers ----
    def _is_active_joint(self, joint_name: str) -> bool:
        """Check if a joint belongs to the active arm."""
        if not self.active_joint_substrings:
            return True
        return any(s in joint_name for s in self.active_joint_substrings)

    def _arm_side(self) -> str | None:
        """Infer arm side ('left' or 'right') from the configured ee_link_name."""
        ee_link = self.config.ee_link_name or self.robot_cfg["kinematics"].get("ee_link")
        if not isinstance(ee_link, str):
            return None
        if "left_" in ee_link or "_L_" in ee_link:
            return "left"
        if "right_" in ee_link or "_R_" in ee_link:
            return "right"
        return None

    def _active_arm_links(self) -> list[str]:
        """Derive active arm link names based on side; keep essential trunk/base links enabled.

        Mirrors demo_motion_planning's approach of selecting the kinematic chain for the chosen arm
        while retaining trunk links like waist/base/torso/pelvis.
        If collision_active_link_substrings is set, links matching any of these substrings are kept
        active in addition to trunk links.
        """
        all_links = list(self.robot_cfg["kinematics"]["collision_link_names"])

        # Keep essential trunk/base links in the collision model
        trunk_tokens = ("waist_", "base_link", "torso_", "pelvis_")
        trunk_links = [link for link in all_links if any(t in link for t in trunk_tokens)]

        # If explicitly provided, keep collisions active for these link subsets
        if self.collision_active_link_substrings:
            keep_links = [link for link in all_links if any(s in link for s in self.collision_active_link_substrings)]
            # Preserve the configured attached object link if present
            attached = getattr(self.config, "attached_object_link_name", None)
            if attached and attached not in keep_links and attached in all_links:
                keep_links.append(attached)
            return list({*keep_links, *trunk_links})

        side = self._arm_side()
        if side is None:
            # Fallback: if side cannot be inferred, keep all links active
            return all_links

        arm_token = f"{side}_"
        arm_links = [link for link in all_links if arm_token in link]

        # Preserve the configured attached object link if present
        attached = getattr(self.config, "attached_object_link_name", None)
        if attached and attached not in arm_links and attached in all_links:
            trunk_links.append(attached)

        return list({*arm_links, *trunk_links})

    def _inactive_collision_links(self) -> list[str]:
        """Get collision links to disable for the inactive arm using side-aware link selection."""
        all_links = list(self.robot_cfg["kinematics"]["collision_link_names"])
        active_links = set(self._active_arm_links())

        inactive = [link for link in all_links if link not in active_links]

        # Ensure we don't disable the attached object link
        if self.config.attached_object_link_name in inactive:
            inactive.remove(self.config.attached_object_link_name)

        return inactive

    def _get_current_joint_state_for_curobo(self) -> JointState:
        """
        Construct the current joint state for cuRobo with zero velocity and acceleration.
        """
        js = super()._get_current_joint_state_for_curobo()

        limits = self.motion_gen.kinematics.get_joint_limits().position
        low, high = limits[0], limits[1]
        margin = 1e-4

        pos_tensor = (
            js.position
            if isinstance(js.position, torch.Tensor)
            else torch.tensor(js.position, device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        )
        pos = torch.clamp(pos_tensor, low + margin, high - margin)
        if not torch.allclose(pos, pos_tensor):
            self.logger.debug("Clamped start state within joint limits")

        # Debug: Log what the humanoid planner reads as start state
        import os
        if os.environ.get("DEBUG_REPLAY", "0") == "1":
            try:
                from isaaclab_mimic.datagen.data_generator_refactored import get_debug_logger
                logger = get_debug_logger()
                if logger:
                    arm_side = self._arm_side() or "unknown"
                    logger.log(f"[HUMANOID PLANNER _get_current_joint_state] arm={arm_side}")
                    logger.log(f"  planner joints (clamped): {pos[0, :10].cpu().numpy()} ... (first 10)")
            except ImportError:
                pass

        return JointState(
            position=pos,
            velocity=torch.zeros_like(pos),
            acceleration=torch.zeros_like(pos),
            joint_names=js.joint_names,
            tensor_args=self.tensor_args,
        ).get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

    def _get_joint_state_for_fk(
        self, start_joint_state: torch.Tensor | None = None
    ) -> JointState:
        """Get joint state for FK computations.

        If start_joint_state is provided (offline planning), use it.
        Otherwise fall back to reading from articulation buffer.
        """
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
            self.logger.debug("Using provided start_joint_state for FK")

            return JointState(
                position=pos,
                velocity=torch.zeros_like(pos),
                acceleration=torch.zeros_like(pos),
                joint_names=self.motion_gen.kinematics.joint_names,
                tensor_args=self.tensor_args,
            )
        else:
            return self._get_current_joint_state_for_curobo()

    # ---- Main planning entry ----
    def update_world_and_plan_motion(
        self,
        target_pose: torch.Tensor,
        expected_attached_object: str | None = None,
        env_id: int = 0,
        step_size: float | None = None,
        enable_retiming: bool | None = None,
        link_target_poses_base: dict[str, torch.Tensor] | None = None,
        *,
        input_is_site_frame: bool = False,
        start_joint_state: torch.Tensor | None = None,
        skip_world_update: bool = False,
    ) -> bool:
        """Plan motion for single humanoid arm with collision management.

        Accepts target_pose as a world-frame tool pose and converts to planner frame.

        Args:
            start_joint_state: Optional start joint state in cuRobo planner ordering.
                If provided, bypasses reading from articulation buffer (useful for offline planning).
            skip_world_update: If True, skip world synchronization (for offline planning where
                object poses are pre-set in the collision world).
        """
        # Convert controller-site/world input to tool/world if requested
        target_pose_world_tool: torch.Tensor
        if input_is_site_frame:
            # For bimanual humanoids, T_tool_site is approximately identity at the home/reset position.
            # The controller's site frame and cuRobo's tool frame are essentially the same when
            # the robot is at home. Computing T_tool_site from env.get_robot_eef_pose() is unreliable
            # in offline mode because the physics state isn't updated.
            # Using identity simplifies the conversion: world->site becomes world->tool directly.
            target_pose_world_tool = target_pose.to(device=self.env.device, dtype=torch.float32).clone()
        else:
            target_pose_world_tool = target_pose.to(device=self.env.device, dtype=torch.float32)

        # Convert world tool pose to planner (base) frame
        if isinstance(target_pose_world_tool, torch.Tensor) and target_pose_world_tool.shape == (4, 4):
            base_pos = (self.robot.data.root_pos_w[self.env_id] - self.env.scene.env_origins[self.env_id]).to(
                device=self.env.device, dtype=torch.float32
            )
            base_rot = PoseUtils.matrix_from_quat(
                self.robot.data.root_quat_w[self.env_id].unsqueeze(0).to(device=self.env.device, dtype=torch.float32)
            )[0]
            T_env_base = PoseUtils.make_pose(base_pos.unsqueeze(0), base_rot.unsqueeze(0))[0]
            T_base_env = torch.linalg.inv(T_env_base)
            target_pose_base_tool = (T_base_env @ target_pose_world_tool).clone()
        else:
            target_pose_base_tool = target_pose_world_tool

        # Guard: if target is effectively current, synthesize a trivial plan to avoid optimizer edge cases
        try:
            current_js = self._get_joint_state_for_fk(start_joint_state)
            ee_pose_bt = self.get_ee_pose(current_js)  # base->tool
            cur_pos_bt = self._to_env_device(ee_pose_bt.position).reshape(-1, 3)[0]
            if hasattr(ee_pose_bt, "quaternion"):
                cur_quat_bt = self._to_env_device(ee_pose_bt.quaternion).view(1, 4)
                cur_rot_bt = PoseUtils.matrix_from_quat(cur_quat_bt)[0]
            else:
                cur_rot_bt = self._to_env_device(ee_pose_bt.get_rotation())
                if cur_rot_bt.dim() == 3:
                    cur_rot_bt = cur_rot_bt[0]
            T_base_tool_cur = PoseUtils.make_pose(cur_pos_bt.unsqueeze(0), cur_rot_bt.unsqueeze(0))[0]
            base_pos_w = (self.robot.data.root_pos_w[self.env_id] - self.env.scene.env_origins[self.env_id]).to(
                device=self.env.device, dtype=torch.float32
            )
            base_rot_w = PoseUtils.matrix_from_quat(
                self.robot.data.root_quat_w[self.env_id].unsqueeze(0).to(device=self.env.device, dtype=torch.float32)
            )[0]
            T_world_base = PoseUtils.make_pose(base_pos_w.unsqueeze(0), base_rot_w.unsqueeze(0))[0]
            T_world_tool_cur = (T_world_base @ T_base_tool_cur).clone()
            # Compare in world frame for clarity
            d_pos = torch.linalg.vector_norm(T_world_tool_cur[:3, 3] - target_pose_world_tool[:3, 3]).item()
            d_rot_mat = T_world_tool_cur[:3, :3].T @ target_pose_world_tool[:3, :3]
            d_rot = torch.acos(torch.clamp((torch.trace(d_rot_mat) - 1.0) / 2.0, -1.0, 1.0)).item()
            if d_pos < 1e-4 and d_rot < 1e-3:
                # Build a single-waypoint plan at current state
                self._current_plan = current_js
                self._plan_index = 0
                return True
        except Exception:
            pass

        inactive_links = self._inactive_collision_links()
        try:
            if inactive_links:
                self.logger.debug(f"Disabling collision for {len(inactive_links)} inactive links")
                self._set_active_links(inactive_links, active=False)

            result = super().update_world_and_plan_motion(
                target_pose=target_pose_base_tool,
                expected_attached_object=expected_attached_object,
                env_id=env_id,
                step_size=step_size,
                enable_retiming=enable_retiming,
                link_target_poses_base=link_target_poses_base,
                start_joint_state=start_joint_state,
                skip_world_update=skip_world_update,
            )
            return result
        finally:
            if inactive_links:
                self._set_active_links(inactive_links, active=True)
