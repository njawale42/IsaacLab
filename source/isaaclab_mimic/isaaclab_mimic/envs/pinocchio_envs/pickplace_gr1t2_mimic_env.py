# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class PickPlaceGR1T2MimicEnv(ManagerBasedRLMimicEnv):
    """Mimic environment for GR1T2 bimanual pick-place tasks.

    IMPORTANT: The GR1T2 robot's hand joints are ordered in an INTERLEAVED pattern:
        L proximal (5), R proximal (5), L intermediate (5), R intermediate (5), L distal (1), R distal (1)

    This means the action tensor's 22 hand joint values have LEFT and RIGHT joints mixed together.
    The methods in this class properly separate/combine left and right gripper actions
    to work with this interleaved joint ordering.
    """

    # Indices of LEFT hand joints within the 22-element hand joint action (interleaved order)
    # These correspond to: L_proximal(5) + L_intermediate(5) + L_distal(1) = 11 joints
    LEFT_HAND_JOINT_INDICES = [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20]

    # Indices of RIGHT hand joints within the 22-element hand joint action (interleaved order)
    # These correspond to: R_proximal(5) + R_intermediate(5) + R_distal(1) = 11 joints
    RIGHT_HAND_JOINT_INDICES = [5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 21]

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """
        Get current robot end effector pose. Should be the same frame as used by the robot end-effector controller.

        Args:
            eef_name: Name of the end effector.
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        if env_ids is None:
            env_ids = slice(None)

        eef_pos_name = f"{eef_name}_eef_pos"
        eef_quat_name = f"{eef_name}_eef_quat"

        target_wrist_position = self.obs_buf["policy"][eef_pos_name][env_ids]
        target_rot_mat = PoseUtils.matrix_from_quat(self.obs_buf["policy"][eef_quat_name][env_ids])

        return PoseUtils.make_pose(target_wrist_position, target_rot_mat)

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,  # Unused, but required to conform to interface
    ) -> torch.Tensor:
        """
        Takes a target pose and gripper action for the end effector controller and returns an action
        (usually a normalized delta pose action) to try and achieve that target pose.
        Noise is added to the target pose action if specified.

        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            action_noise_dict: Noise to add to the action. If None, no noise is added.
            env_id: Environment index to get the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().
        """

        # target position and rotation
        target_left_eef_pos, left_target_rot = PoseUtils.unmake_pose(target_eef_pose_dict["left"])
        target_right_eef_pos, right_target_rot = PoseUtils.unmake_pose(target_eef_pose_dict["right"])

        target_left_eef_rot_quat = PoseUtils.quat_from_matrix(left_target_rot)
        target_right_eef_rot_quat = PoseUtils.quat_from_matrix(right_target_rot)

        # Canonicalize quaternions to w >= 0 to avoid sign ambiguity in recorded actions
        target_left_eef_rot_quat = PoseUtils.quat_unique(target_left_eef_rot_quat)
        target_right_eef_rot_quat = PoseUtils.quat_unique(target_right_eef_rot_quat)

        # gripper actions - need to interleave left and right to match the hand_joint_names order
        left_gripper_action = gripper_action_dict["left"]  # 11 LEFT hand joint values
        right_gripper_action = gripper_action_dict["right"]  # 11 RIGHT hand joint values

        if action_noise_dict is not None:
            pos_noise_left = action_noise_dict["left"] * torch.randn_like(target_left_eef_pos)
            pos_noise_right = action_noise_dict["right"] * torch.randn_like(target_right_eef_pos)
            quat_noise_left = action_noise_dict["left"] * torch.randn_like(target_left_eef_rot_quat)
            quat_noise_right = action_noise_dict["right"] * torch.randn_like(target_right_eef_rot_quat)

            target_left_eef_pos += pos_noise_left
            target_right_eef_pos += pos_noise_right
            target_left_eef_rot_quat += quat_noise_left
            target_right_eef_rot_quat += quat_noise_right

        # Interleave left and right gripper actions to match the hand_joint_names order:
        # L_proximal(5), R_proximal(5), L_intermediate(5), R_intermediate(5), L_distal(1), R_distal(1)
        # This creates a 22-element tensor from the 11 left + 11 right values
        interleaved_gripper = torch.zeros(22, device=left_gripper_action.device, dtype=left_gripper_action.dtype)
        for i, idx in enumerate(self.LEFT_HAND_JOINT_INDICES):
            interleaved_gripper[idx] = left_gripper_action[i]
        for i, idx in enumerate(self.RIGHT_HAND_JOINT_INDICES):
            interleaved_gripper[idx] = right_gripper_action[i]

        return torch.cat(
            (
                target_left_eef_pos,
                target_left_eef_rot_quat,
                target_right_eef_pos,
                target_right_eef_rot_quat,
                interleaved_gripper,
            ),
            dim=0,
        )

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Converts action (compatible with env.step) to a target pose for the end effector controller.
        Inverse of @target_eef_pose_to_action. Usually used to infer a sequence of target controller poses
        from a demonstration trajectory using the recorded actions.

        Args:
            action: Environment action. Shape is (num_envs, action_dim).

        Returns:
            A dictionary of eef pose torch.Tensor that @action corresponds to.
        """
        target_poses = {}

        target_left_wrist_position = action[:, 0:3]
        target_left_rot_mat = PoseUtils.matrix_from_quat(action[:, 3:7])
        target_pose_left = PoseUtils.make_pose(target_left_wrist_position, target_left_rot_mat)
        target_poses["left"] = target_pose_left

        target_right_wrist_position = action[:, 7:10]
        target_right_rot_mat = PoseUtils.matrix_from_quat(action[:, 10:14])
        target_pose_right = PoseUtils.make_pose(target_right_wrist_position, target_right_rot_mat)
        target_poses["right"] = target_pose_right

        return target_poses

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Extracts the gripper actuation part from a sequence of env actions (compatible with env.step).

        The GR1T2 robot has interleaved hand joint ordering, so we need to extract
        LEFT and RIGHT hand joints from their respective positions in the 22-element
        hand joint action (indices 14:36 in the full action tensor).

        Args:
            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).

        Returns:
            A dictionary of torch.Tensor gripper actions. Key to each dict is an eef_name.
            Each value is a tensor of shape (num_envs, num_steps, 11) containing only
            that hand's joint values.
        """
        # Extract the 22 hand joint values from the action tensor
        hand_joints = actions[:, 14:36]  # Shape: (num_envs, num_steps, 22) or (num_steps, 22)

        # Handle both 2D and 3D tensor inputs
        if hand_joints.dim() == 2:
            # Shape: (num_steps, 22)
            left_gripper = hand_joints[:, self.LEFT_HAND_JOINT_INDICES]
            right_gripper = hand_joints[:, self.RIGHT_HAND_JOINT_INDICES]
        else:
            # Shape: (num_envs, num_steps, 22)
            left_gripper = hand_joints[:, :, self.LEFT_HAND_JOINT_INDICES]
            right_gripper = hand_joints[:, :, self.RIGHT_HAND_JOINT_INDICES]

        return {"left": left_gripper, "right": right_gripper}

    def get_expected_attached_object(self, eef_name: str, subtask_index: int, env_cfg) -> str | None:
        """
        (SkillGen) Return the expected attached object for the given EEF/subtask.

        For GR1T2 bimanual tasks, determines which object should be attached based on
        the current subtask and whether a grasp has occurred in the preceding subtask.

        Args:
            eef_name: Name of the end-effector ("left" or "right")
            subtask_index: Current subtask index
            env_cfg: Environment configuration containing subtask_configs

        Returns:
            Object name to attach, or None if no attachment expected
        """
        if eef_name not in env_cfg.subtask_configs:
            return None

        subtask_configs = env_cfg.subtask_configs[eef_name]
        if not (0 <= subtask_index < len(subtask_configs)):
            return None

        # Check if we're past a grasp subtask - if so, we should have the object attached
        if subtask_index > 0:
            prev_cfg = subtask_configs[subtask_index - 1]
            prev_signal = str(prev_cfg.subtask_term_signal).lower()

            # If the previous subtask was a grasp, we should have the object attached
            if "grasp" in prev_signal:
                attached_object = prev_cfg.object_ref
                print(f"[GR1T2 Attachment] EEF '{eef_name}' subtask {subtask_index}: "
                      f"expecting '{attached_object}' attached (grasped in subtask {subtask_index - 1})")
                return attached_object

        return None
