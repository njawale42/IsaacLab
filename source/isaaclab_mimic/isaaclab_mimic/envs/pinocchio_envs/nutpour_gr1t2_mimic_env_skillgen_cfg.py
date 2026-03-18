# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab.envs.mimic_env_cfg import (
    MimicEnvCfg,
    SubTaskConfig,
    SubTaskConstraintConfig,
    SubTaskConstraintType,
)
from isaaclab.utils.configclass import configclass
from isaaclab.envs.common import ViewerCfg

from isaaclab_tasks.manager_based.manipulation.pick_place.nutpour_gr1t2_pink_ik_env_cfg import NutPourGR1T2PinkIKEnvCfg


@configclass
class NutPourGR1T2MimicEnvSkillGenCfg(NutPourGR1T2PinkIKEnvCfg, MimicEnvCfg):

    def __post_init__(self):
        # Calling post init of parents
        super().__post_init__()
        # from isaaclab.envs.common import ViewerCfg
        self.viewer = ViewerCfg(
            eye=(0.0, 2.0, 2.0), lookat=(0.0, 0.0, 0.2), origin_type="asset_body", asset_name="robot", body_name="base_link"
        )

        # Enable SkillGen to consume start boundaries and plan transitions
        self.datagen_config.use_skillgen = True

        # Override the existing values
        self.datagen_config.name = "gr1t2_nut_pouring_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 1000
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_relative = False
        self.datagen_config.generation_joint_pos = False
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.num_demo_to_render = 10
        self.datagen_config.num_fail_demo_to_render = 25
        self.datagen_config.seed = 10
        # Scheduling parameters for bimanual collision avoidance
        # Higher densify_factor = more waypoints = less skipping during discretization
        self.datagen_config.schedule_densify_factor = 1
        # Split blocks so delayed arm holds further back (e.g. at wp 20 instead of at bowl wp 31)
        self.datagen_config.schedule_max_block_size = 30
        self.datagen_config.schedule_pair_batch = 4096
        # Margin must exceed min_pen of right 20-30 vs left pour (~0.02–0.12m) so those
        # chunks are mutex; then right holds at wp 20 instead of wp 31 (at bowl).
        self.datagen_config.schedule_collision_margin = 0.0
        # False: hold edge targets latter's skill block (latter MP can run in parallel).
        # True: hold edge targets latter's first block (MP); latter does not start subtask until former finishes.
        self.datagen_config.schedule_hold_latter_entire_subtask = False
        # Former part the latter waits for: "mp" (latter can only start its MP after former's MP
        # for the constrained subtask completes), "skill", or "entire".
        self.datagen_config.schedule_hold_former_part = "mp"
        # Log which link pair has minimum penetration for each colliding block pair (for debugging).
        self.datagen_config.schedule_debug_collision_links = True
        # If True, skip gripper injection into planner joints (for testing if injection causes false collisions).
        self.datagen_config.schedule_debug_disable_gripper_injection = False
        # If False, each arm uses its own torso from its waypoint (tests shared-torso false-collision hypothesis).
        self.datagen_config.schedule_collision_use_shared_torso = False
        # (Non-DAG only) Cap injected hold waypoints; 0 = no cap. Use 1 to minimize; DAG does not inject.
        self.datagen_config.schedule_max_hold_waypoints = 1
        self.datagen_config.debug_schedule_replay = False  # Enable debug prints for schedule replay
        self.datagen_config.debug_gripper_detail = False  # Show every gripper command (verbose)
        self.datagen_config.skill_gripper_delay_steps = 8 #18 #18  # ~1.5 seconds delay at 50Hz
        self.datagen_config.skill_gripper_interp_steps = 4  # ~1 second smooth close at 50Hz
        # self.datagen_config.schedule_min_dt = 0.05  # Match step_dt for consistency
        self.datagen_config.schedule_all_offline = True
        self.datagen_config.max_joint_step_rad = 0.1 #0.01  # Smoother wrist motion (2.9 degrees max per tick)
        self.datagen_config.final_hold_steps = 2
        self.datagen_config.goal_correction_steps = 10
        self.datagen_config.goal_correction_threshold = 0.005

        # The following are the subtask configurations for the stack task.
        subtask_configs = []
        # subtask_configs.append(
        #     SubTaskConfig(
        #         # Each subtask involves manipulation with respect to a single object frame.
        #         object_ref="sorting_bowl",
        #         # This key corresponds to the binary indicator in "datagen_info" that signals
        #         # when this subtask is finished (e.g., on a 0 to 1 edge).
        #         subtask_term_signal="idle_right",
        #         first_subtask_start_offset_range=(0, 0),
        #         # Randomization range for starting index of the first subtask
        #         subtask_term_offset_range=(0, 0),
        #         # Selection strategy for the source subtask segment during data generation
        #         selection_strategy="nearest_neighbor_object",
        #         # Optional parameters for the selection strategy function
        #         selection_strategy_kwargs={"nn_k": 3},
        #         # Amount of action noise to apply during this subtask
        #         action_noise=0.0,
        #         # Number of interpolation steps to bridge to this subtask segment
        #         num_interpolation_steps=5,
        #         # Additional fixed steps for the robot to reach the necessary pose
        #         num_fixed_steps=0,
        #         # If True, apply action noise during the interpolation phase and execution
        #         apply_noise_during_interpolation=False,
        #     )
        # )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="sorting_bowl",
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="grasp_right",
                first_subtask_start_offset_range=(0, 0),
                # Randomization range for starting index of the first subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.0,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=3,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="sorting_scale",
                # Corresponding key for the binary indicator in "datagen_info" for completion
                subtask_term_signal="place_bowl_on_scale_right",
                # Time offsets for data generation when splitting a trajectory
                subtask_term_offset_range=(0, 0),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.0,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=3,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["right"] = subtask_configs

        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="sorting_beaker",
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="grasp_left",
                first_subtask_start_offset_range=(0, 0),
                # Randomization range for starting index of the first subtask
                subtask_term_offset_range=(0, 0),
                # Selection strategy for the source subtask segment during data generatio
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.0,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="sorting_bowl",
                # Corresponding key for the binary indicator in "datagen_info" for completion
                subtask_term_signal="pour_left",
                # Time offsets for data generation when splitting a trajectory
                subtask_term_offset_range=(0, 0),
                # Selection strategy for source subtask segment
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.0,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=0,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
            )
        )
        self.subtask_configs["left"] = subtask_configs

        self.task_constraint_configs.append(
            SubTaskConstraintConfig(
                eef_subtask_constraint_tuple=[("left", 1), ("right", 0)],
                constraint_type=SubTaskConstraintType.SEQUENTIAL,
                sequential_min_time_diff=1,
            )
        )
