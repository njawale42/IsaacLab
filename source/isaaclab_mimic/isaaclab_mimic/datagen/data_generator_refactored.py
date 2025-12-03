# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Base class for data generator.
"""
import asyncio
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np
import torch
from typing import Any, cast

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import (
    ManagerBasedRLMimicEnv,
    MimicEnvCfg,
    SubTaskConstraintCoordinationScheme,
    SubTaskConstraintType,
)
from isaaclab.managers import TerminationTermCfg

from isaaclab_mimic.datagen.datagen_info import DatagenInfo
from isaaclab_mimic.datagen.scheduling import ArmPath, DiscreteSchedule, build_collision_aware_schedule
from isaaclab_mimic.datagen.selection_strategy import make_selection_strategy
from isaaclab_mimic.datagen.waypoint import MultiWaypoint, Waypoint, WaypointSequence, WaypointTrajectory

from .datagen_info_pool import DataGenInfoPool


def transform_source_data_segment_using_delta_object_pose(
    src_eef_poses: torch.Tensor,
    delta_obj_pose: torch.Tensor,
) -> torch.Tensor:
    """
    Transform a source data segment (object-centric subtask segment from source demonstration) using
    a delta object pose.

    Args:
        src_eef_poses: pose sequence (shape [T, 4, 4]) for the sequence of end effector control poses
            from the source demonstration
        delta_obj_pose: 4x4 delta object pose

    Returns:
        transformed_eef_poses: transformed pose sequence (shape [T, 4, 4])
    """
    return PoseUtils.pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses,
        pose_A_in_B=delta_obj_pose[None],
    )


def transform_source_data_segment_using_object_pose(
    obj_pose: torch.Tensor,
    src_eef_poses: torch.Tensor,
    src_obj_pose: torch.Tensor,
) -> torch.Tensor:
    """
    Transform a source data segment (object-centric subtask segment from source demonstration) such that
    the relative poses between the target eef pose frame and the object frame are preserved. Recall that
    each object-centric subtask segment corresponds to one object, and consists of a sequence of
    target eef poses.

    Args:
        obj_pose: 4x4 object pose in current scene
        src_eef_poses: pose sequence (shape [T, 4, 4]) for the sequence of end effector control poses
            from the source demonstration
        src_obj_pose: 4x4 object pose from the source demonstration

    Returns:
        transformed_eef_poses: transformed pose sequence (shape [T, 4, 4])
    """

    # Transform source end effector poses to be relative to source object frame
    src_eef_poses_rel_obj = PoseUtils.pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses,
        pose_A_in_B=PoseUtils.pose_inv(src_obj_pose[None]),
    )

    # Apply relative poses to current object frame to obtain new target eef poses
    transformed_eef_poses = PoseUtils.pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses_rel_obj,
        pose_A_in_B=obj_pose[None],
    )
    return transformed_eef_poses


def get_delta_pose_with_scheme(
    src_obj_pose: torch.Tensor,
    cur_obj_pose: torch.Tensor,
    task_constraint: dict,
) -> torch.Tensor:
    """
    Get the delta pose with the given coordination scheme.

    Args:
        src_obj_pose: 4x4 object pose in source scene
        cur_obj_pose: 4x4 object pose in current scene
        task_constraint: task constraint dictionary

    Returns:
        delta_pose: 4x4 delta pose
    """
    coord_transform_scheme = task_constraint["coordination_scheme"]
    device = src_obj_pose.device
    if coord_transform_scheme == SubTaskConstraintCoordinationScheme.TRANSFORM:
        delta_pose = PoseUtils.get_delta_object_pose(cur_obj_pose, src_obj_pose)
        # add noise to delta pose position
    elif coord_transform_scheme == SubTaskConstraintCoordinationScheme.TRANSLATE:
        delta_pose = torch.eye(4, device=device)
        delta_pose[:3, 3] = cur_obj_pose[:3, 3] - src_obj_pose[:3, 3]
    elif coord_transform_scheme == SubTaskConstraintCoordinationScheme.REPLAY:
        delta_pose = torch.eye(4, device=device)
    else:
        raise ValueError(
            f"coordination coord_transform_scheme {coord_transform_scheme} not supported, only"
            f" {[e.value for e in SubTaskConstraintCoordinationScheme]} are supported"
        )

    pos_noise_scale = task_constraint["coordination_scheme_pos_noise_scale"]
    rot_noise_scale = task_constraint["coordination_scheme_rot_noise_scale"]
    if pos_noise_scale != 0.0 or rot_noise_scale != 0.0:
        pos = delta_pose[:3, 3]
        rot = delta_pose[:3, :3]
        pos_new, rot_new = PoseUtils.add_uniform_noise_to_pose(pos, rot, pos_noise_scale, rot_noise_scale)
        delta_pose = torch.eye(4, device=device)
        delta_pose[:3, 3] = pos_new
        delta_pose[:3, :3] = rot_new
    return delta_pose


@dataclass
class GenerationBuffers:
    """
    Lightweight container that mirrors the buffers recorded by the environment.

    The generator accumulates raw simulator outputs (states, observations, actions, and
    a rolling success flag) at every control tick. Bundling them inside a dataclass keeps
    the main loop uncluttered and makes it obvious which structures are mutated in-place.

    Attributes:
        states: List of environment states returned by the recorder across timesteps.
        observations: Raw observation dictionaries paired with each executed action.
        actions: Torch tensors representing the low-level actions issued at each tick.
        success: Aggregate boolean indicating whether any execution window satisfied the
            provided success termination condition.
        joint_positions: Snapshot of the robot's joint configuration (per tick) which
            can be post-processed into per-arm joint paths for collision-aware scheduling.
    """

    states: list[Any] = field(default_factory=list)
    observations: list[Any] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    success: bool = False
    joint_positions: list[torch.Tensor] = field(default_factory=list)


@dataclass
class EEFGenerationState:
    """
    Tracks the per-end-effector state machine used while stitching subtasks.

    Each arm/EEF independently progresses through its subtask list, occasionally pausing
    for coordination or being temporarily swapped out for a motion-planned transition.
    Keeping the bookkeeping in a dedicated structure avoids large clusters of parallel
    dictionaries and makes serialization/debugging straightforward.

    Attributes:
        current_subtask_index: Index into `env_cfg.subtask_configs[eef_name]` that is
            currently being generated/executed. Set to -1 while a motion plan is active.
        current_trajectory: Waypoints (post interpolation) that are currently executing.
        subtask_step_index: Pointer into `current_trajectory`. `None` signals "ready to
            build the next subtask trajectory".
        next_subtask_index_after_motion: Cached index used to resume the real skill after
            finishing a motion planner transit segment.
        next_subtask_trajectory_after_motion: Stored `WaypointTrajectory` representing the
            actual skill to resume after the motion-planned path completes.
        subtasks_done: Flag raised once the end-effector has finished its final skill; the
            final waypoint is duplicated to keep the arm stationary during other arms' work.
    """

    current_subtask_index: int = 0
    current_trajectory: list[Waypoint] = field(default_factory=list)
    subtask_step_index: int | None = None
    next_subtask_index_after_motion: int | None = None
    next_subtask_trajectory_after_motion: WaypointTrajectory | None = None
    subtasks_done: bool = False
    waiting_on_constraint: bool = False
    constraint_hold_waypoint: Waypoint | None = None
    last_commanded_gripper_action: torch.Tensor | None = None
    last_commanded_joint_position: torch.Tensor | None = None
    last_joint_position: torch.Tensor | None = None
    subtask_started: bool = False
    current_joint_trajectory: list[torch.Tensor | None] = field(default_factory=list)


class DataGeneratorRefactored:
    """
    The main data generator class that generates new trajectories from source datasets.

    The data generator, inspired by the MimicGen, enables the generation of new datasets based on a few human
    collected source demonstrations.

    The data generator works by parsing demonstrations into object-centric subtask segments, stored in DataGenInfoPool.
    It then adapts these subtask segments to new scenes by transforming each segment according to the new scene's context,
    stitching them into a coherent trajectory for a robotic end-effector to execute.
    """

    def __init__(
        self,
        env: ManagerBasedRLMimicEnv,
        src_demo_datagen_info_pool: DataGenInfoPool | None = None,
        dataset_path: str | None = None,
        demo_keys: list[str] | None = None,
        *,
        skillgen_type: str = "single_arm",
        schedule_all: bool = False,
    ):
        """
        Args:
            env: environment to use for data generation
            src_demo_datagen_info_pool: source demo datagen info pool
            dataset_path: path to hdf5 dataset to use for generation
            demo_keys: list of demonstration keys to use in file. If not provided, all demonstration keys
                will be used.
            schedule_all: When True, constructs an offline schedule of all planner+skill segments prior to
                execution and replays it with collision-aware retiming (bimanual only).
        """
        self.env = env
        self.env_cfg = env.cfg
        assert isinstance(self.env_cfg, MimicEnvCfg)
        try:
            self._robot_articulation = self.env.scene["robot"]
        except (AttributeError, KeyError) as exc:
            raise AttributeError(
                "DataGeneratorRefactored expects the environment scene to expose a 'robot' articulation"
            ) from exc
        self.dataset_path = dataset_path
        self.skillgen_type = skillgen_type
        self.schedule_all = schedule_all
        self.schedule_all_offline = getattr(self.env_cfg.datagen_config, "schedule_all_offline", False)
        if self.schedule_all_offline:
            self.schedule_all = True

        # Sanity check on task spec offset ranges - final subtask should not have any offset randomization
        for subtask_configs in self.env_cfg.subtask_configs.values():
            assert subtask_configs[-1].subtask_term_offset_range[0] == 0
            assert subtask_configs[-1].subtask_term_offset_range[1] == 0

        self.demo_keys = demo_keys

        if src_demo_datagen_info_pool is not None:
            self.src_demo_datagen_info_pool = src_demo_datagen_info_pool
        elif dataset_path is not None:
            self.src_demo_datagen_info_pool = DataGenInfoPool(
                env=self.env, env_cfg=self.env_cfg, device=self.env.device
            )
            self.src_demo_datagen_info_pool.load_from_dataset_file(dataset_path, select_demo_keys=self.demo_keys)
        else:
            raise ValueError("Either src_demo_datagen_info_pool or dataset_path must be provided")

    def __repr__(self):
        """
        Pretty print this object.
        """
        msg = str(self.__class__.__name__)
        msg += " (\n\tdataset_path={}\n\tdemo_keys={}\n)".format(
            self.dataset_path,
            self.demo_keys,
        )
        return msg

    def randomize_subtask_boundaries(self) -> dict[str, np.ndarray]:
        """
        Apply random offsets to sample subtask boundaries according to the task spec.
        Recall that each demonstration is segmented into a set of subtask segments, and the
        end index (and start index when skillgen is enabled) of each subtask can have a random offset.
        """

        randomized_subtask_boundaries = {}

        for eef_name, subtask_boundaries in self.src_demo_datagen_info_pool.subtask_boundaries.items():
            # Initial subtask start and end indices - shape (N, S, 2)
            subtask_boundaries = np.array(subtask_boundaries)

            # Randomize the start of the first subtask
            first_subtask_start_offsets = np.random.randint(
                low=self.env_cfg.subtask_configs[eef_name][0].first_subtask_start_offset_range[0],
                high=self.env_cfg.subtask_configs[eef_name][0].first_subtask_start_offset_range[0] + 1,
                size=subtask_boundaries.shape[0],
            )
            subtask_boundaries[:, 0, 0] += first_subtask_start_offsets

            # For each subtask, sample all end offsets at once for each demonstration
            # Add them to subtask end indices, and then set them as the start indices of next subtask too
            for i in range(subtask_boundaries.shape[1]):
                # If skillgen is enabled, sample a random start offset to increase demonstration variety.
                if self.env_cfg.datagen_config.use_skillgen:
                    start_offset = np.random.randint(
                        low=self.env_cfg.subtask_configs[eef_name][i].subtask_start_offset_range[0],
                        high=self.env_cfg.subtask_configs[eef_name][i].subtask_start_offset_range[1] + 1,
                        size=subtask_boundaries.shape[0],
                    )
                    subtask_boundaries[:, i, 0] += start_offset
                elif i > 0:
                    # Without skillgen, the start of a subtask is the end of the previous one.
                    subtask_boundaries[:, i, 0] = subtask_boundaries[:, i - 1, 1]

                # Sample end offset for each demonstration
                end_offsets = np.random.randint(
                    low=self.env_cfg.subtask_configs[eef_name][i].subtask_term_offset_range[0],
                    high=self.env_cfg.subtask_configs[eef_name][i].subtask_term_offset_range[1] + 1,
                    size=subtask_boundaries.shape[0],
                )
                subtask_boundaries[:, i, 1] = subtask_boundaries[:, i, 1] + end_offsets

            # Ensure non-empty subtasks
            assert np.all((subtask_boundaries[:, :, 1] - subtask_boundaries[:, :, 0]) > 0), "got empty subtasks!"

            # Ensure subtask indices increase (both starts and ends)
            assert np.all(
                (subtask_boundaries[:, 1:, :] - subtask_boundaries[:, :-1, :]) > 0
            ), "subtask indices do not strictly increase"

            # Ensure subtasks are in order
            subtask_inds_flat = subtask_boundaries.reshape(subtask_boundaries.shape[0], -1)
            assert np.all((subtask_inds_flat[:, 1:] - subtask_inds_flat[:, :-1]) >= 0), "subtask indices not in order"

            randomized_subtask_boundaries[eef_name] = subtask_boundaries

        return randomized_subtask_boundaries

    def select_source_demo(
        self,
        eef_name: str,
        eef_pose: np.ndarray,
        object_pose: np.ndarray,
        src_demo_current_subtask_boundaries: np.ndarray,
        subtask_object_name: str,
        selection_strategy_name: str,
        selection_strategy_kwargs: dict | None = None,
    ) -> int:
        """
        Helper method to run source subtask segment selection.

        Args:
            eef_name: name of end effector
            eef_pose: current end effector pose
            object_pose: current object pose for this subtask
            src_demo_current_subtask_boundaries: start and end indices for subtask segment in source demonstrations of shape (N, 2)
            subtask_object_name: name of reference object for this subtask
            selection_strategy_name: name of selection strategy
            selection_strategy_kwargs: extra kwargs for running selection strategy

        Returns:
            selected_src_demo_ind: selected source demo index
        """
        if subtask_object_name is None:
            # no reference object - only random selection is supported
            assert selection_strategy_name == "random", selection_strategy_name

        # We need to collect the datagen info objects over the timesteps for the subtask segment in each source
        # demo, so that it can be used by the selection strategy.
        src_subtask_datagen_infos = []
        for i in range(len(self.src_demo_datagen_info_pool.datagen_infos)):
            # Datagen info over all timesteps of the src trajectory
            src_ep_datagen_info = self.src_demo_datagen_info_pool.datagen_infos[i]

            # Time indices for subtask
            subtask_start_ind = src_demo_current_subtask_boundaries[i][0]
            subtask_end_ind = src_demo_current_subtask_boundaries[i][1]

            # Get subtask segment using indices
            src_subtask_datagen_infos.append(
                DatagenInfo(
                    eef_pose=src_ep_datagen_info.eef_pose[eef_name][subtask_start_ind:subtask_end_ind],
                    # Only include object pose for relevant object in subtask
                    object_poses=(
                        {
                            subtask_object_name: src_ep_datagen_info.object_poses[subtask_object_name][
                                subtask_start_ind:subtask_end_ind
                            ]
                        }
                        if (subtask_object_name is not None)
                        else None
                    ),
                    # Subtask termination signal is unused
                    subtask_term_signals=None,
                    target_eef_pose=src_ep_datagen_info.target_eef_pose[eef_name][subtask_start_ind:subtask_end_ind],
                    gripper_action=src_ep_datagen_info.gripper_action[eef_name][subtask_start_ind:subtask_end_ind],
                )
            )

        # Make selection strategy object
        selection_strategy_obj = make_selection_strategy(selection_strategy_name)

        # Run selection
        if selection_strategy_kwargs is None:
            selection_strategy_kwargs = dict()
        selected_src_demo_ind = selection_strategy_obj.select_source_demo(
            eef_pose=eef_pose,
            object_pose=object_pose,
            src_subtask_datagen_infos=src_subtask_datagen_infos,
            **selection_strategy_kwargs,
        )

        return selected_src_demo_ind

    def generate_eef_subtask_trajectory(
        self,
        env_id: int,
        eef_name: str,
        subtask_ind: int,
        all_randomized_subtask_boundaries: dict,
        runtime_subtask_constraints_dict: dict,
        selected_src_demo_inds: dict,
    ) -> WaypointTrajectory:
        """
        Build a transformed waypoint trajectory for a single subtask of an end-effector.

        This method selects a source demonstration segment for the specified subtask,
        slices the corresponding EEF poses/targets/gripper actions using the randomized
        subtask boundaries, optionally prepends the first robot EEF pose (to interpolate
        from the robot pose instead of the first target), applies an object/coordination
        based transform to the pose sequence, and returns the result as a `WaypointTrajectory`.

        Selection and transforms:

        - Source demo selection is controlled by `SubTaskConfig.selection_strategy` (and kwargs) and by
          `datagen_config.generation_select_src_per_subtask` / `generation_select_src_per_arm`.
        - For coordination constraints, the method reuses/sets the selected source demo ID across
          concurrent subtasks, computes `synchronous_steps`, and stores the pose `transform` used
          to ensure consistent relative motion between tasks.
        - Pose transforms are computed either from object poses (`object_ref`) or via a delta pose
          provided by a concurrent task/coordination scheme.


        Args:
            env_id: Environment index used to query current robot/object poses.
            eef_name: End-effector key whose subtask trajectory is being generated.
            subtask_ind: Index of the subtask within `subtask_configs[eef_name]`.
            all_randomized_subtask_boundaries: For each EEF, an array of per-demo
                randomized (start, end) indices for every subtask.
            runtime_subtask_constraints_dict: In/out dictionary carrying runtime fields
                for constraints (e.g., selected source ID, delta transform, synchronous steps).
            selected_src_demo_inds: Per-EEF mapping for the currently selected source demo index
                (may be reused across arms if configured).

        Returns:
            WaypointTrajectory: The transformed trajectory for the selected subtask segment.
        """
        subtask_configs = self.env_cfg.subtask_configs[eef_name]
        # name of object for this subtask
        subtask_object_name = self.env_cfg.subtask_configs[eef_name][subtask_ind].object_ref
        subtask_object_pose = (
            self.env.get_object_poses(env_ids=[env_id])[subtask_object_name][0]
            if (subtask_object_name is not None)
            else None
        )

        is_first_subtask = subtask_ind == 0

        need_source_demo_selection = is_first_subtask or self.env_cfg.datagen_config.generation_select_src_per_subtask

        if not self.env_cfg.datagen_config.generation_select_src_per_arm:
            need_source_demo_selection = need_source_demo_selection and selected_src_demo_inds[eef_name] is None

        use_delta_transform = None
        coord_transform_scheme = None
        if (eef_name, subtask_ind) in runtime_subtask_constraints_dict:
            if runtime_subtask_constraints_dict[(eef_name, subtask_ind)]["type"] == SubTaskConstraintType.COORDINATION:
                # Avoid selecting source demo if it has already been selected by the concurrent task
                concurrent_task_spec_key = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                    "concurrent_task_spec_key"
                ]
                concurrent_subtask_ind = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                    "concurrent_subtask_ind"
                ]
                concurrent_selected_src_ind = runtime_subtask_constraints_dict[
                    (concurrent_task_spec_key, concurrent_subtask_ind)
                ]["selected_src_demo_ind"]
                if concurrent_selected_src_ind is not None:
                    # The concurrent task has started, so we should use the same source demo
                    selected_src_demo_inds[eef_name] = concurrent_selected_src_ind
                    need_source_demo_selection = False
                    # This transform is set at after the first data generation iteration/first run of the main while loop
                    use_delta_transform = runtime_subtask_constraints_dict[
                        (concurrent_task_spec_key, concurrent_subtask_ind)
                    ]["transform"]
                else:
                    assert (
                        "transform" not in runtime_subtask_constraints_dict[(eef_name, subtask_ind)]
                    ), "transform should not be set for concurrent task"
                    # Need to transform demo according to scheme
                    coord_transform_scheme = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                        "coordination_scheme"
                    ]
                    if coord_transform_scheme != SubTaskConstraintCoordinationScheme.REPLAY:
                        assert (
                            subtask_object_name is not None
                        ), f"object reference should not be None for {coord_transform_scheme} coordination scheme"

        if need_source_demo_selection:
            selected_src_demo_inds[eef_name] = self.select_source_demo(
                eef_name=eef_name,
                eef_pose=self.env.get_robot_eef_pose(env_ids=[env_id], eef_name=eef_name)[0],
                object_pose=subtask_object_pose,
                src_demo_current_subtask_boundaries=all_randomized_subtask_boundaries[eef_name][:, subtask_ind],
                subtask_object_name=subtask_object_name,
                selection_strategy_name=self.env_cfg.subtask_configs[eef_name][subtask_ind].selection_strategy,
                selection_strategy_kwargs=self.env_cfg.subtask_configs[eef_name][subtask_ind].selection_strategy_kwargs,
            )

        assert selected_src_demo_inds[eef_name] is not None
        selected_src_demo_ind = selected_src_demo_inds[eef_name]

        if not self.env_cfg.datagen_config.generation_select_src_per_arm and need_source_demo_selection:
            for itrated_eef_name in self.env_cfg.subtask_configs.keys():
                selected_src_demo_inds[itrated_eef_name] = selected_src_demo_ind

        # Selected subtask segment time indices
        selected_src_subtask_boundary = all_randomized_subtask_boundaries[eef_name][selected_src_demo_ind, subtask_ind]

        if (eef_name, subtask_ind) in runtime_subtask_constraints_dict:
            if runtime_subtask_constraints_dict[(eef_name, subtask_ind)]["type"] == SubTaskConstraintType.COORDINATION:
                # Store selected source demo ind for concurrent task
                runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                    "selected_src_demo_ind"
                ] = selected_src_demo_ind
                concurrent_task_spec_key = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                    "concurrent_task_spec_key"
                ]
                concurrent_subtask_ind = runtime_subtask_constraints_dict[(eef_name, subtask_ind)][
                    "concurrent_subtask_ind"
                ]
                concurrent_src_subtask_inds = all_randomized_subtask_boundaries[concurrent_task_spec_key][
                    selected_src_demo_ind, concurrent_subtask_ind
                ]
                subtask_len = selected_src_subtask_boundary[1] - selected_src_subtask_boundary[0]
                concurrent_subtask_len = concurrent_src_subtask_inds[1] - concurrent_src_subtask_inds[0]
                runtime_subtask_constraints_dict[(eef_name, subtask_ind)]["synchronous_steps"] = min(
                    subtask_len, concurrent_subtask_len
                )

        # Get subtask segment, consisting of the sequence of robot eef poses, target poses, gripper actions
        src_ep_datagen_info = self.src_demo_datagen_info_pool.datagen_infos[selected_src_demo_ind]
        src_subtask_eef_poses = src_ep_datagen_info.eef_pose[eef_name][
            selected_src_subtask_boundary[0] : selected_src_subtask_boundary[1]
        ]
        src_subtask_target_poses = src_ep_datagen_info.target_eef_pose[eef_name][
            selected_src_subtask_boundary[0] : selected_src_subtask_boundary[1]
        ]
        src_subtask_gripper_actions = src_ep_datagen_info.gripper_action[eef_name][
            selected_src_subtask_boundary[0] : selected_src_subtask_boundary[1]
        ]
        src_subtask_joint_positions = None
        if getattr(src_ep_datagen_info, "joint_position", None) is not None:
            src_subtask_joint_positions = src_ep_datagen_info.joint_position[
                selected_src_subtask_boundary[0] : selected_src_subtask_boundary[1]
            ]

        # Get reference object pose from source demo
        src_subtask_object_pose = (
            src_ep_datagen_info.object_poses[subtask_object_name][selected_src_subtask_boundary[0]]
            if (subtask_object_name is not None)
            else None
        )

        prepend_current_joint = is_first_subtask or self.env_cfg.datagen_config.generation_transform_first_robot_pose
        if prepend_current_joint:
            # Source segment consists of first robot eef pose and the target poses. This ensures that
            # We will interpolate to the first robot eef pose in this source segment, instead of the
            # first robot target pose.
            src_eef_poses = torch.cat([src_subtask_eef_poses[0:1], src_subtask_target_poses], dim=0)
            # Account for extra timestep added to @src_eef_poses
            src_subtask_gripper_actions = torch.cat(
                [src_subtask_gripper_actions[0:1], src_subtask_gripper_actions], dim=0
            )
        else:
            # Source segment consists of just the target poses.
            src_eef_poses = src_subtask_target_poses.clone()
            src_subtask_gripper_actions = src_subtask_gripper_actions.clone()
        joint_seed_sequence: list[torch.Tensor | None] | None = None
        if src_subtask_joint_positions is not None:
            joint_seed_sequence = [jp.clone() for jp in src_subtask_joint_positions]
        if prepend_current_joint:
            current_joint = self._robot_articulation.data.joint_pos[env_id].clone()
            if joint_seed_sequence is None:
                joint_seed_sequence = []
            joint_seed_sequence = [current_joint.clone()] + joint_seed_sequence
        if joint_seed_sequence is not None and len(joint_seed_sequence) < src_eef_poses.shape[0]:
            joint_seed_sequence.extend([None] * (src_eef_poses.shape[0] - len(joint_seed_sequence)))

        # Transform source demonstration segment using relevant object pose.
        if use_delta_transform is not None:
            # Use delta transform from concurrent task
            transformed_eef_poses = transform_source_data_segment_using_delta_object_pose(
                src_eef_poses, use_delta_transform
            )

        else:
            if coord_transform_scheme is not None:
                delta_obj_pose = get_delta_pose_with_scheme(
                    src_subtask_object_pose,
                    subtask_object_pose,
                    runtime_subtask_constraints_dict[(eef_name, subtask_ind)],
                )
                transformed_eef_poses = transform_source_data_segment_using_delta_object_pose(
                    src_eef_poses, delta_obj_pose
                )
                runtime_subtask_constraints_dict[(eef_name, subtask_ind)]["transform"] = delta_obj_pose
            else:
                if subtask_object_name is not None:
                    transformed_eef_poses = transform_source_data_segment_using_object_pose(
                        subtask_object_pose,
                        src_eef_poses,
                        src_subtask_object_pose,
                    )
                else:
                    print(f"skipping transformation for {subtask_object_name}")

                    # Skip transformation if no reference object is provided
                    transformed_eef_poses = src_eef_poses

        # Construct trajectory for the transformed segment.
        transformed_joint_positions = joint_seed_sequence
        transformed_seq = WaypointSequence.from_poses(
            poses=transformed_eef_poses,
            gripper_actions=src_subtask_gripper_actions,
            action_noise=subtask_configs[subtask_ind].action_noise,
            joint_positions=transformed_joint_positions,
        )
        transformed_traj = WaypointTrajectory()
        transformed_traj.add_waypoint_sequence(transformed_seq)

        return transformed_traj

    def merge_eef_subtask_trajectory(
        self,
        env_id: int,
        eef_name: str,
        subtask_index: int,
        prev_executed_traj: list[Waypoint] | None,
        subtask_trajectory: WaypointTrajectory,
    ) -> list[Waypoint]:
        """
        Merge a subtask trajectory into an executable trajectory for the robot end-effector.

        This constructs a new `WaypointTrajectory` by first creating an initial
        interpolation segment, then merging the provided `subtask_trajectory` onto it.
        The initial segment begins either from the last executed target waypoint of the
        previous subtask (if configured) or from the robot's current end-effector pose.

        Behavior:

        - If `datagen_config.generation_interpolate_from_last_target_pose` is True and
          this is not the first subtask, interpolation starts from the last waypoint of
          `prev_executed_traj`.
        - Otherwise, interpolation starts from the current robot EEF pose (queried from the env)
          and uses the first waypoint's gripper action and the subtask's action noise.
        - The merge uses `num_interpolation_steps`, `num_fixed_steps`, and optionally
          `apply_noise_during_interpolation` from the corresponding `SubTaskConfig`.
        - The temporary initial waypoint used to enable interpolation is removed before returning.

        Args:
            env_id: Environment index to query the current robot EEF pose when needed.
            eef_name: Name/key of the end-effector whose trajectory is being merged.
            subtask_index: Index of the subtask within `subtask_configs[eef_name]` driving interpolation parameters.
            prev_executed_traj: The previously executed trajectory used to
                seed interpolation from its last target waypoint. Required when interpolation-from-last-target
                is enabled and this is not the first subtask.
            subtask_trajectory:
                Trajectory segment for the current subtask that will be merged after the initial interpolation segment.

        Returns:
            list[Waypoint]: The full sequence of waypoints to execute (initial interpolation segment followed by the subtask segment),
            with the temporary initial waypoint removed.
        """
        is_first_subtask = subtask_index == 0
        # We will construct a WaypointTrajectory instance to keep track of robot control targets
        # and then execute it once we have the trajectory.
        traj_to_execute = WaypointTrajectory()

        if self.env_cfg.datagen_config.generation_interpolate_from_last_target_pose and (not is_first_subtask):
            # Interpolation segment will start from last target pose (which may not have been achieved).
            assert prev_executed_traj is not None
            last_waypoint = prev_executed_traj[-1]
            init_sequence = WaypointSequence(sequence=[last_waypoint])
        else:
            # Interpolation segment will start from current robot eef pose.
            current_joint = self._robot_articulation.data.joint_pos[env_id].clone()
            init_sequence = WaypointSequence.from_poses(
                poses=self.env.get_robot_eef_pose(env_ids=[env_id], eef_name=eef_name)[0].unsqueeze(0),
                gripper_actions=subtask_trajectory[0].gripper_action.unsqueeze(0),
                action_noise=self.env_cfg.subtask_configs[eef_name][subtask_index].action_noise,
                joint_positions=[current_joint],
            )
        traj_to_execute.add_waypoint_sequence(init_sequence)

        # Merge this trajectory into our trajectory using linear interpolation.
        # Interpolation will happen from the initial pose (@init_sequence) to the first element of @transformed_seq.
        traj_to_execute.merge(
            subtask_trajectory,
            num_steps_interp=self.env_cfg.subtask_configs[eef_name][subtask_index].num_interpolation_steps,
            num_steps_fixed=self.env_cfg.subtask_configs[eef_name][subtask_index].num_fixed_steps,
            action_noise=(
                float(self.env_cfg.subtask_configs[eef_name][subtask_index].apply_noise_during_interpolation)
                * self.env_cfg.subtask_configs[eef_name][subtask_index].action_noise
            ),
        )

        # We initialized @traj_to_execute with a pose to allow @merge to handle linear interpolation
        # for us. However, we can safely discard that first waypoint now, and just start by executing
        # the rest of the trajectory (interpolation segment and transformed subtask segment).
        traj_to_execute.pop_first()

        # Return the generated trajectory
        return traj_to_execute.get_full_sequence().sequence

    def _get_mp_gripper_actions_from_source_demo(
        self,
        eef_name: str,
        subtask_ind: int,
        selected_demo_ind: int,
        num_waypoints: int,
        randomized_subtask_boundaries: dict[str, np.ndarray],
    ) -> torch.Tensor:
        """
        Compute gripper actions for motion-planned waypoints by replaying the source
        demo gripper pattern between the previous subtask end and the current subtask
        start (similar to DexMimicGen's `source_demo` scheme).

        This keeps the motion-planning segment's gripper evolution consistent with the
        original demonstration and prevents subtask-specific gripper changes (e.g.,
        grasp / release) from being executed during transit.
        """
        # Subtask bounds for this end-effector and demo: use ORIGINAL (non-randomized)
        # boundaries from the datagen info pool, to mirror DexMimicGen behavior.
        # Shape [S, 2] after conversion from list[tuple[int, int]].
        orig_bounds_list = self.src_demo_datagen_info_pool.subtask_boundaries[eef_name][selected_demo_ind]
        subtask_bounds = np.array(orig_bounds_list, dtype=int)
        current_start = int(subtask_bounds[subtask_ind, 0])
        if subtask_ind > 0:
            prev_end = int(subtask_bounds[subtask_ind - 1, 1])
        else:
            prev_end = 0

        src_ep_datagen_info = self.src_demo_datagen_info_pool.datagen_infos[selected_demo_ind]
        # Gripper actions between previous subtask end and current subtask start (inclusive)
        src_actions = src_ep_datagen_info.gripper_action[eef_name][prev_end : current_start + 1]

        # Fallback: no in-between segment — hold the skill start value
        if src_actions.shape[0] == 0:
            target_val = src_ep_datagen_info.gripper_action[eef_name][current_start]
            return target_val.unsqueeze(0).repeat(num_waypoints, 1)

        L = src_actions.shape[0]
        if L == 1:
            return src_actions.repeat(num_waypoints, 1)

        device = src_actions.device
        # Normalize source timesteps to [0, 1]
        source_timesteps = torch.linspace(0.0, 1.0, L, device=device)
        target_timesteps = torch.linspace(0.0, 1.0, num_waypoints, device=device)

        interpolated = []
        current_idx = 0
        for t in target_timesteps:
            while (current_idx < L - 1) and (source_timesteps[current_idx] < t):
                current_idx += 1
            interpolated.append(src_actions[current_idx])
        return torch.stack(interpolated, dim=0)

    async def generate(
        self,
        env_id: int,
        success_term: TerminationTermCfg,
        env_reset_queue: asyncio.Queue | None = None,
        env_action_queue: asyncio.Queue | None = None,
        pause_subtask: bool = False,
        export_demo: bool = True,
        motion_planner: Any | None = None,
    ) -> dict:
        """
        Attempt to generate a new demonstration.

        Args:
            env_id: environment ID.
            success_term: success function to check if the task is successful.
            env_reset_queue: queue to store environment IDs for reset.
            env_action_queue: queue to store actions for each environment.
            pause_subtask: whether to pause the subtask generation.
            export_demo: whether to export the demo.
            motion_planner: motion planner to use when skillgen is enabled.

        Returns:
            dict containing simulator states, observations, actions, and success flag.
        """
        self._require_motion_planner_if_skillgen(motion_planner)

        # The outer task runners frequently invoke the generator under `torch.inference_mode()`
        # to save dispatcher overhead. That causes tensors created here to become inference
        # tensors, which cuRobo later mutates in-place, tripping PyTorch runtime errors after
        # the first demo. Force the entire generation pass to run in eager mode so planners
        # never inherit inference-only buffers between episodes.
        with torch.inference_mode(False):
            if self.schedule_all:
                try:
                    if self.schedule_all_offline:
                        return await self._generate_with_global_schedule_offline(
                            env_id=env_id,
                            success_term=success_term,
                            env_reset_queue=env_reset_queue,
                            env_action_queue=env_action_queue,
                            pause_subtask=pause_subtask,
                            export_demo=export_demo,
                            motion_planner=motion_planner,
                        )
                    return await self._generate_with_global_schedule(
                        env_id=env_id,
                        success_term=success_term,
                        env_reset_queue=env_reset_queue,
                        env_action_queue=env_action_queue,
                        pause_subtask=pause_subtask,
                        export_demo=export_demo,
                        motion_planner=motion_planner,
                    )
                except Exception as exc:
                    print(f"[DataGenerator] schedule_all execution failed: {exc}")
                    import traceback
                    traceback.print_exc()
                    raise

            return await self._run_episode_once(
                env_id=env_id,
                success_term=success_term,
                env_reset_queue=env_reset_queue,
                env_action_queue=env_action_queue,
                pause_subtask=pause_subtask,
                export_demo=export_demo,
                motion_planner=motion_planner,
                capture_joint_positions=False,
            )

    async def _run_episode_once(
        self,
        env_id: int,
        success_term: TerminationTermCfg,
        env_reset_queue: asyncio.Queue | None,
        env_action_queue: asyncio.Queue | None,
        pause_subtask: bool,
        export_demo: bool,
        motion_planner: Any | None,
        *,
        capture_joint_positions: bool,
    ) -> dict:
        """
        Execute the standard generation loop once and optionally capture joint states per tick.

        Args:
            env_id: Environment replica to operate on.
            success_term: Termination callable.
            env_reset_queue: Sync queue for resets.
            env_action_queue: Queue used when stepping via a remote simulator task.
            pause_subtask: Whether to pause between subtasks.
            export_demo: Whether to export to the recorder manager.
            motion_planner: Motion planner instance (may be None if SkillGen disabled).
            capture_joint_positions: If True, append the robot's joint positions after each
                env step to `GenerationBuffers.joint_positions`.

        Returns:
            A result dictionary akin to `generate`, with an additional `joint_positions`
            entry when `capture_joint_positions` is True.
        """
        env_id_tensor, new_initial_state = await self._reset_environment_for_generation(
            env_id=env_id,
            env_reset_queue=env_reset_queue,
        )

        runtime_subtask_constraints = self._build_runtime_subtask_constraints()
        eef_states = self._initialize_eef_states()
        selected_src_demo_inds: dict[str, int | None] = {
            eef_name: None for eef_name in self.env_cfg.subtask_configs.keys()
        }
        buffers = GenerationBuffers()

        randomized_subtask_boundaries: dict[str, np.ndarray] | None = None
        prev_src_demo_datagen_info_pool_size = 0

        while True:
            pool_lock = self.src_demo_datagen_info_pool.asyncio_lock
            assert pool_lock is not None
            async with pool_lock:
                randomized_subtask_boundaries, prev_src_demo_datagen_info_pool_size = (
                    self._maybe_refresh_randomized_subtask_boundaries(
                        randomized_subtask_boundaries=randomized_subtask_boundaries,
                        prev_pool_size=prev_src_demo_datagen_info_pool_size,
                    )
                )
                assert randomized_subtask_boundaries is not None

                for eef_name, eef_state in eef_states.items():
                    if eef_state.subtasks_done or eef_state.subtask_step_index is not None:
                        continue

                    if eef_state.waiting_on_constraint:
                        if self._should_wait_for_sequential_constraint(
                            eef_name=eef_name,
                            eef_state=eef_state,
                            subtask_index=eef_state.current_subtask_index,
                            runtime_constraints=runtime_subtask_constraints,
                        ):
                            continue
                        self._exit_constraint_hold(eef_state)

                    if self._should_wait_for_sequential_constraint(
                        eef_name=eef_name,
                        eef_state=eef_state,
                        subtask_index=eef_state.current_subtask_index,
                        runtime_constraints=runtime_subtask_constraints,
                    ):
                        self._enter_constraint_hold(
                            env_id=env_id,
                            eef_name=eef_name,
                            eef_state=eef_state,
                        )
                        continue

                    if eef_state.next_subtask_index_after_motion is None:
                        eef_subtask_trajectory = self.generate_eef_subtask_trajectory(
                            env_id=env_id,
                            eef_name=eef_name,
                            subtask_ind=eef_state.current_subtask_index,
                            all_randomized_subtask_boundaries=randomized_subtask_boundaries,
                            runtime_subtask_constraints_dict=runtime_subtask_constraints,
                            selected_src_demo_inds=selected_src_demo_inds,
                        )

                        if self.env_cfg.datagen_config.use_skillgen:
                            transition_started, failure_result = self._start_motion_planned_transition_if_needed(
                                env_id=env_id,
                                eef_name=eef_name,
                                eef_state=eef_state,
                                eef_subtask_trajectory=eef_subtask_trajectory,
                                selected_src_demo_inds=selected_src_demo_inds,
                                randomized_subtask_boundaries=randomized_subtask_boundaries,
                                motion_planner=motion_planner,
                            )
                            if failure_result is not None:
                                return failure_result
                            if transition_started:
                                continue

                        eef_state.current_trajectory = self.merge_eef_subtask_trajectory(
                            env_id=env_id,
                            eef_name=eef_name,
                            subtask_index=eef_state.current_subtask_index,
                            prev_executed_traj=eef_state.current_trajectory,
                            subtask_trajectory=eef_subtask_trajectory,
                        )
                        eef_state.current_joint_trajectory = []
                        eef_state.subtask_step_index = 0
                        eef_state.subtask_started = True
                    else:
                        self._resume_motion_planned_subtask(
                            env_id=env_id,
                            eef_name=eef_name,
                            eef_state=eef_state,
                        )

            eef_waypoints = self._collect_eef_waypoints(
                env_id=env_id,
                runtime_constraints=runtime_subtask_constraints,
                eef_states=eef_states,
                motion_planner=motion_planner,
            )
            multi_waypoint = MultiWaypoint(eef_waypoints)

            exec_results = await multi_waypoint.execute(
                env=self.env,
                success_term=success_term,
                env_id=env_id,
                env_action_queue=env_action_queue,
            )
            self._update_execution_buffers(
                exec_results,
                buffers,
                env_id=env_id,
                capture_joint_positions=capture_joint_positions,
            )

            self._advance_subtask_progress(
                eef_states=eef_states,
                runtime_constraints=runtime_subtask_constraints,
                pause_subtask=pause_subtask,
            )

            if self._all_subtasks_completed(eef_states):
                break

        generated_actions: list[torch.Tensor] | torch.Tensor
        if buffers.actions:
            generated_actions = torch.cat(buffers.actions, dim=0)
        else:
            generated_actions = buffers.actions

        joint_positions: torch.Tensor | None = None
        if capture_joint_positions and buffers.joint_positions:
            joint_positions = torch.stack(buffers.joint_positions, dim=0)

        self.env.recorder_manager.set_success_to_episodes(
            env_id_tensor,
            torch.tensor([[buffers.success]], dtype=torch.bool, device=self.env.device),
        )
        if export_demo:
            self.env.recorder_manager.export_episodes(env_id_tensor)

        results = dict(
            initial_state=new_initial_state,
            states=buffers.states,
            observations=buffers.observations,
            actions=generated_actions,
            success=buffers.success,
            joint_positions=joint_positions,
        )
        return results

    async def _generate_with_global_schedule(
        self,
        env_id: int,
        success_term: TerminationTermCfg,
        env_reset_queue: asyncio.Queue | None,
        env_action_queue: asyncio.Queue | None,
        pause_subtask: bool,
        export_demo: bool,
        motion_planner: Any | None,
    ) -> dict:
        """
        Two-stage pipeline: gather per-arm trajectories then replay them with collision-aware scheduling.
        """
        if self.skillgen_type != "bimanual":
            raise ValueError("--schedule_all is only supported for bimanual SkillGen workflows.")
        planner_left, planner_right = self._resolve_arm_specific_planners(motion_planner)

        first_pass = await self._run_episode_once(
            env_id=env_id,
            success_term=success_term,
            env_reset_queue=env_reset_queue,
            env_action_queue=env_action_queue,
            pause_subtask=pause_subtask,
            export_demo=False,
            motion_planner=motion_planner,
            capture_joint_positions=True,
        )

        actions_tensor = first_pass["actions"]
        actions_tensor = self._coerce_action_history(actions_tensor)
        joint_positions = self._coerce_joint_history(first_pass.get("joint_positions"))
        if joint_positions is None or joint_positions.numel() == 0:
            return first_pass

        arm_paths = self._build_arm_paths_for_schedule(
            actions_tensor=actions_tensor,
            joint_history=joint_positions,
            planner_left=planner_left,
            planner_right=planner_right,
        )

        schedule = build_collision_aware_schedule(
            arm_right=arm_paths["right"],
            arm_left=arm_paths["left"],
            planner_right=planner_right,
            planner_left=planner_left,
            step_dt=self.env.step_dt,
            densify_factor=int(getattr(self.env_cfg.datagen_config, "schedule_densify_factor", 4)),
            pair_batch=int(getattr(self.env_cfg.datagen_config, "schedule_pair_batch", 4096)),
            collision_margin=float(getattr(self.env_cfg.datagen_config, "schedule_collision_margin", 0.01)),
            min_dt=getattr(self.env_cfg.datagen_config, "schedule_min_dt", None),
        )
        schedule.append_hold(int(getattr(self.env_cfg.datagen_config, "final_hold_steps", 0)))

        env_id_tensor = torch.tensor([env_id], dtype=torch.int64, device=self.env.device)
        self.env.scene.reset_to(first_pass["initial_state"], env_ids=env_id_tensor, is_relative=True)
        self.env.recorder_manager.reset(env_ids=env_id_tensor)

        return await self._replay_discrete_schedule(
            env_id=env_id,
            env_id_tensor=env_id_tensor,
            initial_state=first_pass["initial_state"],
            success_term=success_term,
            env_action_queue=env_action_queue,
            arm_paths=arm_paths,
            schedule=schedule,
            export_demo=export_demo,
            planner_map={"left": planner_left, "right": planner_right},
        )

    async def _generate_with_global_schedule_offline(
        self,
        env_id: int,
        success_term: TerminationTermCfg,
        env_reset_queue: asyncio.Queue | None,
        env_action_queue: asyncio.Queue | None,
        pause_subtask: bool,
        export_demo: bool,
        motion_planner: Any | None,
    ) -> dict:
        """
        Offline variant of schedule_all that synthesizes per-arm paths without executing the warmup pass.
        """
        if self.skillgen_type != "bimanual":
            raise ValueError("--schedule_all_offline is only supported for bimanual SkillGen workflows.")
        planner_left, planner_right = self._resolve_arm_specific_planners(motion_planner)

        rollout_result = await self._offline_rollout_waypoints(
            env_id=env_id,
            success_term=success_term,
            env_reset_queue=env_reset_queue,
            pause_subtask=pause_subtask,
            motion_planner=motion_planner,
        )
        if isinstance(rollout_result, dict):
            return rollout_result

        initial_state, waypoint_logs = rollout_result
        arm_paths = self._build_arm_paths_from_waypoints(
            waypoint_logs=waypoint_logs,
            planner_map={"left": planner_left, "right": planner_right},
            env_id=env_id,
        )

        schedule = build_collision_aware_schedule(
            arm_right=arm_paths["right"],
            arm_left=arm_paths["left"],
            planner_right=planner_right,
            planner_left=planner_left,
            step_dt=self.env.step_dt,
            densify_factor=int(getattr(self.env_cfg.datagen_config, "schedule_densify_factor", 4)),
            pair_batch=int(getattr(self.env_cfg.datagen_config, "schedule_pair_batch", 4096)),
            collision_margin=float(getattr(self.env_cfg.datagen_config, "schedule_collision_margin", 0.01)),
            min_dt=getattr(self.env_cfg.datagen_config, "schedule_min_dt", None),
        )
        schedule.append_hold(int(getattr(self.env_cfg.datagen_config, "final_hold_steps", 0)))

        env_id_tensor = torch.tensor([env_id], dtype=torch.int64, device=self.env.device)
        self.env.scene.reset_to(initial_state, env_ids=env_id_tensor, is_relative=True)
        self.env.recorder_manager.reset(env_ids=env_id_tensor)

        return await self._replay_discrete_schedule(
            env_id=env_id,
            env_id_tensor=env_id_tensor,
            initial_state=initial_state,
            success_term=success_term,
            env_action_queue=env_action_queue,
            arm_paths=arm_paths,
            schedule=schedule,
            export_demo=export_demo,
            planner_map={"left": planner_left, "right": planner_right},
        )

    async def _offline_rollout_waypoints(
        self,
        env_id: int,
        success_term: TerminationTermCfg,
        env_reset_queue: asyncio.Queue | None,
        pause_subtask: bool,
        motion_planner: Any | None,
    ) -> tuple[dict, dict[str, list[dict[str, Any]]]] | dict:
        """
        Run the generator state machine without stepping the simulator to record ideal waypoint streams.
        """
        _, initial_state = await self._reset_environment_for_generation(
            env_id=env_id,
            env_reset_queue=env_reset_queue,
        )

        runtime_subtask_constraints = self._build_runtime_subtask_constraints()
        eef_states = self._initialize_eef_states()
        selected_src_demo_inds: dict[str, int | None] = {
            eef_name: None for eef_name in self.env_cfg.subtask_configs.keys()
        }
        waypoint_logs: dict[str, list[dict[str, Any]]] = {
            eef_name: [] for eef_name in self.env_cfg.subtask_configs.keys()
        }

        randomized_subtask_boundaries: dict[str, np.ndarray] | None = None
        prev_src_demo_datagen_info_pool_size = 0

        pool_lock = self.src_demo_datagen_info_pool.asyncio_lock
        assert pool_lock is not None

        while True:
            async with pool_lock:
                randomized_subtask_boundaries, prev_src_demo_datagen_info_pool_size = (
                    self._maybe_refresh_randomized_subtask_boundaries(
                        randomized_subtask_boundaries=randomized_subtask_boundaries,
                        prev_pool_size=prev_src_demo_datagen_info_pool_size,
                    )
                )
                assert randomized_subtask_boundaries is not None

                for eef_name, eef_state in eef_states.items():
                    if eef_state.subtasks_done or eef_state.subtask_step_index is not None:
                        continue

                    if eef_state.waiting_on_constraint:
                        if self._should_wait_for_sequential_constraint(
                            eef_name=eef_name,
                            eef_state=eef_state,
                            subtask_index=eef_state.current_subtask_index,
                            runtime_constraints=runtime_subtask_constraints,
                        ):
                            continue
                        self._exit_constraint_hold(eef_state)

                    if self._should_wait_for_sequential_constraint(
                        eef_name=eef_name,
                        eef_state=eef_state,
                        subtask_index=eef_state.current_subtask_index,
                        runtime_constraints=runtime_subtask_constraints,
                    ):
                        self._enter_constraint_hold(
                            env_id=env_id,
                            eef_name=eef_name,
                            eef_state=eef_state,
                        )
                        continue

                    if eef_state.next_subtask_index_after_motion is None:
                        eef_subtask_trajectory = self.generate_eef_subtask_trajectory(
                            env_id=env_id,
                            eef_name=eef_name,
                            subtask_ind=eef_state.current_subtask_index,
                            all_randomized_subtask_boundaries=randomized_subtask_boundaries,
                            runtime_subtask_constraints_dict=runtime_subtask_constraints,
                            selected_src_demo_inds=selected_src_demo_inds,
                        )

                        if self.env_cfg.datagen_config.use_skillgen:
                            transition_started, failure_result = self._start_motion_planned_transition_if_needed(
                                env_id=env_id,
                                eef_name=eef_name,
                                eef_state=eef_state,
                                eef_subtask_trajectory=eef_subtask_trajectory,
                                selected_src_demo_inds=selected_src_demo_inds,
                                randomized_subtask_boundaries=randomized_subtask_boundaries,
                                motion_planner=motion_planner,
                            )
                            if failure_result is not None:
                                return failure_result
                            if transition_started:
                                continue

                        eef_state.current_trajectory = self.merge_eef_subtask_trajectory(
                            env_id=env_id,
                            eef_name=eef_name,
                            subtask_index=eef_state.current_subtask_index,
                            prev_executed_traj=eef_state.current_trajectory,
                            subtask_trajectory=eef_subtask_trajectory,
                        )
                        eef_state.current_joint_trajectory = cast(
                            list[torch.Tensor | None],
                            [waypoint.joint_seed for waypoint in eef_state.current_trajectory],
                        )
                        eef_state.subtask_step_index = 0
                        eef_state.subtask_started = True
                    else:
                        self._resume_motion_planned_subtask(
                            env_id=env_id,
                            eef_name=eef_name,
                            eef_state=eef_state,
                        )

            eef_waypoints = self._collect_eef_waypoints(
                env_id=env_id,
                runtime_constraints=runtime_subtask_constraints,
                eef_states=eef_states,
                motion_planner=motion_planner,
            )
            if not eef_waypoints:
                raise RuntimeError("Offline rollout produced no waypoints for active timestep.")

            for eef_name, waypoint in eef_waypoints.items():
                eef_state = eef_states[eef_name]
                joint_vec = eef_state.last_commanded_joint_position
                # Only store joints from motion-planned segments (subtask_index == -1).
                # Skill segment joints come from source demo and are not valid for
                # the current scene, so we leave them as None to force IK solving later.
                in_motion_planned_phase = eef_state.current_subtask_index == -1
                stored_joint = joint_vec.clone() if (joint_vec is not None and in_motion_planned_phase) else None
                joint_seed = waypoint.joint_seed
                if joint_seed is not None:
                    joint_seed = joint_seed.to(device=self.env.device)
                waypoint_logs[eef_name].append(
                    {
                        "waypoint": deepcopy(waypoint),
                        "joint": stored_joint,
                        "joint_seed": None if joint_seed is None else joint_seed.clone(),
                    }
                )

            self._advance_subtask_progress(
                eef_states=eef_states,
                runtime_constraints=runtime_subtask_constraints,
                pause_subtask=pause_subtask,
            )

            if self._all_subtasks_completed(eef_states):
                break

        return initial_state, waypoint_logs

    def _joint_tensor_to_list(
        self, joint_tensor: torch.Tensor | None, expected_len: int
    ) -> list[torch.Tensor]:
        if joint_tensor is None:
            return []
        tensor = joint_tensor
        if tensor.shape[0] > expected_len:
            tensor = tensor[:expected_len]
        elif tensor.shape[0] < expected_len:
            pad = tensor[-1:].repeat(expected_len - tensor.shape[0], 1)
            tensor = torch.cat([tensor, pad], dim=0)
        return [tensor[i].clone() for i in range(expected_len)]

    def _build_arm_paths_from_waypoints(
        self,
        waypoint_logs: dict[str, list[dict[str, Any]]],
        planner_map: dict[str, Any],
        env_id: int,
    ) -> dict[str, ArmPath]:
        """Convert offline waypoint logs into ArmPath objects usable by the scheduler."""
        arm_paths: dict[str, ArmPath] = {}
        for eef_name, entries in waypoint_logs.items():
            if not entries:
                continue
            poses = torch.stack([entry["waypoint"].pose.clone() for entry in entries], dim=0)
            gripper_actions = torch.stack(
                [entry["waypoint"].gripper_action.clone() for entry in entries],
                dim=0,
            )
            joint_positions = self._extract_joint_history_from_entries(
                eef_name=eef_name,
                entries=entries,
                planner=planner_map.get(eef_name),
                env_id=env_id,
            )
            arm_paths[eef_name] = ArmPath(
                name=eef_name,
                poses=poses,
                gripper_actions=gripper_actions,
                joint_positions=joint_positions,
            )
        if "left" not in arm_paths or "right" not in arm_paths:
            missing = [k for k in ("left", "right") if k not in arm_paths]
            raise ValueError(f"Offline scheduling requires left/right trajectories; missing {missing}.")
        return arm_paths

    def _extract_joint_history_from_entries(
        self,
        eef_name: str,
        entries: list[dict[str, Any]],
        planner: Any,
        env_id: int,
    ) -> torch.Tensor:
        """Compose a contiguous joint history, re-solving IK only for entries lacking stored data.

        For skill segments (entries without stored joints), IK is solved to recover joint positions.
        At skill→motion-plan boundaries, we use the motion plan's start joint as the target for the
        skill segment's last pose to ensure continuity.
        """
        if planner is None:
            raise ValueError(f"No planner provided for end effector '{eef_name}'.")

        joint_list: list[torch.Tensor] = []
        last_joint: torch.Tensor | None = None
        pose_buffer: list[torch.Tensor] = []
        index_buffer: list[int] = []
        seed_buffer: list[torch.Tensor | None] = []
        planner_device = None
        tensor_args = getattr(planner, "tensor_args", None)
        if tensor_args is None and hasattr(planner, "motion_gen"):
            tensor_args = getattr(planner.motion_gen, "tensor_args", None)
        if tensor_args is not None:
            planner_device = getattr(tensor_args, "device", None)
        target_device = planner_device or self.env.device

        def _flush_buffer(
            initial_joint: torch.Tensor | None,
            end_target_joint: torch.Tensor | None = None,
        ) -> None:
            """Solve IK for buffered poses, ensuring continuity at MP boundaries.

            When end_target_joint is provided (the motion plan's start joint), we ensure
            the skill segment's last joint exactly matches it for smooth transitions.
            We solve IK for all poses except the last, then splice in the MP start joint.
            """
            nonlocal last_joint
            if not pose_buffer:
                return

            start_joint = initial_joint
            if start_joint is None:
                for seed in seed_buffer:
                    if seed is not None:
                        start_joint = seed
                        break

            if end_target_joint is not None and len(pose_buffer) > 1:
                # Solve IK for all poses except the last
                poses_except_last = torch.stack(pose_buffer[:-1], dim=0)
                seeds_except_last = seed_buffer[:-1]
                solved = self._solve_waypoints_to_joint_positions(
                    poses=poses_except_last,
                    planner=planner,
                    env_id=env_id,
                    initial_joint=start_joint,
                    seed_sequence=seeds_except_last,
                )
                for offset, idx in enumerate(index_buffer[:-1]):
                    joint_list[idx] = solved[offset]
                # For the last pose, use the MP start joint directly for exact continuity
                joint_list[index_buffer[-1]] = end_target_joint.to(device=target_device)
            elif end_target_joint is not None and len(pose_buffer) == 1:
                # Only one pose - use MP start directly
                joint_list[index_buffer[0]] = end_target_joint.to(device=target_device)
            else:
                # No MP boundary - solve IK normally
                poses = torch.stack(pose_buffer, dim=0)
                solved = self._solve_waypoints_to_joint_positions(
                    poses=poses,
                    planner=planner,
                    env_id=env_id,
                    initial_joint=start_joint,
                    seed_sequence=seed_buffer,
                )
                for offset, idx in enumerate(index_buffer):
                    joint_list[idx] = solved[offset]

            last_joint = joint_list[index_buffer[-1]]
            pose_buffer.clear()
            index_buffer.clear()
            seed_buffer.clear()

        planner_dof = self._get_planner_dof(planner)

        for idx, entry in enumerate(entries):
            joint_tensor = entry.get("joint", None)
            if idx >= len(joint_list):
                joint_list.append(torch.zeros(planner_dof, device=target_device))
            if joint_tensor is None:
                pose_buffer.append(entry["waypoint"].pose)
                index_buffer.append(idx)
                seed_tensor = entry.get("joint_seed", None)
                if seed_tensor is not None:
                    seed_tensor = self._project_joint_vector_to_planner(seed_tensor, planner).to(device=target_device)
                seed_buffer.append(seed_tensor)
            else:
                # Use the motion plan's start joint as end target for skill segment
                mp_start_joint = self._project_joint_vector_to_planner(joint_tensor, planner).to(device=target_device)
                _flush_buffer(last_joint, end_target_joint=mp_start_joint)
                joint_list[idx] = mp_start_joint
                last_joint = mp_start_joint

        _flush_buffer(last_joint, end_target_joint=None)
        joint_history = torch.stack(joint_list, dim=0)
        planner_dof = self._get_planner_dof(planner)
        if joint_history.shape[1] == planner_dof:
            return joint_history
        return self._project_joint_history_to_planner(joint_history, planner)

    def _get_arm_specific_planner(self, motion_planner: Any, eef_name: str) -> Any:
        """Resolve the underlying planner instance for the specified arm."""
        if motion_planner is None:
            return None
        if hasattr(motion_planner, "_planner_left") and hasattr(motion_planner, "_planner_right"):
            return motion_planner._planner_left if eef_name == "left" else motion_planner._planner_right
        return motion_planner

    def _solve_ik_for_start_state(
        self,
        eef_name: str,
        pose: torch.Tensor,
        planner: Any,
        env_id: int,
    ) -> torch.Tensor | None:
        """Solve IK for the current EEF pose to get a valid start state for motion planning.

        Used when last_joint_position is not available (e.g., first motion plan).

        Args:
            eef_name: Name of the end-effector.
            pose: Current EEF pose (4x4 matrix).
            planner: Arm-specific planner for IK solving.
            env_id: Environment ID.

        Returns:
            Joint configuration that achieves the pose, or None if IK fails.
        """
        if planner is None or pose is None:
            return None

        # Use current articulation joints as seed - clone to avoid inference mode issues
        current_joints = self._robot_articulation.data.joint_pos[env_id].clone().detach()
        seed_joint = self._project_joint_vector_to_planner(current_joints, planner)
        if seed_joint is not None:
            seed_joint = seed_joint.clone().detach()

        # Clone pose to ensure it's not an inference tensor
        poses = pose.clone().detach().unsqueeze(0)
        solved = self._solve_waypoints_to_joint_positions(
            poses=poses,
            planner=planner,
            env_id=env_id,
            initial_joint=seed_joint,
            seed_sequence=[seed_joint] if seed_joint is not None else None,
        )
        if solved is not None and solved.shape[0] > 0:
            return solved[0]
        return seed_joint  # Fallback to current projected joints

    def _solve_ik_for_waypoint_offline(
        self,
        eef_name: str,
        waypoint: Waypoint,
        eef_state: EEFGenerationState,
        motion_planner: Any | None,
        env_id: int,
    ) -> torch.Tensor | None:
        """Solve IK for a skill segment waypoint during offline rollout.

        This ensures that `last_joint_position` reflects IK-consistent joints rather than
        source demo joints, allowing motion planners to start from the correct configuration.

        Args:
            eef_name: Name of the end-effector ("left" or "right").
            waypoint: The current waypoint containing the target pose.
            eef_state: EEF generation state with previous joint info for seeding.
            motion_planner: Motion planner instance (needed for IK solver access).
            env_id: Environment ID.

        Returns:
            IK-solved joint vector, or None if IK fails or planner unavailable.
        """
        if motion_planner is None:
            # Fallback to source demo joints if no planner
            return waypoint.joint_seed

        planner = self._get_arm_specific_planner(motion_planner, eef_name)
        if planner is None:
            return waypoint.joint_seed

        # Use previous joint position as seed for continuity
        seed_joint = eef_state.last_joint_position
        if seed_joint is None:
            seed_joint = waypoint.joint_seed

        pose = waypoint.pose
        if pose is None:
            return seed_joint

        # Solve IK for this single pose
        poses = pose.unsqueeze(0)
        seed_sequence = [seed_joint] if seed_joint is not None else None

        solved = self._solve_waypoints_to_joint_positions(
            poses=poses,
            planner=planner,
            env_id=env_id,
            initial_joint=seed_joint,
            seed_sequence=seed_sequence,
        )
        if solved is not None and solved.shape[0] > 0:
            return solved[0]
        return seed_joint

    def _solve_waypoints_to_joint_positions(
        self,
        poses: torch.Tensor,
        planner: Any,
        env_id: int,
        initial_joint: torch.Tensor | None = None,
        seed_sequence: list[torch.Tensor | None] | None = None,
    ) -> torch.Tensor:
        """Solve IK for each waypoint pose using the cuRobo planner to recover joint trajectories."""
        joints: list[torch.Tensor] = []
        last_valid: torch.Tensor | None = None
        planner_device = None
        tensor_args = getattr(planner, "tensor_args", None)
        if tensor_args is None and hasattr(planner, "motion_gen"):
            tensor_args = getattr(planner.motion_gen, "tensor_args", None)
        if tensor_args is not None:
            planner_device = getattr(tensor_args, "device", None)
        joint_store_device = planner_device if planner_device is not None else self.env.device
        if initial_joint is not None:
            last_valid = initial_joint.to(device=joint_store_device).clone()
            if self.schedule_all_offline:
                self._set_robot_joint_state(env_id, initial_joint, planner)
        for idx_pose, pose in enumerate(poses):
            try:
                device_ctx = (
                    torch.cuda.device(planner_device)
                    if planner_device is not None and planner_device.type == "cuda"
                    else nullcontext()
                )
                with device_ctx:
                    pose_bt = self._convert_world_pose_to_planner_frame(pose, env_id=env_id)
                    pos_bt, rot_bt = PoseUtils.unmake_pose(pose_bt.unsqueeze(0))
                    quat_bt = PoseUtils.quat_from_matrix(rot_bt)[0]
                    pose_obj = planner._make_pose(position=pos_bt[0], quaternion=quat_bt)
                    seed_config = None
                    retract_config = None
                    seed_tensor = None
                    if seed_sequence is not None and idx_pose < len(seed_sequence):
                        seed_tensor = seed_sequence[idx_pose]
                    active_seed = seed_tensor if seed_tensor is not None else last_valid
                    if active_seed is not None:
                        seed_vec = active_seed.to(device=joint_store_device)
                        retract_config = seed_vec.reshape(1, -1)
                        seed_config = seed_vec.reshape(1, 1, -1)
                    try:
                        ik_result = planner.motion_gen.ik_solver.solve_single(
                            pose_obj,
                            retract_config=retract_config,
                            seed_config=seed_config,
                        )
                    except AttributeError:
                        ik_result = planner.motion_gen.ik_solver.solve(
                            pose_obj,
                            retract_config=retract_config,
                            seed_config=seed_config,
                        )
                js_solution = getattr(ik_result, "js_solution", None)
                if js_solution is not None:
                    joint_vec = js_solution.position
                else:
                    joint_vec = ik_result.solution
                if isinstance(joint_vec, torch.Tensor) and joint_vec.ndim > 1:
                    joint_vec = joint_vec[0]
                joint_vec = (
                    torch.as_tensor(joint_vec, dtype=torch.float32)
                    .to(device=joint_store_device)
                    .view(-1)
                )
                joint_vec = self._project_joint_vector_to_planner(joint_vec, planner)
                last_valid = joint_vec
                if self.schedule_all_offline:
                    self._set_robot_joint_state(env_id, joint_vec, planner)
            except Exception:
                if last_valid is None:
                    raise
                joint_vec = last_valid
            joints.append(joint_vec)
        return torch.stack(joints, dim=0)

    def _convert_world_pose_to_planner_frame(self, pose: torch.Tensor, env_id: int) -> torch.Tensor:
        """Convert a world-frame controller pose to the planner's base frame."""
        base_pos = (self._robot_articulation.data.root_pos_w[env_id] - self.env.scene.env_origins[env_id]).to(
            device=self.env.device,
            dtype=torch.float32,
        )
        base_rot = PoseUtils.matrix_from_quat(
            self._robot_articulation.data.root_quat_w[env_id].unsqueeze(0).to(device=self.env.device, dtype=torch.float32)
        )[0]
        T_env_base = PoseUtils.make_pose(base_pos.unsqueeze(0), base_rot.unsqueeze(0))[0]
        T_base_env = torch.linalg.inv(T_env_base)
        return (T_base_env @ pose.to(device=self.env.device, dtype=torch.float32)).clone()

    def _build_arm_paths_for_schedule(
        self,
        actions_tensor: torch.Tensor,
        joint_history: torch.Tensor,
        planner_left: Any,
        planner_right: Any,
    ) -> dict[str, ArmPath]:
        """Convert recorded actions and joint positions into ArmPath objects per arm."""
        # Ensure inputs are tensors (schedule_all path can pass lists/ndarrays)
        if not isinstance(actions_tensor, torch.Tensor):
            # List of tensors / lists / ndarrays -> tensor
            if isinstance(actions_tensor, list):
                actions_tensor = torch.stack(
                    [
                        a if isinstance(a, torch.Tensor) else torch.as_tensor(a)
                        for a in actions_tensor
                    ],
                    dim=0,
                )
            else:
                actions_tensor = torch.as_tensor(actions_tensor)

        if not isinstance(joint_history, torch.Tensor):
            if isinstance(joint_history, list):
                joint_history = torch.stack(
                    [
                        j if isinstance(j, torch.Tensor) else torch.as_tensor(j)
                        for j in joint_history
                    ],
                    dim=0,
                )
            else:
                joint_history = torch.as_tensor(joint_history)

        if actions_tensor.ndim == 1:
            actions_tensor = actions_tensor.unsqueeze(0)
        device = self.env.device
        actions_tensor = actions_tensor.to(device=device)
        joint_history = joint_history.to(device=device)

        target_poses = self.env.action_to_target_eef_pose(actions_tensor)
        gripper_actions = self.env.actions_to_gripper_actions(actions_tensor)

        left_joints = self._project_joint_history_to_planner(joint_history, planner_left)
        right_joints = self._project_joint_history_to_planner(joint_history, planner_right)

        arm_paths = {
            "left": ArmPath(
                name="left",
                poses=target_poses["left"],
                gripper_actions=gripper_actions["left"],
                joint_positions=left_joints,
            ),
            "right": ArmPath(
                name="right",
                poses=target_poses["right"],
                gripper_actions=gripper_actions["right"],
                joint_positions=right_joints,
            ),
        }
        return arm_paths

    def _project_joint_history_to_planner(self, joint_history: torch.Tensor, planner: Any) -> torch.Tensor:
        """Gather the subset of joints (in planner order) from the full articulation history."""
        target_device = getattr(planner.tensor_args, "device", joint_history.device)
        target_dtype = getattr(planner.tensor_args, "dtype", joint_history.dtype)
        joint_names_env = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in self._robot_articulation.data.joint_names
        ]
        planner_joint_names = list(planner.motion_gen.kinematics.joint_names)
        indices: list[int] = []
        for name in planner_joint_names:
            if name not in joint_names_env:
                raise KeyError(f"Joint {name} not found in articulation.")
            indices.append(joint_names_env.index(name))
        index_tensor = torch.tensor(indices, dtype=torch.long, device=joint_history.device)
        sliced = joint_history.index_select(dim=1, index=index_tensor)
        return sliced.to(device=target_device, dtype=target_dtype, copy=True)

    def _project_joint_vector_to_planner(self, joint_vec: torch.Tensor, planner: Any) -> torch.Tensor:
        """Project a single joint vector into the planner's joint order."""
        planner_dof = self._get_planner_dof(planner)
        if joint_vec.shape[-1] == planner_dof:
            return joint_vec
        vec = joint_vec
        if vec.ndim == 1:
            vec = vec.unsqueeze(0)
        projected = self._project_joint_history_to_planner(vec, planner)
        return projected[0]

    def _get_planner_dof(self, planner: Any) -> int:
        """Return the number of joints controlled by the planner."""
        motion_gen = getattr(planner, "motion_gen", None)
        if motion_gen is not None and hasattr(motion_gen, "kinematics"):
            kin = motion_gen.kinematics
            if hasattr(kin, "get_dof"):
                try:
                    return int(kin.get_dof())
                except Exception:
                    pass
            joint_names = getattr(kin, "joint_names", None)
            if joint_names is not None:
                return len(joint_names)
        current_plan = getattr(planner, "_current_plan", None)
        if current_plan is not None and hasattr(current_plan, "position"):
            return current_plan.position.shape[-1]
        raise ValueError("Unable to determine planner DOF for joint projection.")

    def _expand_planner_joint_vector(self, joint_vec: torch.Tensor, planner: Any, env_id: int = 0) -> torch.Tensor:
        """Expand a planner joint vector back into the environment's articulation joint ordering.

        Args:
            joint_vec: Joint vector in planner ordering (subset of full articulation).
            planner: Planner instance providing joint name mapping.
            env_id: Environment ID to use as base for non-planner joints.
        """
        full_size = self._robot_articulation.data.joint_pos.shape[1]
        if joint_vec.shape[0] == full_size:
            return joint_vec.to(device=self.env.device)
        env_joint_names = [
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
            for name in self._robot_articulation.data.joint_names
        ]
        planner_joint_names = list(planner.motion_gen.kinematics.joint_names)
        full = self._robot_articulation.data.joint_pos[env_id].clone()
        vec = joint_vec.to(device=full.device)
        for idx, name in enumerate(planner_joint_names):
            if name not in env_joint_names:
                continue
            env_idx = env_joint_names.index(name)
            full[env_idx] = vec[idx]
        return full

    def _set_robot_joint_state(self, env_id: int, joint_values: torch.Tensor, planner: Any | None = None) -> None:
        """Directly overwrite the articulation joint state for the specified env."""
        target = joint_values.to(device=self.env.device)
        if planner is not None and target.shape[0] != self._robot_articulation.data.joint_pos.shape[1]:
            target = self._expand_planner_joint_vector(target, planner, env_id)
        self._robot_articulation.data.joint_pos[env_id] = target.clone()
        self._robot_articulation.data.joint_pos_target[env_id] = target.clone()
        if hasattr(self._robot_articulation.data, "joint_vel"):
            self._robot_articulation.data.joint_vel[env_id] = torch.zeros_like(target)
        self._robot_articulation.write_data_to_sim()
        self._robot_articulation.update(0.0)

    def _coerce_action_history(self, actions: torch.Tensor | list[Any]) -> torch.Tensor:
        """Convert recorder action history into a contiguous tensor [T, action_dim]."""
        if isinstance(actions, torch.Tensor):
            return actions
        if not actions:
            return torch.zeros((0, self.env.action_manager.total_action_dim), device=self.env.device)

        action_tensors: list[torch.Tensor] = []
        for entry in actions:
            tensor = entry if isinstance(entry, torch.Tensor) else torch.as_tensor(entry)
            tensor = tensor.to(device=self.env.device, dtype=torch.float32)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            action_tensors.append(tensor)
        return torch.cat(action_tensors, dim=0)

    def _coerce_joint_history(self, joints: torch.Tensor | list[Any] | None) -> torch.Tensor | None:
        """Ensure joint snapshot history is a tensor [T, num_joints] on env device."""
        if joints is None:
            return None
        if isinstance(joints, torch.Tensor):
            return joints.to(self.env.device)
        if not joints:
            return None
        joint_tensors: list[torch.Tensor] = []
        for entry in joints:
            tensor = entry if isinstance(entry, torch.Tensor) else torch.as_tensor(entry)
            tensor = tensor.to(device=self.env.device, dtype=torch.float32)
            joint_tensors.append(tensor)
        stacked = torch.stack(joint_tensors, dim=0)
        return stacked

    def _resolve_arm_specific_planners(self, motion_planner: Any | None) -> tuple[Any, Any]:
        """Expose the underlying HumanoidArmCuroboPlanner instances for left/right arms."""
        if motion_planner is None:
            raise ValueError("schedule_all requires a bimanual motion planner instance.")
        planner_left = getattr(motion_planner, "_planner_left", None)
        planner_right = getattr(motion_planner, "_planner_right", None)
        if planner_left is None or planner_right is None:
            raise ValueError("Bimanual motion planner must expose _planner_left/_planner_right for scheduling.")
        return planner_left, planner_right

    def _compute_interpolation_steps(
        self,
        arm_paths: dict[str, ArmPath],
        prev_left_idx: int,
        prev_right_idx: int,
        curr_left_idx: int,
        curr_right_idx: int,
        max_joint_step: float,
    ) -> int:
        """Compute number of interpolation steps needed to smooth large joint jumps.

        Args:
            arm_paths: Per-arm trajectory data containing joint_positions.
            prev_left_idx: Previous left arm waypoint index.
            prev_right_idx: Previous right arm waypoint index.
            curr_left_idx: Current left arm waypoint index.
            curr_right_idx: Current right arm waypoint index.
            max_joint_step: Maximum allowed joint change per step (radians).

        Returns:
            Number of interpolation steps to insert (0 if no smoothing needed).
        """
        if prev_left_idx < 0 or prev_right_idx < 0:
            return 0

        max_delta = 0.0

        # Check left arm joint delta
        left_joints = arm_paths["left"].joint_positions
        if left_joints is not None and left_joints.numel() > 0:
            if prev_left_idx < left_joints.shape[0] and curr_left_idx < left_joints.shape[0]:
                delta = (left_joints[curr_left_idx] - left_joints[prev_left_idx]).abs().max().item()
                max_delta = max(max_delta, delta)

        # Check right arm joint delta
        right_joints = arm_paths["right"].joint_positions
        if right_joints is not None and right_joints.numel() > 0:
            if prev_right_idx < right_joints.shape[0] and curr_right_idx < right_joints.shape[0]:
                delta = (right_joints[curr_right_idx] - right_joints[prev_right_idx]).abs().max().item()
                max_delta = max(max_delta, delta)

        if max_delta <= max_joint_step:
            return 0

        # Compute steps needed to keep joint changes below threshold
        import math
        return int(math.ceil(max_delta / max_joint_step)) - 1

    def _interpolate_waypoints(
        self,
        arm_paths: dict[str, ArmPath],
        prev_left_idx: int,
        prev_right_idx: int,
        curr_left_idx: int,
        curr_right_idx: int,
        alpha: float,
    ) -> dict[str, Waypoint]:
        """Create interpolated waypoints between previous and current indices.

        Args:
            arm_paths: Per-arm trajectory data.
            prev_left_idx: Previous left arm waypoint index.
            prev_right_idx: Previous right arm waypoint index.
            curr_left_idx: Current left arm waypoint index.
            curr_right_idx: Current right arm waypoint index.
            alpha: Interpolation factor in [0, 1] where 0 = prev, 1 = curr.

        Returns:
            Dictionary of interpolated Waypoint objects for each arm.
        """
        def _interp_tensor(t0: torch.Tensor, t1: torch.Tensor, a: float) -> torch.Tensor:
            return t0 * (1.0 - a) + t1 * a

        def _interp_pose(p0: torch.Tensor, p1: torch.Tensor, a: float) -> torch.Tensor:
            """Interpolate 4x4 pose matrices (linear translation, blended rotation)."""
            t_interp = p0[:3, 3] * (1.0 - a) + p1[:3, 3] * a
            R_blend = p0[:3, :3] * (1.0 - a) + p1[:3, :3] * a
            # Orthogonalize via SVD for valid rotation
            U, _, Vh = torch.linalg.svd(R_blend)
            R_interp = U @ Vh
            result = torch.eye(4, device=p0.device, dtype=p0.dtype)
            result[:3, :3] = R_interp
            result[:3, 3] = t_interp
            return result

        result = {}
        arm_indices = [
            ("left", prev_left_idx, curr_left_idx),
            ("right", prev_right_idx, curr_right_idx),
        ]
        for arm_name, prev_idx, curr_idx in arm_indices:
            arm_path = arm_paths[arm_name]
            p0 = arm_path.poses[prev_idx].to(device=self.env.device, dtype=torch.float32)
            p1 = arm_path.poses[curr_idx].to(device=self.env.device, dtype=torch.float32)
            g0 = arm_path.gripper_actions[prev_idx].to(device=self.env.device, dtype=torch.float32)
            g1 = arm_path.gripper_actions[curr_idx].to(device=self.env.device, dtype=torch.float32)

            result[arm_name] = Waypoint(
                pose=_interp_pose(p0, p1, alpha),
                gripper_action=_interp_tensor(g0, g1, alpha),
                noise=0.0,
            )
        return result

    def _apply_scheduled_joint_state(
        self,
        env_id: int,
        arm_paths: dict[str, ArmPath],
        left_idx: int,
        right_idx: int,
        planner_map: dict[str, Any] | None,
    ) -> None:
        """Set the robot articulation to the recorded joint state for the scheduled tick.

        When replaying a collision-aware schedule, the controller receives pose targets but
        the articulation should already be at the configuration the planner computed. This
        helper reads the stored joint vectors from arm_paths and applies them so the robot
        doesn't snap back to its initial configuration before each motion-planned segment.

        Args:
            env_id: Environment ID for which to set joint state.
            arm_paths: Per-arm trajectory data containing joint_positions tensors.
            left_idx: Index into left arm joint_positions for the current tick.
            right_idx: Index into right arm joint_positions for the current tick.
            planner_map: Mapping from arm name to planner used for joint expansion.
        """
        if planner_map is None:
            return

        # Apply joint state for each arm sequentially; _expand_planner_joint_vector
        # preserves other joints when expanding so order matters for overlapping DOFs.
        for arm_name, idx in [("left", left_idx), ("right", right_idx)]:
            arm_path = arm_paths.get(arm_name)
            if arm_path is None:
                continue
            joint_positions = arm_path.joint_positions
            if joint_positions is None or joint_positions.numel() == 0:
                continue
            if idx >= joint_positions.shape[0]:
                continue
            joint_vec = joint_positions[idx]
            planner = planner_map.get(arm_name)
            self._set_robot_joint_state(env_id, joint_vec, planner)

    async def _replay_discrete_schedule(
        self,
        env_id: int,
        env_id_tensor: torch.Tensor,
        initial_state: dict,
        success_term: TerminationTermCfg,
        env_action_queue: asyncio.Queue | None,
        arm_paths: dict[str, ArmPath],
        schedule: DiscreteSchedule,
        export_demo: bool,
        planner_map: dict[str, Any] | None = None,
    ) -> dict:
        """Replay the pre-built schedule and record the resulting buffers.

        Args:
            env_id: Environment ID to replay within.
            env_id_tensor: Tensor version of env_id for recorder calls.
            initial_state: State snapshot used to reset the environment before replay.
            success_term: Termination condition to check for success.
            env_action_queue: Queue for asynchronous action dispatch.
            arm_paths: Per-arm trajectory data (poses, gripper_actions, joint_positions).
            schedule: Discrete schedule mapping ticks to waypoint indices.
            export_demo: Whether to export the resulting demonstration.
            planner_map: Optional mapping from arm name to planner instance used to
                expand planner-space joint vectors to full articulation ordering.
        """
        buffers = GenerationBuffers()

        # Set the initial joint state ONCE at the start of replay to align with first waypoints
        if planner_map is not None and schedule.left_indices.numel() > 0 and schedule.right_indices.numel() > 0:
            left_idx_0 = int(schedule.left_indices[0].item())
            right_idx_0 = int(schedule.right_indices[0].item())
            self._apply_scheduled_joint_state(
                env_id=env_id,
                arm_paths=arm_paths,
                left_idx=left_idx_0,
                right_idx=right_idx_0,
                planner_map=planner_map,
            )

        # Configurable smoothing parameters
        max_joint_step_rad = float(getattr(self.env_cfg.datagen_config, "max_joint_step_rad", 0.1))

        def _ensure_tensor(value: torch.Tensor | list | np.ndarray) -> torch.Tensor:
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            return tensor.to(device=self.env.device, dtype=torch.float32)

        # Track previous indices for detecting large jumps
        prev_left_idx = -1
        prev_right_idx = -1

        num_ticks = schedule.left_indices.shape[0]
        for tick in range(num_ticks):
            left_idx = int(schedule.left_indices[tick].item()) if schedule.left_indices.numel() > 0 else 0
            right_idx = int(schedule.right_indices[tick].item()) if schedule.right_indices.numel() > 0 else 0

            # Check if we need interpolation steps for smooth motion
            interp_steps = self._compute_interpolation_steps(
                arm_paths=arm_paths,
                prev_left_idx=prev_left_idx,
                prev_right_idx=prev_right_idx,
                curr_left_idx=left_idx,
                curr_right_idx=right_idx,
                max_joint_step=max_joint_step_rad,
            )

            # Execute interpolation steps if needed
            for step in range(interp_steps):
                alpha = float(step + 1) / float(interp_steps + 1)
                interp_waypoint_dict = self._interpolate_waypoints(
                    arm_paths=arm_paths,
                    prev_left_idx=prev_left_idx if prev_left_idx >= 0 else left_idx,
                    prev_right_idx=prev_right_idx if prev_right_idx >= 0 else right_idx,
                    curr_left_idx=left_idx,
                    curr_right_idx=right_idx,
                    alpha=alpha,
                )
                interp_multi_waypoint = MultiWaypoint(interp_waypoint_dict)
                exec_results = await interp_multi_waypoint.execute(
                    env=self.env,
                    success_term=success_term,
                    env_id=env_id,
                    env_action_queue=env_action_queue,
                )
                self._update_execution_buffers(exec_results, buffers, env_id=env_id, capture_joint_positions=False)

            # Execute the actual scheduled waypoint
            waypoint_dict = {
                "left": Waypoint(
                    pose=_ensure_tensor(arm_paths["left"].poses[left_idx]),
                    gripper_action=_ensure_tensor(arm_paths["left"].gripper_actions[left_idx]),
                    noise=0.0,
                ),
                "right": Waypoint(
                    pose=_ensure_tensor(arm_paths["right"].poses[right_idx]),
                    gripper_action=_ensure_tensor(arm_paths["right"].gripper_actions[right_idx]),
                    noise=0.0,
                ),
            }
            multi_waypoint = MultiWaypoint(waypoint_dict)
            exec_results = await multi_waypoint.execute(
                env=self.env,
                success_term=success_term,
                env_id=env_id,
                env_action_queue=env_action_queue,
            )
            self._update_execution_buffers(exec_results, buffers, env_id=env_id, capture_joint_positions=False)

            prev_left_idx = left_idx
            prev_right_idx = right_idx

        generated_actions: list[torch.Tensor] | torch.Tensor
        if buffers.actions:
            generated_actions = torch.cat(buffers.actions, dim=0)
        else:
            generated_actions = buffers.actions

        self.env.recorder_manager.set_success_to_episodes(
            env_id_tensor,
            torch.tensor([[buffers.success]], dtype=torch.bool, device=self.env.device),
        )
        if export_demo:
            self.env.recorder_manager.export_episodes(env_id_tensor)

        return dict(
            initial_state=initial_state,
            states=buffers.states,
            observations=buffers.observations,
            actions=generated_actions,
            success=buffers.success,
        )

    def _get_planned_joint_positions(self, motion_planner: Any) -> torch.Tensor | None:
        """Extract the most recent planner joint trajectory if available."""
        candidate_plan = getattr(motion_planner, "_current_plan", None)
        if candidate_plan is None and hasattr(motion_planner, "_planner_left"):
            last_arm = getattr(motion_planner, "_last_arm", "right")
            arm_planner = motion_planner._planner_left if last_arm == "left" else motion_planner._planner_right
            candidate_plan = getattr(arm_planner, "_current_plan", None)

        if candidate_plan is None:
            return None

        position = getattr(candidate_plan, "position", None)
        if position is None:
            return None

        if isinstance(position, torch.Tensor):
            tensor = position.clone().to(device=self.env.device, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(position, device=self.env.device, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _require_motion_planner_if_skillgen(self, motion_planner: Any | None) -> None:
        """
        Ensure runtime configuration is coherent with the current datagen mode.

        SkillGen uses explicit motion-planning segments between subtasks. Running the
        generator without a planner would result in undefined transitions, so this guard
        fails early with a descriptive error instead of allowing the main loop to crash
        much later.

        Args:
            motion_planner: Optional planner provided by the caller. Must be non-null when
                `env_cfg.datagen_config.use_skillgen` is True.

        Raises:
            ValueError: If SkillGen is enabled but no planner instance was supplied.
        """
        if self.env_cfg.datagen_config.use_skillgen and motion_planner is None:
            raise ValueError("motion_planner must be provided if use_skillgen is True")

    async def _reset_environment_for_generation(
        self,
        env_id: int,
        env_reset_queue: asyncio.Queue | None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Reset the simulator-side env replica and capture its fresh initial state.

        The generator runs in tandem with a simulator/recorder loop; this helper pushes
        the environment ID through the shared reset queue, waits for acknowledgement, and
        then snapshots the scene in relative coordinates so the caller can later replay
        the generated trajectory exactly.

        Args:
            env_id: Integer identifier for the vectorized environment instance.
            env_reset_queue: Async queue used to signal the simulator process to perform
                the actual reset. Required to be non-null for predictable synchronization.

        Returns:
            Tuple of (`env_id_tensor`, `initial_state`) where the tensor is the device-
            resident version of `env_id` used by the recorder manager and the dictionary
            captures the simulator state immediately after reset.

        Raises:
            ValueError: If `env_reset_queue` was not provided by the caller.
        """
        env_id_tensor = torch.tensor([env_id], dtype=torch.int64, device=self.env.device)
        self.env.recorder_manager.reset(env_ids=env_id_tensor)
        if env_reset_queue is None:
            raise ValueError("env_reset_queue must be provided")
        await env_reset_queue.put(env_id)
        await env_reset_queue.join()
        new_initial_state = self.env.scene.get_state(is_relative=True)
        return env_id_tensor, new_initial_state

    def _build_runtime_subtask_constraints(self) -> dict:
        """
        Instantiate the mutable constraint dictionaries used during generation.

        Task constraints (coordination, sequential ordering, etc.) are defined in config
        as lightweight dataclasses. Each provides a factory that returns a dictionary
        capturing its runtime state (fulfilled flags, synchronous windows, transforms).
        This method simply merges them all into a single lookup map keyed by
        `(eef_name, subtask_index)`.

        Returns:
            Dictionary mapping `(task_spec_key, subtask_index)` pairs to mutable constraint
            records that the generator will update in-place while executing.
        """
        runtime_constraints: dict = {}
        for subtask_constraint in self.env_cfg.task_constraint_configs:
            runtime_constraints.update(subtask_constraint.generate_runtime_subtask_constraints())
        return runtime_constraints

    def _initialize_eef_states(self) -> dict[str, EEFGenerationState]:
        """
        Allocate the working state structures for every configured end-effector.

        Each EEF maintains its own progress counters and cached trajectories. Initializing
        them up-front keeps the main loop simple and makes it easy to extend the state in
        the future without editing scattered dict literals.

        Returns:
            Dictionary mapping every `eef_name` to a freshly constructed
            `EEFGenerationState`.
        """
        return {eef_name: EEFGenerationState() for eef_name in self.env_cfg.subtask_configs.keys()}

    def _maybe_refresh_randomized_subtask_boundaries(
        self,
        randomized_subtask_boundaries: dict[str, np.ndarray] | None,
        prev_pool_size: int,
    ) -> tuple[dict[str, np.ndarray], int]:
        """
        Re-sample subtask boundaries when new source demonstrations become available.

        The data generator can ingest new source demos while running. Whenever the
        underlying pool grows we re-run the boundary randomization so the latest demos
        participate immediately. If the pool size stays constant, we re-use the existing
        randomization to avoid thrashing.

        Args:
            randomized_subtask_boundaries: Cached mapping from EEF -> randomized [N,S,2]
                bounds. May be `None` the first time through the loop.
            prev_pool_size: Number of demos observed the last time we sampled boundaries.

        Returns:
            Tuple of `(boundaries, new_pool_size)` where `boundaries` is guaranteed to be
            non-None.
        """
        current_pool_size = len(self.src_demo_datagen_info_pool.datagen_infos)
        if randomized_subtask_boundaries is None or current_pool_size > prev_pool_size:
            randomized_subtask_boundaries = self.randomize_subtask_boundaries()
            prev_pool_size = current_pool_size
        return randomized_subtask_boundaries, prev_pool_size

    def _start_motion_planned_transition_if_needed(
        self,
        env_id: int,
        eef_name: str,
        eef_state: EEFGenerationState,
        eef_subtask_trajectory: WaypointTrajectory,
        selected_src_demo_inds: dict[str, int | None],
        randomized_subtask_boundaries: dict[str, np.ndarray],
        motion_planner: Any | None,
    ) -> tuple[bool, dict | None]:
        """
        Launch a motion-planned transition between subtasks for a SkillGen run.

        When SkillGen is enabled, the generator inserts collision-free transit motions
        between subtasks. This helper handles the entire lifecycle of the transit phase:
        it logs intent, asks the planner to update its internal world, converts the
        planned poses into `Waypoint`s (including source-demo gripper replay), and updates
        the per-EEF state so the main loop knows a motion plan is in progress.

        Args:
            env_id: Environment instance being controlled.
            eef_name: Name of the end-effector requesting the transition.
            eef_state: Mutable `EEFGenerationState` for the requesting EEF.
            eef_subtask_trajectory: The upcoming subtask trajectory that we will execute
                after the transit finishes (used to determine the target pose).
            selected_src_demo_inds: Dict tracking which source demo each EEF currently
                mirrors; the motion plan inherits the same gripper evolution.
            randomized_subtask_boundaries: Randomized boundaries used to compute the
                "between skills" gripper actions for the transit.
            motion_planner: Planner instance responsible for producing the transit motion.

        Returns:
            Tuple `(transition_started, failure_result)` where:
                * `transition_started` is True when a motion plan was successfully created
                  and injected into the EEF state (the caller should skip straight to
                  execution in that case).
                * `failure_result` is a result dictionary returned to the caller when
                  planning fails (the generator aborts immediately).
        """
        if motion_planner is None:
            return False, None

        # Ensure robot articulation state is set correctly before motion planning.
        # The planner reads its start state from robot.data.joint_pos, so we must
        # sync the articulation to match where we expect the robot to be.
        if self.schedule_all_offline:
            import torch as _torch
            planner_for_arm = self._get_arm_specific_planner(motion_planner, eef_name)
            if eef_state.last_joint_position is not None:
                # Use the tracked joint position from previous segment
                start_joints = eef_state.last_joint_position.clone()
            else:
                # No previous joint position - solve IK for current EEF pose as fallback
                # Must disable inference mode for cuRobo IK solver
                with _torch.inference_mode(False):
                    current_eef_pose = self.env.get_robot_eef_pose(eef_name, env_ids=[env_id])[0].clone()
                    start_joints = self._solve_ik_for_start_state(
                        eef_name=eef_name,
                        pose=current_eef_pose,
                        planner=planner_for_arm,
                        env_id=env_id,
                    )
            if start_joints is not None:
                self._set_robot_joint_state(env_id, start_joints, planner_for_arm)
                # Force articulation update to ensure planner sees latest state
                self._robot_articulation.write_data_to_sim()
                self._robot_articulation.update(0.0)

        target_pose = eef_subtask_trajectory[0].pose
        target_gripper_action = eef_subtask_trajectory[0].gripper_action

        expected_attached_object = None
        if hasattr(self.env, "get_expected_attached_object"):
            expected_attached_object = self.env.get_expected_attached_object(
                eef_name,
                eef_state.current_subtask_index,
                self.env.cfg,
            )

        print(f"\n--- Environment {env_id}: Planning motion to target pose ---")
        print(f"Target pose: {target_pose}")
        print(f"Expected attached object: {expected_attached_object}")

        import torch as _torch

        with _torch.inference_mode(False):
            planner_kwargs = dict(
                target_pose=target_pose,
                expected_attached_object=expected_attached_object,
                env_id=env_id,
                step_size=getattr(motion_planner, "step_size", None),
                enable_retiming=(
                    hasattr(motion_planner, "step_size") and motion_planner.step_size is not None
                ),
            )
            if self.skillgen_type == "bimanual":
                planner_kwargs["input_is_site_frame"] = True
            planning_success = motion_planner.update_world_and_plan_motion(**planner_kwargs)

        if not planning_success:
            print(f"Env {env_id}: Motion planning failed for {eef_name}")
            return False, {"success": False}

        print(f"Env {env_id}: Motion planning succeeded")
        target_subtask_index = eef_state.current_subtask_index
        eef_state.next_subtask_index_after_motion = target_subtask_index
        eef_state.next_subtask_trajectory_after_motion = eef_subtask_trajectory
        eef_state.current_subtask_index = -1

        arm_planner = self._get_arm_specific_planner(motion_planner, eef_name) or motion_planner
        mp_waypoints, mp_joint_positions = self._convert_planned_trajectory_to_waypoints(
            arm_planner, target_gripper_action
        )

        if len(mp_waypoints) == 0:
            # Empty motion plan - likely start equals target or planning issue
            # Skip the motion plan phase entirely and proceed directly to the skill
            print(f"Info: Motion plan for {eef_name} produced 0 waypoints, skipping transition")
            eef_state.next_subtask_index_after_motion = None
            eef_state.next_subtask_trajectory_after_motion = None
            eef_state.current_subtask_index = target_subtask_index
            return False, None

        selected_demo_ind = selected_src_demo_inds[eef_name]
        if selected_demo_ind is not None:
            mp_gripper_actions = self._get_mp_gripper_actions_from_source_demo(
                eef_name=eef_name,
                subtask_ind=target_subtask_index,
                selected_demo_ind=selected_demo_ind,
                num_waypoints=len(mp_waypoints),
                randomized_subtask_boundaries=randomized_subtask_boundaries,
            )
            for idx, waypoint in enumerate(mp_waypoints):
                waypoint.gripper_action = mp_gripper_actions[idx]

        eef_state.current_trajectory = mp_waypoints
        joint_trajectory = self._joint_tensor_to_list(mp_joint_positions, len(mp_waypoints))
        eef_state.current_joint_trajectory = cast(list[torch.Tensor | None], joint_trajectory)
        if (
            self.schedule_all_offline
            and mp_joint_positions is not None
            and mp_joint_positions.shape[0] > 0
        ):
            self._set_robot_joint_state(env_id, mp_joint_positions[0], arm_planner)
        eef_state.subtask_step_index = 0
        return True, None

    def _resume_motion_planned_subtask(self, env_id: int, eef_name: str, eef_state: EEFGenerationState) -> None:
        """
        Resume the original skill trajectory after finishing a motion-planned transit.

        Once the planner-produced waypoints have been executed, we need to continue with
        the skill that was originally scheduled for this EEF. This helper restores the
        cached target subtask, splices it with interpolation from the last transit pose,
        and resets the motion-plan bookkeeping fields.

        Args:
            env_id: Environment index used for querying the current robot pose.
            eef_name: Name of the end-effector that is resuming normal execution.
            eef_state: Mutable state container storing both the completed transit
                trajectory and the queued skill we are about to re-activate.
        """
        print("Finished executing motion-planned trajectory")
        assert eef_state.next_subtask_index_after_motion is not None
        assert eef_state.next_subtask_trajectory_after_motion is not None
        prev_executed_traj = eef_state.current_trajectory
        eef_state.current_subtask_index = eef_state.next_subtask_index_after_motion
        eef_state.current_trajectory = self.merge_eef_subtask_trajectory(
            env_id=env_id,
            eef_name=eef_name,
            subtask_index=eef_state.current_subtask_index,
            prev_executed_traj=prev_executed_traj,
            subtask_trajectory=eef_state.next_subtask_trajectory_after_motion,
        )
        eef_state.current_joint_trajectory = cast(
            list[torch.Tensor | None],
            [waypoint.joint_seed for waypoint in eef_state.current_trajectory],
        )
        eef_state.subtask_step_index = 0
        eef_state.next_subtask_index_after_motion = None
        eef_state.next_subtask_trajectory_after_motion = None
        eef_state.subtask_started = True

    def _should_wait_for_sequential_constraint(
        self,
        eef_name: str,
        eef_state: EEFGenerationState,
        subtask_index: int,
        runtime_constraints: dict,
    ) -> bool:
        """Return True if the subtask is blocked by an unmet sequential pre-condition."""
        key = (eef_name, subtask_index)
        if key not in runtime_constraints:
            return False
        constraint = runtime_constraints[key]
        if constraint["type"] != SubTaskConstraintType._SEQUENTIAL_LATTER:
            return False
        if constraint["fulfilled"]:
            return False
        min_time_diff = constraint.get("min_time_diff", 0)
        if min_time_diff < 0:
            return True
        return not eef_state.subtask_started

    def _enter_constraint_hold(self, env_id: int, eef_name: str, eef_state: EEFGenerationState) -> None:
        """Freeze an end-effector at its last command while waiting for sequential constraints."""
        if eef_state.waiting_on_constraint:
            return

        hold_waypoint = eef_state.constraint_hold_waypoint
        if hold_waypoint is None:
            pose = self.env.get_robot_eef_pose(env_ids=[env_id], eef_name=eef_name)[0]
            gripper_action = eef_state.last_commanded_gripper_action
            if gripper_action is None:
                gripper_action = torch.zeros(
                    (1,),
                    device=pose.device,
                    dtype=pose.dtype,
                )
            hold_waypoint = Waypoint(pose=pose, gripper_action=gripper_action, noise=0.0)
        eef_state.waiting_on_constraint = True
        eef_state.current_trajectory = [deepcopy(hold_waypoint)]
        eef_state.subtask_step_index = 0

    def _exit_constraint_hold(self, eef_state: EEFGenerationState) -> None:
        """Release an end-effector from a sequential wait state so it can build the next subtask."""
        eef_state.waiting_on_constraint = False
        eef_state.current_trajectory = []
        eef_state.subtask_step_index = None
        eef_state.subtask_started = False

    def _collect_eef_waypoints(
        self,
        env_id: int,
        runtime_constraints: dict,
        eef_states: dict[str, EEFGenerationState],
        motion_planner: Any | None,
    ) -> dict[str, Waypoint]:
        """
        Select the next executable waypoint for every active end-effector.

        The generator runs all EEFs in lockstep. Before asking the environment to execute
        the next tick we ensure each EEF is allowed to advance (respecting sequential and
        coordination constraints) and then grab the appropriate waypoint from its current
        trajectory.

        Args:
            env_id: Environment index (only needed when updating motion-planner visuals).
            runtime_constraints: Mutable constraint table keyed by `(eef_name, subtask)`.
            eef_states: Current `EEFGenerationState` per end-effector.
            motion_planner: Optional planner used for visualization overlays.

        Returns:
            Dictionary mapping each active `eef_name` to the `Waypoint` it should execute
            next.
        """
        eef_waypoint_dict: dict[str, Waypoint] = {}
        for eef_name in sorted(self.env_cfg.subtask_configs.keys()):
            eef_state = eef_states[eef_name]
            if eef_state.subtask_step_index is None:
                continue

            if (
                self.schedule_all_offline
                and eef_state.subtask_step_index == 0
                and eef_state.current_joint_trajectory
                and eef_state.current_joint_trajectory[0] is not None
            ):
                planner_for_arm = self._get_arm_specific_planner(motion_planner, eef_name)
                self._set_robot_joint_state(
                    env_id=env_id,
                    joint_values=eef_state.current_joint_trajectory[0],
                    planner=planner_for_arm,
                )

            if (
                self.schedule_all_offline
                and eef_state.subtask_step_index == 0
                and eef_state.current_joint_trajectory
                and eef_state.current_joint_trajectory[0] is not None
            ):
                planner_for_arm = self._get_arm_specific_planner(motion_planner, eef_name)
                self._set_robot_joint_state(
                    env_id=env_id,
                    joint_values=eef_state.current_joint_trajectory[0],
                    planner=planner_for_arm,
                )

            self._apply_constraint_progression_rules(
                eef_name=eef_name,
                eef_state=eef_state,
                runtime_constraints=runtime_constraints,
                eef_states=eef_states,
            )
            waypoint = eef_state.current_trajectory[eef_state.subtask_step_index]

            # For motion-planned segments (subtask_index == -1), use planner joints directly.
            # For skill segments, solve IK for the waypoint pose to get consistent joints.
            in_motion_planned_phase = eef_state.current_subtask_index == -1
            if in_motion_planned_phase:
                # Motion plan: use joints from current_joint_trajectory (from planner)
                joint_vec = waypoint.joint_seed
                if joint_vec is None and eef_state.current_joint_trajectory:
                    idx = eef_state.subtask_step_index
                    if idx is not None and idx < len(eef_state.current_joint_trajectory):
                        joint_vec = eef_state.current_joint_trajectory[idx]
            else:
                # Skill segment: solve IK for waypoint pose to get consistent joints
                joint_vec = self._solve_ik_for_waypoint_offline(
                    eef_name=eef_name,
                    waypoint=waypoint,
                    eef_state=eef_state,
                    motion_planner=motion_planner,
                    env_id=env_id,
                )
            eef_state.last_joint_position = joint_vec
            eef_state.last_commanded_joint_position = joint_vec

            if motion_planner and getattr(motion_planner, "visualize_spheres", False):
                current_joints = self.env.scene["robot"].data.joint_pos[env_id]
                motion_planner._update_visualization_at_joint_positions(current_joints)

            eef_waypoint_dict[eef_name] = waypoint
            eef_state.last_commanded_gripper_action = waypoint.gripper_action
        return eef_waypoint_dict

    def _apply_constraint_progression_rules(
        self,
        eef_name: str,
        eef_state: EEFGenerationState,
        runtime_constraints: dict,
        eef_states: dict[str, EEFGenerationState],
    ) -> None:
        """
        Enforce sequential and coordination constraints for a single EEF.

        Some subtasks must lag others (sequential constraints) or stay synchronized with
        their partners (coordination). This helper adjusts the per-EEF `subtask_step_index`
        in-place so the subsequent execution call honors those temporal requirements.

        Args:
            eef_name: Identifier of the EEF being inspected.
            eef_state: Mutable execution state for that EEF.
            runtime_constraints: Global constraint dictionary populated during init.
            eef_states: Full collection of EEF states (needed for coordination checks).
        """
        subtask_key = (eef_name, eef_state.current_subtask_index)
        if subtask_key not in runtime_constraints:
            return

        step_index = eef_state.subtask_step_index
        if step_index is None:
            return

        task_constraint = runtime_constraints[subtask_key]
        if task_constraint["type"] == SubTaskConstraintType._SEQUENTIAL_LATTER:
            if task_constraint["fulfilled"]:
                return
            min_time_diff = task_constraint["min_time_diff"]
            traj_len = len(eef_state.current_trajectory)
            if min_time_diff < 0:
                hold_start_idx = 0
            else:
                hold_start_idx = max(0, traj_len - min_time_diff)
            stall_idx = max(0, hold_start_idx - 1)
            if step_index >= hold_start_idx:
                eef_state.subtask_step_index = stall_idx
            return

        if task_constraint["type"] != SubTaskConstraintType.COORDINATION:
            return

        synchronous_steps = task_constraint["synchronous_steps"]
        concurrent_task_spec_key = task_constraint["concurrent_task_spec_key"]
        concurrent_subtask_ind = task_constraint["concurrent_subtask_ind"]

        concurrent_constraint = runtime_constraints[(concurrent_task_spec_key, concurrent_subtask_ind)]
        concurrent_state = eef_states[concurrent_task_spec_key]

        if (
            task_constraint["coordination_synchronize_start"]
            and concurrent_state.current_subtask_index < concurrent_subtask_ind
        ):
            eef_state.subtask_step_index = 0
            return

        if (
            not concurrent_constraint["fulfilled"]
            and step_index >= len(eef_state.current_trajectory) - synchronous_steps
        ):
            concurrent_constraint["fulfilled"] = True

        if not task_constraint["fulfilled"]:
            if step_index >= len(eef_state.current_trajectory) - synchronous_steps:
                if step_index > 0:
                    step_index -= 1
                    eef_state.subtask_step_index = step_index

    def _update_execution_buffers(
        self,
        exec_results: dict,
        buffers: GenerationBuffers,
        *,
        env_id: int,
        capture_joint_positions: bool = False,
    ) -> None:
        """
        Append simulator outputs from the latest control tick to the shared buffers.

        The environment's `MultiWaypoint.execute` call returns a dictionary containing
        per-tick states/observations/actions and a success bit. This method simply
        extends the buffer lists in-place, guarding against empty batches (which can
        occur if no simulator step was required).

        Args:
            exec_results: Dictionary produced by `MultiWaypoint.execute`.
            buffers: Mutable `GenerationBuffers` instance that accumulates results.
        """
        if len(exec_results["states"]) == 0:
            return

        buffers.states.extend(exec_results["states"])
        buffers.observations.extend(exec_results["observations"])
        buffers.actions.extend(exec_results["actions"])
        buffers.success = buffers.success or exec_results["success"]

        if capture_joint_positions:
            joint_tensor = self._robot_articulation.data.joint_pos
            if isinstance(joint_tensor, torch.Tensor):
                buffers.joint_positions.append(joint_tensor[env_id].clone())

    def _advance_subtask_progress(
        self,
        eef_states: dict[str, EEFGenerationState],
        runtime_constraints: dict,
        pause_subtask: bool,
    ) -> None:
        """
        Increment trajectory indices and trigger completion handling when needed.

        After each execution step we advance the per-EEF waypoint indices. Whenever an
        EEF finishes all waypoints for its current subtask we mark the relevant
        constraints as fulfilled and either queue the next subtask or freeze the EEF if it
        has completed its final skill.

        Args:
            eef_states: All mutable EEF states.
            runtime_constraints: Constraint table to update when subtasks complete.
            pause_subtask: Whether to prompt the operator after each completed subtask
                (useful for debugging interactive runs).
        """
        for eef_name, eef_state in eef_states.items():
            if eef_state.subtask_step_index is None:
                continue
            if eef_state.waiting_on_constraint:
                continue

            eef_state.subtask_step_index += 1
            if eef_state.subtask_step_index == len(eef_state.current_trajectory):
                self._handle_subtask_completion(
                    eef_name=eef_name,
                    eef_state=eef_state,
                    eef_states=eef_states,
                    runtime_constraints=runtime_constraints,
                    pause_subtask=pause_subtask,
                )

    def _handle_subtask_completion(
        self,
        eef_name: str,
        eef_state: EEFGenerationState,
        eef_states: dict[str, EEFGenerationState],
        runtime_constraints: dict,
        pause_subtask: bool,
    ) -> None:
        """
        Finalize constraint updates and prepare the next subtask for an EEF.

        This method handles the edge cases that occur when a subtask completes: it raises
        the relevant constraint flags, optionally pauses for operator input, duplicates
        the final waypoint if the EEF is done with all subtasks, and resets the internal
        indices so the next call to `generate_eef_subtask_trajectory` knows to run.

        Args:
            eef_name: Name of the end-effector that just finished its subtask.
            eef_state: Mutable state for that EEF.
            eef_states: Full state table, needed when validating coordination partners.
            runtime_constraints: Constraint records to mark as fulfilled/finished.
            pause_subtask: Whether to interactively pause between subtasks.
        """
        if eef_state.current_trajectory:
            eef_state.constraint_hold_waypoint = deepcopy(eef_state.current_trajectory[-1])
        eef_state.current_joint_trajectory = []

        subtask_key = (eef_name, eef_state.current_subtask_index)
        if subtask_key in runtime_constraints:
            task_constraint = runtime_constraints[subtask_key]
            if task_constraint["type"] == SubTaskConstraintType._SEQUENTIAL_FORMER:
                constrained_task_spec_key = task_constraint["constrained_task_spec_key"]
                constrained_subtask_ind = task_constraint["constrained_subtask_ind"]
                runtime_constraints[(constrained_task_spec_key, constrained_subtask_ind)]["fulfilled"] = True
            elif task_constraint["type"] == SubTaskConstraintType.COORDINATION:
                concurrent_task_spec_key = task_constraint["concurrent_task_spec_key"]
                concurrent_subtask_ind = task_constraint["concurrent_subtask_ind"]
                task_constraint["finished"] = True
                concurrent_constraint = runtime_constraints[(concurrent_task_spec_key, concurrent_subtask_ind)]
                concurrent_state = eef_states[concurrent_task_spec_key]
                assert (
                    concurrent_constraint["finished"]
                    or (
                        concurrent_state.subtask_step_index is not None
                        and concurrent_state.subtask_step_index
                        >= len(concurrent_state.current_trajectory) - 1
                    )
                )

        if pause_subtask:
            input(
                f"Pausing after subtask {eef_state.current_subtask_index} of {eef_name} execution."
                " Press any key to continue..."
            )

        last_subtask_index = len(self.env_cfg.subtask_configs[eef_name]) - 1
        if eef_state.current_subtask_index == last_subtask_index:
            eef_state.subtasks_done = True
            eef_state.current_trajectory.append(eef_state.current_trajectory[-1])
            return

        eef_state.subtask_step_index = None
        eef_state.current_subtask_index += 1
        eef_state.subtask_started = False

    @staticmethod
    def _all_subtasks_completed(eef_states: dict[str, EEFGenerationState]) -> bool:
        """
        Check whether every end-effector has completed its full subtask list.

        Returns:
            True if all `EEFGenerationState.subtasks_done` flags are set, False otherwise.
        """
        return all(state.subtasks_done for state in eef_states.values())

    def _convert_planned_trajectory_to_waypoints(
        self, motion_planner: Any, gripper_action: torch.Tensor
    ) -> tuple[list[Waypoint], torch.Tensor | None]:
        """
        Convert the planner's raw pose sequence into executable `Waypoint` objects.

        Motion planners operate in their own reference frames (base->tool, site frames,
        etc.). The controller, however, expects waypoints expressed in the controller site
        frame with an associated gripper command. This helper performs the appropriate
        frame conversions (including the bimanual base->site transform) and annotates
        each pose with the constant gripper action used during transits.

        Args:
            motion_planner: Planner instance exposing `get_planned_poses()` and, optionally,
                configuration data such as `motion_noise_scale`, `_last_arm`, or `_arm_side`.
            gripper_action: Gripper tensor to attach to every waypoint in the transit.

        Returns:
            List of `Waypoint` objects ready to be executed by the MultiWaypoint controller.
        """
        # Get motion noise scale from the planner's configuration
        motion_noise_scale = getattr(motion_planner.config, "motion_noise_scale", 0.0)

        planned_poses = motion_planner.get_planned_poses()
        planned_joint_positions = self._get_planned_joint_positions(motion_planner)
        if planned_joint_positions is not None:
            expanded: list[torch.Tensor] = []
            for idx in range(planned_joint_positions.shape[0]):
                expanded.append(
                    self._expand_planner_joint_vector(planned_joint_positions[idx], motion_planner)
                )
            planned_joint_positions = torch.stack(expanded, dim=0)

        # For tabletop tasks, planned poses are already in the env's expected frame
        if self.skillgen_type != "bimanual":
            waypoints = []
            for idx, p in enumerate(planned_poses):
                joint_seed = None
                if planned_joint_positions is not None and idx < planned_joint_positions.shape[0]:
                    joint_seed = planned_joint_positions[idx]
                waypoints.append(
                    Waypoint(
                        pose=p,
                        gripper_action=gripper_action,
                        noise=motion_noise_scale,
                        joint_seed=joint_seed,
                    )
                )
            return waypoints, planned_joint_positions

        # Bimanual/humanoid: convert base->tool to site/world before wrapping as waypoints
        # Determine which arm this plan corresponds to
        eef_name = None
        if hasattr(motion_planner, "_last_arm") and motion_planner._last_arm is not None:
            eef_name = motion_planner._last_arm
        elif hasattr(motion_planner, "_arm_side"):
            try:
                eef_name = motion_planner._arm_side()
            except Exception:
                eef_name = None
        if eef_name is None:
            eef_name = "right"

        env_id = getattr(motion_planner, "env_id", 0)

        # Compute world->base transform
        base_pos_world = (self.env.scene["robot"].data.root_pos_w[env_id] - self.env.scene.env_origins[env_id]).to(
            device=self.env.device, dtype=torch.float32
        )
        base_rot_world = PoseUtils.matrix_from_quat(
            self.env.scene["robot"]
            .data.root_quat_w[env_id]
            .unsqueeze(0)
            .to(device=self.env.device, dtype=torch.float32)
        )[0]
        T_world_base = PoseUtils.make_pose(base_pos_world.unsqueeze(0), base_rot_world.unsqueeze(0))[0]

        # Compute fixed tool->site mapping at current configuration
        try:
            cu_js = motion_planner._get_current_joint_state_for_curobo()
            ee_pose_bt = motion_planner.get_ee_pose(cu_js)  # base->tool
            pos_bt = (
                ee_pose_bt.position
                if isinstance(ee_pose_bt.position, torch.Tensor)
                else torch.tensor(ee_pose_bt.position)
            )
            pos_bt = pos_bt.to(device=self.env.device, dtype=torch.float32).reshape(-1, 3)[0]
            if hasattr(ee_pose_bt, "quaternion"):
                quat_bt = ee_pose_bt.quaternion.view(1, 4).to(device=self.env.device, dtype=torch.float32)
                rot_bt = PoseUtils.matrix_from_quat(quat_bt)[0]
            else:
                rot_bt = ee_pose_bt.get_rotation()
                if isinstance(rot_bt, torch.Tensor) and rot_bt.dim() == 3:
                    rot_bt = rot_bt[0]
                rot_bt = rot_bt.to(device=self.env.device, dtype=torch.float32)
            T_base_tool_now = PoseUtils.make_pose(pos_bt.unsqueeze(0), rot_bt.unsqueeze(0))[0]
            T_world_tool_now = (T_world_base @ T_base_tool_now).clone()
            ctrl_site_env = self.env.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
            T_tool_site = torch.linalg.solve(T_world_tool_now, ctrl_site_env).clone()
        except Exception:
            # Fallback: assume identity tool->site (may be okay if controller site == tool)
            # TODO: Neel We need to figure this all out for humanoid
            T_tool_site = torch.eye(4, device=self.env.device, dtype=torch.float32)

        waypoints = []
        for idx, planned_pose in enumerate(planned_poses):
            # planned_pose is base->tool; map to world->tool and then to world->site
            p_bt = planned_pose.to(device=self.env.device, dtype=torch.float32)
            T_world_tool = (T_world_base @ p_bt).clone()
            T_world_site = (T_world_tool @ T_tool_site).clone()
            joint_seed = None
            if planned_joint_positions is not None and idx < planned_joint_positions.shape[0]:
                joint_seed = planned_joint_positions[idx]
            waypoint = Waypoint(
                pose=T_world_site,
                gripper_action=gripper_action,
                noise=motion_noise_scale,
                joint_seed=joint_seed,
            )
            waypoints.append(waypoint)

        return waypoints, planned_joint_positions
