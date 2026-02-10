# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Base class for data generator.
"""
import asyncio
from copy import deepcopy
from collections import deque
from dataclasses import dataclass, field, replace as dc_replace
import numpy as np
import torch
from typing import Any

import isaaclab.utils.math as PoseUtils
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
from isaaclab.envs import (
    ManagerBasedRLMimicEnv,
    MimicEnvCfg,
    SubTaskConstraintCoordinationScheme,
    SubTaskConstraintType,
)
from isaaclab.managers import TerminationTermCfg

from isaaclab_mimic.datagen.datagen_info import DatagenInfo
from isaaclab_mimic.datagen.offline_scheduling_helpers import (
    JointMapperCache,
    OfflineObjectHelper,
    compute_interpolation_steps,
    force_grasp_check,
    get_ee_pose_from_fk,
    get_world_to_base_transform,
    interpolate_pose,
    is_grasp_subtask,
    is_place_subtask,
    get_subtask_object_ref,
    resolve_eef_name_from_planner,
    transform_base_to_world,
)
from isaaclab_mimic.datagen.scheduling import ArmPath, DiscreteSchedule, HoldConstraint, build_collision_aware_schedule
from isaaclab_mimic.datagen.selection_strategy import make_selection_strategy
from isaaclab_mimic.datagen.waypoint import MultiWaypoint, Waypoint, WaypointSequence, WaypointTrajectory

from .datagen_info_pool import DataGenInfoPool

# Shared queue for goal visualization requests to be flushed on the main thread (env loop).
_goal_viz_queue: deque[tuple[int, str, torch.Tensor]] = deque()
_goal_viz_visualizers: dict[tuple[int, str], VisualizationMarkers] = {}
_goal_viz_warning_emitted: bool = False


def enqueue_goal_visualization(env_id: int, eef_name: str, target_pose: torch.Tensor) -> None:
    """Queue a goal pose for visualization; drained in the main env loop."""
    _goal_viz_queue.append((env_id, eef_name, target_pose.detach().clone()))


def drain_goal_visualizations(env: ManagerBasedRLMimicEnv) -> None:
    """Flush queued goal visualizations on the main thread to avoid asyncio re-entrancy."""
    global _goal_viz_warning_emitted
    while _goal_viz_queue:
        env_id, eef_name, target_pose = _goal_viz_queue.popleft()
        key = (env_id, eef_name)
        if key not in _goal_viz_visualizers:
            prim_path = f"/Visuals/MotionPlanGoals/env_{env_id}_{eef_name}"
            marker_cfg = dc_replace(FRAME_MARKER_CFG, prim_path=prim_path)
            marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            _goal_viz_visualizers[key] = VisualizationMarkers(marker_cfg)
        goal_visualizer = _goal_viz_visualizers[key]
        try:
            goal_pos = target_pose[:3, 3].to(dtype=torch.float32, device=env.device)
            # Offset by env origin for multi-env layouts so markers sit in each env tile.
            if hasattr(env.scene, "env_origins"):
                goal_pos = goal_pos + env.scene.env_origins[env_id, :3].to(device=env.device, dtype=goal_pos.dtype)
            goal_quat = PoseUtils.quat_from_matrix(target_pose[:3, :3].unsqueeze(0))[0].to(
                dtype=torch.float32, device=env.device
            )
            goal_visualizer.visualize(translations=goal_pos.unsqueeze(0), orientations=goal_quat.unsqueeze(0))
        except Exception as exc:
            if not _goal_viz_warning_emitted:
                print(f"Goal visualization failed; continuing without markers. Reason: {exc}")
                _goal_viz_warning_emitted = True


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
    """

    states: list[Any] = field(default_factory=list)
    observations: list[Any] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    success: bool = False


@dataclass
class EEFGenerationState:
    """
    Tracks the per-end-effector state machine used while stitching subtasks.

    Each arm/EEF independently progresses through its subtask list, occasionally pausing
    for coordination constraints. Keeping the bookkeeping in a dedicated structure avoids
    large clusters of parallel dictionaries and makes serialization/debugging straightforward.

    Attributes:
        current_subtask_index: Index into `env_cfg.subtask_configs[eef_name]` that is
            currently being generated/executed. Set to -1 while a motion plan is active.
        current_trajectory: Waypoints (post interpolation) that are currently executing.
            For SkillGen, this contains both MP waypoints and skill waypoints combined,
            so constraints apply to the full (MP + skill) trajectory length.
        subtask_step_index: Pointer into `current_trajectory`. `None` signals "ready to
            build the next subtask trajectory".
        subtasks_done: Flag raised once the end-effector has finished its final skill; the
            final waypoint is duplicated to keep the arm stationary during other arms' work.
        constraint_hold_waypoint: Cached waypoint used when the EEF is paused due to a constraint.
        last_commanded_gripper_action: The last gripper action sent to this EEF, used for
            continuity during motion planning transitions.
        last_commanded_joint_position: The last joint position commanded/tracked for this EEF,
            used for offline scheduling joint continuity.
        subtask_started: Flag indicating whether the current subtask has started execution.
        current_joint_trajectory: Joint positions corresponding to waypoints in current_trajectory.
    """

    current_subtask_index: int = 0
    current_trajectory: list[Waypoint] = field(default_factory=list)
    subtask_step_index: int | None = None
    subtasks_done: bool = False
    constraint_hold_waypoint: Waypoint | None = None
    last_commanded_gripper_action: torch.Tensor | None = None
    last_commanded_joint_position: torch.Tensor | None = None
    subtask_started: bool = False
    is_currently_paused: bool = False  # Track if arm is paused due to constraint - used to skip step increment
    _was_paused_prev_iter: bool = False  # Internal: track pause state transitions for proper waypoint capture
    current_joint_trajectory: list[torch.Tensor | None] = field(default_factory=list)


class DataGeneratorScheduled:
    """
    The main data generator class that generates new trajectories from source datasets.

    The data generator, inspired by the MimicGen, enables the generation of new datasets based on a few human
    collected source demonstrations.

    The data generator works by parsing demonstrations into object-centric subtask segments, stored in DataGenInfoPool.
    It then adapts these subtask segments to new scenes by transforming each segment according to the new scene’s context,
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
        schedule_offline: bool = True,
    ):
        """
        Args:
            env: environment to use for data generation
            src_demo_datagen_info_pool: source demo datagen info pool
            dataset_path: path to hdf5 dataset to use for generation
            demo_keys: list of demonstration keys to use in file. If not provided, all demonstration keys
                will be used.
            skillgen_type: type of skillgen workflow ("single_arm" or "bimanual")
            schedule_offline: when True, builds complete arm paths offline without stepping
                the simulator, then schedules them for collision-aware execution.
        """
        self.env = env
        self.env_cfg = env.cfg
        assert isinstance(self.env_cfg, MimicEnvCfg)
        self.dataset_path = dataset_path
        self.skillgen_type = skillgen_type
        self.schedule_offline = schedule_offline
        self._goal_visualizers: dict[tuple[int, str], VisualizationMarkers] = {}
        self._goal_viz_warning_emitted: bool = False

        # Cache robot articulation for joint state access
        try:
            self._robot_articulation = self.env.scene["robot"]
        except (AttributeError, KeyError) as exc:
            raise AttributeError(
                "DataGeneratorRefactored expects the environment scene to expose a 'robot' articulation"
            ) from exc

        # Initialize helpers for offline scheduling
        self._joint_mapper_cache = JointMapperCache(self._robot_articulation, self.env.device)
        self._object_helper = OfflineObjectHelper(self.env)

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

        # Get reference object pose from source demo
        src_subtask_object_pose = (
            src_ep_datagen_info.object_poses[subtask_object_name][selected_src_subtask_boundary[0]]
            if (subtask_object_name is not None)
            else None
        )

        if is_first_subtask or self.env_cfg.datagen_config.generation_transform_first_robot_pose:
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
        transformed_seq = WaypointSequence.from_poses(
            poses=transformed_eef_poses,
            gripper_actions=src_subtask_gripper_actions,
            action_noise=subtask_configs[subtask_ind].action_noise,
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
        force_use_prev_traj: bool = False,
    ) -> list[Waypoint]:
        """
        Merge a subtask trajectory into an executable trajectory for the robot end-effector.

        This constructs a new `WaypointTrajectory` by first creating an initial
        interpolation segment, then merging the provided `subtask_trajectory` onto it.
        The initial segment begins either from the last executed target waypoint of the
        previous subtask (if configured) or from the robot's current end-effector pose.

        Behavior:

        - If `force_use_prev_traj` is True and `prev_executed_traj` is provided,
          interpolation starts from the last waypoint of `prev_executed_traj`.
          This is used when combining MP + skill trajectories upfront.
        - Else if `datagen_config.generation_interpolate_from_last_target_pose` is True and
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
            force_use_prev_traj: If True, always use the last waypoint from prev_executed_traj
                for interpolation, regardless of is_first_subtask. Used for SkillGen's combined
                MP + skill trajectory building.

        Returns:
            list[Waypoint]: The full sequence of waypoints to execute (initial interpolation segment followed by the subtask segment),
            with the temporary initial waypoint removed.
        """
        is_first_subtask = subtask_index == 0
        # We will construct a WaypointTrajectory instance to keep track of robot control targets
        # and then execute it once we have the trajectory.
        traj_to_execute = WaypointTrajectory()

        # Determine whether to use prev_executed_traj's last waypoint or current robot pose
        use_prev_traj = (
            (force_use_prev_traj and prev_executed_traj)
            or (self.env_cfg.datagen_config.generation_interpolate_from_last_target_pose and (not is_first_subtask))
        )

        if use_prev_traj:
            # Interpolation segment will start from last target pose (which may not have been achieved).
            assert prev_executed_traj is not None
            last_waypoint = prev_executed_traj[-1]
            init_sequence = WaypointSequence(sequence=[last_waypoint])
        else:
            # Interpolation segment will start from current robot eef pose.
            init_sequence = WaypointSequence.from_poses(
                poses=self.env.get_robot_eef_pose(env_ids=[env_id], eef_name=eef_name)[0].unsqueeze(0),
                gripper_actions=subtask_trajectory[0].gripper_action.unsqueeze(0),
                action_noise=self.env_cfg.subtask_configs[eef_name][subtask_index].action_noise,
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

        # Use offline scheduling pipeline if enabled (bimanual only)
        if self.schedule_offline:
            if self.skillgen_type != "bimanual":
                raise ValueError("schedule_offline is only supported for bimanual SkillGen workflows.")
            with torch.inference_mode(False):
                return await self._generate_with_offline_scheduling(
                    env_id=env_id,
                    success_term=success_term,
                    env_reset_queue=env_reset_queue,
                    env_action_queue=env_action_queue,
                    pause_subtask=pause_subtask,
                    export_demo=export_demo,
                    motion_planner=motion_planner,
                )

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

                    # NOTE: Removed pre-trajectory constraint hold logic here.
                    # Sequential constraints are handled purely during execution in
                    # _apply_constraint_progression_rules, matching the original
                    # data_generator.py behavior. Trajectories are always generated;
                    # constraints just stall execution near the end of the trajectory.

                    eef_subtask_trajectory = self.generate_eef_subtask_trajectory(
                        env_id=env_id,
                        eef_name=eef_name,
                        subtask_ind=eef_state.current_subtask_index,
                        all_randomized_subtask_boundaries=randomized_subtask_boundaries,
                        runtime_subtask_constraints_dict=runtime_subtask_constraints,
                        selected_src_demo_inds=selected_src_demo_inds,
                    )

                    if self.env_cfg.datagen_config.use_skillgen:
                        # SkillGen: combine MP waypoints + skill waypoints into one trajectory
                        # so that constraints apply to the full (MP + skill) length,
                        # matching data_gen_bimanual.py behavior.
                        transition_started, combined_waypoints, failure_result = (
                            self._start_motion_planned_transition_if_needed(
                                env_id=env_id,
                                eef_name=eef_name,
                                eef_state=eef_state,
                                eef_subtask_trajectory=eef_subtask_trajectory,
                                selected_src_demo_inds=selected_src_demo_inds,
                                randomized_subtask_boundaries=randomized_subtask_boundaries,
                                motion_planner=motion_planner,
                            )
                        )
                        if failure_result is not None:
                            return failure_result
                        if transition_started and combined_waypoints is not None:
                            eef_state.current_trajectory = combined_waypoints
                            eef_state.subtask_step_index = 0
                            eef_state.subtask_started = True
                            continue

                    # Non-SkillGen path or no motion planner: just use skill waypoints
                    eef_state.current_trajectory = self.merge_eef_subtask_trajectory(
                        env_id=env_id,
                        eef_name=eef_name,
                        subtask_index=eef_state.current_subtask_index,
                        prev_executed_traj=eef_state.current_trajectory,
                        subtask_trajectory=eef_subtask_trajectory,
                    )
                    eef_state.subtask_step_index = 0
                    eef_state.subtask_started = True

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
            self._update_execution_buffers(exec_results, buffers)

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
        )
        return results

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

        # Debug: Print constraint setup
        if runtime_constraints:
            print(f"[Constraints] Built {len(runtime_constraints)} runtime constraints:")
            for key, constraint in runtime_constraints.items():
                print(f"  {key}: type={constraint.get('type')}, fulfilled={constraint.get('fulfilled', 'N/A')}")
        else:
            print("[Constraints] No runtime constraints configured")

        # Validate constraint indices against actual subtask counts
        for eef_name, subtask_configs in self.env_cfg.subtask_configs.items():
            print(f"[Constraints] {eef_name} has {len(subtask_configs)} subtasks (indices 0-{len(subtask_configs)-1})")

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
    ) -> tuple[bool, list[Waypoint] | None, dict | None]:
        """
        Launch a motion-planned transition and return combined MP + skill waypoints.

        When SkillGen is enabled, the generator inserts collision-free transit motions
        between subtasks. This helper plans the transition and returns a combined
        trajectory of MP waypoints followed by skill waypoints, matching the behavior
        of data_gen_bimanual.py where constraints apply to the full (MP + skill) length.

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
            Tuple `(transition_started, combined_waypoints, failure_result)` where:
                * `transition_started` is True when a motion plan was successfully created.
                * `combined_waypoints` is the list of MP + skill waypoints if successful.
                * `failure_result` is a result dictionary returned to the caller when
                  planning fails (the generator aborts immediately).
        """
        if motion_planner is None:
            return False, None, None

        target_pose = eef_subtask_trajectory[0].pose

        # Determine the gripper action to use during the motion planning phase.
        # Priority (matching data_gen_bimanual.py behavior):
        # 1. Use last_commanded_gripper_action for continuity (if available)
        # 2. For first subtask: use initial gripper action from source demo (should be open)
        # 3. Fallback: use skill trajectory's first gripper action
        if eef_state.last_commanded_gripper_action is not None:
            base_gripper_action = eef_state.last_commanded_gripper_action
        elif eef_state.current_subtask_index == 0:
            # First subtask: use the FIRST gripper action from the source demo (timestep 0).
            # This matches data_gen_bimanual.py: "HACK: reading first source demo gripper action
            # at start of demo - just to have an 'open' gripper action"
            first_demo = self.src_demo_datagen_info_pool.datagen_infos[0]
            base_gripper_action = first_demo.gripper_action[eef_name][0]
        elif len(eef_subtask_trajectory) > 0:
            # Non-first subtask: use skill trajectory's first gripper action
            base_gripper_action = eef_subtask_trajectory[0].gripper_action
        else:
            # Fallback: use the FIRST gripper action from the source demo (timestep 0).
            first_demo = self.src_demo_datagen_info_pool.datagen_infos[0]
            base_gripper_action = first_demo.gripper_action[eef_name][0]

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
                # Force the specific arm to avoid auto-selection based on proximity
                planner_kwargs["arm"] = eef_name
            planning_success = motion_planner.update_world_and_plan_motion(**planner_kwargs)

        if not planning_success:
            print(f"Env {env_id}: Motion planning failed for {eef_name}")
            return False, None, {"success": False}

        print(f"Env {env_id}: Motion planning succeeded")
        # Enqueue goal visualization to be flushed on the main env loop thread.
        # enqueue_goal_visualization(env_id=env_id, eef_name=eef_name, target_pose=target_pose)

        # Convert the planned Cartesian trajectory into waypoints, holding the gripper fixed
        # at the last skill segment action for the entire motion-planned transit.
        result = self._convert_planned_trajectory_to_waypoints(
            motion_planner,
            base_gripper_action,
        )
        mp_waypoints: list[Waypoint] = result if isinstance(result, list) else result[0]

        # Merge the skill trajectory (with interpolation from MP end pose) and combine
        # MP waypoints + skill waypoints into a single trajectory.
        # IMPORTANT: force_use_prev_traj=True ensures we interpolate from the LAST MP WAYPOINT
        # (where the robot will be after MP), not from the current robot pose (before MP).
        skill_waypoints = self.merge_eef_subtask_trajectory(
            env_id=env_id,
            eef_name=eef_name,
            subtask_index=eef_state.current_subtask_index,
            prev_executed_traj=mp_waypoints if mp_waypoints else eef_state.current_trajectory,
            subtask_trajectory=eef_subtask_trajectory,
            force_use_prev_traj=True,
        )

        # Combined trajectory: MP waypoints followed by skill waypoints
        combined_waypoints = mp_waypoints + skill_waypoints
        return True, combined_waypoints, None

    def _get_gripper_action_dim(self, eef_name: str) -> int:
        """
        Infer the gripper action dimensionality for the given end-effector.

        Priority order:
        1) Use the recorded source demo gripper action shape (always available during generation).
        2) Derive from the action space assuming per-EEF pose components are 7D (pos + quat).
        3) Fallback to 1 to remain robust even if shapes are misconfigured.
        """
        if self.src_demo_datagen_info_pool.datagen_infos:
            first_demo = self.src_demo_datagen_info_pool.datagen_infos[0]
            if eef_name in first_demo.gripper_action:
                return int(first_demo.gripper_action[eef_name].shape[-1])

        action_dim = int(self.env.action_space.shape[-1])
        num_eefs = max(len(self.env_cfg.subtask_configs), 1)
        remaining = action_dim - 7 * num_eefs
        if remaining > 0 and remaining % num_eefs == 0:
            return remaining // num_eefs
        return 1

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

            is_paused = self._apply_constraint_progression_rules(
                eef_name=eef_name,
                eef_state=eef_state,
                runtime_constraints=runtime_constraints,
                eef_states=eef_states,
            )

            # Track pause state transitions for proper waypoint capture.
            # This matches data_gen_bimanual.py behavior where get_paused_arm_waypoint()
            # uses prev_executed_waypoints_in_base_frame instead of trajectory[step_index].
            was_paused = getattr(eef_state, '_was_paused_prev_iter', False)
            just_paused = not was_paused and is_paused  # First iteration of being paused
            eef_state._was_paused_prev_iter = is_paused
            # Store current pause status so _advance_subtask_progress can skip incrementing
            eef_state.is_currently_paused = is_paused

            if is_paused:
                # On the FIRST iteration of being paused, capture the CURRENT trajectory
                # waypoint as the hold waypoint. This ensures we hold the correct pose/gripper
                # state rather than using a stale waypoint from a previous step.
                if just_paused:
                    eef_state.constraint_hold_waypoint = deepcopy(
                        eef_state.current_trajectory[eef_state.subtask_step_index]
                    )

                if eef_state.constraint_hold_waypoint is not None:
                    waypoint = deepcopy(eef_state.constraint_hold_waypoint)
                else:
                    # Fallback: should not happen after the fix above, but just in case
                    waypoint = eef_state.current_trajectory[eef_state.subtask_step_index]
                    eef_state.constraint_hold_waypoint = deepcopy(waypoint)
            else:
                waypoint = eef_state.current_trajectory[eef_state.subtask_step_index]
                # When not paused, keep updating the hold waypoint as a fallback
                eef_state.constraint_hold_waypoint = deepcopy(waypoint)

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
    ) -> bool:
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

        Returns:
            True if the EEF is paused due to a constraint (should use constraint_hold_waypoint),
            False otherwise.
        """
        subtask_key = (eef_name, eef_state.current_subtask_index)
        if subtask_key not in runtime_constraints:
            return False

        step_index = eef_state.subtask_step_index
        if step_index is None:
            return False

        task_constraint = runtime_constraints[subtask_key]
        if task_constraint["type"] == SubTaskConstraintType._SEQUENTIAL_LATTER:
            if task_constraint["fulfilled"]:
                return False
            min_time_diff = task_constraint["min_time_diff"]
            traj_len = len(eef_state.current_trajectory)
            if traj_len == 0:
                return False
            if min_time_diff < 0:
                # Strict ordering (min_time_diff == -1): hold from the very beginning
                # until the precondition is met.
                should_hold = True
            else:
                # Hold once we reach the last @min_time_diff steps of the trajectory.
                # This matches data_gen_bimanual.py: hold when step_ind >= traj_len - min_time_diff
                should_hold = step_index >= traj_len - min_time_diff
            if should_hold:
                # Signal that this EEF is paused. The step index is NOT modified here;
                # _advance_subtask_progress will skip incrementing for paused arms.
                return True
            return False

        if task_constraint["type"] != SubTaskConstraintType.COORDINATION:
            return False

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
            return True  # Signal that this EEF is paused waiting for coordination start

        if (
            not concurrent_constraint["fulfilled"]
            and step_index >= len(eef_state.current_trajectory) - synchronous_steps
        ):
            concurrent_constraint["fulfilled"] = True

        if not task_constraint["fulfilled"]:
            if step_index >= len(eef_state.current_trajectory) - synchronous_steps:
                # Signal that this EEF is paused. The step index is NOT modified here;
                # _advance_subtask_progress will skip incrementing for paused arms.
                return True
        return False

    def _is_blocked_by_sequential_constraint(
        self,
        eef_name: str,
        subtask_index: int,
        runtime_constraints: dict,
        eef_states: dict[str, EEFGenerationState],
    ) -> bool:
        """
        Check if this subtask's trajectory generation should be blocked by a SEQUENTIAL constraint.

        For offline path building, this is critical: we must NOT generate the "latter" subtask's
        trajectory until the "former" subtask completes. Otherwise, object positions read during
        trajectory generation will be stale (e.g., bowl hasn't been moved yet by right arm).

        Args:
            eef_name: End-effector name.
            subtask_index: Current subtask index.
            runtime_constraints: Runtime constraint dictionary.
            eef_states: Current EEF states.

        Returns:
            True if trajectory generation should be skipped for now, False otherwise.
        """
        subtask_key = (eef_name, subtask_index)
        if subtask_key not in runtime_constraints:
            print(f"[Constraints] {eef_name} subtask {subtask_index}: no constraint found")
            return False

        task_constraint = runtime_constraints[subtask_key]
        constraint_type = task_constraint.get("type")
        # print(f"[Constraints] {eef_name} subtask {subtask_index}: type={constraint_type}")

        # Check if this is the "latter" part of a SEQUENTIAL constraint
        if constraint_type == SubTaskConstraintType._SEQUENTIAL_LATTER:
            fulfilled = task_constraint.get("fulfilled", False)
            # pre_cond_eef = task_constraint.get("pre_condition_task_spec_key")
            # pre_cond_idx = task_constraint.get("pre_condition_subtask_ind")
            # Check FORMER's current state
            # if pre_cond_eef in eef_states:
            #     former_state = eef_states[pre_cond_eef]
                # print(f"[Constraints] FORMER {pre_cond_eef} subtask {pre_cond_idx}: "
                #       f"current_idx={former_state.current_subtask_index}, done={former_state.subtasks_done}")
            
            if not fulfilled:
                # print(f"[Constraints] BLOCKING {eef_name} subtask {subtask_index} - "
                #       f"waiting for FORMER {pre_cond_eef}[{pre_cond_idx}] to complete (fulfilled={fulfilled})")
                return True
            # else:
                # print(f"[Constraints] UNBLOCKING {eef_name} subtask {subtask_index} - "
                #       f"FORMER {pre_cond_eef}[{pre_cond_idx}] completed")

        return False

    def _update_execution_buffers(self, exec_results: dict, buffers: GenerationBuffers) -> None:
        """
        Append simulator outputs from the latest control tick to the shared buffers.

        The environment’s `MultiWaypoint.execute` call returns a dictionary containing
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

            # Skip incrementing step_index if the arm is paused due to a constraint.
            # This ensures we resume from the correct trajectory position when released.
            if eef_state.is_currently_paused:
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

        subtask_key = (eef_name, eef_state.current_subtask_index)
        if subtask_key in runtime_constraints:
            task_constraint = runtime_constraints[subtask_key]
            if task_constraint["type"] == SubTaskConstraintType._SEQUENTIAL_FORMER:
                constrained_task_spec_key = task_constraint["constrained_task_spec_key"]
                constrained_subtask_ind = task_constraint["constrained_subtask_ind"]
                runtime_constraints[(constrained_task_spec_key, constrained_subtask_ind)]["fulfilled"] = True
                print(f"[Constraints] FORMER {eef_name} subtask {eef_state.current_subtask_index} completed -> "
                      f"unblocking {constrained_task_spec_key} subtask {constrained_subtask_ind}")
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
            eef_state.subtask_step_index = None  # Prevent re-triggering completion
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

    def _get_initial_gripper_from_articulation(self, env_id: int, eef_name: str) -> torch.Tensor:
        """Get initial gripper state from robot articulation (matches physics reset state).

        After reset, the articulation contains joint positions from robot config's init_state.
        This is what cuRobo should use as initial gripper state for planning.

        Args:
            env_id: Environment ID.
            eef_name: "left" or "right" arm.

        Returns:
            Gripper action tensor for the specified arm.
        """
        actions_cfg = getattr(getattr(self.env, "cfg", None), "actions", None)
        pink_cfg = getattr(actions_cfg, "pink_ik_cfg", None) if actions_cfg else None
        hand_names = getattr(pink_cfg, "hand_joint_names", None) if pink_cfg else None

        if not hand_names:
            return torch.zeros(11, device=self.env.device, dtype=torch.float32)

        robot_joint_names = list(self._robot_articulation.data.joint_names)
        robot_joint_pos = self._robot_articulation.data.joint_pos[env_id]
        name_to_idx = {n: i for i, n in enumerate(robot_joint_names)}

        prefix = "L_" if eef_name == "left" else "R_"
        arm_hand_names = [n for n in hand_names if n.startswith(prefix)]

        result = []
        for name in arm_hand_names:
            if name in name_to_idx:
                result.append(robot_joint_pos[name_to_idx[name]])
            else:
                result.append(torch.tensor(0.0, device=self.env.device))

        return torch.stack(result).to(dtype=torch.float32) if result else torch.zeros(11, device=self.env.device)

    def _extract_gripper_from_joints(
        self,
        joint_positions: torch.Tensor,
        motion_planner: Any,
        eef_name: str,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Extract finger joint positions from cuRobo's planned joint state.

        Similar to plan_humanoid_single_call.py: uses cuRobo's planned joints if available,
        falls back to articulation's current state for missing joints.

        Args:
            joint_positions: Full joint vector from cuRobo (planner ordering).
            motion_planner: Planner instance for joint name lookup.
            eef_name: "left" or "right" to determine finger prefix.
            env_id: Environment ID for fallback values.

        Returns:
            Gripper action tensor for the specified arm.
        """
        # Get fallback values from articulation (matches physics reset state)
        fallback = self._get_initial_gripper_from_articulation(env_id, eef_name)

        actions_cfg = getattr(getattr(self.env, "cfg", None), "actions", None)
        pink_cfg = getattr(actions_cfg, "pink_ik_cfg", None) if actions_cfg else None
        hand_names = getattr(pink_cfg, "hand_joint_names", None) if pink_cfg else None

        if not hand_names or not hasattr(motion_planner, "motion_gen"):
            return fallback

        cu_names = list(motion_planner.motion_gen.kinematics.joint_names)
        name_to_cu_idx = {n: i for i, n in enumerate(cu_names)}

        prefix = "L_" if eef_name == "left" else "R_"
        arm_hand_names = [n for n in hand_names if n.startswith(prefix)]
        if not arm_hand_names:
            return fallback

        jpos = joint_positions.view(-1).to(device=self.env.device, dtype=torch.float32)

        # Like plan_humanoid_single_call.py: use cuRobo joint if available, else fallback
        result = []
        for i, name in enumerate(arm_hand_names):
            if name in name_to_cu_idx and name_to_cu_idx[name] < jpos.shape[0]:
                result.append(jpos[name_to_cu_idx[name]])
            elif i < fallback.shape[0]:
                result.append(fallback[i])
            else:
                result.append(torch.tensor(0.0, device=self.env.device))

        return torch.stack(result).to(dtype=torch.float32)

    def _convert_planned_trajectory_to_waypoints(
        self,
        motion_planner: Any,
        gripper_action: torch.Tensor,
        include_joints: bool = False,
        use_planned_gripper: bool = False,
    ) -> list[Waypoint] | tuple[list[Waypoint], list[torch.Tensor]]:
        """
        Convert the planner's raw pose sequence into executable `Waypoint` objects.

        Motion planners operate in their own reference frames (base->tool, site frames,
        etc.). The controller expects waypoints in the controller site frame. This method
        performs frame conversions and optionally extracts joint positions.

        Args:
            motion_planner: Planner instance exposing `get_planned_poses()`.
            gripper_action: Gripper tensor fallback for each waypoint.
            include_joints: If True, also return joint positions from the plan.
            use_planned_gripper: If True, extract gripper from cuRobo's planned joints.

        Returns:
            List of Waypoint objects, or tuple of (waypoints, joints) if include_joints=True.
        """
        motion_noise_scale = getattr(motion_planner.config, "motion_noise_scale", 0.0)
        planned_poses = motion_planner.get_planned_poses()
        planned_joints = self._get_planned_joint_positions(motion_planner) if include_joints else None

        waypoints: list[Waypoint] = []
        joints: list[torch.Tensor] = []

        # For tabletop tasks, poses are already in the correct frame
        if self.skillgen_type != "bimanual":
            for idx, pose in enumerate(planned_poses):
                joint_vec = None
                if planned_joints is not None and idx < planned_joints.shape[0]:
                    joint_vec = planned_joints[idx].clone()
                waypoints.append(Waypoint(
                    pose=pose.clone() if include_joints else pose,
                    gripper_action=gripper_action.clone() if include_joints else gripper_action,
                    noise=motion_noise_scale,
                    joint_seed=joint_vec,
                ))
                if include_joints:
                    if joint_vec is not None:
                        joints.append(joint_vec)
                    elif joints:
                        joints.append(joints[-1].clone())
            return (waypoints, joints) if include_joints else waypoints

        # Bimanual/humanoid: convert base->tool to world->site frame
        eef_name = resolve_eef_name_from_planner(motion_planner)
        env_id = getattr(motion_planner, "env_id", 0)
        T_world_base = get_world_to_base_transform(
            self._robot_articulation, self.env.scene.env_origins, env_id, self.env.device
        )

        # For offline mode (include_joints=True), use identity for T_tool_site because:
        # - The physics state isn't updated during offline path building
        # - Computing T_tool_site from articulation buffer is unreliable
        # - The controller's site frame and cuRobo's tool frame are essentially the same at home position
        if include_joints:
            T_tool_site = torch.eye(4, device=self.env.device, dtype=torch.float32)
        else:
            T_tool_site = self._compute_tool_to_site_transform(motion_planner, T_world_base, eef_name, env_id)

        for idx, planned_pose in enumerate(planned_poses):
            p_bt = planned_pose.to(device=self.env.device, dtype=torch.float32)
            T_world_site = transform_base_to_world(p_bt, T_world_base, T_tool_site)

            joint_vec = None
            if planned_joints is not None and idx < planned_joints.shape[0]:
                joint_vec = planned_joints[idx].clone()

            # Use gripper from cuRobo's planned joints if requested (first MP)
            wp_gripper = gripper_action
            if use_planned_gripper and joint_vec is not None:
                wp_gripper = self._extract_gripper_from_joints(joint_vec, motion_planner, eef_name, env_id)

            waypoints.append(Waypoint(
                pose=T_world_site,
                gripper_action=wp_gripper.clone() if include_joints else wp_gripper,
                noise=motion_noise_scale,
                joint_seed=joint_vec,
            ))
            if include_joints:
                if joint_vec is not None:
                    joints.append(joint_vec)
                elif joints:
                    joints.append(joints[-1].clone())

        return (waypoints, joints) if include_joints else waypoints

    def _compute_tool_to_site_transform(
        self,
        motion_planner: Any,
        T_world_base: torch.Tensor,
        eef_name: str,
        env_id: int,
    ) -> torch.Tensor:
        """Compute the tool-to-site transform for bimanual robots."""
        try:
            cu_js = motion_planner._get_current_joint_state_for_curobo()
            ee_pose_bt = motion_planner.get_ee_pose(cu_js)
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
            return torch.linalg.solve(T_world_tool_now, ctrl_site_env).clone()
        except Exception:
            return torch.eye(4, device=self.env.device, dtype=torch.float32)

    def _get_goal_visualizer(self, env_id: int, eef_name: str) -> VisualizationMarkers:
        """
        Lazily create or fetch the goal marker visualizer for a given environment/end-effector.
        """
        key = (env_id, eef_name)
        if key not in self._goal_visualizers:
            prim_path = f"/Visuals/MotionPlanGoals/env_{env_id}_{eef_name}"
            marker_cfg = dc_replace(FRAME_MARKER_CFG, prim_path=prim_path)
            marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self._goal_visualizers[key] = VisualizationMarkers(marker_cfg)
        return self._goal_visualizers[key]

# ====================================================================================
# ========================= Offline Scheduling Methods ===============================

    async def _generate_with_offline_scheduling(
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
        Build complete arm paths offline without stepping the simulator, then schedule
        them for collision-aware execution.

        This two-stage pipeline:
        1. Runs the generator state machine to collect all waypoints and plan all motion
           plans, tracking joint positions throughout without stepping the simulator.
        2. Uses the scheduling utilities to build a collision-aware discrete schedule.
        3. Replays the scheduled actions in the simulator.

        Args:
            env_id: Environment ID.
            success_term: Termination condition to check for success.
            env_reset_queue: Queue for reset synchronization.
            env_action_queue: Queue for action dispatch.
            pause_subtask: Whether to pause between subtasks.
            export_demo: Whether to export the resulting demonstration.
            motion_planner: Bimanual motion planner instance.

        Returns:
            Dictionary containing generated trajectory data.
        """
        planner_left, planner_right = self._resolve_arm_planners(motion_planner)

        # Stage 1: Build offline arm paths
        build_result = await self._build_offline_arm_paths(
            env_id=env_id,
            env_reset_queue=env_reset_queue,
            pause_subtask=pause_subtask,
            motion_planner=motion_planner,
        )
        if isinstance(build_result, dict):
            # Early failure (e.g., motion planning failed)
            return build_result

        initial_state, arm_paths, hold_constraints = build_result

        # Stage 2: Build collision-aware schedule
        # Have to add try except or else I cannot see the logs without setting verbose which is annoying
        try:
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
                hold_constraints=hold_constraints,
            )
        except Exception as e:
            print(f"Scheduling failed: {e}")
            return {"success": False}

        schedule.append_hold(int(getattr(self.env_cfg.datagen_config, "final_hold_steps", 0)))

        # Stage 3: Replay scheduled actions in simulator
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

    def _resolve_arm_planners(self, motion_planner: Any | None) -> tuple[Any, Any]:
        """
        Extract the underlying arm-specific planner instances from a bimanual planner.

        Args:
            motion_planner: Bimanual motion planner instance.

        Returns:
            Tuple of (planner_left, planner_right).

        Raises:
            ValueError: If the planner doesn't expose the expected arm-specific attributes.
        """
        if motion_planner is None:
            raise ValueError("schedule_offline requires a bimanual motion planner instance.")
        planner_left = getattr(motion_planner, "_planner_left", None)
        planner_right = getattr(motion_planner, "_planner_right", None)
        if planner_left is None or planner_right is None:
            raise ValueError("Bimanual motion planner must expose _planner_left/_planner_right for scheduling.")
        return planner_left, planner_right

    def _get_arm_planner(self, motion_planner: Any, eef_name: str) -> Any:
        """Get the arm-specific planner for the given end-effector."""
        if hasattr(motion_planner, "_planner_left") and hasattr(motion_planner, "_planner_right"):
            return motion_planner._planner_left if eef_name == "left" else motion_planner._planner_right
        return motion_planner

    async def _build_offline_arm_paths(
        self,
        env_id: int,
        env_reset_queue: asyncio.Queue | None,
        pause_subtask: bool,
        motion_planner: Any | None,
    ) -> tuple[dict, dict[str, ArmPath], list[HoldConstraint]] | dict:
        """
        Run the generator state machine to collect all waypoints and joint positions
        without stepping the simulator.

        This method simulates what the online generation would do, but instead of
        executing actions, it collects waypoints and tracks joint positions through:
        - Motion planner output for MP segments
        - IK-solved positions for skill waypoints

        Args:
            env_id: Environment ID.
            env_reset_queue: Queue for reset synchronization.
            pause_subtask: Whether to pause between subtasks (for debugging).
            motion_planner: Motion planner instance.

        Returns:
            Tuple of (initial_state, arm_paths) on success, or a failure dict on error.
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

        # Initialize joint positions from current robot state for each EEF
        # This is critical for offline planning - motion planner needs correct start state
        initial_joints = self._robot_articulation.data.joint_pos[env_id].clone()

        # For bimanual robots, track global robot state that accumulates updates from both arms
        # This prevents one arm's update from being overwritten by the other arm's initial state
        global_robot_joint_state = initial_joints.clone()

        for eef_name in self.env_cfg.subtask_configs.keys():
            if motion_planner is not None:
                arm_planner = self._get_arm_planner(motion_planner, eef_name)
                eef_states[eef_name].last_commanded_joint_position = self._project_joint_to_planner(
                    initial_joints, arm_planner
                )
            else:
                eef_states[eef_name].last_commanded_joint_position = initial_joints.clone()
        # Waypoint logs: list of dicts per EEF with waypoint, joint position
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

                    # Check if SEQUENTIAL constraint blocks this subtask's trajectory generation.
                    # For offline path building, we must not generate the trajectory until the
                    # constraining subtask completes - otherwise we read wrong object positions.
                    is_blocked = self._is_blocked_by_sequential_constraint(
                        eef_name=eef_name,
                        subtask_index=eef_state.current_subtask_index,
                        runtime_constraints=runtime_subtask_constraints,
                        eef_states=eef_states,
                    )
                    if is_blocked:
                        # print(f"[Offline Build] SKIPPING {eef_name} subtask {eef_state.current_subtask_index} - blocked by SEQUENTIAL")
                        continue
                    print(f"[Offline Build] GENERATING trajectory for {eef_name} subtask {eef_state.current_subtask_index}")

                    # Generate subtask trajectory
                    eef_subtask_trajectory = self.generate_eef_subtask_trajectory(
                        env_id=env_id,
                        eef_name=eef_name,
                        subtask_ind=eef_state.current_subtask_index,
                        all_randomized_subtask_boundaries=randomized_subtask_boundaries,
                        runtime_subtask_constraints_dict=runtime_subtask_constraints,
                        selected_src_demo_inds=selected_src_demo_inds,
                    )

                    if self.env_cfg.datagen_config.use_skillgen:
                        # Plan motion and combine with skill trajectory
                        # Pass global robot state so planner starts from accumulated state
                        transition_result = self._plan_motion_offline(
                            env_id=env_id,
                            eef_name=eef_name,
                            eef_state=eef_state,
                            eef_subtask_trajectory=eef_subtask_trajectory,
                            selected_src_demo_inds=selected_src_demo_inds,
                            randomized_subtask_boundaries=randomized_subtask_boundaries,
                            motion_planner=motion_planner,
                            global_robot_joint_state=global_robot_joint_state,
                        )
                        if isinstance(transition_result, dict):
                            return transition_result  # Failure
                        if transition_result is not None:
                            mp_waypoints, mp_joints = transition_result
                            # Update global robot state with this arm's final position
                            if mp_joints:
                                arm_planner = self._get_arm_planner(motion_planner, eef_name)
                                global_robot_joint_state = self._merge_arm_joints_to_global(
                                    global_state=global_robot_joint_state,
                                    arm_joints=mp_joints[-1],
                                    arm_planner=arm_planner,
                                )

                                # NOTE: Object pose updates are now deferred to subtask COMPLETION
                                # (in _update_world_state_after_subtask_offline) to ensure correct
                                # object positions for subtasks that reference the same object.
                                # Previously, preemptive updates caused incorrect planning targets
                                # when one arm's grasp updated the object position before another
                                # arm's subtask could plan with the original position.

                            # Merge skill waypoints after MP
                            skill_waypoints = self.merge_eef_subtask_trajectory(
                                env_id=env_id,
                                eef_name=eef_name,
                                subtask_index=eef_state.current_subtask_index,
                                prev_executed_traj=mp_waypoints,
                                subtask_trajectory=eef_subtask_trajectory,
                                force_use_prev_traj=True,
                            )
                            eef_state.current_trajectory = mp_waypoints + skill_waypoints
                            # Build joint trajectory: MP joints + None for skill (to be IK-solved)
                            eef_state.current_joint_trajectory = list(mp_joints) + [None] * len(skill_waypoints)
                            eef_state.subtask_step_index = 0
                            eef_state.subtask_started = True
                            continue

                    # Non SkillGen or no motion plan needed
                    eef_state.current_trajectory = self.merge_eef_subtask_trajectory(
                        env_id=env_id,
                        eef_name=eef_name,
                        subtask_index=eef_state.current_subtask_index,
                        prev_executed_traj=eef_state.current_trajectory,
                        subtask_trajectory=eef_subtask_trajectory,
                    )
                    eef_state.current_joint_trajectory = [None] * len(eef_state.current_trajectory)
                    eef_state.subtask_step_index = 0
                    eef_state.subtask_started = True

            # Collect waypoints for this timestep (without stepping simulator)
            eef_waypoints, global_robot_joint_state = self._collect_offline_waypoints(
                env_id=env_id,
                eef_states=eef_states,
                runtime_constraints=runtime_subtask_constraints,
                motion_planner=motion_planner,
                global_robot_joint_state=global_robot_joint_state,
            )
            if not eef_waypoints:
                raise RuntimeError("Offline build produced no waypoints for active timestep.")

            # Store per-arm 21-joint vectors (already in planner space).
            # These come from last_commanded_joint_position which is set during MP or IK solving.
            for eef_name, waypoint in eef_waypoints.items():
                eef_state = eef_states[eef_name]
                joint_vec = eef_state.last_commanded_joint_position
                waypoint_logs[eef_name].append({
                    "waypoint": deepcopy(waypoint),
                    "joint": joint_vec.clone() if joint_vec is not None else None,
                    "subtask_index": eef_state.current_subtask_index,
                })

            # Track which subtasks are about to complete (for world state updates)
            completing_subtasks = {}
            for eef_name, eef_state in eef_states.items():
                is_completing = (
                    eef_state.subtask_step_index is not None
                    and not eef_state.is_currently_paused
                    and eef_state.subtask_step_index + 1 == len(eef_state.current_trajectory)
                )
                if is_completing:
                    completing_subtasks[eef_name] = eef_state.current_subtask_index

            # Advance progress (without executing)
            self._advance_subtask_progress_offline(
                eef_states=eef_states,
                runtime_constraints=runtime_subtask_constraints,
                pause_subtask=pause_subtask,
            )

            # Update world state for completed subtasks (robot pose, object positions)
            if completing_subtasks:
                self._update_world_state_after_subtask_offline(
                    env_id=env_id,
                    completing_subtasks=completing_subtasks,
                    eef_states=eef_states,
                    motion_planner=motion_planner,
                )

            if self._all_subtasks_completed(eef_states):
                break

        # Build ArmPath objects from collected waypoints
        arm_paths = self._build_arm_paths_from_logs(
            waypoint_logs=waypoint_logs,
            motion_planner=motion_planner,
            env_id=env_id,
        )

        # Inject hold waypoints for SEQUENTIAL constraints
        # The LATTER arm must wait at the start of its trajectory for sequential_min_time_diff
        arm_paths, hold_constraints = self._inject_sequential_hold_waypoints(
            arm_paths=arm_paths,
            runtime_constraints=runtime_subtask_constraints,
            step_dt=self.env.step_dt,
        )

        return initial_state, arm_paths, hold_constraints

    def _plan_motion_offline(
        self,
        env_id: int,
        eef_name: str,
        eef_state: EEFGenerationState,
        eef_subtask_trajectory: WaypointTrajectory,
        selected_src_demo_inds: dict[str, int | None],
        randomized_subtask_boundaries: dict[str, np.ndarray],
        motion_planner: Any | None,
        global_robot_joint_state: torch.Tensor | None = None,
    ) -> tuple[list[Waypoint], list[torch.Tensor]] | dict | None:
        """
        Plan a motion-planned transition offline, tracking joint positions.

        Similar to _start_motion_planned_transition_if_needed but designed for
        offline path building. Returns the planned waypoints and joint positions
        without executing them.

        Args:
            env_id: Environment ID.
            eef_name: End-effector name.
            eef_state: Current EEF state.
            eef_subtask_trajectory: The upcoming skill trajectory.
            selected_src_demo_inds: Source demo selection mapping.
            randomized_subtask_boundaries: Randomized subtask boundaries.
            motion_planner: Motion planner instance.
            global_robot_joint_state: Global robot joint state accumulating all arm updates.

        Returns:
            Tuple of (mp_waypoints, mp_joint_positions) on success,
            None if no motion plan needed,
            dict with success=False on planning failure.
        """
        if motion_planner is None:
            return None

        target_pose = eef_subtask_trajectory[0].pose

        # Determine gripper action for MP phase
        # For first MP (no prior skill), use articulation's initial state (matches physics reset)
        if eef_state.last_commanded_gripper_action is not None:
            base_gripper_action = eef_state.last_commanded_gripper_action
        elif eef_state.current_subtask_index == 0:
            # Use initial gripper from robot articulation (what physics produces after reset)
            base_gripper_action = self._get_initial_gripper_from_articulation(env_id, eef_name)
        elif len(eef_subtask_trajectory) > 0:
            base_gripper_action = eef_subtask_trajectory[0].gripper_action
        else:
            base_gripper_action = self._get_initial_gripper_from_articulation(env_id, eef_name)

        expected_attached_object = None
        if hasattr(self.env, "get_expected_attached_object"):
            expected_attached_object = self.env.get_expected_attached_object(
                eef_name,
                eef_state.current_subtask_index,
                self.env.cfg,
            )

        # Get arm-specific planner for this EEF
        arm_planner = self._get_arm_planner(motion_planner, eef_name)

        # Prepare start joint state for motion planner (in planner ordering)
        # PRIORITY: Use last_commanded_joint_position if available, as it's already in planner
        # ordering and contains IK-solved values from the skill segment.
        start_joint_state_for_planner = None
        if eef_state.last_commanded_joint_position is not None:
            start_joint_state_for_planner = eef_state.last_commanded_joint_position.clone()
        elif global_robot_joint_state is not None:
            start_joint_state_for_planner = self._project_joint_to_planner(
                global_robot_joint_state, arm_planner
            )

        # Update articulation buffer with the start joint state for frame conversions
        # The planner's frame conversion (T_tool_site) reads from articulation buffer
        if start_joint_state_for_planner is not None:
            full_joints = self._expand_planner_joints_to_full(
                start_joint_state_for_planner, arm_planner, env_id
            )
            self._robot_articulation.data.joint_pos[env_id] = full_joints.clone()
            self._robot_articulation.data.joint_pos_target[env_id] = full_joints.clone()
            # NOTE: We skip write_data_to_sim/update as they cause sync issues in offline mode
        elif global_robot_joint_state is not None:
            self._robot_articulation.data.joint_pos[env_id] = global_robot_joint_state.clone()
            self._robot_articulation.data.joint_pos_target[env_id] = global_robot_joint_state.clone()

        # Build planner kwargs
        planner_kwargs = dict(
            target_pose=target_pose,
            expected_attached_object=expected_attached_object,
            env_id=env_id,
            step_size=getattr(motion_planner, "step_size", None),
            enable_retiming=(hasattr(motion_planner, "step_size") and motion_planner.step_size is not None),
        )
        if self.skillgen_type == "bimanual":
            planner_kwargs["input_is_site_frame"] = True
            planner_kwargs["arm"] = eef_name
            if start_joint_state_for_planner is not None:
                planner_kwargs["start_joint_state"] = start_joint_state_for_planner

        # Execute motion planning with optional object attachment setup
        planning_success = self._execute_motion_planning_with_attachment(
            env_id=env_id,
            arm_planner=arm_planner,
            expected_attached_object=expected_attached_object,
            motion_planner=motion_planner,
            planner_kwargs=planner_kwargs,
        )

        if not planning_success:
            print(f"Motion planning failed for {eef_name}")
            return {"success": False}

        # For first MP (no prior skill), use gripper from cuRobo's collision-free plan
        is_first_mp = eef_state.last_commanded_gripper_action is None

        # Extract planned poses and joint positions
        result = self._convert_planned_trajectory_to_waypoints(
            arm_planner, base_gripper_action, include_joints=True, use_planned_gripper=is_first_mp
        )
        mp_waypoints: list[Waypoint] = result[0] if isinstance(result, tuple) else result
        mp_joints: list[torch.Tensor] = result[1] if isinstance(result, tuple) else []

        if len(mp_waypoints) == 0:
            return None

        # Update last commanded joint position for next motion plan
        if mp_joints:
            eef_state.last_commanded_joint_position = mp_joints[-1].clone()
            self._set_robot_joint_state(env_id, mp_joints[-1], arm_planner, skip_sim_sync=True)

        # Update last gripper action from plan for subsequent segments
        if is_first_mp and mp_waypoints:
            eef_state.last_commanded_gripper_action = mp_waypoints[-1].gripper_action.clone()

        return mp_waypoints, mp_joints

    def _move_object_to_ee_for_offline(
        self,
        env_id: int,
        object_name: str,
        arm_planner: Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Move an object to the EE position for offline planning.

        Only POSITION is changed:
        - Position: palm/grasp center (wrist + palm_offset)
        - Orientation: kept as original (how object naturally sits when grasped)

        The orientation is preserved because the object's mesh orientation relative to
        the hand is what gets captured by CuRobo's attachment mechanism.
        """
        # Get original pose for debugging
        orig = self._object_helper.get_object_pose(env_id, object_name)
        if orig:
            print(f"[MOVE OBJECT] {object_name}: Original pos (world) = {orig[0].cpu().tolist()}")

        ee_pose = get_ee_pose_from_fk(self._robot_articulation, arm_planner, env_id)
        if ee_pose is None:
            return None
        ee_pos, _ = ee_pose  # Only use position

        print(f"[MOVE OBJECT] {object_name}: Target EE/palm pos (world) = {ee_pos.cpu().tolist()}")

        result = self._object_helper.move_to_ee_position(env_id, object_name, ee_pos)

        # Verify the move happened
        new = self._object_helper.get_object_pose(env_id, object_name)
        if new:
            print(f"[MOVE OBJECT] {object_name}: After move pos (world) = {new[0].cpu().tolist()}")

        return result

    def _restore_object_pose_for_offline(
        self,
        env_id: int,
        object_name: str,
        original_pose: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Restore an object's original pose after offline planning."""
        self._object_helper.restore_pose(env_id, object_name, original_pose)

    def _execute_motion_planning_with_attachment(
        self,
        env_id: int,
        arm_planner: Any,
        expected_attached_object: str | None,
        motion_planner: Any,
        planner_kwargs: dict,
    ) -> bool:
        """
        Execute motion planning with optional object attachment setup.

        Uses context manager for grasp check patching and handles object pose
        save/restore for correct attachment computation in offline mode.
        """
        if expected_attached_object is None:
            return motion_planner.update_world_and_plan_motion(**planner_kwargs)

        # Move object to EE and save original pose
        original_object_pose = self._move_object_to_ee_for_offline(
            env_id=env_id,
            object_name=expected_attached_object,
            arm_planner=arm_planner,
        )

        try:
            with force_grasp_check(arm_planner, expected_attached_object):
                planning_success = motion_planner.update_world_and_plan_motion(**planner_kwargs)
        finally:
            if original_object_pose is not None:
                self._restore_object_pose_for_offline(
                    env_id=env_id,
                    object_name=expected_attached_object,
                    original_pose=original_object_pose,
                )

        return planning_success

    def _get_subtask_config(self, eef_name: str, subtask_index: int) -> Any | None:
        """Get subtask configuration for the given EEF and index."""
        if eef_name not in self.env_cfg.subtask_configs:
            return None
        subtask_configs = self.env_cfg.subtask_configs[eef_name]
        if not (0 <= subtask_index < len(subtask_configs)):
            return None
        return subtask_configs[subtask_index]

    def _get_planned_joint_positions(self, motion_planner: Any) -> torch.Tensor | None:
        """Extract joint positions from the planner's current plan."""
        current_plan = getattr(motion_planner, "_current_plan", None)
        if current_plan is None:
            return None
        position = getattr(current_plan, "position", None)
        if position is None:
            return None
        if isinstance(position, torch.Tensor):
            return position.clone().to(device=self.env.device, dtype=torch.float32)
        return torch.as_tensor(position, device=self.env.device, dtype=torch.float32)

    def _collect_offline_waypoints(
        self,
        env_id: int,
        eef_states: dict[str, EEFGenerationState],
        runtime_constraints: dict,
        motion_planner: Any | None,
        global_robot_joint_state: torch.Tensor,
    ) -> tuple[dict[str, Waypoint], torch.Tensor]:
        """
        Collect waypoints for the current timestep without stepping the simulator.

        This mirrors _collect_eef_waypoints but also tracks joint positions and
        solves IK for skill segment waypoints. Returns the updated global robot
        joint state to maintain consistency across arms for collision detection.
        """
        eef_waypoint_dict: dict[str, Waypoint] = {}
        for eef_name in sorted(self.env_cfg.subtask_configs.keys()):
            eef_state = eef_states[eef_name]
            if eef_state.subtask_step_index is None:
                continue

            # Cache arm planner for this EEF (used for IK and joint merging)
            arm_planner = self._get_arm_planner(motion_planner, eef_name) if motion_planner else None

            is_paused = self._apply_constraint_progression_rules(
                eef_name=eef_name,
                eef_state=eef_state,
                runtime_constraints=runtime_constraints,
                eef_states=eef_states,
            )

            was_paused = getattr(eef_state, '_was_paused_prev_iter', False)
            just_paused = not was_paused and is_paused
            eef_state._was_paused_prev_iter = is_paused
            eef_state.is_currently_paused = is_paused

            if is_paused:
                if just_paused:
                    eef_state.constraint_hold_waypoint = deepcopy(
                        eef_state.current_trajectory[eef_state.subtask_step_index]
                    )
                waypoint = deepcopy(eef_state.constraint_hold_waypoint) if eef_state.constraint_hold_waypoint else \
                    eef_state.current_trajectory[eef_state.subtask_step_index]
            else:
                waypoint = eef_state.current_trajectory[eef_state.subtask_step_index]
                eef_state.constraint_hold_waypoint = deepcopy(waypoint)

            # Get or solve for joint position
            step_idx = eef_state.subtask_step_index
            joint_vec = None
            if step_idx < len(eef_state.current_joint_trajectory):
                joint_vec = eef_state.current_joint_trajectory[step_idx]

            if joint_vec is None and waypoint.joint_seed is not None:
                joint_vec = waypoint.joint_seed

            if joint_vec is None and arm_planner is not None:
                # Solve IK for this waypoint
                joint_vec = self._solve_ik_for_waypoint(
                    waypoint.pose,
                    arm_planner,
                    eef_state.last_commanded_joint_position,
                )

            if joint_vec is not None:
                eef_state.last_commanded_joint_position = joint_vec.clone()
                # Merge this arm's joints into the global state for consistency
                if arm_planner is not None:
                    global_robot_joint_state = self._merge_arm_joints_to_global(
                        global_state=global_robot_joint_state,
                        arm_joints=joint_vec,
                        arm_planner=arm_planner,
                    )
                    # Update robot state for subsequent IK/motion planning (skip sim sync in offline mode)
                    self._set_robot_joint_state(env_id, joint_vec, arm_planner, skip_sim_sync=True)

            eef_waypoint_dict[eef_name] = waypoint
            eef_state.last_commanded_gripper_action = waypoint.gripper_action

        return eef_waypoint_dict, global_robot_joint_state

    def _solve_ik_for_waypoint(
        self,
        pose: torch.Tensor,
        planner: Any,
        seed_joint: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """
        Solve IK for a single waypoint pose.

        Args:
            pose: 4x4 target pose matrix.
            planner: Arm-specific planner with IK solver.
            seed_joint: Optional seed joint configuration.

        Returns:
            Solved joint configuration, or seed_joint if IK fails.
        """
        if planner is None or pose is None:
            return seed_joint

        # Determine planner device (cuRobo typically runs on CUDA)
        planner_device = self.env.device
        if hasattr(planner, "motion_gen") and hasattr(planner.motion_gen, "tensor_args"):
            planner_device = planner.motion_gen.tensor_args.device

        # Convert pose to planner frame, ensuring correct device
        pose_bt = self._convert_world_pose_to_planner_frame(pose, env_id=0).to(device=planner_device, dtype=torch.float32)
        pos_bt, rot_bt = PoseUtils.unmake_pose(pose_bt.unsqueeze(0))
        quat_bt = PoseUtils.quat_from_matrix(rot_bt)[0]

        pose_obj = planner._make_pose(position=pos_bt[0], quaternion=quat_bt)

        seed_config = None
        retract_config = None
        if seed_joint is not None:
            seed_vec = seed_joint.to(device=planner_device, dtype=torch.float32)
            retract_config = seed_vec.reshape(1, -1)
            seed_config = seed_vec.reshape(1, 1, -1)

        ik_result = planner.motion_gen.ik_solver.solve_single(
            pose_obj,
            retract_config=retract_config,
            seed_config=seed_config,
        )

        js_solution = getattr(ik_result, "js_solution", None)
        if js_solution is not None:
            joint_vec = js_solution.position
        else:
            joint_vec = ik_result.solution

        if joint_vec is None:
            return seed_joint

        if isinstance(joint_vec, torch.Tensor) and joint_vec.ndim > 1:
            joint_vec = joint_vec[0]

        result = torch.as_tensor(joint_vec, dtype=torch.float32, device=self.env.device).view(-1)

        # Clamp IK solution to joint limits to prevent controller jerking
        if hasattr(planner, "motion_gen") and hasattr(planner.motion_gen, "kinematics"):
            try:
                limits = planner.motion_gen.kinematics.get_joint_limits().position
                low = limits[0].to(device=result.device, dtype=result.dtype)
                high = limits[1].to(device=result.device, dtype=result.dtype)
                # Apply clamping with small margin to avoid boundary issues
                margin = 1e-3
                result = torch.clamp(result, low + margin, high - margin)
            except Exception:
                pass  # If limit retrieval fails, use unclamped result

        return result

    def _convert_world_pose_to_planner_frame(self, pose: torch.Tensor, env_id: int) -> torch.Tensor:
        """Convert a world-frame pose to the planner's base frame."""
        T_world_base = get_world_to_base_transform(
            self._robot_articulation, self.env.scene.env_origins, env_id, self.env.device
        )
        T_base_world = torch.linalg.inv(T_world_base)
        return (T_base_world @ pose.to(device=self.env.device, dtype=torch.float32)).clone()

    def _set_robot_joint_state(
        self, env_id: int, joint_values: torch.Tensor, planner: Any | None = None, skip_sim_sync: bool = False
    ) -> None:
        """
        Directly update the articulation joint state for the specified env.

        This is critical for offline planning - the planner reads its start state
        from robot.data.joint_pos, so we must sync it before each motion plan.

        Args:
            env_id: Environment ID.
            joint_values: Joint position values.
            planner: Optional planner for joint expansion.
            skip_sim_sync: If True, skip write_data_to_sim/update to avoid physics reset
                          in offline mode.
        """
        target = joint_values.to(device=self.env.device)
        if planner is not None and target.shape[0] != self._robot_articulation.data.joint_pos.shape[1]:
            target = self._expand_planner_joints_to_full(target, planner, env_id)
        self._robot_articulation.data.joint_pos[env_id] = target.clone()
        self._robot_articulation.data.joint_pos_target[env_id] = target.clone()
        if hasattr(self._robot_articulation.data, "joint_vel"):
            self._robot_articulation.data.joint_vel[env_id] = torch.zeros_like(target)
        if not skip_sim_sync:
            self._robot_articulation.write_data_to_sim()
            self._robot_articulation.update(0.0)

    def _expand_planner_joints_to_full(
        self,
        joint_vec: torch.Tensor,
        planner: Any,
        env_id: int,
    ) -> torch.Tensor:
        """Expand planner joint vector to full articulation ordering."""
        mapper = self._joint_mapper_cache.get(planner)
        return mapper.expand_to_full(joint_vec, env_id)

    def _merge_arm_joints_to_global(
        self,
        global_state: torch.Tensor,
        arm_joints: torch.Tensor,
        arm_planner: Any,
    ) -> torch.Tensor:
        """Merge an arm's joint positions into the global robot state."""
        mapper = self._joint_mapper_cache.get(arm_planner)
        return mapper.merge_to_global(global_state, arm_joints)

    def _advance_subtask_progress_offline(
        self,
        eef_states: dict[str, EEFGenerationState],
        runtime_constraints: dict,
        pause_subtask: bool,
    ) -> None:
        """Advance subtask progress for offline path building."""
        for eef_name, eef_state in eef_states.items():
            if eef_state.subtask_step_index is None:
                continue
            if eef_state.is_currently_paused:
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

    def _update_world_state_after_subtask_offline(
        self,
        env_id: int,
        completing_subtasks: dict[str, int],
        eef_states: dict[str, EEFGenerationState],
        motion_planner: Any | None,
    ) -> None:
        """
        Update world state after subtask completion in offline mode.

        This emulates what happens in online mode after executing a subtask:
        - Robot joint state is updated to the final trajectory position
        - If the subtask was a "grasp", the object is moved to the EE position
        - If the subtask was a "place" or "release", the object is moved to final pose

        Object pose updates are DEFERRED to this point (subtask completion) rather than
        immediately after motion planning. This ensures that other subtasks referencing
        the same object will plan with the correct (pre-grasp) object position.

        Args:
            env_id: Environment ID.
            completing_subtasks: Dict mapping eef_name -> subtask_index that just completed.
            eef_states: Current EEF states (after subtask completion).
            motion_planner: Motion planner for updating world model.
        """
        for eef_name, completed_subtask_idx in completing_subtasks.items():
            eef_state = eef_states[eef_name]

            # Update robot joint state to final trajectory position
            if eef_state.last_commanded_joint_position is not None and motion_planner is not None:
                arm_planner = self._get_arm_planner(motion_planner, eef_name)
                self._set_robot_joint_state(env_id, eef_state.last_commanded_joint_position, arm_planner, skip_sim_sync=True)

            subtask_cfg = self._get_subtask_config(eef_name, completed_subtask_idx)
            if subtask_cfg is None:
                continue

            object_ref = get_subtask_object_ref(subtask_cfg)
            if not object_ref or not eef_state.current_trajectory:
                continue

            # Grasp subtask: move object to EE position (now robot is "holding" it)
            if is_grasp_subtask(subtask_cfg):
                final_pose = eef_state.current_trajectory[-1].pose
                print(f"[Offline] Grasp complete: {eef_name} subtask {completed_subtask_idx} - moving {object_ref} to EE")
                self._update_object_position_after_place(env_id, object_ref, final_pose)

            # Place/release subtask: move object to final EE position (object is released there)
            elif is_place_subtask(subtask_cfg):
                final_pose = eef_state.current_trajectory[-1].pose
                print(f"[Offline] Place complete: {eef_name} subtask {completed_subtask_idx} - moving {object_ref} to final pose")
                self._update_object_position_after_place(env_id, object_ref, final_pose)

    def _update_object_position_after_place(
        self,
        env_id: int,
        object_name: str,
        final_ee_pose: torch.Tensor,
    ) -> None:
        """Update an object's position to reflect where it was placed."""
        new_pos = final_ee_pose[:3, 3].clone()
        self._object_helper.set_object_position(env_id, object_name, new_pos)

    def _apply_gripper_delay_and_interpolation(
        self,
        gripper_actions: torch.Tensor,
        delay_steps: int,
        interp_steps: int,
        change_threshold: float = 0.5,
        eef_name: str = "",
    ) -> torch.Tensor:
        """
        Delay and interpolate major gripper transitions for smooth grasping.

        When a major gripper change is detected (delta > threshold), this function:
        1. Extends the PREVIOUS gripper state for `delay_steps` waypoints (arm settles)
        2. Linearly interpolates to the TARGET gripper over `interp_steps` (smooth close)

        This prevents premature finger curl and ensures smooth gripper motion.

        Args:
            gripper_actions: [N, D] tensor of gripper actions.
            delay_steps: Number of steps to delay after detecting a major change.
            interp_steps: Number of steps to interpolate from previous to target gripper.
            change_threshold: Minimum delta to consider as a major gripper transition.
            eef_name: Name of the end-effector for debug logging.

        Returns:
            Modified gripper_actions tensor with delayed and interpolated transitions.
        """
        if (delay_steps <= 0 and interp_steps <= 0) or gripper_actions.shape[0] < 2:
            return gripper_actions

        modified = gripper_actions.clone()
        n_waypoints = gripper_actions.shape[0]
        debug = getattr(self.env_cfg.datagen_config, "debug_gripper_detail", False)

        i = 1
        while i < n_waypoints:
            delta = (gripper_actions[i] - gripper_actions[i - 1]).abs().max().item()
            if delta > change_threshold:
                # Major gripper change detected at index i
                prev_gripper = gripper_actions[i - 1].clone()
                target_gripper = gripper_actions[i].clone()

                # Phase 1: Delay - keep previous gripper for delay_steps
                delay_end = min(i + delay_steps, n_waypoints)
                for j in range(i, delay_end):
                    modified[j] = prev_gripper

                # Phase 2: Interpolate - smoothly transition to target gripper
                interp_start = delay_end
                interp_end = min(interp_start + interp_steps, n_waypoints)
                actual_interp_steps = interp_end - interp_start

                if actual_interp_steps > 0:
                    for step, j in enumerate(range(interp_start, interp_end)):
                        alpha = (step + 1) / (actual_interp_steps + 1)
                        modified[j] = prev_gripper * (1.0 - alpha) + target_gripper * alpha

                if debug:
                    print(f"[GRIPPER DELAY+INTERP] {eef_name}: waypoint {i}, "
                          f"delay={delay_end - i} steps, interp={actual_interp_steps} steps "
                          f"(delta={delta:.3f})")

                # Skip past the processed region
                i = interp_end if interp_end > i else i + 1
            else:
                i += 1

        return modified

    def _build_arm_paths_from_logs(
        self,
        waypoint_logs: dict[str, list[dict[str, Any]]],
        motion_planner: Any | None,
        env_id: int,
    ) -> dict[str, ArmPath]:
        """Convert collected waypoint logs into ArmPath objects for scheduling."""
        arm_paths: dict[str, ArmPath] = {}

        # Get gripper delay config (number of steps to delay gripper at skill start)
        gripper_delay_steps = int(getattr(self.env_cfg.datagen_config, "skill_gripper_delay_steps", 0))

        for eef_name, entries in waypoint_logs.items():
            if not entries:
                continue

            poses = torch.stack([e["waypoint"].pose.clone() for e in entries], dim=0)
            gripper_actions = torch.stack([e["waypoint"].gripper_action.clone() for e in entries], dim=0)

            # Compute subtask boundaries: subtask_index -> (start_idx, end_idx)
            # Each subtask includes both MP transition and skill waypoints.
            subtask_boundaries: dict[int, tuple[int, int]] = {}
            current_subtask: int | None = None
            subtask_start = 0
            for i, entry in enumerate(entries):
                si = entry.get("subtask_index")
                if si != current_subtask:
                    if current_subtask is not None:
                        subtask_boundaries[current_subtask] = (subtask_start, i)
                    current_subtask = si
                    subtask_start = i
            if current_subtask is not None:
                subtask_boundaries[current_subtask] = (subtask_start, len(entries))

            print(f"[ArmPath] {eef_name} subtask boundaries (MP+skill): {subtask_boundaries}")

            # Apply gripper delay and interpolation if configured
            gripper_interp_steps = int(getattr(self.env_cfg.datagen_config, "skill_gripper_interp_steps", 20))
            if gripper_delay_steps > 0 or gripper_interp_steps > 0:
                gripper_actions = self._apply_gripper_delay_and_interpolation(
                    gripper_actions,
                    delay_steps=gripper_delay_steps,
                    interp_steps=gripper_interp_steps,
                    change_threshold=0.5,  # Threshold for major gripper change
                    eef_name=eef_name,
                )

            # Build joint position tensor, solving IK for missing entries
            joint_positions = self._extract_joint_history_from_logs(
                eef_name=eef_name,
                entries=entries,
                motion_planner=motion_planner,
                env_id=env_id,
            )

            arm_paths[eef_name] = ArmPath(
                name=eef_name,
                poses=poses,
                gripper_actions=gripper_actions,
                joint_positions=joint_positions,
                subtask_boundaries=subtask_boundaries,
            )

        if "left" not in arm_paths or "right" not in arm_paths:
            missing = [k for k in ("left", "right") if k not in arm_paths]
            raise ValueError(f"Offline scheduling requires left/right trajectories; missing {missing}.")

        return arm_paths

    def _inject_sequential_hold_waypoints(
        self,
        arm_paths: dict[str, ArmPath],
        runtime_constraints: dict,
        step_dt: float,
    ) -> tuple[dict[str, ArmPath], list[HoldConstraint]]:
        """
        Inject hold waypoints into LATTER arm's trajectory to match online SEQUENTIAL behavior.

        In online mode, the LATTER arm holds within its *constrained subtask*:
          hold when step_index >= subtask_traj_len - min_time_diff

        The offline code must replicate this by computing hold_idx relative to the
        constrained subtask's boundaries (MP + skill) within the concatenated trajectory,
        NOT relative to the full trajectory length.

        Args:
            arm_paths: Dict of arm name -> ArmPath (with subtask_boundaries populated).
            runtime_constraints: Runtime constraint dictionary.
            step_dt: Time step for discretization.

        Returns:
            Tuple of (modified arm_paths, list of HoldConstraints for scheduler).
        """
        hold_constraints: list[HoldConstraint] = []

        for subtask_key, constraint in runtime_constraints.items():
            if constraint.get("type") != SubTaskConstraintType._SEQUENTIAL_LATTER:
                continue

            latter_eef, latter_subtask_idx = subtask_key
            former_eef = constraint.get("pre_condition_task_spec_key")
            former_subtask_idx = constraint.get("pre_condition_subtask_ind")
            min_time_diff = constraint.get("min_time_diff", 0)

            if min_time_diff <= 0:
                continue

            if latter_eef not in arm_paths or former_eef not in arm_paths:
                continue

            latter_path = arm_paths[latter_eef]
            former_path = arm_paths[former_eef]
            latter_len = latter_path.joint_positions.shape[0]

            # FORMER length: use FORMER subtask's end boundary (MP+skill), not full trajectory.
            # Online, hold releases when that specific FORMER subtask completes.
            former_boundaries = former_path.subtask_boundaries
            if former_boundaries is not None and former_subtask_idx in former_boundaries:
                _, former_subtask_end = former_boundaries[former_subtask_idx]
                former_len = former_subtask_end
            else:
                former_len = former_path.joint_positions.shape[0]
                print(f"[Sequential Hold] WARNING: No subtask boundary for FORMER {former_eef} "
                      f"subtask {former_subtask_idx}, using full trajectory length {former_len}")

            # LATTER: look up the constrained subtask's boundaries (MP+skill)
            latter_boundaries = latter_path.subtask_boundaries
            if latter_boundaries is None or latter_subtask_idx not in latter_boundaries:
                print(f"[Sequential Hold] WARNING: No subtask boundary for {latter_eef} "
                      f"subtask {latter_subtask_idx}, falling back to full trajectory")
                subtask_start = 0
                subtask_end = latter_len
            else:
                subtask_start, subtask_end = latter_boundaries[latter_subtask_idx]

            subtask_len = subtask_end - subtask_start

            # Online behavior: hold when step_index >= subtask_traj_len - min_time_diff
            # hold_idx is in the concatenated trajectory space
            hold_idx = subtask_start + max(0, subtask_len - min_time_diff)

            # How many hold waypoints: FORMER subtask must complete, then LATTER resumes.
            # Both arms start at time 0, so hold count = former_len - hold_idx.
            n_hold = max(0, former_len - hold_idx)

            if n_hold <= 0:
                print(f"[Sequential Hold] No hold needed for {latter_eef} subtask {latter_subtask_idx} "
                      f"(FORMER completes in time)")
                continue

            print(f"[Sequential Hold] {latter_eef} subtask {latter_subtask_idx}: "
                  f"subtask range [{subtask_start}:{subtask_end}] (len={subtask_len}), "
                  f"hold at global idx {hold_idx} for {n_hold} steps")
            print(f"  FORMER {former_eef} subtask {former_subtask_idx} ends at idx {former_len}, "
                  f"LATTER subtask pre-hold: {hold_idx - subtask_start}, "
                  f"LATTER subtask post-hold: {subtask_end - hold_idx}")

            # Get the waypoint to hold at
            hold_pose = latter_path.poses[hold_idx:hold_idx + 1]
            hold_gripper = latter_path.gripper_actions[hold_idx:hold_idx + 1]
            hold_joint = latter_path.joint_positions[hold_idx:hold_idx + 1]

            # Create hold waypoints
            hold_poses = hold_pose.repeat(n_hold, 1, 1)
            hold_gripper_actions = hold_gripper.repeat(n_hold, 1)
            hold_joints = hold_joint.repeat(n_hold, 1)

            # Split trajectory at hold position and insert hold waypoints
            # [0...hold_idx-1] + [hold_idx repeated n_hold times] + [hold_idx...end]
            new_poses = torch.cat([
                latter_path.poses[:hold_idx], hold_poses, latter_path.poses[hold_idx:]
            ], dim=0)
            new_gripper = torch.cat([
                latter_path.gripper_actions[:hold_idx], hold_gripper_actions, latter_path.gripper_actions[hold_idx:]
            ], dim=0)
            new_joints = torch.cat([
                latter_path.joint_positions[:hold_idx], hold_joints, latter_path.joint_positions[hold_idx:]
            ], dim=0)

            arm_paths[latter_eef] = ArmPath(
                name=latter_path.name,
                poses=new_poses,
                gripper_actions=new_gripper,
                joint_positions=new_joints,
            )

            print(f"[Sequential Hold] {latter_eef} trajectory: {latter_len} -> {new_joints.shape[0]} waypoints")
            print(f"  Structure: [{hold_idx} pre-hold] + [{n_hold} hold] + "
                  f"[{latter_len - hold_idx} post-hold]")

            # Create hold constraint for the scheduler
            hold_constraints.append(HoldConstraint(
                holding_arm=latter_eef,
                other_arm=former_eef,
                hold_start_idx=hold_idx,
                hold_duration=n_hold,
                other_arm_len=former_len,
            ))
        
        return arm_paths, hold_constraints

    def _extract_joint_history_from_logs(
        self,
        eef_name: str,
        entries: list[dict[str, Any]],
        motion_planner: Any | None,
        env_id: int,
    ) -> torch.Tensor:
        """Extract or solve for joint positions from waypoint log entries."""
        if motion_planner is None:
            raise ValueError(f"No motion planner provided for end effector '{eef_name}'.")

        arm_planner = self._get_arm_planner(motion_planner, eef_name)
        planner_dof = self._get_planner_dof(arm_planner)

        # Get planner's device (CUDA) - cuRobo kinematics requires CUDA tensors
        tensor_args = getattr(arm_planner, "tensor_args", None)
        if tensor_args is None and hasattr(arm_planner, "motion_gen"):
            tensor_args = getattr(arm_planner.motion_gen, "tensor_args", None)
        planner_device = getattr(tensor_args, "device", self.env.device) if tensor_args else self.env.device
        planner_dtype = getattr(tensor_args, "dtype", torch.float32) if tensor_args else torch.float32

        joints: list[torch.Tensor] = []
        last_valid: torch.Tensor | None = None

        for entry in entries:
            joint_vec = entry.get("joint")
            if joint_vec is not None:
                # Use recorded joint position (already in planner space, just ensure device/dtype)
                projected = self._project_joint_to_planner(joint_vec, arm_planner)
                projected = projected.to(device=planner_device, dtype=planner_dtype)
                joints.append(projected)
                last_valid = projected
            else:
                # Solve IK for this waypoint
                waypoint = entry["waypoint"]
                solved = self._solve_ik_for_waypoint(waypoint.pose, arm_planner, last_valid)
                if solved is not None:
                    projected = self._project_joint_to_planner(solved, arm_planner)
                    projected = projected.to(device=planner_device, dtype=planner_dtype)
                    joints.append(projected)
                    last_valid = projected
                elif last_valid is not None:
                    joints.append(last_valid.clone())
                else:
                    joints.append(torch.zeros(planner_dof, device=planner_device, dtype=planner_dtype))

        return torch.stack(joints, dim=0)

    def _project_joint_to_planner(self, joint_vec: torch.Tensor, planner: Any) -> torch.Tensor:
        """Project a joint vector to the planner's expected joint ordering."""
        mapper = self._joint_mapper_cache.get(planner)
        return mapper.project_to_planner(joint_vec)

    def _get_planner_dof(self, planner: Any) -> int:
        """Get the number of DOFs for the planner."""
        mapper = self._joint_mapper_cache.get(planner)
        return mapper.planner_dof

    def _apply_scheduled_joint_state(
        self,
        env_id: int,
        arm_paths: dict[str, ArmPath],
        left_idx: int,
        right_idx: int,
        planner_map: dict[str, Any] | None,
    ) -> None:
        """
        Set the robot articulation to the recorded joint state for the scheduled tick.

        When replaying a collision-aware schedule, the controller receives pose targets but
        the articulation should already be at the configuration the planner computed.
        """
        if planner_map is None:
            return

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

    def _apply_initial_gripper_state(
        self,
        env_id: int,
        arm_paths: dict[str, ArmPath],
        left_idx: int,
        right_idx: int,
    ) -> None:
        """Apply gripper joint state from first waypoints to match cuRobo's planned start.

        Ensures the physical gripper matches what cuRobo assumed when planning
        collision-free trajectories. This prevents mismatch after env reset.
        """
        actions_cfg = getattr(getattr(self.env, "cfg", None), "actions", None)
        pink_cfg = getattr(actions_cfg, "pink_ik_cfg", None) if actions_cfg else None
        hand_names = getattr(pink_cfg, "hand_joint_names", None) if pink_cfg else None
        if not hand_names:
            return

        robot_joint_names = list(self._robot_articulation.data.joint_names)
        name_to_idx = {n: i for i, n in enumerate(robot_joint_names)}

        for arm_name, idx in [("left", left_idx), ("right", right_idx)]:
            arm_path = arm_paths.get(arm_name)
            if arm_path is None or arm_path.gripper_actions is None:
                continue
            if idx >= arm_path.gripper_actions.shape[0]:
                continue

            gripper = arm_path.gripper_actions[idx].to(device=self.env.device, dtype=torch.float32)
            prefix = "L_" if arm_name == "left" else "R_"
            arm_hand_names = [n for n in hand_names if n.startswith(prefix)]

            for i, hand_name in enumerate(arm_hand_names):
                if hand_name in name_to_idx and i < gripper.shape[0]:
                    self._robot_articulation.data.joint_pos[env_id, name_to_idx[hand_name]] = gripper[i]
                    self._robot_articulation.data.joint_pos_target[env_id, name_to_idx[hand_name]] = gripper[i]

        self._robot_articulation.write_data_to_sim()
        self._robot_articulation.update(0.0)

    def _compute_interpolation_steps(
        self,
        arm_paths: dict[str, ArmPath],
        prev_left_idx: int,
        prev_right_idx: int,
        curr_left_idx: int,
        curr_right_idx: int,
        max_joint_step: float,
    ) -> int:
        """Compute number of interpolation steps needed to smooth large joint jumps."""
        if prev_left_idx < 0 or prev_right_idx < 0:
            return 0

        max_steps = 0
        for arm_name, prev_idx, curr_idx in [("left", prev_left_idx, curr_left_idx), ("right", prev_right_idx, curr_right_idx)]:
            joints = arm_paths[arm_name].joint_positions
            if joints is None or joints.numel() == 0:
                continue
            if prev_idx >= joints.shape[0] or curr_idx >= joints.shape[0]:
                continue
            steps = compute_interpolation_steps(joints[prev_idx], joints[curr_idx], max_joint_step)
            max_steps = max(max_steps, steps)

        return max_steps

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
        
        IMPORTANT: Gripper actions are NOT interpolated. We keep the previous
        gripper state during interpolation to prevent premature gripper closing/opening.
        The gripper only changes when we reach the actual target waypoint.
        """
        result = {}
        for arm_name, prev_idx, curr_idx in [("left", prev_left_idx, curr_left_idx), ("right", prev_right_idx, curr_right_idx)]:
            arm_path = arm_paths[arm_name]
            p0 = arm_path.poses[prev_idx].to(device=self.env.device, dtype=torch.float32)
            p1 = arm_path.poses[curr_idx].to(device=self.env.device, dtype=torch.float32)
            g0 = arm_path.gripper_actions[prev_idx].to(device=self.env.device, dtype=torch.float32)

            result[arm_name] = Waypoint(
                pose=interpolate_pose(p0, p1, alpha),
                # Keep previous gripper state during interpolation to prevent
                # premature grasp/release before arm reaches target position
                gripper_action=g0,
                noise=0.0,
            )
        return result

    def _compute_tracking_correction_steps(
        self,
        env_id: int,
        target_waypoints: dict[str, Waypoint],
        max_pose_error: float = 0.05,
    ) -> int:
        """
        Compute interpolation steps needed to correct tracking error.

        Reads the actual robot EE pose and compares to the target waypoint.
        If the error exceeds the threshold, returns the number of interpolation
        steps needed to smoothly reach the target.

        Args:
            env_id: Environment ID.
            target_waypoints: Target waypoints for each arm.
            max_pose_error: Maximum allowed position error (meters) before correction.

        Returns:
            Number of interpolation steps needed (0 if no correction needed).
        """
        max_error = 0.0
        for eef_name, waypoint in target_waypoints.items():
            actual_pose = self.env.get_robot_eef_pose(eef_name, env_ids=[env_id])
            if actual_pose is None or actual_pose.numel() == 0:
                continue
            actual_pos = actual_pose[0, :3, 3]
            target_pos = waypoint.pose[:3, 3].to(device=actual_pos.device)
            error = (actual_pos - target_pos).norm().item()
            max_error = max(max_error, error)

        if max_error <= max_pose_error:
            return 0

        # More steps for larger errors (1 step per 2cm of error beyond threshold)
        extra_steps = int((max_error - max_pose_error) / 0.02)
        return min(extra_steps + 1, 10)  # Cap at 10 interpolation steps

    def _interpolate_from_actual(
        self,
        env_id: int,
        target_waypoints: dict[str, Waypoint],
        alpha: float,
        prev_gripper_actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Waypoint]:
        """
        Interpolate from actual robot pose to target waypoint.

        Args:
            env_id: Environment ID.
            target_waypoints: Target waypoints for each arm.
            alpha: Interpolation factor (0 = actual, 1 = target).
            prev_gripper_actions: Previous gripper actions to maintain during correction.
                If None, uses target gripper action (legacy behavior).

        Returns:
            Interpolated waypoints from actual to target.
        """
        result = {}
        for eef_name, waypoint in target_waypoints.items():
            actual_pose = self.env.get_robot_eef_pose(eef_name, env_ids=[env_id])
            if actual_pose is None or actual_pose.numel() == 0:
                result[eef_name] = waypoint
                continue

            actual_pose_4x4 = actual_pose[0].to(device=self.env.device, dtype=torch.float32)
            target_pose = waypoint.pose.to(device=self.env.device, dtype=torch.float32)

            interp_pose = interpolate_pose(actual_pose_4x4, target_pose, alpha)
            
            # Use previous gripper state during tracking correction to prevent
            # premature gripper changes before arm reaches position
            if prev_gripper_actions is not None and eef_name in prev_gripper_actions:
                gripper = prev_gripper_actions[eef_name]
            else:
                gripper = waypoint.gripper_action
            
            result[eef_name] = Waypoint(
                pose=interp_pose,
                gripper_action=gripper,
                noise=0.0,
            )
        return result

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
        """
        Replay the scheduled trajectory in the simulator.

        Args:
            env_id: Environment ID.
            env_id_tensor: Tensor version of env_id.
            initial_state: Initial state to reset to before replay.
            success_term: Termination condition.
            env_action_queue: Action queue for async dispatch.
            arm_paths: Per-arm trajectory data.
            schedule: Discrete schedule mapping ticks to waypoint indices.
            export_demo: Whether to export the resulting demonstration.
            planner_map: Mapping from arm name to planner for joint expansion.

        Returns:
            Dictionary containing generated trajectory data.
        """
        buffers = GenerationBuffers()

        # Set the initial joint state ONCE at the start of replay to align with first waypoints.
        # This is important even for humanoids - we need the robot to start from the correct
        # configuration before the controller takes over with pose targets.
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
            # Apply gripper state from first waypoints to match cuRobo's planned start
            self._apply_initial_gripper_state(
                env_id=env_id,
                arm_paths=arm_paths,
                left_idx=left_idx_0,
                right_idx=right_idx_0,
            )

        # Configurable smoothing: max joint change per tick (radians)
        max_joint_step_rad = float(getattr(self.env_cfg.datagen_config, "max_joint_step_rad", 0.1))
        prev_left_idx = -1
        prev_right_idx = -1

        # Debug: track gripper changes
        prev_gripper_left: torch.Tensor | None = None
        prev_gripper_right: torch.Tensor | None = None
        # Track last COMMANDED gripper (what we actually sent to the robot)
        last_commanded_gripper: dict[str, torch.Tensor] = {}
        debug_schedule = getattr(self.env_cfg.datagen_config, "debug_schedule_replay", False)
        debug_gripper_detail = getattr(self.env_cfg.datagen_config, "debug_gripper_detail", False)

        # Analyze and print gripper transition points in the original trajectory
        if debug_schedule:
            print("\n[GRIPPER ANALYSIS] Analyzing original trajectory gripper transitions...")
            for arm_name in ["left", "right"]:
                arm_path = arm_paths[arm_name]
                g_actions = arm_path.gripper_actions
                print(f"  {arm_name.upper()} arm: {len(g_actions)} waypoints")
                
                # Find all significant gripper change points
                change_points = []
                for i in range(1, len(g_actions)):
                    delta = (g_actions[i] - g_actions[i - 1]).abs().max().item()
                    if delta > 0.05:  # Any notable change
                        change_points.append((i, delta, g_actions[i - 1].tolist(), g_actions[i].tolist()))
                
                if change_points:
                    print("Gripper change points:")
                    for idx, delta, before, after in change_points:
                        pos = arm_path.poses[idx][:3, 3].tolist()
                        print(f"waypoint {idx}: delta={delta:.3f}, {before} → {after}")
                        print(f"EE pos: {[f'{p:.3f}' for p in pos]}")
                else:
                    print("    No significant gripper changes found in trajectory")
            print()

        num_ticks = schedule.left_indices.shape[0]
        for tick in range(num_ticks):
            left_idx = int(schedule.left_indices[tick].item())
            right_idx = int(schedule.right_indices[tick].item())

            # DEBUG: Detect large index jumps (potential cause of jerks)
            if debug_schedule and prev_left_idx >= 0:
                left_jump = left_idx - prev_left_idx
                right_jump = right_idx - prev_right_idx
                if abs(left_jump) > 3 or abs(right_jump) > 3:
                    print(f"[DEBUG] Tick {tick}: Large index jump! L:{prev_left_idx}→{left_idx} (+{left_jump}), R:{prev_right_idx}→{right_idx} (+{right_jump})")

            # DEBUG: Detect gripper changes in trajectory
            if debug_schedule:
                g_left = arm_paths["left"].gripper_actions[left_idx]
                g_right = arm_paths["right"].gripper_actions[right_idx]
                if prev_gripper_left is not None and prev_gripper_right is not None:
                    delta_left = (g_left - prev_gripper_left).abs().max().item()
                    delta_right = (g_right - prev_gripper_right).abs().max().item()
                    if delta_left > 0.1 or delta_right > 0.1:
                        # Get current EE pose for context
                        left_pose = arm_paths["left"].poses[left_idx]
                        right_pose = arm_paths["right"].poses[right_idx]
                        print(f"[DEBUG] Tick {tick}: Gripper TRAJECTORY change! L_delta={delta_left:.3f} R_delta={delta_right:.3f}")
                        print(f"L_idx={left_idx}, R_idx={right_idx}")
                        print(f"L_gripper: {prev_gripper_left.tolist()} → {g_left.tolist()}")
                        print(f"R_gripper: {prev_gripper_right.tolist()} → {g_right.tolist()}")
                        print(f"L_pos={left_pose[:3,3].tolist()}, R_pos={right_pose[:3,3].tolist()}")
                        # Also show gripper values at surrounding waypoints
                        if left_idx > 0:
                            g_left_prev = arm_paths["left"].gripper_actions[left_idx - 1]
                            print(f"L_gripper[{left_idx-1}]={g_left_prev.tolist()}")
                        if left_idx + 1 < len(arm_paths["left"].gripper_actions):
                            g_left_next = arm_paths["left"].gripper_actions[left_idx + 1]
                            print(f"L_gripper[{left_idx+1}]={g_left_next.tolist()}")
                prev_gripper_left = g_left.clone()
                prev_gripper_right = g_right.clone()

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
            if debug_schedule and interp_steps > 0:
                print(f"[DEBUG] Tick {tick}: Adding {interp_steps} interpolation steps (L:{prev_left_idx}→{left_idx}, R:{prev_right_idx}→{right_idx})")

            for step in range(interp_steps):
                alpha = float(step + 1) / float(interp_steps + 1)
                interp_waypoints = self._interpolate_waypoints(
                    arm_paths=arm_paths,
                    prev_left_idx=prev_left_idx if prev_left_idx >= 0 else left_idx,
                    prev_right_idx=prev_right_idx if prev_right_idx >= 0 else right_idx,
                    curr_left_idx=left_idx,
                    curr_right_idx=right_idx,
                    alpha=alpha,
                )
                if debug_gripper_detail:
                    for eef_name, wp in interp_waypoints.items():
                        print(f"  [GRIPPER] Tick {tick} interp step {step}/{interp_steps}: {eef_name} = {wp.gripper_action.tolist()}")
                interp_multi = MultiWaypoint(interp_waypoints)
                exec_results = await interp_multi.execute(
                    env=self.env,
                    success_term=success_term,
                    env_id=env_id,
                    env_action_queue=env_action_queue,
                )
                self._update_execution_buffers(exec_results, buffers)

            # Build the target waypoint
            waypoint_dict = {
                "left": Waypoint(
                    pose=arm_paths["left"].poses[left_idx].to(device=self.env.device, dtype=torch.float32),
                    gripper_action=arm_paths["left"].gripper_actions[left_idx].to(device=self.env.device, dtype=torch.float32),
                    noise=0.0,
                ),
                "right": Waypoint(
                    pose=arm_paths["right"].poses[right_idx].to(device=self.env.device, dtype=torch.float32),
                    gripper_action=arm_paths["right"].gripper_actions[right_idx].to(device=self.env.device, dtype=torch.float32),
                    noise=0.0,
                ),
            }

            # Check if we need tracking correction interpolation from ACTUAL pose to target
            # This handles controller tracking error by smoothing large jumps
            tracking_interp_steps = self._compute_tracking_correction_steps(
                env_id=env_id,
                target_waypoints=waypoint_dict,
                max_pose_error=0.05,  # 5cm threshold for correction
            )

            if tracking_interp_steps > 0:
                if debug_schedule:
                    print(f"[DEBUG] Tick {tick}: Tracking correction {tracking_interp_steps} steps (controller lag)")
                for step in range(tracking_interp_steps):
                    alpha = float(step + 1) / float(tracking_interp_steps + 1)
                    corrected_waypoints = self._interpolate_from_actual(
                        env_id=env_id,
                        target_waypoints=waypoint_dict,
                        alpha=alpha,
                        prev_gripper_actions=last_commanded_gripper if last_commanded_gripper else None,
                    )
                    if debug_gripper_detail:
                        for eef_name, wp in corrected_waypoints.items():
                            print(f"  [GRIPPER] Tick {tick} tracking step {step}: {eef_name} = {wp.gripper_action.tolist()}")
                    correction_multi = MultiWaypoint(corrected_waypoints)
                    exec_results = await correction_multi.execute(
                        env=self.env,
                        success_term=success_term,
                        env_id=env_id,
                        env_action_queue=env_action_queue,
                    )
                    self._update_execution_buffers(exec_results, buffers)

            # Execute the actual scheduled waypoint
            if debug_gripper_detail:
                for eef_name, wp in waypoint_dict.items():
                    print(f"  [GRIPPER] Tick {tick} TARGET: {eef_name} = {wp.gripper_action.tolist()}")
            multi_waypoint = MultiWaypoint(waypoint_dict)
            exec_results = await multi_waypoint.execute(
                env=self.env,
                success_term=success_term,
                env_id=env_id,
                env_action_queue=env_action_queue,
            )
            self._update_execution_buffers(exec_results, buffers)

            # Update last commanded gripper for next iteration
            last_commanded_gripper = {
                eef_name: wp.gripper_action.clone()
                for eef_name, wp in waypoint_dict.items()
            }

            prev_left_idx = left_idx
            prev_right_idx = right_idx

        print(f"[Replay] Completed {num_ticks} ticks. Success: {buffers.success}")

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
