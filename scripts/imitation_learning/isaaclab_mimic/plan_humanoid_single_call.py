# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Plan and execute a bimanual motion with a single cuRobo call.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--enable_pinocchio", action="store_true", default=False)

parser.add_argument("--goal", type=str, default="up", choices=["up", "lateral", "forward", "random"])
parser.add_argument("--dz", type=float, default=0.05)
parser.add_argument("--dx", type=float, default=0.05)
parser.add_argument("--dy", type=float, default=0.05)

parser.add_argument("--retime_deg", type=float, default=0.0)
parser.add_argument("--rest", type=int, default=10)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--visualize_goal", action="store_true")

# Optional per-arm overrides
parser.add_argument("--right_dx", type=float, default=None)
parser.add_argument("--right_dy", type=float, default=None)
parser.add_argument("--right_dz", type=float, default=None)
parser.add_argument("--left_dx", type=float, default=None)
parser.add_argument("--left_dy", type=float, default=None)
parser.add_argument("--left_dz", type=float, default=None)
parser.add_argument("--replay_trials", type=int, default=2)

# Isaac app args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import os
import tempfile
import torch
import yaml
from dataclasses import replace as dc_replace
from typing import Any, cast

from nvplan.applications.custream.config import create_robot_config
from nvplan.applications.custream.spheres import load_spheres

import isaaclab.utils.math as PoseUtils
from isaaclab.controllers import utils as ControllerUtils
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
from isaaclab_mimic.envs.pinocchio_envs.pickplace_gr1t2_mimic_env_cfg import PickPlaceGR1T2MimicEnvCfg
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
from isaaclab_mimic.motion_planners.curobo.curobo_planner_humanoid import HumanoidArmCuroboPlanner

import isaaclab_tasks  # noqa: F401


def _tool_link_for_arm(arm: str) -> str:
    """Return the tool link name for the specified arm.

    Args:
        arm: Arm side identifier, typically "right" or "left".

    Returns:
        The full link name for the fourier 6-DoF hand pitch link on the given arm.
    """
    return f"GR1T2_fourier_hand_6dof_{arm}_hand_pitch_link"


def to_python(obj):
    """Recursively convert numpy types to native Python types for YAML/JSON.

    This utility ensures compatibility when dumping configs by mapping numpy scalars
    and arrays to native Python lists and numbers.

    Args:
        obj: Arbitrary Python/numpy object possibly containing numpy types.

    Returns:
        A structure with numpy types converted to Python built-ins.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj


def _build_temp_robot_yaml_both_arms(usd_path: str, inactive_joints) -> str:
    """Create a temporary robot YAML for cuRobo with both arm tool links.

    Converts the provided USD robot to URDF, loads collision spheres, strips
    per-arm lock/cspace keys to avoid over-constraints, and writes a YAML file
    consumable by cuRobo. Returns the temporary file path.

    Args:
        usd_path: Path to the robot USD file.
        inactive_joints: Iterable of joint names to disable (fixed) in cuRobo.

    Returns:
        Absolute path to the generated temporary YAML file.
    """
    tmp_dir = tempfile.mkdtemp(prefix="gr1_curobo_full_")
    urdf_path, _ = ControllerUtils.convert_usd_to_urdf(usd_path, tmp_dir, force_conversion=True)

    tool_links = [
        "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
        "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
    ]
    robot_config = create_robot_config(
        urdf_path,
        tool_links=tool_links,
        ignore_arms=True,
        inactive_joints=inactive_joints or [],
        depth=2,
        verbose=False,
    )
    max_spheres = 225 - len(tool_links) * 50
    load_spheres(robot_config, max_spheres=max_spheres, max_link_spheres=int(1e9))

    robot_cfg_yaml = robot_config["robot_cfg"]
    robot_cfg_yaml = to_python(robot_cfg_yaml)

    def _strip_keys(obj, keys):
        if isinstance(obj, dict):
            for k in list(obj.keys()):
                if k in keys:
                    obj.pop(k, None)
                else:
                    _strip_keys(obj[k], keys)
        elif isinstance(obj, list):
            for v in obj:
                _strip_keys(v, keys)

    _strip_keys(robot_cfg_yaml, {"lock_joints", "cspace"})
    if isinstance(robot_cfg_yaml, dict):
        kin = robot_cfg_yaml.get("kinematics", None)
        if isinstance(kin, dict):
            kin["ee_link"] = "GR1T2_fourier_hand_6dof_right_hand_pitch_link"

    out_dir = tempfile.mkdtemp(prefix="curobo_robot_cfg_full_")
    out_path = os.path.join(out_dir, "gr1_full.yml")
    with open(out_path, "w") as f:
        yaml.safe_dump({"robot_cfg": robot_cfg_yaml}, f, sort_keys=False)
    return out_path


def rest_with_idle_action(env, steps=10):
    """Advance the simulation for a given number of steps using the idle action.

    Validates the shape of the environment's configured idle action, broadcasts it
    across all envs, and steps the environment without changing state.

    Args:
        env: Isaac Lab environment with an `idle_action` on its cfg.
        steps: Number of sim steps to execute.

    Raises:
        AttributeError: If `cfg.idle_action` is missing.
        ValueError: If idle action shape doesn't match the action space.
    """
    cfg = getattr(env, "cfg", None)
    if cfg is None or not hasattr(cfg, "idle_action"):
        raise AttributeError("This env has no cfg.idle_action defined.")
    idle = cfg.idle_action
    if not isinstance(idle, torch.Tensor):
        idle = torch.tensor(idle, dtype=torch.float32)
    else:
        idle = idle.to(dtype=torch.float32)
    idle = idle.to(device=env.device)
    act_dim = env.action_manager.total_action_dim
    if idle.shape[-1] != act_dim:
        raise ValueError(f"Idle action dim mismatch ({idle.shape[-1]} != {act_dim}).")
    if idle.dim() == 1:
        idle_batched = idle.unsqueeze(0).repeat(env.num_envs, 1)
    elif idle.dim() == 2 and idle.size(0) == 1:
        idle_batched = idle.repeat(env.num_envs, 1)
    elif idle.dim() == 2 and idle.size(0) == env.num_envs:
        idle_batched = idle
    else:
        raise ValueError(f"Unexpected idle_action shape: {tuple(idle.shape)}")
    for _ in range(steps):
        env.step(idle_batched)


def _build_env_and_planner_bimanual_single(args_cli):
    """Build a single-env mimic environment and a humanoid cuRobo planner instance.

    Sets up a temporary cuRobo robot config for both arms, creates the environment,
    optionally rests the sim, and constructs a `HumanoidArmCuroboPlanner` with
    visualization enabled.

    Args:
        args_cli: Parsed CLI args with rest/debug and planner options.

    Returns:
        Tuple of (env, robot, planner).
    """
    env_name = "Isaac-PickPlace-GR1T2-Abs-Mimic-v0"
    env_cfg = PickPlaceGR1T2MimicEnvCfg()
    env_cfg.scene.num_envs = 1

    usd_path = cast(Any, env_cfg).scene.robot.spawn.usd_path
    inactive_joint_names = list(env_cfg.actions.pink_ik_cfg.ik_urdf_fixed_joint_names) or []
    # inactive_joint_names.extend(env_cfg.actions.pink_ik_cfg.hand_joint_names)  # add finger joints here
    print(f"Inactive joint names: {inactive_joint_names}")
    robot_yaml_both = _build_temp_robot_yaml_both_arms(usd_path, inactive_joints=inactive_joint_names)

    env = gym.make(env_name, cfg=env_cfg).unwrapped
    env.reset()
    if args_cli.rest > 0:
        rest_with_idle_action(env, steps=args_cli.rest)

    cfg = CuroboPlannerCfg()
    cfg.visualize_plan = True
    cfg.visualize_spheres = False
    cfg.debug_planner = True  # args_cli.debug
    cfg.robot_config_file = robot_yaml_both
    cfg.robot_name = "gr1"
    cfg.approach_distance = 0.0
    cfg.retreat_distance = 0.0
    cfg.time_dilation_factor = 0.5
    # cfg.collision_activation_distance = 0.05
    # cfg.collision_sphere_buffer = 0.015
    cfg.enable_finetune_trajopt = True
    # cfg.ee_link_name = _tool_link_for_arm("right")  # primary for MotionGen; we'll set link_poses for both
    cfg.enable_graph = True
    cfg.enable_graph_attempt = 4
    cfg.max_planning_attempts = 10
    cfg.gripper_open_positions = {}
    cfg.gripper_closed_positions = {}

    robot = cast(Any, env).scene["robot"]
    # For single-call bimanual, keep collisions active only for the primary arm (right) and trunk
    # to avoid inter-arm self-collision over-constraints during planning
    planner = HumanoidArmCuroboPlanner(
        env=cast(Any, env),
        robot=robot,
        config=cfg,
        env_id=0,
        # active_joint_substrings=("right_",),
        # hand_link_substrings=("GR1T2_fourier_hand_6dof_right_",),
        # collision_active_link_substrings=["right_", "left_"],
    )
    return env, robot, planner


def _compute_world_base(env, robot):
    """Compute world->base transform for env_id=0, relative to env origin.

    Args:
        env: Isaac Lab environment instance.
        robot: Robot articulation instance from the scene.

    Returns:
        Tuple of (env_origin translation, T_world_base 4x4 transform).
    """
    env_origin = env.scene.env_origins[0].to(device=env.device, dtype=torch.float32)
    base_pos_world = (robot.data.root_pos_w[0] - env_origin).to(device=env.device, dtype=torch.float32)
    base_rot_world = PoseUtils.matrix_from_quat(
        robot.data.root_quat_w[0].unsqueeze(0).to(device=env.device, dtype=torch.float32)
    )[0]
    T_world_base = PoseUtils.make_pose(base_pos_world.unsqueeze(0), base_rot_world.unsqueeze(0))[0]
    return env_origin, T_world_base


def _link_pose_base(planner: CuroboPlanner, link_name: str, device) -> torch.Tensor:
    """Compute current base->link pose using cuRobo kinematics.

    Uses the planner's current joint state and kinematics to get the pose
    of a specific link in the base frame.

    Args:
        planner: Active cuRobo planner instance.
        link_name: Name of the link whose pose is requested.
        device: Torch device for the returned tensor.

    Returns:
        4x4 homogeneous transform from base to specified link.
    """
    js = planner._get_current_joint_state_for_curobo()
    kin = planner.motion_gen.kinematics
    q = (
        js.position
        if isinstance(js.position, torch.Tensor)
        else torch.tensor(js.position, dtype=planner.tensor_args.dtype, device=planner.tensor_args.device)
    )
    if q.dim() == 1:
        q = q.unsqueeze(0)
    state = kin.get_state(q)
    # Resolve link index and build 4x4 from position/quaternion
    names = list(cast(Any, state).link_names)
    idx = names.index(link_name)
    pos = cast(Any, state).links_position[..., idx, :].to(device=device, dtype=torch.float32)
    quat = cast(Any, state).links_quaternion[..., idx, :].to(device=device, dtype=torch.float32)
    rot = PoseUtils.matrix_from_quat(quat.view(1, 4))[0]
    return PoseUtils.make_pose(pos.view(1, 3), rot.unsqueeze(0))[0]


def _compute_site_mapping_for_link(env, planner, link_name: str, ctrl_site_eef_name: str):
    """Derive tool->site mapping for a given link and controller site.

    Computes the mapping between the actual tool pose (via FK) and the controller
    site pose (as exposed by the env) to translate targets between frames.

    Args:
        env: Isaac Lab environment.
        planner: cuRobo planner providing FK.
        link_name: Tool link name in cuRobo kinematics.
        ctrl_site_eef_name: Controller site identifier (e.g., "right"/"left").

    Returns:
        Tuple (env_origin, T_world_base, ctrl_site_env, T_tool_site) where
        T_tool_site maps tool-frame to controller-site frame.
    """
    device = env.device
    env_origin, T_world_base = _compute_world_base(env, planner.robot)
    # controller site pose (world/env-origin) for the arm
    ctrl_site_env = env.get_robot_eef_pose(ctrl_site_eef_name)[0].to(device=device, dtype=torch.float32)
    # current tool pose base->tool via cuRobo FK
    T_base_tool_now = _link_pose_base(planner, link_name, device)
    # world->tool now
    T_world_tool_now = (T_world_base @ T_base_tool_now).clone()
    # mapping tool->site: T_T_S
    T_tool_site = torch.linalg.solve(T_world_tool_now, ctrl_site_env)
    return env_origin, T_world_base, ctrl_site_env, T_tool_site


def _build_site_goal_independent(ctrl_site_env_r, ctrl_site_env_l, args_cli, device):
    """Create independent per-arm site/world goals based on CLI deltas.

    Args:
        ctrl_site_env_r: Current right controller-site pose in world/env origin frame.
        ctrl_site_env_l: Current left controller-site pose in world/env origin frame.
        args_cli: Parsed CLI args containing optional per-arm dx/dy/dz overrides.
        device: Torch device for computations (unused, kept for signature stability).

    Returns:
        Tuple (goal_r, goal_l) of 4x4 target poses for right and left sites.
    """

    def pick(val_specific, val_global, default):
        return (
            float(val_specific)
            if val_specific is not None
            else (float(val_global) if val_global is not None else default)
        )

    rdx = pick(args_cli.right_dx, args_cli.dx, 0.0)
    rdy = pick(args_cli.right_dy, args_cli.dy, 0.0)
    rdz = pick(args_cli.right_dz, args_cli.dz, 0.0)
    ldx = pick(args_cli.left_dx, args_cli.dx, 0.0)
    ldy = pick(args_cli.left_dy, args_cli.dy, 0.0)
    ldz = pick(args_cli.left_dz, args_cli.dz, 0.0)

    goal_r = ctrl_site_env_r.clone()
    goal_l = ctrl_site_env_l.clone()
    goal_r[0, 3] = goal_r[0, 3] + rdx
    goal_r[1, 3] = goal_r[1, 3] + rdy
    goal_r[2, 3] = goal_r[2, 3] + rdz
    goal_l[0, 3] = goal_l[0, 3] + ldx
    goal_l[1, 3] = goal_l[1, 3] + ldy
    goal_l[2, 3] = goal_l[2, 3] + ldz
    print(
        f"[SingleCall] Independent goals | R dpos=({rdx:.3f},{rdy:.3f},{rdz:.3f}) m | L"
        f" dpos=({ldx:.3f},{ldy:.3f},{ldz:.3f}) m"
    )
    return goal_r, goal_l


def _build_site_goal_bimanual(ctrl_site_env_r, ctrl_site_env_l, args_cli, device):
    """Create symmetric bimanual site/world goals around a target midpoint.

    Maintains a specified horizontal gap while shifting forward/upward from the
    current midpoint between the sites.

    Args:
        ctrl_site_env_r: Current right controller-site pose in world/env origin frame.
        ctrl_site_env_l: Current left controller-site pose in world/env origin frame.
        args_cli: Parsed CLI args providing dx (gap/2), dy (forward), dz (up).
        device: Torch device for computations (unused, kept for signature stability).

    Returns:
        Tuple (goal_r, goal_l) of 4x4 target poses for right and left sites.
    """
    goal_r = ctrl_site_env_r.clone()
    goal_l = ctrl_site_env_l.clone()
    cur_mid = 0.5 * (ctrl_site_env_r[:3, 3] + ctrl_site_env_l[:3, 3])
    half_gap = float(args_cli.dx)
    forward = float(args_cli.dy)
    upward = float(args_cli.dz)
    target_mid = cur_mid.clone()
    target_mid[1] = target_mid[1] + forward
    target_mid[2] = target_mid[2] + upward
    goal_r[:3, 3] = target_mid
    goal_l[:3, 3] = target_mid
    goal_r[0, 3] = goal_r[0, 3] + half_gap
    goal_l[0, 3] = goal_l[0, 3] - half_gap
    print(
        f"[SingleCall] Bimanual goals | mid={target_mid.cpu().numpy()} | gap={2*half_gap:.3f}m, forward={forward:.3f}m,"
        f"up={upward:.3f}m"
    )
    return goal_r, goal_l


def _build_site_goals(ctrl_site_env_r, ctrl_site_env_l, args_cli, device):
    """Build per-arm site/world goals based on CLI args.

    Dispatches to independent or bimanual goal builders without changing behavior.

    Args:
        ctrl_site_env_r: Current right controller-site pose (world/env origin frame).
        ctrl_site_env_l: Current left controller-site pose (world/env origin frame).
        args_cli: Parsed CLI args with optional per-arm deltas.
        device: Torch device for computations.

    Returns:
        Tuple (goal_env_site_r, goal_env_site_l) of site/world goals.
    """
    use_independent = any(
        x is not None
        for x in (
            args_cli.right_dx,
            args_cli.right_dy,
            args_cli.right_dz,
            args_cli.left_dx,
            args_cli.left_dy,
            args_cli.left_dz,
        )
    )
    if use_independent:
        return _build_site_goal_independent(ctrl_site_env_r, ctrl_site_env_l, args_cli, device)
    return _build_site_goal_bimanual(ctrl_site_env_r, ctrl_site_env_l, args_cli, device)


def _compute_targets_for_planning(T_world_base, goal_env_site_r, goal_env_site_l, T_tool_site_r, T_tool_site_l):
    """Convert site/world goals to world/tool and base/tool frames required for planning.

    Args:
        T_world_base: 4x4 world->base transform.
        goal_env_site_r: Right target in site/world frame.
        goal_env_site_l: Left target in site/world frame.
        T_tool_site_r: Tool->site mapping for right.
        T_tool_site_l: Tool->site mapping for left.

    Returns:
        Tuple (T_world_tool_goal_r, T_world_tool_goal_l, T_base_world, T_base_tool_goal_l).
    """
    T_site_inv_r = torch.linalg.inv(T_tool_site_r)
    T_site_inv_l = torch.linalg.inv(T_tool_site_l)
    T_world_tool_goal_r = (goal_env_site_r @ T_site_inv_r).clone()
    T_world_tool_goal_l = (goal_env_site_l @ T_site_inv_l).clone()
    T_base_world = torch.linalg.inv(T_world_base)
    T_base_tool_goal_l = (T_base_world @ T_world_tool_goal_l).clone()
    return T_world_tool_goal_r, T_world_tool_goal_l, T_base_world, T_base_tool_goal_l


def _get_step_size_from_args(args_cli):
    """Return retiming step size in radians if requested via CLI, else None.

    Args:
        args_cli: Parsed CLI args with `retime_deg` degrees-per-step value.

    Returns:
        Step size in radians, or None if retiming not requested.
    """
    return np.deg2rad(args_cli.retime_deg) if args_cli.retime_deg > 0 else None


def _attempt_plan_with_debug(
    planner,
    link_r,
    link_l,
    T_world_tool_goal_r,
    T_world_tool_goal_l,
    T_base_world,
    T_base_tool_goal_l,
    step_size,
):
    """Perform diagnostics and attempt planning in a guarded block.

    Prints the same debug lines as the original inline flow, then calls the
    planner's high-level planning method. Exceptions are caught and printed to
    preserve identical behavior.

    Args:
        planner: Humanoid cuRobo planner instance.
        link_r: Right tool link name.
        link_l: Left tool link name.
        T_world_tool_goal_r: World->tool target for right.
        T_world_tool_goal_l: World->tool target for left (for checks only).
        T_base_world: Base<-world transform.
        T_base_tool_goal_l: Base->tool target for left, used as link constraint.
        step_size: Optional step size for retiming.

    Returns:
        Tuple (ok, had_exception) where ok indicates planning success and
        had_exception indicates whether an exception was thrown.
    """
    try:
        print("[SingleCall] Planning bimanual in one call via planner...")
        # Mirror demo: use a single primary goal (right tool) and constrain only the other arm's tool via link_poses
        # Keep frames: base->tool for link_poses; world->tool for primary target
        link_targets = {link_l: T_base_tool_goal_l}

        # --- Diagnostics to validate frames and link names before planning ---
        kin_names = {str(n) for n in planner.motion_gen.kinematics.link_names}
        print(f"[SingleCall][Debug] ee_link_name (planner): {planner.config.ee_link_name}")
        print(f"[SingleCall][Debug] link_targets keys: {list(link_targets.keys())}")
        for ln in [link_r, link_l]:
            print(f"[SingleCall][Debug] link '{ln}' in kinematics: {ln in kin_names}")

        ok = planner.update_world_and_plan_motion(
            target_pose=T_world_tool_goal_r,
            step_size=step_size,
            enable_retiming=step_size is not None,
            link_target_poses_base=link_targets,
        )
        return ok, False
    except Exception as e:
        traceback.print_exc()
        print(f"[SingleCall] Planning error: {e}")
        return False, True


def _normalize_plan_joint_order(planner, plan_js):
    """Expand to full joint state and align to environment joint ordering subset.

    Args:
        planner: cuRobo planner instance.
        plan_js: Planned `JointState` from cuRobo.

    Returns:
        `JointState` aligned to the environment's joint name subset.
    """
    plan_js = planner.motion_gen.get_full_js(plan_js)
    names_list = plan_js.joint_names or []
    common_js_names = [x for x in planner.robot.data.joint_names if x in names_list]
    return plan_js.get_ordered_joint_state(common_js_names)


def _retime_plan_if_needed(planner, plan_js, step_size):
    """Optionally apply linear retiming to the plan using the planner helper.

    Args:
        planner: cuRobo planner instance.
        plan_js: Planned `JointState` to potentially retime.
        step_size: Step size to apply; if None, returns plan unchanged.

    Returns:
        The retimed `JointState` if applicable, otherwise the original plan.
    """
    if step_size is None:
        return plan_js
    tmp = planner._linearly_retime_plan(step_size=step_size, plan=plan_js)
    if tmp is not None:
        return tmp
    return plan_js


def _post_plan_diagnostics(
    planner,
    plan_js,
    link_r,
    link_l,
    T_world_base,
    T_tool_site_r,
    goal_env_site_r,
    T_base_tool_goal_l,
):
    """Emit diagnostics comparing final planned poses against goals.

    Computes FK at the final waypoint and reports pose/rotation errors for:
    - Right arm in site/world frame
    - Left arm in base/tool frame

    Args:
        planner: cuRobo planner instance.
        plan_js: Final (aligned/retimed) `JointState` plan.
        link_r: Right tool link name.
        link_l: Left tool link name.
        T_world_base: World->base transform.
        T_tool_site_r: Tool->site mapping for right.
        goal_env_site_r: Right target in site/world frame.
        T_base_tool_goal_l: Left target in base/tool frame.
    """
    # Reorder to cuRobo kinematics joint order for FK diagnostics
    cu_order = list(planner.motion_gen.kinematics.joint_names)
    cu_plan = plan_js.get_ordered_joint_state(cu_order)
    q_last_val = cu_plan.position[-1]
    q_last = (
        q_last_val
        if isinstance(q_last_val, torch.Tensor)
        else torch.tensor(q_last_val, dtype=planner.tensor_args.dtype, device=planner.tensor_args.device)
    )
    if q_last.dim() == 1:
        q_last = q_last.unsqueeze(0)
    state_last = planner.motion_gen.kinematics.get_state(q_last)
    names_last = list(cast(Any, state_last).link_names)
    idx_r = names_last.index(link_r)
    idx_l = names_last.index(link_l)

    pos_r = cast(Any, state_last).links_position[..., idx_r, :]
    quat_r = cast(Any, state_last).links_quaternion[..., idx_r, :]
    R_r = PoseUtils.matrix_from_quat(quat_r.view(1, 4))[0]
    pose_r_bt = PoseUtils.make_pose(pos_r.view(1, 3), R_r.unsqueeze(0))[0]

    pos_l = cast(Any, state_last).links_position[..., idx_l, :]
    quat_l = cast(Any, state_last).links_quaternion[..., idx_l, :]
    R_l = PoseUtils.matrix_from_quat(quat_l.view(1, 4))[0]
    pose_l_bt = PoseUtils.make_pose(pos_l.view(1, 3), R_l.unsqueeze(0))[0]

    T_world_tool_r_final = (T_world_base @ pose_r_bt).clone()
    # Compare in site/world for right
    T_world_site_r_final = (T_world_tool_r_final @ T_tool_site_r).clone()
    pos_err_r = torch.linalg.vector_norm(T_world_site_r_final[:3, 3] - goal_env_site_r[:3, 3]).item()
    rot_err_mat_r = T_world_site_r_final[:3, :3].T @ goal_env_site_r[:3, :3]
    rot_err_trace_r = torch.clamp((torch.trace(rot_err_mat_r) - 1.0) / 2.0, -1.0, 1.0)
    rot_err_r = torch.acos(rot_err_trace_r).item()
    print(
        f"[SingleCall][PlanDiag] RIGHT final vs goal (site) | pos_err={pos_err_r:.4e} m | rot_err={rot_err_r:.4e} rad"
    )

    # Compare base/tool for left
    pos_err_l = torch.linalg.vector_norm(pose_l_bt[:3, 3] - T_base_tool_goal_l[:3, 3]).item()
    rot_err_mat_l = pose_l_bt[:3, :3].T @ T_base_tool_goal_l[:3, :3]
    rot_err_trace_l = torch.clamp((torch.trace(rot_err_mat_l) - 1.0) / 2.0, -1.0, 1.0)
    rot_err_l = torch.acos(rot_err_trace_l).item()
    print(f"[SingleCall][PlanDiag] LEFT final vs goal (base) | pos_err={pos_err_l:.4e} m | rot_err={rot_err_l:.4e} rad")
    print(f"[SingleCall] Planned waypoints: {len(plan_js.position)}")


def _visualize_goals(args_cli, env, goal_r_env_site, goal_l_env_site):
    """Optionally render goal pose frames for left/right targets.

    Args:
        args_cli: CLI args containing `visualize_goal` flag.
        env: Isaac Lab environment.
        goal_r_env_site: Right target in site/world frame.
        goal_l_env_site: Left target in site/world frame.

    Returns:
        Tuple of optional `VisualizationMarkers` instances for right and left goals.
    """
    if not args_cli.visualize_goal:
        return None, None
    try:
        viz_goal_r = "/World/Visuals/goal_pose_marker_right_single"
        viz_goal_l = "/World/Visuals/goal_pose_marker_left_single"
        frame_goal_r = dc_replace(FRAME_MARKER_CFG, prim_path=viz_goal_r)
        frame_goal_l = dc_replace(FRAME_MARKER_CFG, prim_path=viz_goal_l)
        cast(Any, frame_goal_r).markers["frame"].scale = (0.1, 0.1, 0.1)
        cast(Any, frame_goal_l).markers["frame"].scale = (0.1, 0.1, 0.1)
        goal_vis_r = VisualizationMarkers(frame_goal_r)
        goal_vis_l = VisualizationMarkers(frame_goal_l)

        def as_quat(T):
            return PoseUtils.quat_from_matrix(T[:3, :3].unsqueeze(0))[0].detach().to(dtype=torch.float32)

        pos_r, quat_r = goal_r_env_site[:3, 3].float(), as_quat(goal_r_env_site)
        pos_l, quat_l = goal_l_env_site[:3, 3].float(), as_quat(goal_l_env_site)
        goal_vis_r.visualize(translations=pos_r.unsqueeze(0), orientations=quat_r.unsqueeze(0))
        goal_vis_l.visualize(translations=pos_l.unsqueeze(0), orientations=quat_l.unsqueeze(0))
        return goal_vis_r, goal_vis_l
    except Exception as e:
        print(f"[SingleCall] Goal visualization failed: {e}")
        return None, None


def _execute_bimanual_plan(env, robot, planner: CuroboPlanner, T_tool_site_r, T_tool_site_l, env_origin, plan_js):
    """Replay a joint-space plan while commanding target site poses for both arms.

    For each waypoint, computes tool poses via FK, maps them into controller site
    frames, constructs action tensors (including per-finger commands), and steps
    the environment. Includes periodic inversion diagnostics to verify controller
    mappings.

    Args:
        env: Isaac Lab environment instance.
        robot: Robot articulation used for base pose.
        planner: cuRobo planner instance.
        T_tool_site_r: Tool->site mapping for right.
        T_tool_site_l: Tool->site mapping for left.
        env_origin: Environment origin translation for env_id=0.
        plan_js: Planned trajectory as `JointState`.
    """
    idle = env.cfg.idle_action
    if not isinstance(idle, torch.Tensor):
        idle = torch.tensor(idle, dtype=torch.float32)
    else:
        idle = idle.to(dtype=torch.float32)
    idle = idle.to(device=env.device)

    kin = planner.motion_gen.kinematics
    link_r = _tool_link_for_arm("right")
    link_l = _tool_link_for_arm("left")

    # Precompute hand joint names and index map in cuRobo order (same as plan_exec)
    hand_names = env.cfg.actions.pink_ik_cfg.hand_joint_names
    left_hand_names = [n for n in hand_names if n.startswith("L_")]
    right_hand_names = [n for n in hand_names if n.startswith("R_")]
    cu_order = list(planner.motion_gen.kinematics.joint_names)
    name_to_cu_idx = {n: i for i, n in enumerate(cu_order)}
    # Idle fallbacks for any missing joints in plan
    idle_left = idle[14:25].to(device=env.device, dtype=torch.float32)
    idle_right = idle[25:36].to(device=env.device, dtype=torch.float32)

    for trial in range(args_cli.replay_trials if hasattr(args_cli, "replay_trials") else 1):
        print(f"[SingleCall] Replaying trial {trial + 1}")
        env.reset()
        # Recompute tool->site mapping after reset to avoid stale mapping
        try:
            _, _, ctrl_site_env_r_new, T_tool_site_r_new = _compute_site_mapping_for_link(env, planner, link_r, "right")
            _, _, ctrl_site_env_l_new, T_tool_site_l_new = _compute_site_mapping_for_link(env, planner, link_l, "left")
            T_tool_site_r_exec = T_tool_site_r_new
            T_tool_site_l_exec = T_tool_site_l_new
        except Exception:
            T_tool_site_r_exec = T_tool_site_r
            T_tool_site_l_exec = T_tool_site_l
        # Reorder plan to cuRobo kinematic order for FK-consistent playback and gripper sampling
        plan_exec = plan_js.get_ordered_joint_state(cu_order)
        for k in range(len(plan_js.position)):
            base_pos_world = (robot.data.root_pos_w[0] - env_origin).to(device=env.device, dtype=torch.float32)
            base_rot_world = PoseUtils.matrix_from_quat(
                robot.data.root_quat_w[0].unsqueeze(0).to(device=env.device, dtype=torch.float32)
            )[0]
            T_world_base = PoseUtils.make_pose(base_pos_world.unsqueeze(0), base_rot_world.unsqueeze(0))[0]

            js_k = plan_exec[k]
            qk = (
                js_k.position
                if isinstance(js_k.position, torch.Tensor)
                else torch.tensor(js_k.position, dtype=planner.tensor_args.dtype, device=planner.tensor_args.device)
            )
            if qk.dim() == 1:
                qk = qk.unsqueeze(0)
            state = kin.get_state(qk)
            # base->tool for both links using links_position/links_quaternion
            names = list(cast(Any, state).link_names)
            idx_r = names.index(link_r)
            idx_l = names.index(link_l)
            pos_r = cast(Any, state).links_position[..., idx_r, :]
            quat_r = cast(Any, state).links_quaternion[..., idx_r, :]
            pos_l = cast(Any, state).links_position[..., idx_l, :]
            quat_l = cast(Any, state).links_quaternion[..., idx_l, :]
            R_r = PoseUtils.matrix_from_quat(quat_r.view(1, 4))[0]
            R_l = PoseUtils.matrix_from_quat(quat_l.view(1, 4))[0]
            pose_r_bt = PoseUtils.make_pose(pos_r.view(1, 3), R_r.unsqueeze(0))[0]
            pose_l_bt = PoseUtils.make_pose(pos_l.view(1, 3), R_l.unsqueeze(0))[0]

            # world->tool
            T_world_tool_r = (T_world_base @ pose_r_bt).clone()
            T_world_tool_l = (T_world_base @ pose_l_bt).clone()
            # site/world (use recomputed mapping)
            T_site_r = (T_world_tool_r @ T_tool_site_r_exec).clone()
            T_site_l = (T_world_tool_l @ T_tool_site_l_exec).clone()
            T_site_r[3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device, dtype=torch.float32)
            T_site_l[3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device, dtype=torch.float32)

            # Build per-step gripper commands from the plan in cuRobo order (fallback to idle per joint if absent)
            js_k = plan_exec[k]
            pos_vec = (
                js_k.position
                if isinstance(js_k.position, torch.Tensor)
                else torch.tensor(js_k.position, dtype=planner.tensor_args.dtype, device=planner.tensor_args.device)
            )
            pos_vec = pos_vec.to(device=env.device, dtype=torch.float32)
            if pos_vec.dim() == 2:
                pos_vec = pos_vec.squeeze(0)

            gr_left = torch.stack([
                pos_vec[name_to_cu_idx[n]] if n in name_to_cu_idx else idle_left[i]
                for i, n in enumerate(left_hand_names)
            ])
            gr_right = torch.stack([
                pos_vec[name_to_cu_idx[n]] if n in name_to_cu_idx else idle_right[i]
                for i, n in enumerate(right_hand_names)
            ])

            action = env.target_eef_pose_to_action(
                target_eef_pose_dict={"left": T_site_l, "right": T_site_r},
                gripper_action_dict={"left": gr_left, "right": gr_right},
                action_noise_dict=None,
                env_id=0,
            )
            if action.ndim == 1:
                action = action.unsqueeze(0)
            # Action inversion diagnostics to ensure controller mapping matches target site frames
            try:
                inferred_targets = env.action_to_target_eef_pose(action)
                inf_r = inferred_targets["right"][0].to(device=env.device)
                inf_l = inferred_targets["left"][0].to(device=env.device)
                re_r_pos = torch.linalg.vector_norm(inf_r[:3, 3] - T_site_r[:3, 3]).item()
                re_r_rot_m = inf_r[:3, :3].T @ T_site_r[:3, :3]
                re_r_rot = torch.acos(torch.clamp((torch.trace(re_r_rot_m) - 1.0) / 2.0, -1.0, 1.0)).item()
                re_l_pos = torch.linalg.vector_norm(inf_l[:3, 3] - T_site_l[:3, 3]).item()
                re_l_rot_m = inf_l[:3, :3].T @ T_site_l[:3, :3]
                re_l_rot = torch.acos(torch.clamp((torch.trace(re_l_rot_m) - 1.0) / 2.0, -1.0, 1.0)).item()
                if (k == 0) or ((k + 1) % 25 == 0) or (k == len(plan_js.position) - 1):
                    print(
                        f"[SingleCall][ExecDiag] inversion | R pos={re_r_pos:.3e} rot={re_r_rot:.3e} | L"
                        f" pos={re_l_pos:.3e} rot={re_l_rot:.3e}"
                    )
            except Exception as e:
                print(f"[SingleCall][ExecDiag] inversion failed: {e}")
            env.step(action)


def main():
    """Entry point to plan and execute a single-call bimanual motion demo.

    Workflow:
    >> Build env and planner, seed RNGs for reproducibility
    >> Compute current tool->site mappings
    >> Build site/world goals from CLI args and visualize if requested
    >> Convert targets to required frames and run planning with diagnostics
    >> Emit post-plan diagnostics, then execute the plan
    """
    # import pdb; pdb.set_trace()
    np.random.seed(42)
    torch.manual_seed(42)

    env, robot, planner = _build_env_and_planner_bimanual_single(args_cli)

    device = cast(Any, env).device
    env_origin, T_world_base = _compute_world_base(env, robot)

    # Compute current tool->site mappings
    link_r = _tool_link_for_arm("right")
    link_l = _tool_link_for_arm("left")
    _, _, ctrl_site_env_r, T_tool_site_r = _compute_site_mapping_for_link(env, planner, link_r, "right")
    _, _, ctrl_site_env_l, T_tool_site_l = _compute_site_mapping_for_link(env, planner, link_l, "left")

    # Build target goals in site/world
    goal_env_site_r, goal_env_site_l = _build_site_goals(ctrl_site_env_r, ctrl_site_env_l, args_cli, device)

    _visualize_goals(args_cli, env, goal_env_site_r, goal_env_site_l)

    # Convert site/world -> world/tool -> base/tool targets
    (
        T_world_tool_goal_r,
        T_world_tool_goal_l,
        T_base_world,
        T_base_tool_goal_l,
    ) = _compute_targets_for_planning(T_world_base, goal_env_site_r, goal_env_site_l, T_tool_site_r, T_tool_site_l)

    # Plan once via humanoid wrapper; pass world-frame target (wrapper converts to base)
    step_size = _get_step_size_from_args(args_cli)

    # --- Diagnostics identical to original pre-plan checks ---
    T_world_site_goal_r_calc = (T_world_tool_goal_r @ T_tool_site_r).clone()
    pos_err_r = torch.linalg.vector_norm(T_world_site_goal_r_calc[:3, 3] - goal_env_site_r[:3, 3]).item()
    rot_err_mat_r = T_world_site_goal_r_calc[:3, :3].T @ goal_env_site_r[:3, :3]
    rot_err_trace_r = torch.clamp((torch.trace(rot_err_mat_r) - 1.0) / 2.0, -1.0, 1.0)
    rot_err_r = torch.acos(rot_err_trace_r).item()
    print(f"[SingleCall][Debug] RIGHT site frame check | pos_err={pos_err_r:.4e} m | rot_err={rot_err_r:.4e} rad")

    T_base_tool_goal_l_from_world = (T_base_world @ T_world_tool_goal_l).clone()
    pos_err_l = torch.linalg.vector_norm(T_base_tool_goal_l_from_world[:3, 3] - T_base_tool_goal_l[:3, 3]).item()
    rot_err_mat_l = T_base_tool_goal_l_from_world[:3, :3].T @ T_base_tool_goal_l[:3, :3]
    rot_err_trace_l = torch.clamp((torch.trace(rot_err_mat_l) - 1.0) / 2.0, -1.0, 1.0)
    rot_err_l = torch.acos(rot_err_trace_l).item()
    print(f"[SingleCall][Debug] LEFT base frame check | pos_err={pos_err_l:.4e} m | rot_err={rot_err_l:.4e} rad")

    ok, had_exception = _attempt_plan_with_debug(
        planner,
        link_r,
        link_l,
        T_world_tool_goal_r,
        T_world_tool_goal_l,
        T_base_world,
        T_base_tool_goal_l,
        step_size,
    )
    if had_exception:
        return

    if not ok or planner.current_plan is None:
        print("[SingleCall] Planning failed: no plan")
        return

    plan_js = _normalize_plan_joint_order(planner, planner.current_plan)

    # naive linear retime on joint path (if requested)
    plan_js = _retime_plan_if_needed(planner, plan_js, step_size)

    # --- Post-plan diagnostics: does the last planned pose match the goals? ---
    _post_plan_diagnostics(
        planner,
        plan_js,
        link_r,
        link_l,
        T_world_base,
        T_tool_site_r,
        goal_env_site_r,
        T_base_tool_goal_l,
    )

    # Execute the plan
    _execute_bimanual_plan(env, robot, planner, T_tool_site_r, T_tool_site_l, env_origin, plan_js)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    finally:
        simulation_app.close()
