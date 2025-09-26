#!/usr/bin/env python3
# launch the simulator

# import pinnochio
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Plan and execute a humanoid arm lift with cuRobo.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--enable_pinocchio", action="store_true", default=False)

parser.add_argument("--arm", type=str, default="right", choices=["right", "left"])
parser.add_argument("--dz", type=float, default=0.05, help="Upward lift in meters.")
parser.add_argument("--retime_deg", type=float, default=1.0, help="Joint retime step (deg); 0 disables retiming.")


# append AppLauncher cli args and parse
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import tempfile
import json
import numpy as np
import torch
import gymnasium as gym
import yaml

import isaaclab_tasks  # noqa: F401
import isaaclab_mimic.envs  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg import PickPlaceGR1T2EnvCfg

# Register pinocchio envs if requested
if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401

# from isaaclab_mimic.datagen.generation import setup_env_config
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
from isaaclab_mimic.motion_planners.curobo.curobo_planner_humanoid import HumanoidArmCuroboPlanner
from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
import isaaclab.utils.math as PoseUtils

# Controller utils to convert USD->URDF
from isaaclab.controllers import utils as ControllerUtils


def _detect_robot_usd_path(env):
    try:
        usd_path = env.cfg.scene.robot.spawn.usd_path
        if isinstance(usd_path, str) and len(usd_path) > 0:
            return usd_path
    except Exception:
        pass
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    return f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd"


def _tool_link_for_arm(arm: str) -> str:
    # Matches the converted GR1T2 URDF link names
    return f"GR1T2_fourier_hand_6dof_{arm}_hand_pitch_link"

def to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj


def _build_temp_robot_yaml_from_usd(usd_path: str, arm: str) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="gr1_curobo_")
    print(f"[PlanHumanoid] Converting USD to URDF...")
    urdf_path, _ = ControllerUtils.convert_usd_to_urdf(usd_path, tmp_dir, force_conversion=True)
    print(f"[PlanHumanoid] URDF: {urdf_path}")
    robot_cfg_dict = None
    print(f"[PlanHumanoid] Creating robot config...")
    try:
        from nvplan.applications.custream.config import create_robot_config
        from nvplan.applications.custream.spheres import load_spheres
    except Exception as e:
        print(f"[PlanHumanoid] Error importing custream: {e}")
        raise e

    print(f"[PlanHumanoid] Loading spheres...")
    tool_links = [_tool_link_for_arm(arm)]
    try:
        robot_config = create_robot_config(
            urdf_path,
            tool_links=tool_links,
            inactive_joints=[],
            depth=2,
            verbose=False,
        )
    except Exception as e:
        # print traceback
        import traceback
        traceback.print_exc()
        print(f"[PlanHumanoid] Error creating robot config: {e}")
        raise e

    print(f"[PlanHumanoid] Robot config: {robot_config}")
    max_spheres = 225 - len(tool_links) * 50
    load_spheres(robot_config, max_spheres=max_spheres, max_link_spheres=int(1e9))
    robot_cfg_dict = robot_config["robot_cfg"]
    robot_cfg_yaml = to_python(robot_cfg_dict)

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

    # Ensure ee_link points to the selected arm’s tool link
    if "kinematics" in robot_cfg_yaml:
        robot_cfg_yaml["kinematics"]["ee_link"] = _tool_link_for_arm(arm)

    out_dir = tempfile.mkdtemp(prefix="curobo_robot_cfg_")
    out_path = os.path.join(out_dir, "gr1_generated.yml")
    print(f"[PlanHumanoid] Writing robot YAML to {out_path} ...")
    with open(out_path, "w") as f:
        yaml.safe_dump({"robot_cfg": robot_cfg_yaml}, f, sort_keys=False)
    print("[PlanHumanoid] Robot YAML written.")
    return out_path


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # load env
    env_name = "Isaac-PickPlace-GR1T2-Abs-v0"  # get_env_name_from_dataset(args_cli.input_file)
    print(f"[PlanHumanoid] Env: {env_name}")

    print("[PlanHumanoid] Building env config...")
    # env_cfg, _ = setup_env_config(
    #     env_name=env_name,
    #     output_dir=".",
    #     output_file_name="tmp.hdf5",
    #     num_envs=args_cli.num_envs,
    #     device=args_cli.device,
    #     generation_num_trials=1,
    # )
    env_cfg = PickPlaceGR1T2EnvCfg()
    print("[PlanHumanoid] Creating env...")
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    env.reset()
    print("[PlanHumanoid] Env ready.")

    # build planner config
    ###### TODO: #1 Correctly configure planner config ######
    planner_cfg = CuroboPlannerCfg()
    planner_cfg.visualize_plan = True
    planner_cfg.visualize_spheres = True
    print(f"[PlanHumanoid] Planner config: {planner_cfg}")

    if planner_cfg.robot_name.lower() != "gr1":
        usd_path = _detect_robot_usd_path(env)
        print(f"[PlanHumanoid] Detected robot USD: {usd_path}")
        robot_yaml = _build_temp_robot_yaml_from_usd(usd_path, arm=args_cli.arm)
        print(f"[PlanHumanoid] Generated cuRobo robot YAML: {robot_yaml}")
        planner_cfg.robot_config_file = robot_yaml
        planner_cfg.robot_name = "gr1"
        planner_cfg.approach_distance = 0.0
        planner_cfg.retreat_distance = 0.0
        planner_cfg.time_dilation_factor = 0.5
        planner_cfg.enable_finetune_trajopt = True
        planner_cfg.debug_planner = True
        planner_cfg.ee_link_name = _tool_link_for_arm(args_cli.arm)
    if args_cli.arm == "right":
        active_joint_substrings = ("right_",)
        hand_link_substrings = ("GR1T2_fourier_hand_6dof_right_",)
    else:
        active_joint_substrings = ("left_",)
        hand_link_substrings = ("GR1T2_fourier_hand_6dof_left_",)

    print("[PlanHumanoid] Creating planner...")

    ###### TODO: #2 Correctly use planner for the humanoid arm ######
    try:
        planner = HumanoidArmCuroboPlanner(
            env=env,
            robot=env.scene["robot"],
            config=planner_cfg,
            env_id=0,
            active_joint_substrings=active_joint_substrings,
            hand_link_substrings=hand_link_substrings,
        )
    except Exception as e:
        # print traceback
        import traceback
        traceback.print_exc()
        print(f"[PlanHumanoid] Error creating planner: {e}")
        raise e
    print("[PlanHumanoid] Planner ready.")

    cu_js = planner._get_current_joint_state_for_curobo()
    ee_pose_cu = planner.get_ee_pose(cu_js)
    pos = planner._to_env_device(ee_pose_cu.position)
    rot = planner._to_env_device(ee_pose_cu.get_rotation())
    current_pose: torch.Tensor = PoseUtils.make_pose(pos, rot)[0]

    target_pose = current_pose.clone()
    target_pose[2, 3] = target_pose[2, 3] + float(args_cli.dz)

    step_size = np.deg2rad(args_cli.retime_deg) if args_cli.retime_deg > 0 else None
    print("[PlanHumanoid] Planning...")
    try:
        ok = planner.update_world_and_plan_motion(
            target_pose=target_pose,
            expected_attached_object=None,
            env_id=0,
            step_size=step_size,
            enable_retiming=step_size is not None,
        )
    except Exception as e:
        # print traceback
        import traceback
        traceback.print_exc()
        print(f"[PlanHumanoid] Error planning: {e}")
        raise e
    print(f"[PlanHumanoid] Plan success: {ok}")
    if not ok:
        print("Planning failed.")
        import traceback
        traceback.print_exc()
        return

    planned_poses = planner.get_planned_poses()
    print(f"[PlanHumanoid] Executing {len(planned_poses)} waypoints...")
    for idx, pose in enumerate(planned_poses):
        action, _ = env.target_eef_pose_to_action(target_pose=pose, env_id=0, action_noise_dict=None)
        env.step(action)
        if (idx + 1) % 10 == 0:
            print(f"[PlanHumanoid] Step {idx + 1}/{len(planned_poses)}")

    print("[PlanHumanoid] Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    finally:
        simulation_app.close()
