# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bimanual DROID three-cube stacking environment."""

import math

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.droid import DROID_FRANKA_ROBOTIQ_CFG

TABLE_POSITION = (0.5, 0.0, 0.0)
TABLE_ROTATION = (0.707, 0.0, 0.0, 0.707)
TABLE_TOP_Z = 0.0
CUBE_SIZE = 0.0468
CUBE_INITIAL_Z = TABLE_TOP_Z + CUBE_SIZE / 2.0 + 0.005
STACK_XY_TOLERANCE = 0.04
STACK_Z_TOLERANCE = 0.025
STACK_TILT_TOLERANCE = 0.5


def _make_droid_cfg(prim_path: str, y_position: float) -> ArticulationCfg:
    cfg = DROID_FRANKA_ROBOTIQ_CFG.replace(prim_path=prim_path)
    cfg.init_state.pos = (0.0, y_position, 0.0)
    return cfg


def _make_external_camera(prim_path: str, position: tuple, rotation: tuple) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        update_latest_camera_pose=True,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.1,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
            clipping_range=(0.05, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=position, rot=rotation, convention="opengl"),
    )


def _make_wrist_camera(prim_path: str) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        update_latest_camera_pose=True,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.8,
            focus_distance=28.0,
            horizontal_aperture=5.376,
            vertical_aperture=3.024,
            clipping_range=(0.02, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.011, -0.031, -0.074),
            rot=(-0.420, 0.570, 0.576, -0.409),
            convention="opengl",
        ),
    )


_CUBE_RIGID_PROPERTIES = sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=100,
    solver_velocity_iteration_count=20,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=0.5,
)


def _make_cube(
    name: str,
    position: tuple[float, float, float],
    color: tuple[float, float, float],
    mass: float,
    footprint_scale: float = 1.0,
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE * footprint_scale, CUBE_SIZE * footprint_scale, CUBE_SIZE),
            rigid_props=_CUBE_RIGID_PROPERTIES,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


@configclass
class DroidBimanualCubeStackSceneCfg(InteractiveSceneCfg):
    """Two DROID arms, a shared table, four calibrated cameras, and three cubes."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POSITION, rot=TABLE_ROTATION),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
            scale=(1.6, 1.0, 1.0),
            semantic_tags=[("class", "table")],
        ),
    )

    left_arm = _make_droid_cfg("{ENV_REGEX_NS}/LeftArm", 0.4)
    right_arm = _make_droid_cfg("{ENV_REGEX_NS}/RightArm", -0.4)

    cube_0 = _make_cube(
        "cube_0",
        (TABLE_POSITION[0], 0.0, CUBE_INITIAL_Z),
        (0.2, 0.8, 0.2),
        mass=0.5,
        footprint_scale=1.5,
    )
    cube_1 = _make_cube(
        "cube_1",
        (TABLE_POSITION[0], 0.2, CUBE_INITIAL_Z),
        (0.2, 0.4, 1.0),
        mass=0.1,
    )
    cube_2 = _make_cube(
        "cube_2",
        (TABLE_POSITION[0], -0.2, CUBE_INITIAL_Z),
        (1.0, 0.2, 0.2),
        mass=0.1,
    )

    # These poses and intrinsics are the corrected wxyz DROID calibrations.
    external_camera = _make_external_camera(
        "{ENV_REGEX_NS}/external_camera",
        (0.05, 0.57, 0.66),
        (-0.393, -0.195, 0.399, 0.805),
    )
    external_camera_2 = _make_external_camera(
        "{ENV_REGEX_NS}/external_camera_2",
        (0.05, -0.57, 0.66),
        (0.805, 0.399, -0.195, -0.393),
    )
    left_wrist_camera = _make_wrist_camera(
        "{ENV_REGEX_NS}/LeftArm/Gripper/Robotiq_2F_85/base_link/wrist_camera"
    )
    right_wrist_camera = _make_wrist_camera(
        "{ENV_REGEX_NS}/RightArm/Gripper/Robotiq_2F_85/base_link/wrist_camera"
    )


@configclass
class ActionsCfg:
    """Absolute joint-position commands for both arms and grippers."""

    left_arm = mdp.JointPositionActionCfg(
        asset_name="left_arm",
        joint_names=["panda_joint[1-7]"],
        scale=1.0,
        use_default_offset=False,
    )
    left_gripper = mdp.JointPositionActionCfg(
        asset_name="left_arm",
        joint_names=["finger_joint"],
        scale=1.0,
        use_default_offset=False,
    )
    right_arm = mdp.JointPositionActionCfg(
        asset_name="right_arm",
        joint_names=["panda_joint[1-7]"],
        scale=1.0,
        use_default_offset=False,
    )
    right_gripper = mdp.JointPositionActionCfg(
        asset_name="right_arm",
        joint_names=["finger_joint"],
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """Proprioceptive and RGB policy observations."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        left_arm_joint_pos = ObservationTermCfg(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("left_arm", joint_names=["panda_joint[1-7]"])},
        )
        right_arm_joint_pos = ObservationTermCfg(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("right_arm", joint_names=["panda_joint[1-7]"])},
        )
        left_gripper_state = ObservationTermCfg(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("left_arm", joint_names=["finger_joint"])},
        )
        right_gripper_state = ObservationTermCfg(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("right_arm", joint_names=["finger_joint"])},
        )
        external_camera_rgb = ObservationTermCfg(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("external_camera"), "data_type": "rgb", "normalize": False},
        )
        external_camera_2_rgb = ObservationTermCfg(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("external_camera_2"), "data_type": "rgb", "normalize": False},
        )
        left_wrist_camera_rgb = ObservationTermCfg(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("left_wrist_camera"), "data_type": "rgb", "normalize": False},
        )
        right_wrist_camera_rgb = ObservationTermCfg(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_camera"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


def _cube_is_upright(env, cube_name: str) -> torch.Tensor:
    quaternion = env.scene[cube_name].data.root_quat_w
    z_axis_z = 1.0 - 2.0 * (quaternion[:, 1].square() + quaternion[:, 2].square())
    tilt = torch.acos(torch.clamp(torch.abs(z_axis_z), 0.0, 1.0))
    return tilt < STACK_TILT_TOLERANCE


def _cube_is_on_cube(env, top_name: str, bottom_name: str) -> torch.Tensor:
    top_position = env.scene[top_name].data.root_pos_w
    bottom_position = env.scene[bottom_name].data.root_pos_w
    xy_distance = torch.linalg.vector_norm(top_position[:, :2] - bottom_position[:, :2], dim=-1)
    height_error = torch.abs(top_position[:, 2] - bottom_position[:, 2] - CUBE_SIZE)
    return (
        (xy_distance < STACK_XY_TOLERANCE)
        & (height_error < STACK_Z_TOLERANCE)
        & _cube_is_upright(env, top_name)
    )


def cubes_stacked(env) -> torch.Tensor:
    """Return whether the blue and red cubes form a tower on the green cube."""
    return _cube_is_on_cube(env, "cube_1", "cube_0") & _cube_is_on_cube(env, "cube_2", "cube_1")


def stack_reward(env) -> torch.Tensor:
    """Return a sparse reward for completing the stack."""
    return cubes_stacked(env).float()


@configclass
class EventCfg:
    """Reset the robots and randomize cube poses around their defaults."""

    reset_scene = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")
    reset_cube_0 = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "yaw": (-math.radians(10.0), math.radians(10.0)),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_0"),
        },
    )
    reset_cube_1 = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.04, 0.04),
                "y": (-0.04, 0.04),
                "yaw": (-math.radians(15.0), math.radians(15.0)),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_1"),
        },
    )
    reset_cube_2 = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.04, 0.04),
                "y": (-0.04, 0.04),
                "yaw": (-math.radians(15.0), math.radians(15.0)),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_2"),
        },
    )


@configclass
class RewardsCfg:
    success = RewardTermCfg(func=stack_reward, weight=1.0)


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    success = TerminationTermCfg(func=cubes_stacked)
    cube_0_dropped = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": TABLE_TOP_Z - 0.05, "asset_cfg": SceneEntityCfg("cube_0")},
    )
    cube_1_dropped = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": TABLE_TOP_Z - 0.05, "asset_cfg": SceneEntityCfg("cube_1")},
    )
    cube_2_dropped = TerminationTermCfg(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": TABLE_TOP_Z - 0.05, "asset_cfg": SceneEntityCfg("cube_2")},
    )


@configclass
class DroidBimanualCubeStackEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for bimanual DROID cube stacking."""

    scene: DroidBimanualCubeStackSceneCfg = DroidBimanualCubeStackSceneCfg(num_envs=1, env_spacing=3.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    commands = None
    curriculum = None

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 60.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.min_position_iteration_count = 16
        self.sim.physx.min_velocity_iteration_count = 1
        self.num_rerenders_on_reset = 3
        self.viewer.eye = (1.6, 0.0, 1.2)
        self.viewer.lookat = (TABLE_POSITION[0], 0.0, TABLE_TOP_Z + 0.05)
