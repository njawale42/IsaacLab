# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the DROID Franka and Robotiq 2F-85 robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

_ARENA_NUCLEUS_DIR = ISAACLAB_NUCLEUS_DIR.replace("omniverse-content-production", "omniverse-content-staging")

DROID_FRANKA_ROBOTIQ_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{_ARENA_NUCLEUS_DIR}/Arena/assets/robot_library/droid/franka_robotiq_2f_85_flattened.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=0,
        ),
        semantic_tags=[("class", "robot")],
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0444,
            "panda_joint2": -0.1894,
            "panda_joint3": -0.1107,
            "panda_joint4": -2.5148,
            "panda_joint5": 0.0044,
            "panda_joint6": 2.3775,
            "panda_joint7": 0.6952,
            "finger_joint": 0.0,
            "left_inner_finger_joint": 0.0,
            "left_inner_finger_knuckle_joint": 0.0,
            "right_inner_finger_joint": 0.0,
            "right_inner_finger_knuckle_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
        },
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=400.0,
            damping=80.0,
        ),
        "forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=400.0,
            damping=80.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=140.0,
            velocity_limit_sim=1.0,
            stiffness=17.0,
            damping=0.02,
        ),
        "gripper_mimic": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_inner_finger_joint",
                "left_inner_finger_knuckle_joint",
                "right_inner_finger_joint",
                "right_inner_finger_knuckle_joint",
                "right_outer_knuckle_joint",
            ],
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""DROID Franka arm with a Robotiq 2F-85 gripper."""
