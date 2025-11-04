"""
Wrapper planner that selects the appropriate humanoid arm planner (left/right)
per call, while keeping collision spheres active for both arms.

Interface matches MotionPlannerBase for compatibility with DataGenerator.
"""

from __future__ import annotations

import traceback
import torch
from typing import Any

from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_env import ManagerBasedEnv

from isaaclab_mimic.motion_planners.motion_planner_base import MotionPlannerBase
from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg
from isaaclab_mimic.motion_planners.curobo.curobo_planner_humanoid import HumanoidArmCuroboPlanner


class BimanualHumanoidPlanner(MotionPlannerBase):
    """
    Delegates to two HumanoidArmCuroboPlanner instances (right and left).

    Arm selection per update is decided by comparing the target pose to the
    current EEF poses and picking the closest arm.
    """

    def __init__(
        self,
        env: ManagerBasedEnv,
        robot: Articulation,
        *,
        cfg_right: CuroboPlannerCfg,
        cfg_left: CuroboPlannerCfg,
        env_id: int = 0,
    ) -> None:
        super().__init__(env=env, robot=robot, env_id=env_id, debug=bool(getattr(cfg_right, "debug_planner", False)))

        # Create right/left arm planners. Keep both arms' spheres active for collision with the other arm.
        self._planner_right = HumanoidArmCuroboPlanner(
            env=env,
            robot=robot,
            config=cfg_right,
            env_id=env_id,
            active_joint_substrings=("right_",),
            hand_link_substrings=("GR1T2_fourier_hand_6dof_right_",),
            collision_active_link_substrings=("GR1T2_fourier_hand_6dof_left_", "GR1T2_fourier_hand_6dof_right_"),
        )
        self._planner_left = HumanoidArmCuroboPlanner(
            env=env,
            robot=robot,
            config=cfg_left,
            env_id=env_id,
            active_joint_substrings=("left_",),
            hand_link_substrings=("GR1T2_fourier_hand_6dof_left_",),
            collision_active_link_substrings=("GR1T2_fourier_hand_6dof_left_", "GR1T2_fourier_hand_6dof_right_"),
        )

        # Default to right for config accessors
        self.config = cfg_right
        self.visualize_spheres = bool(getattr(cfg_right, "visualize_spheres", False))

        self._last_arm: str | None = None

    def _choose_arm(self, target_pose: torch.Tensor) -> str:
        """
        Choose the arm whose current EEF is closest to the target pose.
        """
        # Current EEF poses from env are in controller frame (world/site). Use positions only.
        left_pose = self.env.get_robot_eef_pose("left", env_ids=[self.env_id])[0]
        right_pose = self.env.get_robot_eef_pose("right", env_ids=[self.env_id])[0]

        tpos = target_pose[:3, 3]
        ldist = torch.linalg.vector_norm(left_pose[:3, 3] - tpos)
        rdist = torch.linalg.vector_norm(right_pose[:3, 3] - tpos)
        try:
            print(
                f"[BimanualHumanoidPlanner] Distances -> left: {float(ldist):.4f} m, right: {float(rdist):.4f} m"
            )
        except Exception:
            pass
        return "left" if ldist <= rdist else "right"

    def update_world_and_plan_motion(
        self,
        target_pose: torch.Tensor,
        expected_attached_object: str | None = None,
        env_id: int = 0,
        step_size: float | None = None,
        enable_retiming: bool | None = None,
        link_target_poses_base: dict[str, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> bool:
        arm_primary = self._choose_arm(target_pose)
        arm_fallback = "left" if arm_primary == "right" else "right"
        # Minimal debug to aid diagnosis
        print(f"[BimanualHumanoidPlanner] Selected arm: {arm_primary}")

        def _attempt(arm_name: str) -> bool:
            planner_local = self._planner_right if arm_name != "left" else self._planner_left
            try:
                ok = planner_local.update_world_and_plan_motion(
                    target_pose=target_pose,
                    expected_attached_object=expected_attached_object,
                    env_id=env_id,
                    step_size=step_size,
                    enable_retiming=enable_retiming,
                    link_target_poses_base=link_target_poses_base,
                    **kwargs,
                )
                if ok:
                    self._last_arm = arm_name
                return bool(ok)
            except Exception:
                print("[BimanualHumanoidPlanner] ERROR during update_world_and_plan_motion:")
                print(
                    f"  arm={arm_name} env_id={env_id} step_size={step_size} enable_retiming={enable_retiming}"
                )
                print(f"  expected_attached_object={expected_attached_object}")
                if "input_is_site_frame" in kwargs:
                    print(f"  input_is_site_frame={kwargs['input_is_site_frame']}")
                try:
                    print(f"  target_pose=\n{target_pose}")
                except Exception:
                    pass
                traceback.print_exc()
                return False

        # Try primary arm; if it fails, try the other arm once
        if _attempt(arm_primary):
            return True
        print(f"[BimanualHumanoidPlanner] Primary arm {arm_primary} failed; trying fallback arm {arm_fallback}")
        return _attempt(arm_fallback)

    def has_next_waypoint(self) -> bool:
        planner = self._planner_right if self._last_arm != "left" else self._planner_left
        return planner.has_next_waypoint()

    def get_next_waypoint_ee_pose(self) -> Any:
        planner = self._planner_right if self._last_arm != "left" else self._planner_left
        return planner.get_next_waypoint_ee_pose()

    def get_planned_poses(self) -> list[Any]:
        planner = self._planner_right if self._last_arm != "left" else self._planner_left
        return planner.get_planned_poses()

    def reset_plan(self) -> None:
        self._planner_right.reset_plan()
        self._planner_left.reset_plan()
        self._last_arm = None


