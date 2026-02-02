import os
import tempfile
import yaml
import numpy as np
from isaaclab.controllers import utils as ControllerUtils
from nvplan.applications.custream.config import create_robot_config
from nvplan.applications.custream.spheres import load_spheres


def to_python(obj):
    """Convert numpy/torch types to Python native types for YAML serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj


def _scale_collision_sphere_radii(
    robot_cfg: dict,
    radius_scale: float = 1.0,
    per_link_scale: dict[str, float] | None = None,
) -> None:
    """Scale collision sphere radii in-place."""
    spheres = robot_cfg.get("kinematics", {}).get("collision_spheres")
    if not spheres or radius_scale == 1.0 and not per_link_scale:
        return
    for link_name, link_spheres in spheres.items():
        link_scale = per_link_scale.get(link_name, 1.0) if per_link_scale else 1.0
        scale = radius_scale * link_scale
        if scale == 1.0:
            continue
        for sphere in link_spheres:
            if "radius" in sphere:
                sphere["radius"] *= scale


def build_humanoid_yaml_from_usd(
    usd_path: str,
    arm: str,
    inactive_joints: list[str] | None = None,
    radius_scale: float = 1.0,
    per_link_radius_scale: dict[str, float] | None = None,
) -> str:
    """Build cuRobo robot configuration YAML from USD file."""
    tmp_dir = tempfile.mkdtemp(prefix="gr1_curobo_")
    print("[PlanHumanoid] Converting USD to URDF...")
    urdf_path, _ = ControllerUtils.convert_usd_to_urdf(usd_path, tmp_dir, force_conversion=True)
    print(f"[PlanHumanoid] URDF: {urdf_path}")

    print("[PlanHumanoid] Creating robot config...")
    print("[PlanHumanoid] Loading spheres...")
    tool_links = [_tool_link_for_arm(arm)]
    robot_config = create_robot_config(
        urdf_path,
        tool_links=tool_links,
        inactive_joints=inactive_joints or [],
        depth=2,
        verbose=False,
    )

    # Configure sphere generation
    max_spheres = 600 - len(tool_links) * 50
    load_spheres(robot_config, max_spheres=max_spheres, max_link_spheres=int(1e9))
    robot_cfg_dict = robot_config["robot_cfg"]
    _scale_collision_sphere_radii(robot_cfg_dict, radius_scale=radius_scale, per_link_scale=per_link_radius_scale)
    robot_cfg_yaml = to_python(robot_cfg_dict)

    def _strip_keys(obj, keys):
        """Remove specified keys from nested dictionary."""
        if isinstance(obj, dict):
            for k in list(obj.keys()):
                if k in keys:
                    obj.pop(k, None)
                else:
                    _strip_keys(obj[k], keys)
        elif isinstance(obj, list):
            for v in obj:
                _strip_keys(v, keys)

    # Remove lock_joints and cspace as they'll be configured dynamically
    _strip_keys(robot_cfg_yaml, {"lock_joints", "cspace"})

    # Ensure ee_link points to the selected arm's tool link (guard against malformed YAML)
    if isinstance(robot_cfg_yaml, dict):
        kin = robot_cfg_yaml.get("kinematics", None)
        if isinstance(kin, dict):
            kin["ee_link"] = _tool_link_for_arm(arm)

    out_dir = tempfile.mkdtemp(prefix="curobo_robot_cfg_")
    out_path = os.path.join(out_dir, "gr1_generated.yml")
    print(f"[PlanHumanoid] Writing robot YAML to {out_path} ...")
    with open(out_path, "w") as f:
        yaml.safe_dump({"robot_cfg": robot_cfg_yaml}, f, sort_keys=False)
    print("[PlanHumanoid] Robot YAML written.")
    return out_path


def _tool_link_for_arm(arm: str) -> str:
    """Get tool link name for the specified arm."""
    # Matches the converted GR1T2 URDF link names
    return f"GR1T2_fourier_hand_6dof_{arm}_hand_pitch_link"
