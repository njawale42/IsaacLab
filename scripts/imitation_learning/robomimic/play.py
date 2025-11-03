# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy from robomimic.

This script loads a robomimic policy and plays it in an Isaac Lab environment.

Args:
    task: Name of the environment.
    checkpoint: Path to the robomimic policy checkpoint.
    horizon: If provided, override the step horizon of each rollout.
    num_rollouts: If provided, override the number of rollouts.
    seed: If provided, overeride the default random seed.
    norm_factor_min: If provided, minimum value of the action space normalization factor.
    norm_factor_max: If provided, maximum value of the action space normalization factor.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=800, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor."
)
parser.add_argument("--enable_pinocchio", default=False, action="store_true", help="Enable Pinocchio.")
parser.add_argument(
    "--dp_viz_3d",
    action="store_true",
    help="If set, visualize diffusion denoising rollouts in 3D using debug draw.",
)
parser.add_argument(
    "--dp_viz_stride",
    type=int,
    default=5,
    help="Visualize every N denoising steps to reduce clutter.",
)
parser.add_argument(
    "--dp_viz_scale",
    type=float,
    default=0.05,
    help="Meters per unit action for visualizing integrated EEF position.",
)
parser.add_argument(
    "--dp_viz_rot_scale",
    type=float,
    default=0.2,
    help="Radians per unit action for visualizing orientation change (axis-angle length).",
)
parser.add_argument(
    "--dp_viz_live",
    action="store_true",
    help="Run a lightweight diffusion sampling every frame for visualization (does not affect control).",
)
parser.add_argument(
    "--dp_viz_steps",
    type=int,
    default=10,
    help="Number of reverse denoising steps for live visualization per frame.",
)
parser.add_argument(
    "--dp_quat_xyzw",
    action="store_true",
    help="Assume obs eef_quat is in XYZW order (default). If unset, uses WXYZ.",
    default=True,
)

parser.add_argument(
    "--dp_infer_steps",
    type=int,
    default=None,
    help="Optional: override diffusion num_inference_timesteps at replay (smaller is faster).",
)
parser.add_argument(
    "--dp_use_ddim",
    action="store_true",
    help="Optional: use DDIM scheduler for faster inference during replay.",
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import gymnasium as gym
import numpy as np
import random
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
from typing import cast

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg


def rollout(policy, env, success_term, horizon, device):  # noqa: C901
    """Perform a single rollout of the policy in the environment.

    Args:
        policy: The robomimicpolicy to play.
        env: The environment to play in.
        horizon: The step horizon of each rollout.
        device: The device to run the policy on.

    Returns:
        terminated: Whether the rollout terminated.
        traj: The trajectory of the rollout.
    """
    policy.start_episode()
    obs_dict, _ = env.reset()
    traj = dict(actions=[], obs=[], next_obs=[])

    # Detect diffusion policy and set up temporal buffers
    is_diffusion = getattr(policy.policy, "global_config", None) is not None and \
        getattr(policy.policy.global_config, "algo_name", "") == "diffusion_policy"
    obs_horizon = None
    obs_keys_for_policy = None
    if is_diffusion:
        obs_horizon = int(policy.policy.global_config.algo.horizon.observation_horizon)
        obs_keys_for_policy = list(policy.policy.global_config.all_obs_keys)
        from collections import deque
        obs_buffers = {k: deque(maxlen=obs_horizon) for k in obs_keys_for_policy if k in obs_dict["policy"]}
        # prime buffers with initial obs repeated
        initial_obs = obs_dict["policy"]
        for k in obs_buffers:
            val = initial_obs[k]
            obs_buffers[k].extend([val for _ in range(obs_horizon)])
        # action buffering to avoid re-sampling every step (expensive)
        action_buffer = []
        action_horizon = int(policy.policy.global_config.algo.horizon.action_horizon)
        # background prefetch of next trajectory while consuming current buffer
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=1)
        inflight = None
        prefetch_threshold = max(1, action_horizon // 2)

    for i in range(horizon):
        # Prepare observations (temporal window for diffusion)
        if is_diffusion:
            # push current obs into buffers
            for k in list(obs_buffers.keys()):
                if k in obs_dict["policy"]:
                    obs_buffers[k].append(obs_dict["policy"][k])
            # build [B=1, T, D] inputs per key
            obs = {}
            for k in obs_buffers:
                # each element may be tensor with extra dims; squeeze batch/time-like dims
                frames = [torch.as_tensor(f).squeeze(0) for f in list(obs_buffers[k])]
                stacked = torch.stack(frames, dim=0)  # [T, D]
                obs[k] = stacked.unsqueeze(0)  # [1, T, D]

            # Live visualization: run a tiny diffusion sample each frame (no control effect)
            if args_cli.dp_viz_3d and args_cli.dp_viz_live:
                try:
                    _ = _dp_sample_action_sequence(
                        policy,
                        obs,
                        env,
                        return_intermediates=False,
                        override_steps=args_cli.dp_viz_steps,
                    )
                except Exception:
                    pass
        else:
            obs = copy.deepcopy(obs_dict["policy"])
            for ob in obs:
                obs[ob] = torch.squeeze(obs[ob])

        # Check if environment image observations
        if hasattr(env.cfg, "image_obs_list"):
            # Process image observations for robomimic inference
            for image_name in env.cfg.image_obs_list:
                if image_name in obs_dict["policy"].keys():
                    # Convert from chw uint8 to hwc normalized float
                    image = torch.squeeze(obs_dict["policy"][image_name])
                    image = image.permute(2, 0, 1).clone().float()
                    image = image / 255.0
                    image = image.clip(0.0, 1.0)
                    obs[image_name] = image

        traj["obs"].append(obs)

        # Compute actions
        if is_diffusion:
            # Ensure buffer has at least one action:
            # - If a prefetch job exists, block until it completes and fill the buffer
            # - Otherwise, run a synchronous inference
            while len(action_buffer) == 0:
                if inflight is not None:
                    try:
                        res_block = inflight.result()
                        if args_cli.dp_viz_3d and isinstance(res_block, tuple):
                            seq_np_block, intermed_block = res_block
                            try:
                                _dp_draw_intermediates(env, obs, intermed_block, action_horizon, args_cli.dp_viz_scale, args_cli.dp_viz_stride)
                            except Exception:
                                pass
                        else:
                            seq_np_block = res_block if not isinstance(res_block, tuple) else res_block[0]
                        seq_np_block = cast(np.ndarray, np.asarray(seq_np_block))
                        action_buffer.extend([seq_np_block[i] for i in range(len(seq_np_block))])
                    except Exception:
                        # fall back to synchronous inference on failure
                        seq_np = _dp_sample_action_sequence(policy, obs)
                        action_buffer = [seq_np[i] for i in range(len(seq_np))]
                    finally:
                        inflight = None
                else:
                    seq_np = _dp_sample_action_sequence(policy, obs, env if args_cli.dp_viz_3d else None)
                    action_buffer = [seq_np[i] for i in range(len(seq_np))]

            # Background prefetch when buffer gets low
            if len(action_buffer) <= prefetch_threshold and inflight is None:
                snap_obs = {k: v.clone() for k, v in obs.items()}

                def _infer_sequence(o):
                    # when viz is enabled, return intermediates for drawing on main thread
                    if args_cli.dp_viz_3d:
                        return _dp_sample_action_sequence(policy, o, None, return_intermediates=True)
                    return _dp_sample_action_sequence(policy, o)
                inflight = executor.submit(_infer_sequence, snap_obs)
            # If a prefetch finished, extend buffer
            if inflight is not None and inflight.done():
                try:
                    res = inflight.result()
                    if args_cli.dp_viz_3d and isinstance(res, tuple):
                        seq_np2, intermed2 = res
                        try:
                            _dp_draw_intermediates(env, obs, intermed2, action_horizon, args_cli.dp_viz_scale, args_cli.dp_viz_stride)
                            _dp_store_viz(intermed2, action_horizon)
                        except Exception:
                            pass
                    else:
                        seq_np2 = res if not isinstance(res, tuple) else res[0]
                    seq_np2 = cast(np.ndarray, np.asarray(seq_np2))
                    action_buffer.extend([seq_np2[i] for i in range(len(seq_np2))])
                finally:
                    inflight = None
            actions = action_buffer.pop(0)
        else:
            actions = policy(obs)

        # Ensure numpy array for arithmetic
        actions = np.asarray(actions)
        # Unnormalize actions
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = (
                (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min

        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

        # Apply actions
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]

        # Force redraw each step anchored to CURRENT obs so points move
        _dp_redraw_last(env, obs)

        # Record trajectory
        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs)

        # Check if rollout was successful
        if bool(success_term.func(env, **success_term.params)[0]):
            return True, traj
        elif terminated or truncated:
            return False, traj

    return False, traj


_DBG_IFACE = None
_DP_LAST_VIZ = None  # (intermediates, action_horizon)


def _acquire_debug_draw():
    global _DBG_IFACE
    if _DBG_IFACE is not None:
        return _DBG_IFACE
    try:
        import isaacsim.util.debug_draw._debug_draw as _debug_draw_mod  # type: ignore
        _DBG_IFACE = _debug_draw_mod.acquire_debug_draw_interface()
    except Exception:
        _DBG_IFACE = None
    return _DBG_IFACE


def _dp_draw_intermediates(env, obs_dict_bt, intermediates, action_horizon, scale: float, stride: int) -> None:
    draw = _acquire_debug_draw()
    if draw is None:
        return
    try:
        # base EEF pos from observation window's last frame
        if "eef_pos" not in obs_dict_bt:
            return
        base = obs_dict_bt["eef_pos"][0, -1, :].detach().cpu().numpy()
        # orientation from last frame to rotate deltas into world frame
        if "eef_quat" in obs_dict_bt:
            q = obs_dict_bt["eef_quat"][0, -1, :].detach().cpu()
        else:
            q = None

        def _quat_to_R(qt: torch.Tensor) -> torch.Tensor:
            # qt shape (4,)
            x: torch.Tensor
            y: torch.Tensor
            z: torch.Tensor
            w: torch.Tensor
            if args_cli.dp_quat_xyzw:
                x, y, z, w = qt[0], qt[1], qt[2], qt[3]
            else:
                w, x, y, z = qt[0], qt[1], qt[2], qt[3]
            # normalize
            norm = torch.linalg.vector_norm(torch.stack([w, x, y, z])) + 1e-8
            w, x, y, z = w / norm, x / norm, y / norm, z / norm
            # rotation matrix (world_from_eef)
            R = torch.empty((3, 3), dtype=torch.float32)
            R[0, 0] = 1 - 2 * (y * y + z * z)
            R[0, 1] = 2 * (x * y - z * w)
            R[0, 2] = 2 * (x * z + y * w)
            R[1, 0] = 2 * (x * y + z * w)
            R[1, 1] = 1 - 2 * (x * x + z * z)
            R[1, 2] = 2 * (y * z - x * w)
            R[2, 0] = 2 * (x * z - y * w)
            R[2, 1] = 2 * (y * z + x * w)
            R[2, 2] = 1 - 2 * (x * x + y * y)
            return R

        R_we = _quat_to_R(q) if q is not None else None
        num_snaps = len(intermediates)
        points = []
        colors = []
        sizes = []
        # compute which snapshot indices to draw; always include last snapshot
        stride_val = max(1, stride)
        draw_indices = [i for i in range(num_snaps) if (i % stride_val) == 0]
        if (num_snaps - 1) not in draw_indices and num_snaps > 0:
            draw_indices.append(num_snaps - 1)
        for si in draw_indices:
            seq = intermediates[si]
            seq_np = seq[0, :action_horizon, :].detach().cpu().numpy()
            pos = base.copy()
            q_cur = q.clone() if q is not None else None
            R_cur = R_we.clone() if R_we is not None else None
            tval = (si + 1.0) / float(max(1, num_snaps))
            color = (1.0 - tval, 0.2, tval, 0.6 if si == draw_indices[-1] else 0.35)
            for t in range(min(action_horizon, seq_np.shape[0])):
                if seq_np.shape[1] < 3:
                    break
                delta_eef = torch.from_numpy(seq_np[t, :3]).to(dtype=torch.float32)
                if R_cur is not None:
                    delta_world = (R_cur @ (scale * delta_eef)).cpu().numpy()
                else:
                    delta_world = (scale * delta_eef).cpu().numpy()
                pos = pos + delta_world
                points.append((float(pos[0]), float(pos[1]), float(pos[2])))
                colors.append(color)
                sizes.append(10.0 if si == draw_indices[-1] else 6.0)
                # accumulate orientation if rotational dims provided
                if seq_np.shape[1] >= 6 and q_cur is not None:
                    # axis-angle in eef frame
                    delta_aa = torch.from_numpy(seq_np[t, 3:6]).to(dtype=torch.float32) * float(args_cli.dp_viz_rot_scale)
                    # convert axis-angle to quaternion and update
                    angle = torch.linalg.vector_norm(delta_aa)
                    if angle > 1e-8:
                        axis = delta_aa / angle
                        half = 0.5 * angle
                        s = torch.sin(half)
                        if args_cli.dp_quat_xyzw:
                            dq = torch.tensor([axis[0] * s, axis[1] * s, axis[2] * s, torch.cos(half)], dtype=torch.float32)
                        else:
                            dq = torch.tensor([torch.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=torch.float32)
                        # quaternion multiply q_cur = q_cur * dq
                        if args_cli.dp_quat_xyzw:
                            x1, y1, z1, w1 = q_cur
                            x2, y2, z2, w2 = dq
                            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
                            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
                            q_cur = torch.tensor([x, y, z, w], dtype=torch.float32)
                        else:
                            w1, x1, y1, z1 = q_cur
                            w2, x2, y2, z2 = dq
                            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
                            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
                            q_cur = torch.tensor([w, x, y, z], dtype=torch.float32)
                        R_cur = _quat_to_R(q_cur)
        # only clear if we have something new to draw; otherwise keep previous
        if points:
            try:
                draw.clear_points()
            except Exception:
                pass
            draw.draw_points(points, colors, sizes)
    except Exception:
        return


def _dp_store_viz(intermediates, action_horizon):
    global _DP_LAST_VIZ
    _DP_LAST_VIZ = (intermediates, action_horizon)


def _dp_redraw_last(env, obs_bt_current):
    if not args_cli.dp_viz_3d:
        return
    try:
        global _DP_LAST_VIZ
        if _DP_LAST_VIZ is None:
            return
        intermed, Ta = _DP_LAST_VIZ
        _dp_draw_intermediates(env, obs_bt_current, intermed, Ta, args_cli.dp_viz_scale, args_cli.dp_viz_stride)
    except Exception:
        return


def _dp_sample_action_sequence(policy, obs_bt, env_for_viz=None, return_intermediates: bool = False, override_steps: int | None = None):
    # Prepare obs
    prep_obs = policy._prepare_observation(obs_bt, batched_ob=True)
    m = policy.policy
    To = int(m.algo_config.horizon.observation_horizon)
    Ta = int(m.algo_config.horizon.action_horizon)
    Tp = int(m.algo_config.horizon.prediction_horizon)
    device = m.device
    # select nets
    nets = m.ema.averaged_model if m.ema is not None else m.nets
    # encode obs
    inputs = {"obs": prep_obs, "goal": None}
    for k in m.obs_shapes:
        if inputs["obs"][k].ndim - 1 == len(m.obs_shapes[k]):
            inputs["obs"][k] = inputs["obs"][k].unsqueeze(1)
    obs_features = TensorUtils.time_distributed(inputs, nets["policy"]["obs_encoder"], inputs_as_kwargs=True)
    obs_cond = obs_features.flatten(start_dim=1)
    # init noise
    naction = torch.randn((1, Tp, m.ac_dim), device=device)
    # init scheduler timesteps
    if override_steps is not None:
        num_steps = int(override_steps)
    else:
        if m.algo_config.ddpm.enabled:
            num_steps = m.algo_config.ddpm.num_inference_timesteps
        elif m.algo_config.ddim.enabled:
            num_steps = m.algo_config.ddim.num_inference_timesteps
        else:
            num_steps = 50
    m.noise_scheduler.set_timesteps(num_steps)
    intermediates = []
    for k in m.noise_scheduler.timesteps:
        noise_pred = nets["policy"]["noise_pred_net"](sample=naction, timestep=k, global_cond=obs_cond)
        naction = m.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
        if env_for_viz is not None and args_cli.dp_viz_3d:
            intermediates.append(naction.clone())
    start = To - 1
    end = start + Ta
    action = naction[:, start:end]
    if env_for_viz is not None and args_cli.dp_viz_3d and not return_intermediates:
        try:
            _dp_draw_intermediates(env_for_viz, obs_bt, intermediates, Ta, args_cli.dp_viz_scale, args_cli.dp_viz_stride)
            _dp_store_viz(intermediates, Ta)
        except Exception:
            pass
    if return_intermediates:
        return action[0].detach().cpu().numpy(), intermediates
    return action[0].detach().cpu().numpy()


def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    # Set observations to dictionary mode for Robomimic
    env_cfg.observations.policy.concatenate_terms = False

    # Set termination conditions
    env_cfg.terminations.time_out = None

    # Disable recorder
    env_cfg.recorders = None

    # Extract success checking function
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Set seed
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)

    # Acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # Run policy
    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")
        policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device)
        # Optional: speed up diffusion policy inference
        try:
            if getattr(policy.policy, "global_config", None) is not None and \
               getattr(policy.policy.global_config, "algo_name", "") == "diffusion_policy":
                if args_cli.dp_use_ddim:
                    policy.policy.algo_config.ddpm.enabled = False
                    policy.policy.algo_config.ddim.enabled = True
                if args_cli.dp_infer_steps is not None:
                    if policy.policy.algo_config.ddpm.enabled:
                        policy.policy.algo_config.ddpm.num_inference_timesteps = args_cli.dp_infer_steps
                    if policy.policy.algo_config.ddim.enabled:
                        policy.policy.algo_config.ddim.num_inference_timesteps = args_cli.dp_infer_steps
        except Exception:
            pass
        terminated, traj = rollout(policy, env, success_term, args_cli.horizon, device)
        results.append(terminated)
        print(f"[INFO] Trial {trial}: {terminated}\n")

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    print(f"Trial Results: {results}\n")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
