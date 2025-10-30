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

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg


def rollout(policy, env, success_term, horizon, device):
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
                        seq_np_block = inflight.result()
                        action_buffer.extend([seq_np_block[i] for i in range(seq_np_block.shape[0])])
                    except Exception:
                        # fall back to synchronous inference on failure
                        prep_obs = policy._prepare_observation(obs, batched_ob=True)
                        with torch.no_grad():
                            action_seq = policy.policy._get_action_trajectory(obs_dict=prep_obs)
                        seq_np = action_seq[0, :action_horizon].detach().cpu().numpy()
                        action_buffer = [seq_np[i] for i in range(seq_np.shape[0])]
                    finally:
                        inflight = None
                else:
                    prep_obs = policy._prepare_observation(obs, batched_ob=True)
                    with torch.no_grad():
                        action_seq = policy.policy._get_action_trajectory(obs_dict=prep_obs)
                    seq_np = action_seq[0, :action_horizon].detach().cpu().numpy()
                    action_buffer = [seq_np[i] for i in range(seq_np.shape[0])]

            # Background prefetch when buffer gets low
            if len(action_buffer) <= prefetch_threshold and inflight is None:
                snap_obs = {k: v.clone() for k, v in obs.items()}

                def _infer_sequence(o):
                    po = policy._prepare_observation(o, batched_ob=True)
                    with torch.no_grad():
                        aseq = policy.policy._get_action_trajectory(obs_dict=po)
                    return aseq[0, :action_horizon].detach().cpu().numpy()
                inflight = executor.submit(_infer_sequence, snap_obs)
            # If a prefetch finished, extend buffer
            if inflight is not None and inflight.done():
                try:
                    seq_np2 = inflight.result()
                    action_buffer.extend([seq_np2[i] for i in range(seq_np2.shape[0])])
                finally:
                    inflight = None
            actions = action_buffer.pop(0)
        else:
            actions = policy(obs)

        # Unnormalize actions
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = (
                (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min

        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

        # Apply actions
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]

        # Record trajectory
        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs)

        # Check if rollout was successful
        if bool(success_term.func(env, **success_term.params)[0]):
            return True, traj
        elif terminated or truncated:
            return False, traj

    return False, traj


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
