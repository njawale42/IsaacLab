# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to replay SkillGen / Mimic datagen demonstrations.

Unlike replay_demos.py, this script reads the HDF5 schema produced by the
mimic datagen pipeline where per-timestep states live under
``data/<demo>/states/{articulation,rigid_object}/...`` and there is no
dedicated ``initial_state`` key. The first timestep of ``states`` is used
as the initial state for each episode.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay SkillGen / Mimic datagen demonstrations.")
parser.add_argument("--dataset_file", type=str, required=True, help="Path to the HDF5 dataset file.")
parser.add_argument("--task", type=str, default=None, help="Override the task name stored in the dataset.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to replay into.")
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="Episode indices to replay. Empty replays all.",
)
parser.add_argument(
    "--use_processed_actions",
    action="store_true",
    default=False,
    help="Use processed_actions instead of raw actions.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Import pinocchio before AppLauncher (required for Pink IK tasks).",
)
parser.add_argument("--video", action="store_true", default=False, help="Record video during replay.")
parser.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help="Output directory for recorded videos. Defaults to replay_videos/ next to the dataset file.",
)
parser.add_argument(
    "--video_fps",
    type=str,
    default="realtime",
    help=(
        "Video FPS mode. 'realtime' (default) encodes at physics real-time speed (1/step_dt). "
        "'wallclock' encodes at measured sim-window speed. Or pass a number for explicit FPS."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import h5py
import numpy as np
import os
import time
import torch

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

if args_cli.enable_pinocchio:
    import isaaclab_mimic.envs.pinocchio_envs  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

import isaaclab_mimic  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

is_paused = False


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def extract_initial_state(states_group: h5py.Group, device: str) -> dict:
    """Build an initial-state dict from timestep 0 of the per-step states group.

    The returned dict matches the format expected by ``InteractiveScene.reset_to``:
    ``{asset_type: {asset_name: {field: tensor(1, ...)}}}``
    """
    state = {}
    for asset_type in states_group:
        state[asset_type] = {}
        for asset_name in states_group[asset_type]:
            state[asset_type][asset_name] = {}
            for field in states_group[asset_type][asset_name]:
                data = np.array(states_group[asset_type][asset_name][field][0])
                state[asset_type][asset_name][field] = torch.tensor(data, device=device).unsqueeze(0)
    return state


def load_actions(episode_group: h5py.Group, device: str, use_processed: bool) -> torch.Tensor:
    key = "processed_actions" if use_processed else "actions"
    return torch.tensor(np.array(episode_group[key]), device=device)


def resolve_task_name(hdf5_file: h5py.File, cli_task: str | None) -> str:
    """Return the gym task id, preferring the CLI override."""
    if cli_task is not None:
        return cli_task
    env_args_raw = hdf5_file["data"].attrs.get("env_args", None)
    if env_args_raw is None:
        raise ValueError("No env_args in dataset and no --task provided.")
    env_args = json.loads(env_args_raw) if isinstance(env_args_raw, str) else env_args_raw
    return env_args["env_name"]


def _resolve_video_fps(mode: str, metadata_fps: float, measured_fps: float) -> float:
    """Resolve the encoding FPS from the --video_fps argument."""
    if mode == "realtime":
        return metadata_fps
    if mode == "wallclock":
        return measured_fps
    return float(mode)


def _write_video(
    frames: list[np.ndarray],
    video_dir: str,
    wall_elapsed: float,
    fps_mode: str,
    env,
):
    """Encode collected frames into an MP4."""
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

    n = len(frames)
    metadata_fps = env.metadata.get("render_fps", 30)
    measured_fps = n / wall_elapsed if wall_elapsed > 0 else metadata_fps
    chosen_fps = _resolve_video_fps(fps_mode, metadata_fps, measured_fps)

    print(f"Video encoding: {n} frames, wall-clock {wall_elapsed:.1f}s")
    print(f"  realtime fps (1/step_dt): {metadata_fps:.1f}")
    print(f"  wallclock fps (sim-window): {measured_fps:.1f}")
    print(f"  encoding fps: {chosen_fps:.1f}")

    video_path = os.path.join(video_dir, "replay.mp4")
    clip = ImageSequenceClip(list(frames), fps=chosen_fps)
    clip.write_videofile(video_path, logger=None)
    print(f"Saved video: {video_path}")


def main():
    global is_paused

    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"Dataset not found: {args_cli.dataset_file}")

    hdf5_file = h5py.File(args_cli.dataset_file, "r")
    data_group = hdf5_file["data"]

    task_name = resolve_task_name(hdf5_file, args_cli.task)
    episode_names = sorted(data_group.keys(), key=lambda x: int(x.split("_")[-1]))
    episode_count = len(episode_names)
    print(f"Dataset: {args_cli.dataset_file}")
    print(f"Task: {task_name}  |  Episodes: {episode_count}")

    if episode_count == 0:
        print("No episodes found.")
        hdf5_file.close()
        return

    episode_indices = list(args_cli.select_episodes) if args_cli.select_episodes else list(range(episode_count))

    num_envs = args_cli.num_envs
    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=num_envs)
    env_cfg.recorders = {}
    env_cfg.terminations = {}

    render_mode = "rgb_array" if args_cli.video else None
    _gym_env = gym.make(task_name, cfg=env_cfg, render_mode=render_mode)
    env = _gym_env.unwrapped

    video_frames: list[np.ndarray] = []
    video_dir = None
    if args_cli.video:
        video_dir = args_cli.video_dir or os.path.join(
            os.path.dirname(os.path.abspath(args_cli.dataset_file)), "replay_videos"
        )
        os.makedirs(video_dir, exist_ok=True)
        metadata_fps = env.metadata.get("render_fps", 30)
        print(f"Recording video -> {video_dir}")
        print(f"  metadata render_fps (real-time): {metadata_fps:.1f}")
        print(f"  video_fps mode: {args_cli.video_fps}")

    teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    teleop_interface.add_callback("N", play_cb)
    teleop_interface.add_callback("B", pause_cb)
    print('Press "B" to pause, "N" to resume.')

    idle_action = (
        env_cfg.idle_action.repeat(num_envs, 1)
        if hasattr(env_cfg, "idle_action")
        else torch.zeros(env.action_space.shape)
    )

    _gym_env.reset()
    teleop_interface.reset()

    # Per-env tracking: which episode index and its pre-loaded action tensor + step cursor
    env_episode_idx = [None] * num_envs
    env_actions = [None] * num_envs
    env_step = [0] * num_envs

    replayed = 0
    wall_t0 = time.perf_counter()
    pause_duration = 0.0

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            # Assign episodes to idle envs
            for env_id in range(num_envs):
                if env_actions[env_id] is None or env_step[env_id] >= len(env_actions[env_id]):
                    if not episode_indices:
                        env_actions[env_id] = None
                        continue
                    ep_idx = episode_indices.pop(0)
                    if ep_idx >= episode_count:
                        env_actions[env_id] = None
                        continue
                    ep_name = episode_names[ep_idx]
                    replayed += 1
                    print(f"{replayed:4d}: Loading {ep_name} (idx {ep_idx}) -> env_{env_id}")

                    ep_group = data_group[ep_name]
                    initial_state = extract_initial_state(ep_group["states"], env.device)
                    env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)

                    env_actions[env_id] = load_actions(ep_group, env.device, args_cli.use_processed_actions)
                    env_step[env_id] = 0
                    env_episode_idx[env_id] = ep_idx

            # Check if any env still has work
            if all(a is None for a in env_actions):
                break

            # Build batched action
            actions = idle_action.clone()
            for env_id in range(num_envs):
                if env_actions[env_id] is not None and env_step[env_id] < len(env_actions[env_id]):
                    actions[env_id] = env_actions[env_id][env_step[env_id]]
                    env_step[env_id] += 1

            pause_t0 = time.perf_counter()
            while is_paused:
                env.sim.render()
            pause_duration += time.perf_counter() - pause_t0

            _gym_env.step(actions)

            if args_cli.video:
                frame = env.render()
                if frame is not None:
                    video_frames.append(frame)

    wall_elapsed = time.perf_counter() - wall_t0 - pause_duration
    hdf5_file.close()

    if args_cli.video and video_frames:
        _write_video(video_frames, video_dir, wall_elapsed, args_cli.video_fps or "realtime", env)

    suffix = "s" if replayed != 1 else ""
    print(f"Finished replaying {replayed} episode{suffix}.")
    _gym_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
