"""
Utilities for building collision-aware schedules over recorded arm trajectories.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from nvplan.applications.custream.retime import retime_paths


@dataclass
class ArmPath:
    """Container describing the recorded trajectory for a single arm."""

    name: str
    poses: torch.Tensor
    gripper_actions: torch.Tensor
    joint_positions: torch.Tensor


@dataclass
class DiscreteSchedule:
    """
    Final discrete playback schedule (per simulator tick) for both arms.

    Attributes:
        left_indices: Index of the left-arm waypoint to command at each tick.
        right_indices: Index of the right-arm waypoint to command at each tick.
        total_time: Continuous-time horizon covered by the schedule.
        step_dt: Simulation step duration used for discretization.
    """

    left_indices: torch.Tensor
    right_indices: torch.Tensor
    total_time: float
    step_dt: float

    def append_hold(self, hold_steps: int) -> None:
        """Extend the schedule by repeating the final sample for `hold_steps` ticks."""
        if hold_steps <= 0:
            return
        if self.left_indices.numel() == 0 or self.right_indices.numel() == 0:
            return
        last_left = self.left_indices[-1].repeat(hold_steps)
        last_right = self.right_indices[-1].repeat(hold_steps)
        self.left_indices = torch.cat([self.left_indices, last_left], dim=0)
        self.right_indices = torch.cat([self.right_indices, last_right], dim=0)


def _densify_path(path: torch.Tensor, factor: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearly densify a joint path by an integer factor; return dense path and index map."""
    if factor <= 1 or path.shape[0] <= 1:
        mapping = torch.arange(path.shape[0], device=path.device, dtype=torch.long)
        return path, mapping

    segments = path.shape[0] - 1
    dense = []
    mapping = []
    for seg in range(segments):
        p0 = path[seg]
        p1 = path[seg + 1]
        for s in range(factor):
            alpha = float(s) / float(factor)
            dense.append((1.0 - alpha) * p0 + alpha * p1)
            mapping.append(seg)
    dense.append(path[-1])
    mapping.append(segments)
    return torch.stack(dense, dim=0), torch.tensor(mapping, device=path.device, dtype=torch.long)


def _compute_collision_pairs(
    joint_path_r: torch.Tensor,
    joint_path_l: torch.Tensor,
    kin_right,
    kin_left,
    *,
    densify_factor: int = 4,
    pair_batch: int = 4096,
    collision_margin: float = 0.0,
) -> list[tuple[int, int]]:
    """Find colliding waypoint pairs using sphere overlaps produced by cuRobo kinematics."""
    if joint_path_r.numel() == 0 or joint_path_l.numel() == 0:
        return []

    dev = joint_path_r.device
    pos_r_dense, map_r = _densify_path(joint_path_r, densify_factor)
    pos_l_dense, map_l = _densify_path(joint_path_l, densify_factor)

    Nr = int(pos_r_dense.shape[0])
    Nl = int(pos_l_dense.shape[0])
    if Nr == 0 or Nl == 0:
        return []

    q_r = torch.repeat_interleave(pos_r_dense, repeats=Nl, dim=0)
    q_l = pos_l_dense.repeat(Nr, 1)
    total_rows = q_r.shape[0]

    colliding_rows: list[int] = []

    for start in range(0, total_rows, pair_batch):
        end = min(total_rows, start + pair_batch)
        q_r_b = q_r[start:end]
        q_l_b = q_l[start:end]
        batch_size = q_r_b.shape[0]

        state_r = kin_right.get_state(q_r_b)
        state_l = kin_left.get_state(q_l_b)
        sph_r = state_r.link_spheres_tensor.view(batch_size, -1, 4)
        sph_l = state_l.link_spheres_tensor.view(batch_size, -1, 4)

        c_r = sph_r[..., :3]
        r_r = sph_r[..., 3]
        c_l = sph_l[..., :3]
        r_l = sph_l[..., 3]

        aa = (c_r * c_r).sum(dim=-1, keepdim=True)
        bb = (c_l * c_l).sum(dim=-1).unsqueeze(1)
        ab = torch.bmm(c_r, c_l.transpose(1, 2))
        dist2 = torch.clamp(aa + bb - 2.0 * ab, min=0.0)

        radii = r_r.unsqueeze(-1) + r_l.unsqueeze(-2) + collision_margin
        thresh2 = radii * radii
        collide = (dist2 <= thresh2).any(dim=(1, 2))
        rows = torch.nonzero(collide, as_tuple=False).flatten()
        if rows.numel() > 0:
            colliding_rows.extend((start + int(idx.item())) for idx in rows)

    if not colliding_rows:
        return []

    pairs: list[tuple[int, int]] = []
    for row in colliding_rows:
        i_dense = row // Nl
        j_dense = row % Nl
        pairs.append((int(map_r[i_dense].item()), int(map_l[j_dense].item())))
    return pairs


def _retime(len_r: int, len_l: int, pairs: list[tuple[int, int]], min_dt: float) -> tuple[list[float], list[float]] | None:
    """Solve MILP retiming for both arms."""
    if len_r == 0 or len_l == 0:
        return None

    path_r = [None] * len_r
    path_l = [None] * len_l
    # buffer > 0 enforces time separation between colliding waypoint pairs
    # min_dt is the nominal time between waypoints, buffer ensures collision avoidance
    buffer = min_dt * 0.5  # Half a timestep buffer between colliding pairs
    print(f"[Retime] len_r={len_r}, len_l={len_l}, pairs={len(pairs)}, min_dt={min_dt:.4f}, buffer={buffer:.4f}")
    # Debug: show sample collision pairs
    if pairs:
        print(f"[Retime] Sample pairs (first 5): {pairs[:5]}")
        print(f"[Retime] Sample pairs (last 5): {pairs[-5:]}")
        # Check pair value ranges
        r_indices = [p[0] for p in pairs]
        l_indices = [p[1] for p in pairs]
        print(f"[Retime] Right indices range: [{min(r_indices)}, {max(r_indices)}]")
        print(f"[Retime] Left indices range: [{min(l_indices)}, {max(l_indices)}]")

    # Reduce collision pairs to boundary pairs for MILP tractability
    # For each right index i, keep only the min and max left indices j
    # This captures the "earliest" and "latest" collision points
    # boundary_map: dict[int, tuple[int, int]] = {}  # i -> (min_j, max_j)
    # for i, j in pairs:
    #     if i not in boundary_map:
    #         boundary_map[i] = (j, j)
    #     else:
    #         cur_min, cur_max = boundary_map[i]
    #         boundary_map[i] = (min(j, cur_min), max(j, cur_max))
    # reduced_pairs = []
    # for i, (min_j, max_j) in boundary_map.items():
    #     reduced_pairs.append((i, min_j))
    #     if max_j != min_j:
    #         reduced_p airs.append((i, max_j))
    reduced = {}
    for i, j in pairs:
        if (i not in reduced) or (j < reduced[i]):
            reduced[i] = j
    capped = list(reduced.items())
    print(f"[Retime] Reduced {len(pairs)} pairs to {len(capped)} boundary pairs")

    try:
        # linear=False uses binary variables to enforce mutual exclusion for collision pairs
        # Must use reduced_pairs (not full pairs) to keep MILP tractable
        return retime_paths(path_r, path_l, colliding=capped, linear=True, min_dt=min_dt, buffer=0.0, verbose=True)
    except AssertionError:
        try:
            reduced = {}
            for i, j in pairs:
                if (i not in reduced) or (j < reduced[i]):
                    reduced[i] = j
            capped = list(reduced.items())
            return retime_paths(
                path_r,
                path_l,
                colliding=capped,
                linear=True,
                min_dt=max(min_dt * 2.0, 0.1),
                buffer=0.02,
                verbose=False,
            )
        except AssertionError:
            return None


def _discretize(times: Sequence[float], total_time: float, step_dt: float, length: int, device: torch.device) -> torch.Tensor:
    """Map monotonic times to per-tick indices using searchsorted."""
    if length == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    times_tensor = torch.as_tensor(times, dtype=torch.float32, device=device)
    if times_tensor.numel() == 0:
        times_tensor = torch.arange(length, device=device, dtype=torch.float32) * step_dt
    tick_count = max(1, math.ceil(total_time / step_dt))
    tick_times = torch.arange(0.0, (tick_count + 1) * step_dt, step_dt, device=device)
    idx = torch.searchsorted(times_tensor, tick_times, right=True) - 1
    idx = idx.clamp(min=0, max=length - 1)
    return idx


def build_collision_aware_schedule(
    arm_right: ArmPath,
    arm_left: ArmPath,
    planner_right,
    planner_left,
    *,
    step_dt: float,
    densify_factor: int = 4,
    pair_batch: int = 4096,
    collision_margin: float = 0.01,
    min_dt: float | None = None,
) -> DiscreteSchedule:
    """Build a discrete collision-aware schedule for both arm trajectories."""
    print("[Scheduling] Computing collision pairs...")
    print(f"  Right joints: {arm_right.joint_positions.shape}, device: {arm_right.joint_positions.device}")
    print(f"  Left joints: {arm_left.joint_positions.shape}, device: {arm_left.joint_positions.device}")
    # Debug: check if joints are actually changing throughout trajectory
    # First 14 joints are typically shared (torso), last 7 are arm-specific
    if arm_right.joint_positions.shape[0] > 0 and arm_left.joint_positions.shape[0] > 0:
        r0 = arm_right.joint_positions[0].cpu().numpy()
        l0 = arm_left.joint_positions[0].cpu().numpy()
        print(f"  Right[0] (shared): {r0[:6]}, (arm): {r0[-7:]}")
        print(f"  Left[0] (shared): {l0[:6]}, (arm): {l0[-7:]}")
        # Check middle and end of trajectory
        mid = arm_right.joint_positions.shape[0] // 2
        r_mid = arm_right.joint_positions[mid].cpu().numpy()
        l_mid = arm_left.joint_positions[mid].cpu().numpy()
        r_last = arm_right.joint_positions[-1].cpu().numpy()
        l_last = arm_left.joint_positions[-1].cpu().numpy()
        print(f"  Right[{mid}] (shared): {r_mid[:6]}, (arm): {r_mid[-7:]}")
        print(f"  Left[{mid}] (shared): {l_mid[:6]}, (arm): {l_mid[-7:]}")
        print(f"  Right[-1] (shared): {r_last[:6]}, (arm): {r_last[-7:]}")
        print(f"  Left[-1] (shared): {l_last[:6]}, (arm): {l_last[-7:]}")
        # Check for all-zeros (indicates broken projection)
        right_all_zeros = (arm_right.joint_positions.abs().sum().item() < 1e-6)
        left_all_zeros = (arm_left.joint_positions.abs().sum().item() < 1e-6)
        if right_all_zeros:
            print("  WARNING: Right arm joint positions are ALL ZEROS!")
        if left_all_zeros:
            print("  WARNING: Left arm joint positions are ALL ZEROS!")

    joint_pairs = _compute_collision_pairs(
        joint_path_r=arm_right.joint_positions,
        joint_path_l=arm_left.joint_positions,
        kin_right=planner_right.motion_gen.kinematics,
        kin_left=planner_left.motion_gen.kinematics,
        densify_factor=densify_factor,
        pair_batch=pair_batch,
        collision_margin=collision_margin,
    )
    print(f"[Scheduling] Found {len(joint_pairs)} collision pairs")

    len_r = int(arm_right.joint_positions.shape[0])
    len_l = int(arm_left.joint_positions.shape[0])
    base_dt = min_dt if min_dt is not None else max(step_dt, 1e-3)

    times_r: list[float]
    times_l: list[float]

    if joint_pairs:
        retimed = _retime(len_r, len_l, joint_pairs, base_dt)
        if retimed is not None:
            times_r, times_l = retimed
            # Debug: check if retiming actually shifted the times
            print("[Scheduling] Retiming result:")
            print(f"  times_r first 5: {times_r[:5]}")
            print(f"  times_l first 5: {times_l[:5]}")
            print(f"  times_r last 5: {times_r[-5:]}")
            print(f"  times_l last 5: {times_l[-5:]}")
            # Check if times are sequential (no delay applied)
            n_check = min(10, len(times_r) - 1)
            r_deltas = [times_r[i + 1] - times_r[i] for i in range(n_check)]
            l_deltas = [times_l[i + 1] - times_l[i] for i in range(n_check)]
            print(f"  Right time deltas (first 10): {r_deltas}")
            print(f"  Left time deltas (first 10): {l_deltas}")
        else:
            print("[Scheduling] WARNING: Retiming returned None, using sequential times")
            times_r = [i * base_dt for i in range(len_r)]
            times_l = [i * base_dt for i in range(len_l)]
    else:
        times_r = [i * base_dt for i in range(len_r)]
        times_l = [i * base_dt for i in range(len_l)]

    total_time = max(times_r[-1], times_l[-1]) if times_r and times_l else max(len_r, len_l) * base_dt
    print(f"[Scheduling] Total time: {total_time:.3f}, step_dt: {step_dt:.4f}")
    idx_r = _discretize(times_r, total_time, step_dt, len_r, device=arm_right.joint_positions.device)
    idx_l = _discretize(times_l, total_time, step_dt, len_l, device=arm_left.joint_positions.device)
    print(f"[Scheduling] Discretized: idx_r[:10]={idx_r[:10].tolist()}, idx_l[:10]={idx_l[:10].tolist()}")

    return DiscreteSchedule(left_indices=idx_l, right_indices=idx_r, total_time=total_time, step_dt=step_dt)