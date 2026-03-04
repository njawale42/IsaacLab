"""
Utilities for building collision-aware schedules over recorded arm trajectories.
"""

from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass
from typing import Sequence

import torch
from nvplan.applications.custream.retime import retime_paths


# Global debug settings
_COLLISION_DEBUG_ENABLED = os.environ.get("COLLISION_DEBUG", "0") == "1"
_COLLISION_DEBUG_PATH = os.environ.get("COLLISION_DEBUG_PATH", "/tmp/collision_debug.pkl")


@dataclass
class ArmPath:
    """Container describing the recorded trajectory for a single arm."""

    name: str
    poses: torch.Tensor
    gripper_actions: torch.Tensor
    joint_positions: torch.Tensor
    subtask_boundaries: dict[int, tuple[int, int]] | None = None
    """Maps subtask_index -> (start_idx, end_idx) within the concatenated trajectory.
    Includes both MP transition and skill waypoints for each subtask."""


@dataclass
class HoldConstraint:
    """
    Describes a sequential hold constraint for one arm.
    
    The holding arm must wait at `hold_start_idx` until the other arm
    completes at least `wait_for_steps` waypoints, then continues.
    
    Attributes:
        holding_arm: Name of the arm that holds ("left" or "right").
        other_arm: Name of the arm being waited on.
        hold_start_idx: Waypoint index where hold begins (pre-hold length).
        hold_duration: Number of hold waypoints inserted.
        other_arm_len: Total length of the other arm's trajectory.
    """
    
    holding_arm: str
    other_arm: str
    hold_start_idx: int
    hold_duration: int
    other_arm_len: int


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
    n_shared_joints: int = 6,
    debug_save_path: str | None = None,
) -> list[tuple[int, int, float]]:
    """Find colliding waypoint pairs using sphere overlaps.
    
    Returns list of (r_idx, l_idx, penetration) tuples, where penetration is negative for actual collisions.
    
    Synchronizes shared torso joints so both arms compute spheres in consistent coordinates.
    
    Set env COLLISION_DEBUG=1 to enable debug visualization data saving.
    Set env COLLISION_DEBUG_PATH to specify save path (default: /tmp/collision_debug.pkl).
    """
    if joint_path_r.numel() == 0 or joint_path_l.numel() == 0:
        return []

    dev = joint_path_r.device
    pos_r_dense, map_r = _densify_path(joint_path_r, densify_factor)
    pos_l_dense, map_l = _densify_path(joint_path_l, densify_factor)

    Nr = int(pos_r_dense.shape[0])
    Nl = int(pos_l_dense.shape[0])
    if Nr == 0 or Nl == 0:
        return []

    # Get link info for debug
    def get_link_info(kin):
        kin_config = kin.kinematics_config
        link_sphere_idx_map = kin_config.link_sphere_idx_map.cpu().numpy()
        link_name_to_idx = kin_config.link_name_to_idx_map
        idx_to_name = {v: k for k, v in link_name_to_idx.items()}
        return link_sphere_idx_map, idx_to_name

    link_idx_map_r, idx_to_name_r = get_link_info(kin_right)
    link_idx_map_l, idx_to_name_l = get_link_info(kin_left)

    q_r = torch.repeat_interleave(pos_r_dense, repeats=Nl, dim=0)
    q_l = pos_l_dense.repeat(Nr, 1)
    total_rows = q_r.shape[0]

    # CRITICAL: Synchronize shared joints so both arms use the same torso configuration
    if n_shared_joints > 0:
        q_l = q_l.clone()
        q_l[:, :n_shared_joints] = q_r[:, :n_shared_joints]
    
    print(f"[Collision] Synchronized {n_shared_joints} shared joints, checking {total_rows} pairs")

    # Debug: Print sample joint values and resulting sphere positions
    if _COLLISION_DEBUG_ENABLED:
        print("[Collision Debug] Verifying joint->sphere mapping...")
        # Get expected device/dtype from kinematics (tensor_args is on CudaRobotModel, not kinematics_config)
        kin_device = kin_right.tensor_args.device
        kin_dtype = kin_right.tensor_args.dtype
        print(f"  Kinematics device: {kin_device}, dtype: {kin_dtype}")
        print(f"  Input tensor device: {pos_r_dense.device}, dtype: {pos_r_dense.dtype}")

        # Check first and last positions - use batch of 2 to avoid cuRobo buffer reuse
        for name, q_path, kin in [("Right", pos_r_dense, kin_right), ("Left", pos_l_dense, kin_left)]:
            k_device = kin.tensor_args.device
            k_dtype = kin.tensor_args.dtype
            # Batch first and last together to get correct FK for both
            q_batch = torch.cat([q_path[:1], q_path[-1:]], dim=0).to(device=k_device, dtype=k_dtype)
            state = kin.get_state(q_batch)
            sph_batch = state.link_spheres_tensor.view(2, -1, 4)
            sph_first = sph_batch[0]
            sph_last = sph_batch[1]

            valid_first = sph_first[:, 3] > 0
            valid_last = sph_last[:, 3] > 0
            if valid_first.any():
                centers_first = sph_first[valid_first, :3]
                mean_first = centers_first.mean(dim=0).cpu().numpy()
                print(f"  {name} first: joints={q_batch[0, :3].cpu().numpy()}...{q_batch[0, -3:].cpu().numpy()}")
                print(f"    spheres mean={mean_first}")
            if valid_last.any():
                centers_last = sph_last[valid_last, :3]
                mean_last = centers_last.mean(dim=0).cpu().numpy()
                print(f"  {name} last: joints={q_batch[1, :3].cpu().numpy()}...{q_batch[1, -3:].cpu().numpy()}")
                print(f"    spheres mean={mean_last}")

            # Check if spheres actually differ
            if valid_first.any() and valid_last.any():
                diff = (centers_first.mean(dim=0) - centers_last.mean(dim=0)).abs().max().item()
                print(f"  {name} sphere mean change: {diff:.4f}m (should be > 0 if joints differ)")

    colliding_rows: list[tuple[int, float]] = []  # (row_idx, min_penetration)
    global_min_penetration = float('inf')  # Track min across all pairs
    min_pen_pair: tuple[int, int] | None = None  # (r_idx, l_idx) at minimum
    min_pen_spheres: tuple[torch.Tensor, torch.Tensor] | None = None  # Sphere data at minimum

    # Debug data collection
    debug_enabled = _COLLISION_DEBUG_ENABLED or debug_save_path is not None
    debug_data = {
        "sample_pairs": [],
        "min_distances": [],
        "collision_margin": collision_margin,
        "densify_factor": densify_factor,
        "Nr": Nr,
        "Nl": Nl,
        "link_idx_map_r": link_idx_map_r,
        "link_idx_map_l": link_idx_map_l,
        "idx_to_name_r": idx_to_name_r,
        "idx_to_name_l": idx_to_name_l,
    } if debug_enabled else None
    
    # Sample indices for debug visualization
    # Sample a grid of (r_idx, l_idx) combinations, not just diagonal
    debug_sample_indices: set[int] = set()
    if debug_enabled and Nr > 0 and Nl > 0:
        # Sample 5 points along each trajectory
        r_samples = [int(i * (Nr - 1) / 4) for i in range(5)]
        l_samples = [int(i * (Nl - 1) / 4) for i in range(5)]
        for r_idx in r_samples:
            for l_idx in l_samples:
                row_idx = r_idx * Nl + l_idx
                debug_sample_indices.add(row_idx)
        print(f"[Collision Debug] Sampling {len(debug_sample_indices)} grid pairs")
        print(f"[Collision Debug] R samples: {r_samples}, L samples: {l_samples}")

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

        # Filter out spheres with radius <= 0 (disabled spheres)
        valid_r = r_r > 0
        valid_l = r_l > 0

        aa = (c_r * c_r).sum(dim=-1, keepdim=True)
        bb = (c_l * c_l).sum(dim=-1).unsqueeze(1)
        ab = torch.bmm(c_r, c_l.transpose(1, 2))
        dist2 = torch.clamp(aa + bb - 2.0 * ab, min=0.0)

        radii = r_r.unsqueeze(-1) + r_l.unsqueeze(-2) + collision_margin
        valid_pairs = valid_r.unsqueeze(-1) & valid_l.unsqueeze(-2)
        
        # Compute penetration for all pairs (negative = collision)
        dist = torch.sqrt(dist2)
        penetration = dist - radii
        penetration_valid = torch.where(valid_pairs, penetration, torch.tensor(float('inf'), device=dev))
        
        # Find min penetration per batch item
        batch_pen_flat = penetration_valid.view(batch_size, -1)
        min_pen_per_item, _ = batch_pen_flat.min(dim=1)  # [batch_size]
        
        # Collect colliding rows with their penetration depths
        collide = min_pen_per_item <= 0  # Collision if penetration <= 0
        rows = torch.nonzero(collide, as_tuple=False).flatten()
        if rows.numel() > 0:
            for idx in rows:
                local_idx = int(idx.item())
                global_idx = start + local_idx
                pen_value = float(min_pen_per_item[local_idx].item())
                colliding_rows.append((global_idx, pen_value))

        # Find global min penetration in this batch (reuse min_pen_per_item computed above)
        overall_batch_min_idx = int(min_pen_per_item.argmin().item())
        batch_min = float(min_pen_per_item[overall_batch_min_idx].item())

        if batch_min < global_min_penetration:
            global_min_penetration = batch_min
            # Identify which (r_idx, l_idx) pair this is
            global_row = start + overall_batch_min_idx
            i_dense = global_row // Nl
            j_dense = global_row % Nl
            min_pen_pair = (int(map_r[i_dense].item()), int(map_l[j_dense].item()))
            min_pen_spheres = (sph_r[overall_batch_min_idx].clone(), sph_l[overall_batch_min_idx].clone())

        # Save debug samples
        if debug_data is not None:
            for local_idx in range(batch_size):
                global_idx = start + local_idx
                if global_idx in debug_sample_indices:
                    i_dense = global_idx // Nl
                    j_dense = global_idx % Nl

                    # Get min distance for this pair
                    d2 = dist2[local_idx]
                    r_sum = radii[local_idx]
                    valid = valid_pairs[local_idx]

                    # Compute penetration depth (negative = collision)
                    dist = torch.sqrt(d2)
                    penetration = dist - r_sum
                    penetration_masked = torch.where(valid, penetration, torch.tensor(float('inf'), device=dev))
                    min_pen = penetration_masked.min().item()

                    debug_data["sample_pairs"].append({
                        "r_idx": int(map_r[i_dense].item()),
                        "l_idx": int(map_l[j_dense].item()),
                        "spheres_r": sph_r[local_idx].cpu().numpy(),
                        "spheres_l": sph_l[local_idx].cpu().numpy(),
                        "min_penetration": min_pen,
                        "is_colliding": min_pen <= 0,
                    })
                    debug_data["min_distances"].append(min_pen)

    print(f"[Collision] Found {len(colliding_rows)} colliding pairs out of {total_rows}")
    print(f"[Collision] Global min penetration: {global_min_penetration:.4f}m (negative=collision, margin={collision_margin})")
    if min_pen_pair is not None:
        print(f"[Collision] Closest approach at R[{min_pen_pair[0]}] vs L[{min_pen_pair[1]}]")
        if min_pen_spheres is not None:
            sph_r_min, sph_l_min = min_pen_spheres
            valid_r = sph_r_min[:, 3] > 0
            valid_l = sph_l_min[:, 3] > 0
            if valid_r.any() and valid_l.any():
                c_r = sph_r_min[valid_r, :3]
                c_l = sph_l_min[valid_l, :3]
                print(f"[Collision] Right spheres center mean: {c_r.mean(dim=0).cpu().numpy()}")
                print(f"[Collision] Left spheres center mean: {c_l.mean(dim=0).cpu().numpy()}")

    # Add the minimum distance pair to debug data
    if debug_data is not None and min_pen_pair is not None and min_pen_spheres is not None:
        debug_data["min_pair"] = {
            "r_idx": min_pen_pair[0],
            "l_idx": min_pen_pair[1],
            "spheres_r": min_pen_spheres[0].cpu().numpy(),
            "spheres_l": min_pen_spheres[1].cpu().numpy(),
            "min_penetration": global_min_penetration,
            "is_colliding": global_min_penetration <= 0,
        }
        # Also add to sample_pairs so it's visualized
        debug_data["sample_pairs"].append(debug_data["min_pair"])
        debug_data["min_distances"].append(global_min_penetration)

    # Save debug data
    if debug_data is not None and debug_data["sample_pairs"]:
        save_path = debug_save_path or _COLLISION_DEBUG_PATH
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(debug_data, f)
        print(f"[Collision Debug] Saved {len(debug_data['sample_pairs'])} samples to {save_path}")
        print(f"[Collision Debug] Min penetration range: [{min(debug_data['min_distances']):.4f}, {max(debug_data['min_distances']):.4f}]")
        print(f"[Collision Debug] Run: python -m isaaclab_mimic.datagen.visualize_collision_debug {save_path}")
    
    if not colliding_rows:
        return []

    pairs: list[tuple[int, int, float]] = []
    for row_idx, pen_value in colliding_rows:
        i_dense = row_idx // Nl
        j_dense = row_idx % Nl
        pairs.append((int(map_r[i_dense].item()), int(map_l[j_dense].item()), pen_value))
    return pairs


def _reduce_pairs_smart(
    pairs: list[tuple[int, int, float]], 
    top_k_worst: int = 20,
) -> list[tuple[int, int]]:
    """Reduce collision pairs to a tractable set for MILP.
    
    Strategy:
    1. Keep boundary pairs (min and max j for each i) - these define collision windows
    2. Keep top K pairs with worst (most negative) penetration - these are critical collisions
    
    Returns pairs without penetration values (just (i, j) tuples) for MILP.
    """
    if not pairs:
        return []
    
    # Build boundary map: for each i, track min_j and max_j
    boundary_map: dict[int, tuple[int, int]] = {}  # i -> (min_j, max_j)
    for i, j, _ in pairs:
        if i not in boundary_map:
            boundary_map[i] = (j, j)
        else:
            cur_min, cur_max = boundary_map[i]
            boundary_map[i] = (min(j, cur_min), max(j, cur_max))
    
    # Collect boundary pairs
    reduced_set: set[tuple[int, int]] = set()
    for i, (min_j, max_j) in boundary_map.items():
        reduced_set.add((i, min_j))
        if max_j != min_j:
            reduced_set.add((i, max_j))
    
    # Sort by penetration (most negative first = worst collisions)
    sorted_by_pen = sorted(pairs, key=lambda x: x[2])
    
    # Add top K worst penetration pairs
    for i, j, pen in sorted_by_pen[:top_k_worst]:
        reduced_set.add((i, j))
    
    result = list(reduced_set)
    print(f"[Retime] Smart reduction: {len(pairs)} -> {len(result)} pairs")
    print(f"  Boundary pairs: {len(boundary_map) * 2} (min/max j per i)")
    print(f"  Worst penetration: {sorted_by_pen[0][2]:.4f}m at ({sorted_by_pen[0][0]}, {sorted_by_pen[0][1]})")
    return result


def _retime(
    len_r: int, 
    len_l: int, 
    pairs: list[tuple[int, int, float]], 
    min_dt: float,
) -> tuple[list[float], list[float]] | None:
    """Solve MILP retiming for both arms.
    
    Args:
        pairs: List of (r_idx, l_idx, penetration) tuples. Penetration < 0 means collision.
    """
    if len_r == 0 or len_l == 0:
        return None
    if not pairs:
        return None

    path_r = [None] * len_r
    path_l = [None] * len_l
    
    print(f"[Retime] len_r={len_r}, len_l={len_l}, pairs={len(pairs)}, min_dt={min_dt:.4f}")
    
    # Extract just (i, j) for MILP (strip penetration)
    pairs_ij = [(i, j) for i, j, _ in pairs]
    
    # Debug: show sample collision pairs
    r_indices = [p[0] for p in pairs]
    l_indices = [p[1] for p in pairs]
    print(f"[Retime] Right indices range: [{min(r_indices)}, {max(r_indices)}]")
    print(f"[Retime] Left indices range: [{min(l_indices)}, {max(l_indices)}]")
    
    # First try with all pairs
    # synchronize=False allows different trajectory lengths
    try:
        result = retime_paths(
            path_r, path_l,
            colliding=pairs_ij,
            linear=True,
            min_dt=min_dt,
            buffer=0.03,
            synchronize=False,
            verbose=True,
            max_time=10.0,
        )
        if result is not None:
            return result
        print("[Retime] Full pairs failed, trying smart reduction...")
    except (TypeError, AssertionError, Exception) as e:
        print(f"[Retime] Full pairs error: {e}")

    # Second try with smart-reduced pairs (boundary + worst penetration)
    reduced = _reduce_pairs_smart(pairs, top_k_worst=30)
    try:
        result = retime_paths(
            path_r, path_l,
            colliding=reduced,
            linear=True,
            min_dt=min_dt,
            buffer=0.01,
            synchronize=False,
            verbose=True,
            max_time=15.0,
        )
        if result is not None:
            return result
    except (TypeError, AssertionError, Exception) as e:
        print(f"[Retime] Smart reduction error: {e}")

    # Third try with buffer (more relaxed constraints)
    try:
        result = retime_paths(
            path_r, path_l,
            colliding=reduced,
            linear=True,
            min_dt=min_dt * 0.5,
            buffer=0.01,
            synchronize=False,
            verbose=True,
            max_time=20.0,
        )
        if result is not None:
            return result
        print("[Retime] Relaxed attempt also failed")
    except (TypeError, AssertionError, Exception) as e2:
        print(f"[Retime] Relaxed attempt error: {e2}")

    return None


def _discretize(
    times: Sequence[float],
    total_time: float,
    step_dt: float,
    length: int,
    device: torch.device,
    gripper_actions: torch.Tensor | None = None,
    gripper_change_threshold: float = 0.1,
) -> torch.Tensor:
    """Map monotonic times to per-tick indices, preserving gripper transitions.

    Basic discretization uses searchsorted to map sim ticks to waypoint indices.
    If gripper_actions is provided, ensures indices with significant gripper
    changes are never skipped (critical for grasp/release commands).
    """
    if length == 0:
        return torch.zeros(0, dtype=torch.long, device=device)

    times_tensor = torch.as_tensor(times, dtype=torch.float32, device=device)
    if times_tensor.numel() == 0:
        times_tensor = torch.arange(length, device=device, dtype=torch.float32) * step_dt
    tick_count = max(1, math.ceil(total_time / step_dt))
    tick_times = torch.arange(0.0, (tick_count + 1) * step_dt, step_dt, device=device)
    idx = torch.searchsorted(times_tensor, tick_times, right=True) - 1
    idx = idx.clamp(min=0, max=length - 1)

    # Post-process: ensure no gripper transitions are skipped
    if gripper_actions is not None and gripper_actions.numel() > 0:
        gripper = gripper_actions.to(device=device, dtype=torch.float32)
        # Compute gripper change magnitude at each waypoint
        gripper_delta = torch.zeros(length, device=device)
        if length > 1:
            for i in range(1, length):
                gripper_delta[i] = (gripper[i] - gripper[i - 1]).abs().max()

        # Find indices with significant gripper changes
        critical_indices = (gripper_delta > gripper_change_threshold).nonzero(as_tuple=True)[0]

        # Ensure each critical index appears in the output
        for crit_idx in critical_indices:
            crit_idx_val = int(crit_idx.item())
            # Check if this index is present in the discretized output
            if not (idx == crit_idx_val).any():
                # Find the tick where we should insert this index
                # (the tick just before we would have skipped past it)
                for tick in range(len(idx) - 1):
                    if idx[tick] < crit_idx_val <= idx[tick + 1]:
                        # Insert the critical index at this tick
                        idx[tick + 1] = crit_idx_val
                        break

    return idx


def _apply_hold_constraint_timing(
    hold: HoldConstraint,
    len_r: int,
    len_l: int,
    base_dt: float,
) -> tuple[list[float], list[float]]:
    """
    Compute timing that respects a sequential hold constraint.
    
    Online behavior (which we replicate):
    1. Both arms START simultaneously at time 0
    2. LATTER plays waypoints 0 to (hold_start_idx - 1) at normal speed
    3. When LATTER reaches hold_start_idx, it HOLDS until FORMER completes
    4. Then LATTER finishes remaining waypoints (post-hold)
    
    The hold waypoints represent the time LATTER stays in place while FORMER catches up.
    
    Returns (times_r, times_l) with proper hold timing.
    """
    hold_end_idx = hold.hold_start_idx + hold.hold_duration
    post_hold_count = len_r - hold_end_idx if hold.holding_arm == "right" else len_l - hold_end_idx
    
    # FORMER end time
    former_end_time = (hold.other_arm_len - 1) * base_dt
    
    print(f"[Hold Timing] LATTER ({hold.holding_arm}) holds at idx {hold.hold_start_idx}, FORMER ends at {former_end_time:.2f}s")
    print(f"[Hold Timing] hold_duration: {hold.hold_duration}, post_hold_count: {post_hold_count}")
    
    if hold.holding_arm == "right":
        # Left (FORMER) plays continuously from time 0
        times_l = [i * base_dt for i in range(len_l)]
        times_r = []
        
        # Pre-hold (0 to hold_start_idx-1): Right plays at normal speed, starting at time 0
        for i in range(hold.hold_start_idx):
            times_r.append(i * base_dt)
        
        pre_hold_end_time = hold.hold_start_idx * base_dt
        
        # Hold phase: Right stays at hold_start_idx until left completes
        # The hold waypoints span from pre_hold_end_time to former_end_time
        if pre_hold_end_time >= former_end_time:
            # Left already done when right reaches hold position - no waiting needed
            # Compress hold waypoints (but still need to play them at increasing times)
            for i in range(hold.hold_duration):
                times_r.append(pre_hold_end_time + i * base_dt)
            actual_hold_end_time = pre_hold_end_time + (hold.hold_duration - 1) * base_dt if hold.hold_duration > 0 else pre_hold_end_time
        else:
            # Right holds while left catches up
            # Spread hold waypoints from pre_hold_end_time to former_end_time
            hold_time_span = former_end_time - pre_hold_end_time
            if hold.hold_duration > 1:
                hold_dt = hold_time_span / (hold.hold_duration - 1)
            else:
                hold_dt = 0.0
            for i in range(hold.hold_duration):
                times_r.append(pre_hold_end_time + i * hold_dt)
            actual_hold_end_time = former_end_time
        
        # Post-hold: Right resumes after left finishes
        for i in range(post_hold_count):
            times_r.append(actual_hold_end_time + (i + 1) * base_dt)
            
    else:
        # Right (FORMER) plays continuously from time 0
        times_r = [i * base_dt for i in range(len_r)]
        times_l = []
        
        # Pre-hold: Left plays at normal speed, starting at time 0
        for i in range(hold.hold_start_idx):
            times_l.append(i * base_dt)
        
        pre_hold_end_time = hold.hold_start_idx * base_dt
        
        if pre_hold_end_time >= former_end_time:
            for i in range(hold.hold_duration):
                times_l.append(pre_hold_end_time + i * base_dt)
            actual_hold_end_time = pre_hold_end_time + (hold.hold_duration - 1) * base_dt if hold.hold_duration > 0 else pre_hold_end_time
        else:
            hold_time_span = former_end_time - pre_hold_end_time
            if hold.hold_duration > 1:
                hold_dt = hold_time_span / (hold.hold_duration - 1)
            else:
                hold_dt = 0.0
            for i in range(hold.hold_duration):
                times_l.append(pre_hold_end_time + i * hold_dt)
            actual_hold_end_time = former_end_time
        
        for i in range(post_hold_count):
            times_l.append(actual_hold_end_time + (i + 1) * base_dt)
    
    return times_r, times_l


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
    hold_constraints: list[HoldConstraint] | None = None,
) -> DiscreteSchedule:
    """Build a discrete collision-aware schedule for both arm trajectories.
    
    Args:
        arm_right: Right arm trajectory.
        arm_left: Left arm trajectory.
        planner_right: Right arm motion planner (for kinematics).
        planner_left: Left arm motion planner (for kinematics).
        step_dt: Simulation step duration.
        densify_factor: Factor for densifying paths during collision checking.
        pair_batch: Batch size for collision checking.
        collision_margin: Collision margin in meters.
        min_dt: Minimum time delta between waypoints.
        hold_constraints: List of hold constraints from sequential subtask ordering.
    """
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
            # Show hold phase timing if hold constraints exist
            if hold_constraints:
                hold = hold_constraints[0]
                hold_end_idx = hold.hold_start_idx + hold.hold_duration
                print(f"[Scheduling] MILP scheduled hold phase (hold waypoints {hold.hold_start_idx} to {hold_end_idx-1}):")
                print(f"  times_r around hold start: {times_r[max(0, hold.hold_start_idx-2):hold.hold_start_idx+5]}")
                print(f"  times_r around hold end: {times_r[max(0, hold_end_idx-5):min(len_r, hold_end_idx+3)]}")
        elif hold_constraints:
            # MILP failed but we have hold constraints - apply hold timing as fallback
            print("[Scheduling] WARNING: Retiming returned None, applying hold constraints")
            hold = hold_constraints[0]
            times_r, times_l = _apply_hold_constraint_timing(hold, len_r, len_l, base_dt)
        else:
            print("[Scheduling] WARNING: Retiming returned None, using sequential times")
            times_r = [i * base_dt for i in range(len_r)]
            times_l = [i * base_dt for i in range(len_l)]
    elif hold_constraints:
        # No collisions but we have hold constraints - apply hold timing
        print(f"[Scheduling] No collisions, applying {len(hold_constraints)} hold constraint(s)")
        # Use the first hold constraint (typically only one for sequential tasks)
        hold = hold_constraints[0]
        print(f"  Hold: {hold.holding_arm} holds at idx {hold.hold_start_idx} for {hold.hold_duration} steps")
        print(f"  Other arm ({hold.other_arm}) length: {hold.other_arm_len}")
        times_r, times_l = _apply_hold_constraint_timing(hold, len_r, len_l, base_dt)
        print(f"[Scheduling] Hold timing applied:")
        print(f"  times_r first 5: {times_r[:5]}")
        print(f"  times_l first 5: {times_l[:5]}")
        print(f"  times_r around hold: {times_r[max(0, hold.hold_start_idx-2):hold.hold_start_idx+hold.hold_duration+2]}")
        print(f"  times_r last 5: {times_r[-5:]}")
        print(f"  times_l last 5: {times_l[-5:]}")
    else:
        times_r = [i * base_dt for i in range(len_r)]
        times_l = [i * base_dt for i in range(len_l)]

    total_time = max(times_r[-1], times_l[-1]) if times_r and times_l else max(len_r, len_l) * base_dt
    print(f"[Scheduling] Total time: {total_time:.3f}, step_dt: {step_dt:.4f}")

    # Pass gripper actions to ensure transitions are preserved during discretization
    idx_r = _discretize(
        times_r, total_time, step_dt, len_r,
        device=arm_right.joint_positions.device,
        gripper_actions=arm_right.gripper_actions,
    )
    idx_l = _discretize(
        times_l, total_time, step_dt, len_l,
        device=arm_left.joint_positions.device,
        gripper_actions=arm_left.gripper_actions,
    )
    print(f"[Scheduling] Discretized: idx_r[:10]={idx_r[:10].tolist()}, idx_l[:10]={idx_l[:10].tolist()}")
    # Show more of right arm to verify delayed start
    if len(idx_r) > 300:
        print(f"[Scheduling] idx_r[280:300]={idx_r[280:300].tolist()} (checking for delayed start)")
    print(f"[Scheduling] idx_r[-10:]={idx_r[-10:].tolist()}, idx_l[-10:]={idx_l[-10:].tolist()}")

    return DiscreteSchedule(left_indices=idx_l, right_indices=idx_r, total_time=total_time, step_dt=step_dt)
