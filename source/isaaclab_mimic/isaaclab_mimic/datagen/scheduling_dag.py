"""
DAG-based segment-level scheduler for bimanual collision-aware trajectory playback.

Instead of assigning a continuous time to every individual waypoint (as in scheduling.py),
this module treats each subtask block as an atomic unit and solves a smaller MILP that
decides the *ordering* between cross-arm blocks that would collide. Within each block,
waypoints play out at uniform speed.

Trade-offs vs. the waypoint-level retiming in scheduling.py:
  + Much smaller MILP (O(num_blocks) variables vs O(num_waypoints)).
  + Faster solve times.
  - Cannot slow down mid-block to thread through narrow collision windows.
  - Hold constraints are ordering edges between sub-blocks, not physical waypoint injections.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaaclab_mimic.datagen.scheduling import (
    ArmPath,
    DiscreteSchedule,
    HoldConstraint,
    _discretize,  # pyright: ignore[reportPrivateUsage]
)


@dataclass(eq=False)
class SubtaskBlock:
    """A contiguous block of waypoints for one subtask (or sub-block) on one arm."""

    arm_name: str
    subtask_index: int
    start_idx: int
    end_idx: int  # exclusive
    joint_positions: torch.Tensor  # [block_len, num_joints]

    @property
    def duration(self) -> int:
        return self.end_idx - self.start_idx

    @property
    def var_name(self) -> str:
        return f"{self.arm_name}_{self.start_idx}"

    def __repr__(self) -> str:
        return f"{self.arm_name}[st{self.subtask_index}:{self.start_idx}-{self.end_idx}]"


# ---------------------------------------------------------------------------
# Block extraction and splitting
# ---------------------------------------------------------------------------

def _extract_blocks(arm_path: ArmPath) -> list[SubtaskBlock]:
    """Extract subtask blocks from an ArmPath using its subtask_boundaries."""
    if arm_path.subtask_boundaries is None or len(arm_path.subtask_boundaries) == 0:
        return [
            SubtaskBlock(
                arm_name=arm_path.name,
                subtask_index=0,
                start_idx=0,
                end_idx=int(arm_path.joint_positions.shape[0]),
                joint_positions=arm_path.joint_positions,
            )
        ]

    blocks: list[SubtaskBlock] = []
    for idx in sorted(arm_path.subtask_boundaries.keys()):
        start, end = arm_path.subtask_boundaries[idx]
        blocks.append(
            SubtaskBlock(
                arm_name=arm_path.name,
                subtask_index=idx,
                start_idx=start,
                end_idx=end,
                joint_positions=arm_path.joint_positions[start:end],
            )
        )
    return blocks


def _split_blocks_at_holds(
    blocks_r: list[SubtaskBlock],
    blocks_l: list[SubtaskBlock],
    hold_constraints: list[HoldConstraint] | None,
    arm_right: ArmPath,
    arm_left: ArmPath,
) -> tuple[list[SubtaskBlock], list[SubtaskBlock], list[tuple[SubtaskBlock, SubtaskBlock]]]:
    """Split blocks at hold points so pre-hold portions can run concurrently.

    A hold constraint says: "LATTER arm can play until hold_start_idx, then must
    pause until FORMER arm reaches other_arm_len."  We split the LATTER block at
    hold_start_idx into a pre-hold sub-block (no ordering constraint) and a
    post-hold sub-block (must wait for the FORMER block to finish).

    Returns:
        (blocks_r, blocks_l, hold_orderings) where hold_orderings is a list of
        (former_block, post_hold_block) pairs to add as hard ordering edges.
    """
    if not hold_constraints:
        return blocks_r, blocks_l, []

    blocks_r = list(blocks_r)
    blocks_l = list(blocks_l)
    hold_orderings: list[tuple[SubtaskBlock, SubtaskBlock]] = []

    for hold in hold_constraints:
        latter_blocks = blocks_r if hold.holding_arm == "right" else blocks_l
        latter_path = arm_right if hold.holding_arm == "right" else arm_left
        former_blocks = blocks_l if hold.other_arm == "left" else blocks_r

        # Find the LATTER block containing hold_start_idx
        target_idx: int | None = None
        for i, block in enumerate(latter_blocks):
            if block.start_idx <= hold.hold_start_idx < block.end_idx:
                target_idx = i
                break
        if target_idx is None:
            continue

        block = latter_blocks[target_idx]
        split_at = hold.hold_start_idx

        # Split only if the hold point is strictly inside the block
        if split_at > block.start_idx:
            pre_hold = SubtaskBlock(
                arm_name=block.arm_name,
                subtask_index=block.subtask_index,
                start_idx=block.start_idx,
                end_idx=split_at,
                joint_positions=latter_path.joint_positions[block.start_idx:split_at],
            )
            post_hold = SubtaskBlock(
                arm_name=block.arm_name,
                subtask_index=block.subtask_index,
                start_idx=split_at,
                end_idx=block.end_idx,
                joint_positions=latter_path.joint_positions[split_at:block.end_idx],
            )
            latter_blocks[target_idx:target_idx + 1] = [pre_hold, post_hold]
            post_hold_block = post_hold
            print(
                f"[DAG] Split {block.arm_name}[{block.subtask_index}] at idx {split_at}: "
                f"pre-hold={pre_hold.duration}wp, post-hold={post_hold.duration}wp"
            )
        else:
            post_hold_block = block

        # Find the FORMER block that must complete before the post-hold can start
        former_block: SubtaskBlock | None = None
        for fb in former_blocks:
            if fb.start_idx < hold.other_arm_len <= fb.end_idx:
                former_block = fb
                break
        if former_block is None and former_blocks:
            former_block = former_blocks[-1]
        if former_block is None:
            continue

        hold_orderings.append((former_block, post_hold_block))
        print(f"[DAG] Hold ordering: {former_block} -> {post_hold_block}")

    return blocks_r, blocks_l, hold_orderings


# ---------------------------------------------------------------------------
# Block-level collision check (sphere overlap)
# ---------------------------------------------------------------------------

@dataclass
class BlockCollisionResult:
    """Diagnostics from a block-pair collision check."""

    collides: bool
    n_colliding_pairs: int
    total_pairs: int
    min_penetration: float  # most negative = deepest collision
    closest_r_idx: int  # waypoint index within block (local)
    closest_l_idx: int


def _blocks_collide(
    joints_r: torch.Tensor,
    joints_l: torch.Tensor,
    kin_right,
    kin_left,
    collision_margin: float = 0.0,
    n_shared_joints: int = 6,
    batch_size: int = 4096,
) -> BlockCollisionResult:
    """Check collisions between two joint-position arrays and return diagnostics."""
    Nr = int(joints_r.shape[0])
    Nl = int(joints_l.shape[0])
    if Nr == 0 or Nl == 0:
        return BlockCollisionResult(False, 0, 0, float("inf"), -1, -1)

    dev = joints_r.device
    q_r = joints_r.repeat_interleave(Nl, dim=0)
    q_l = joints_l.repeat(Nr, 1)
    total = q_r.shape[0]

    if n_shared_joints > 0:
        q_l = q_l.clone()
        q_l[:, :n_shared_joints] = q_r[:, :n_shared_joints]

    n_colliding = 0
    global_min_pen = float("inf")
    best_r_idx = -1
    best_l_idx = -1

    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        state_r = kin_right.get_state(q_r[start:end])
        state_l = kin_left.get_state(q_l[start:end])
        bs = end - start

        sph_r = state_r.link_spheres_tensor.view(bs, -1, 4)
        sph_l = state_l.link_spheres_tensor.view(bs, -1, 4)

        c_r, r_r = sph_r[..., :3], sph_r[..., 3]
        c_l, r_l = sph_l[..., :3], sph_l[..., 3]

        valid_r = r_r > 0
        valid_l = r_l > 0

        aa = (c_r * c_r).sum(dim=-1, keepdim=True)
        bb = (c_l * c_l).sum(dim=-1).unsqueeze(1)
        ab = torch.bmm(c_r, c_l.transpose(1, 2))
        dist2 = torch.clamp(aa + bb - 2.0 * ab, min=0.0)

        radii = r_r.unsqueeze(-1) + r_l.unsqueeze(-2) + collision_margin
        valid_pairs = valid_r.unsqueeze(-1) & valid_l.unsqueeze(-2)

        penetration = torch.sqrt(dist2) - radii
        penetration = torch.where(valid_pairs, penetration, torch.tensor(float("inf"), device=dev))

        pen_flat = penetration.view(bs, -1)
        min_pen_per_item, _ = pen_flat.min(dim=1)

        n_colliding += int((min_pen_per_item <= 0).sum().item())

        batch_min_idx = int(min_pen_per_item.argmin().item())
        batch_min_val = float(min_pen_per_item[batch_min_idx].item())
        if batch_min_val < global_min_pen:
            global_min_pen = batch_min_val
            global_row = start + batch_min_idx
            best_r_idx = global_row // Nl
            best_l_idx = global_row % Nl

    return BlockCollisionResult(
        collides=n_colliding > 0,
        n_colliding_pairs=n_colliding,
        total_pairs=total,
        min_penetration=global_min_pen,
        closest_r_idx=best_r_idx,
        closest_l_idx=best_l_idx,
    )


# ---------------------------------------------------------------------------
# MILP construction and solve
# ---------------------------------------------------------------------------

def _solve_block_milp(
    blocks_r: list[SubtaskBlock],
    blocks_l: list[SubtaskBlock],
    mutex_pairs: list[tuple[SubtaskBlock, SubtaskBlock]],
    hold_orderings: list[tuple[SubtaskBlock, SubtaskBlock]],
    base_dt: float,
    max_time: float = 5.0,
) -> dict[SubtaskBlock, float] | None:
    """Solve a MILP that assigns a start time to each block.

    Args:
        blocks_r / blocks_l: Ordered blocks per arm (may include sub-blocks from hold splitting).
        mutex_pairs: Cross-arm block pairs that collide.
        hold_orderings: (former_block, post_hold_block) pairs — hard ordering edges.
        base_dt: Time per waypoint step.
        max_time: Solver timeout (seconds).

    Returns a dict mapping each block to its start time, or None if infeasible.
    """
    from nvplan.common.milp import Constraint, Cost, Variable, solve_milp

    all_blocks = blocks_r + blocks_l
    if not all_blocks:
        return {}

    variables: list[Variable] = []
    constraints: list[Constraint] = []
    costs: list[Cost] = []

    makespan = Variable(name="makespan", lower=0.0)
    variables.append(makespan)
    costs.append(Cost(coefficients={makespan.name: 1.0}))

    block_vars: dict[SubtaskBlock, Variable] = {}
    for block in all_blocks:
        var = Variable(name=block.var_name, lower=0.0)
        block_vars[block] = var
        variables.append(var)

    # Intra-arm sequential: block k+1 starts after block k finishes
    for arm_blocks in (blocks_r, blocks_l):
        for i in range(len(arm_blocks) - 1):
            b1, b2 = arm_blocks[i], arm_blocks[i + 1]
            constraints.append(
                Constraint(
                    lower=b1.duration * base_dt,
                    coefficients={block_vars[b2].name: +1, block_vars[b1].name: -1},
                )
            )
        # Makespan >= last block's end time
        if arm_blocks:
            last = arm_blocks[-1]
            constraints.append(
                Constraint(
                    lower=last.duration * base_dt,
                    coefficients={makespan.name: +1, block_vars[last].name: -1},
                )
            )

    # Hold orderings: FORMER block must finish before post-hold block can start
    for former, latter in hold_orderings:
        constraints.append(
            Constraint(
                lower=former.duration * base_dt,
                coefficients={
                    block_vars[latter].name: +1,
                    block_vars[former].name: -1,
                },
            )
        )

    # Mutex disjunction (big-M)
    total_duration = sum(b.duration for b in all_blocks) * base_dt
    M = 1000.0 * max(total_duration, 1.0)

    z_vars: dict[tuple[SubtaskBlock, SubtaskBlock], Variable] = {}
    for br, bl in mutex_pairs:
        z = Variable(
            name=f"z_{br.var_name}__{bl.var_name}",
            integer=True,
            lower=0.0,
            upper=1.0,
        )
        z_vars[(br, bl)] = z
        variables.append(z)

        dur_r = br.duration * base_dt
        dur_l = bl.duration * base_dt

        # z=0 → right finishes before left starts:  t[l] >= t[r] + dur_r
        # z=1 → left finishes before right starts:   t[r] >= t[l] + dur_l
        constraints.extend(
            [
                Constraint(
                    lower=dur_r,
                    coefficients={block_vars[bl].name: +1, block_vars[br].name: -1, z.name: +M},
                ),
                Constraint(
                    lower=dur_l - M,
                    coefficients={block_vars[br].name: +1, block_vars[bl].name: -1, z.name: -M},
                ),
            ]
        )

    print(
        f"[DAG MILP] blocks_r={len(blocks_r)}, blocks_l={len(blocks_l)}, "
        f"mutexes={len(mutex_pairs)}, hold_edges={len(hold_orderings)}, "
        f"binary_vars={len(z_vars)}"
    )

    solution = solve_milp(
        variables,
        constraints=constraints,
        costs=costs,
        gap_percent=None,
        max_time=max_time,
        verbose=True,
    )
    if solution is None:
        return None

    start_times = {block: solution[block_vars[block].name] for block in all_blocks}
    print(f"[DAG MILP] Makespan: {solution[makespan.name]:.3f}")
    for block in all_blocks:
        t = start_times[block]
        print(f"  {block}: start={t:.3f}, dur={block.duration * base_dt:.3f}")

    return start_times


# ---------------------------------------------------------------------------
# Convert MILP solution → DiscreteSchedule
# ---------------------------------------------------------------------------

def _blocks_to_times(
    blocks: list[SubtaskBlock],
    start_times: dict[SubtaskBlock, float],
    total_waypoints: int,
    base_dt: float,
) -> list[float]:
    """Map each waypoint to a continuous time based on its block's start time."""
    times = [0.0] * total_waypoints
    for block in blocks:
        t0 = start_times[block]
        for j in range(block.start_idx, block.end_idx):
            times[j] = t0 + (j - block.start_idx) * base_dt
    # Ensure monotonicity (fill any unassigned leading/trailing gaps)
    for i in range(1, total_waypoints):
        if times[i] < times[i - 1]:
            times[i] = times[i - 1] + base_dt
    return times


def _build_simultaneous_schedule(
    arm_right: ArmPath,
    arm_left: ArmPath,
    base_dt: float,
    step_dt: float,
) -> DiscreteSchedule:
    """Fallback: both arms play simultaneously at uniform speed."""
    len_r = int(arm_right.joint_positions.shape[0])
    len_l = int(arm_left.joint_positions.shape[0])
    times_r = [i * base_dt for i in range(len_r)]
    times_l = [i * base_dt for i in range(len_l)]
    total_time = max(times_r[-1] if times_r else 0.0, times_l[-1] if times_l else 0.0)

    idx_r = _discretize(
        times_r, total_time, step_dt, len_r,
        arm_right.joint_positions.device, gripper_actions=arm_right.gripper_actions,
    )
    idx_l = _discretize(
        times_l, total_time, step_dt, len_l,
        arm_left.joint_positions.device, gripper_actions=arm_left.gripper_actions,
    )
    return DiscreteSchedule(left_indices=idx_l, right_indices=idx_r, total_time=total_time, step_dt=step_dt)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_dag_schedule(
    arm_right: ArmPath,
    arm_left: ArmPath,
    planner_right,
    planner_left,
    *,
    step_dt: float,
    hold_constraints: list[HoldConstraint] | None = None,
    collision_margin: float = 0.01,
    pair_batch: int = 4096,
    n_shared_joints: int = 6,
    min_dt: float | None = None,
    milp_max_time: float = 5.0,
) -> DiscreteSchedule:
    """Build a discrete schedule using DAG-based segment-level MILP ordering.

    Each subtask block is treated as an atomic unit with fixed duration.  Hold
    constraints split blocks at the hold point so the pre-hold portion can run
    concurrently while only the post-hold portion waits for the other arm.

    The MILP decides the *start time* of each (sub-)block to minimise makespan
    while avoiding inter-arm collisions and respecting hold orderings.

    Args:
        arm_right / arm_left: Arm trajectories with ``subtask_boundaries`` populated.
        planner_right / planner_left: Arm planners (for FK kinematics).
        step_dt: Simulator step duration.
        hold_constraints: Sequential ordering constraints.
        collision_margin: Sphere collision margin (m).
        pair_batch: FK batch size for collision checking.
        n_shared_joints: Number of leading shared (torso) joints to synchronise.
        min_dt: Minimum time delta between consecutive waypoints inside a block.
        milp_max_time: MILP solver timeout (seconds).
    """
    blocks_r = _extract_blocks(arm_right)
    blocks_l = _extract_blocks(arm_left)
    base_dt = min_dt if min_dt is not None else max(step_dt, 1e-3)

    # Split blocks at hold points so pre-hold portions can overlap
    blocks_r, blocks_l, hold_orderings = _split_blocks_at_holds(
        blocks_r, blocks_l, hold_constraints, arm_right, arm_left,
    )

    print(f"[DAG Schedule] Right blocks: {blocks_r}")
    print(f"[DAG Schedule] Left blocks:  {blocks_l}")

    # Check collisions between cross-arm block pairs
    kin_right = planner_right.motion_gen.kinematics
    kin_left = planner_left.motion_gen.kinematics

    mutex_pairs: list[tuple[SubtaskBlock, SubtaskBlock]] = []
    print(f"[DAG Collision] Checking {len(blocks_r)} x {len(blocks_l)} block pairs, margin={collision_margin}m")
    for br in blocks_r:
        for bl in blocks_l:
            result = _blocks_collide(
                br.joint_positions,
                bl.joint_positions,
                kin_right,
                kin_left,
                collision_margin=collision_margin,
                n_shared_joints=n_shared_joints,
                batch_size=pair_batch,
            )
            status = "COLLIDE" if result.collides else "clear"
            print(
                f"[DAG Collision] {br} x {bl}: {status} | "
                f"pairs={result.n_colliding_pairs}/{result.total_pairs} | "
                f"min_pen={result.min_penetration:.4f}m | "
                f"closest=R[{br.start_idx + result.closest_r_idx}] vs L[{bl.start_idx + result.closest_l_idx}]"
            )
            if result.collides:
                mutex_pairs.append((br, bl))

    if not mutex_pairs and not hold_orderings:
        print("[DAG Schedule] No collisions or hold constraints — simultaneous playback")
        return _build_simultaneous_schedule(arm_right, arm_left, base_dt, step_dt)

    # Solve MILP
    start_times = _solve_block_milp(
        blocks_r,
        blocks_l,
        mutex_pairs,
        hold_orderings,
        base_dt,
        max_time=milp_max_time,
    )

    if start_times is None:
        print("[DAG Schedule] MILP infeasible — falling back to simultaneous playback")
        return _build_simultaneous_schedule(arm_right, arm_left, base_dt, step_dt)

    # Convert to time arrays
    len_r = int(arm_right.joint_positions.shape[0])
    len_l = int(arm_left.joint_positions.shape[0])
    times_r = _blocks_to_times(blocks_r, start_times, len_r, base_dt)
    times_l = _blocks_to_times(blocks_l, start_times, len_l, base_dt)

    total_time = max(times_r[-1] if times_r else 0.0, times_l[-1] if times_l else 0.0)
    print(f"[DAG Schedule] Total time: {total_time:.3f}, step_dt: {step_dt:.4f}")

    dev = arm_right.joint_positions.device
    idx_r = _discretize(times_r, total_time, step_dt, len_r, dev, gripper_actions=arm_right.gripper_actions)
    idx_l = _discretize(times_l, total_time, step_dt, len_l, dev, gripper_actions=arm_left.gripper_actions)

    print(f"[DAG Schedule] Discretized: idx_r[:10]={idx_r[:10].tolist()}, idx_l[:10]={idx_l[:10].tolist()}")
    print(f"[DAG Schedule] idx_r[-10:]={idx_r[-10:].tolist()}, idx_l[-10:]={idx_l[-10:].tolist()}")

    return DiscreteSchedule(left_indices=idx_l, right_indices=idx_r, total_time=total_time, step_dt=step_dt)
