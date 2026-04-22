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
    _densify_path,  # pyright: ignore[reportPrivateUsage]
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
    block_type: str = "mixed"  # "mp", "skill", or "mixed"

    @property
    def duration(self) -> int:
        return self.end_idx - self.start_idx

    @property
    def var_name(self) -> str:
        return f"{self.arm_name}_{self.start_idx}"

    def __repr__(self) -> str:
        tag = f"/{self.block_type}" if self.block_type != "mixed" else ""
        return f"{self.arm_name}[st{self.subtask_index}{tag}:{self.start_idx}-{self.end_idx}]"


# ---------------------------------------------------------------------------
# Block extraction and splitting
# ---------------------------------------------------------------------------

def _extract_blocks(arm_path: ArmPath) -> list[SubtaskBlock]:
    """Extract subtask blocks from an ArmPath, splitting each into MP and skill blocks.

    If ``skill_boundaries`` is available, each subtask is split into an MP block
    (motion planner waypoints) and a skill block (demonstrated trajectory), matching
    the separate MotionPolicy/SkillPolicy structure in skill_interface.py.
    """
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
        skill_start = (
            arm_path.skill_boundaries.get(idx, start)
            if arm_path.skill_boundaries is not None
            else start
        )

        # MP block (if there are MP waypoints)
        if skill_start > start:
            blocks.append(SubtaskBlock(
                arm_name=arm_path.name,
                subtask_index=idx,
                start_idx=start,
                end_idx=skill_start,
                joint_positions=arm_path.joint_positions[start:skill_start],
                block_type="mp",
            ))

        # Skill block (if there are skill waypoints)
        if end > skill_start:
            blocks.append(SubtaskBlock(
                arm_name=arm_path.name,
                subtask_index=idx,
                start_idx=skill_start,
                end_idx=end,
                joint_positions=arm_path.joint_positions[skill_start:end],
                block_type="skill",
            ))
    return blocks


@dataclass
class _TaskOrdering:
    """Directed ordering from a SEQUENTIAL constraint.

    When a collision mutex is found between a block on ``yielding_arm``
    (within the yielding subtask range) and a block on ``leading_arm``
    (within the leading subtask range), the yielding arm always waits.
    """

    leading_arm: str
    leading_subtask: int
    yielding_arm: str
    yielding_subtask: int


def _process_hold_constraints(
    blocks_r: list[SubtaskBlock],
    blocks_l: list[SubtaskBlock],
    hold_constraints: list[HoldConstraint] | None,
    arm_right: ArmPath,
    arm_left: ArmPath,
    *,
    hold_latter_entire_subtask: bool = False,
    hold_former_part: str = "skill",
) -> tuple[list[SubtaskBlock], list[SubtaskBlock],
           list[tuple[SubtaskBlock, SubtaskBlock]], list[_TaskOrdering]]:
    """Process hold constraints into hard orderings and/or directed mutex preferences.

    When ``hold_start_idx`` is inside a block (min_time_diff > 0), the block
    is split and a hard ordering edge is added for the post-hold portion.

    Regardless of min_time_diff, a :class:`_TaskOrdering` is always emitted so
    that collision mutexes between the ordered subtask pair are directed (the
    yielding arm always waits, the leading arm never does).

    If ``hold_latter_entire_subtask`` is False (default), the hold edge targets
    the latter arm's *skill* block so its MP can run in parallel. If True, the
    hold edge targets the latter's *first* block (MP) of that subtask so the
    latter does not start the subtask at all until the former finishes.

    ``hold_former_part`` controls which part of the former's subtask the latter
    waits for: "mp" (former's MP only; latter can then start its own MP), "skill"
    (former's skill only), or "entire" (former's full subtask). When "mp", the
    hold edge targets the latter's first block (MP) so the latter can only start
    its planning segment after the former's MP for this constraint completes.

    Returns:
        (blocks_r, blocks_l, hold_orderings, task_orderings)
    """
    task_orderings: list[_TaskOrdering] = []
    if not hold_constraints:
        return blocks_r, blocks_l, [], task_orderings

    blocks_r = list(blocks_r)
    blocks_l = list(blocks_l)
    hold_orderings: list[tuple[SubtaskBlock, SubtaskBlock]] = []

    for hold in hold_constraints:
        latter_blocks = blocks_r if hold.holding_arm == "right" else blocks_l
        # latter_path = arm_right if hold.holding_arm == "right" else arm_left
        former_blocks = blocks_l if hold.other_arm == "left" else blocks_r

        # Always emit a task ordering so collision mutexes are directed
        former_subtask_idx: int | None = None
        for fb in former_blocks:
            if fb.start_idx < hold.other_arm_len <= fb.end_idx:
                former_subtask_idx = fb.subtask_index
                break
        if former_subtask_idx is None and former_blocks:
            former_subtask_idx = former_blocks[-1].subtask_index

        latter_subtask_idx: int | None = None
        for block in latter_blocks:
            if block.start_idx <= hold.hold_start_idx < block.end_idx:
                latter_subtask_idx = block.subtask_index
                break
        # When hold_start_idx == subtask_end (min_time_diff=0), check the previous block
        if latter_subtask_idx is None:
            for block in latter_blocks:
                if block.end_idx == hold.hold_start_idx:
                    latter_subtask_idx = block.subtask_index
                    break

        if former_subtask_idx is not None and latter_subtask_idx is not None:
            task_orderings.append(_TaskOrdering(
                leading_arm=hold.other_arm,
                leading_subtask=former_subtask_idx,
                yielding_arm=hold.holding_arm,
                yielding_subtask=latter_subtask_idx,
            ))
            print(
                f"[DAG] Task ordering: {hold.other_arm}[st{former_subtask_idx}] leads, "
                f"{hold.holding_arm}[st{latter_subtask_idx}] yields on collision"
            )

        # Choose which block on the latter arm to constrain.
        # - hold_former_part="mp": target latter's first block (MP) so latter can only start its
        #   planning segment after former's MP for this constraint's subtask completes.
        # - hold_latter_entire_subtask=True: target first block (latter waits for whole task).
        # - else: target latter's skill block (latter MP can overlap with former).
        latter_target_block: SubtaskBlock | None = None
        if hold_former_part == "mp" or hold_latter_entire_subtask:
            for block in latter_blocks:
                if block.subtask_index == latter_subtask_idx:
                    latter_target_block = block
                    break
        else:
            for block in latter_blocks:
                if (block.subtask_index == latter_subtask_idx
                        and block.block_type in ("skill", "mixed")):
                    latter_target_block = block
                    break
        if latter_target_block is None:
            continue

        # Find the FORMER arm's block to wait for (by hold_former_part)
        former_subtask_blocks = [fb for fb in former_blocks if fb.subtask_index == former_subtask_idx]
        former_block: SubtaskBlock | None = None
        if former_subtask_blocks:
            if hold_former_part == "mp":
                mp_blocks = [b for b in former_subtask_blocks if b.block_type == "mp"]
                former_block = max(mp_blocks, key=lambda b: b.end_idx) if mp_blocks else None
            elif hold_former_part == "skill":
                skill_blocks = [b for b in former_subtask_blocks if b.block_type in ("skill", "mixed")]
                former_block = max(skill_blocks, key=lambda b: b.end_idx) if skill_blocks else None
            elif hold_former_part == "entire":
                former_block = max(former_subtask_blocks, key=lambda b: b.end_idx)
            else:
                former_block = max(
                    [b for b in former_subtask_blocks if b.block_type in ("skill", "mixed")],
                    key=lambda b: b.end_idx,
                ) if any(b.block_type in ("skill", "mixed") for b in former_subtask_blocks) else None
        if former_block is None and former_blocks:
            former_block = former_blocks[-1]
        if former_block is None:
            continue

        hold_orderings.append((former_block, latter_target_block))
        print(f"[DAG] Hold ordering: {former_block} -> {latter_target_block}")

    return blocks_r, blocks_l, hold_orderings, task_orderings


def _split_large_blocks(
    blocks: list[SubtaskBlock],
    arm_path: ArmPath,
    max_block_size: int,
) -> list[SubtaskBlock]:
    """Split blocks that exceed *max_block_size* into consecutive sub-blocks.

    This gives the MILP finer scheduling granularity: only the sub-blocks that
    actually collide with the other arm need to be serialised, while the rest
    can overlap.
    """
    if max_block_size <= 0:
        return blocks

    out: list[SubtaskBlock] = []
    for block in blocks:
        if block.duration <= max_block_size:
            out.append(block)
            continue

        n_before = len(out)
        cursor = block.start_idx
        while cursor < block.end_idx:
            chunk_end = min(cursor + max_block_size, block.end_idx)
            out.append(
                SubtaskBlock(
                    arm_name=block.arm_name,
                    subtask_index=block.subtask_index,
                    start_idx=cursor,
                    end_idx=chunk_end,
                    joint_positions=arm_path.joint_positions[cursor:chunk_end],
                )
            )
            cursor = chunk_end

        print(
            f"[DAG] Split {block} ({block.duration}wp) into "
            f"{len(out) - n_before} chunks of <={max_block_size}wp"
        )
    return out


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
    closest_r_link: str | None = None  # link name on right arm at worst penetration
    closest_l_link: str | None = None  # link name on left arm at worst penetration

# TODO: Neel: Simplify this function (remove redundant checks & code)
def _sphere_index_to_link_name(kin, sphere_idx: int) -> str | None:
    """Resolve a single sphere index to link name using kinematics_config."""
    if not hasattr(kin, "kinematics_config"):
        return None
    kc = kin.kinematics_config
    if not hasattr(kc, "link_sphere_idx_map") or not hasattr(kc, "link_name_to_idx_map"):
        return None
    link_sphere_idx_map = kc.link_sphere_idx_map
    link_name_to_idx = getattr(kc, "link_name_to_idx_map", None)
    if link_name_to_idx is None:
        return None
    if hasattr(link_sphere_idx_map, "cpu"):
        link_sphere_idx_map = link_sphere_idx_map.cpu().numpy()
    idx_to_name = {v: k for k, v in link_name_to_idx.items()}
    if sphere_idx >= link_sphere_idx_map.size:
        return None
    link_idx = int(link_sphere_idx_map.flat[sphere_idx])
    return idx_to_name.get(link_idx)


def _blocks_collide(
    joints_r: torch.Tensor,
    joints_l: torch.Tensor,
    kin_right,
    kin_left,
    collision_margin: float = 0.0,
    n_shared_joints: int = 6,
    batch_size: int = 4096,
    densify_factor: int = 4,
    debug_links: bool = False,
    use_shared_torso: bool = True,
    base_pos_right: torch.Tensor | None = None,
    base_pos_left: torch.Tensor | None = None,
) -> BlockCollisionResult:
    """Check collisions between two joint-position arrays and return diagnostics.

    Linearly densifies both paths by ``densify_factor`` before checking, so
    intermediate configurations between waypoints are also tested.  Reported
    indices (closest_r_idx, closest_l_idx) are mapped back to the original
    (non-densified) waypoint indices.

    When ``use_shared_torso`` is True (default), left's first ``n_shared_joints``
    are overwritten with right's so both arms are evaluated in the same base frame.
    When False, each arm uses its own torso from its own waypoint (tests whether
    shared-torso causes false collisions when torso differs between waypoints).
    """
    Nr_orig = int(joints_r.shape[0])
    Nl_orig = int(joints_l.shape[0])
    if Nr_orig == 0 or Nl_orig == 0:
        return BlockCollisionResult(False, 0, 0, float("inf"), -1, -1)

    dev = joints_r.device

    dense_r, map_r = _densify_path(joints_r, densify_factor)
    dense_l, map_l = _densify_path(joints_l, densify_factor)
    Nr = int(dense_r.shape[0])
    Nl = int(dense_l.shape[0])

    q_r = dense_r.repeat_interleave(Nl, dim=0)
    q_l = dense_l.repeat(Nr, 1)
    total = q_r.shape[0]

    if n_shared_joints > 0 and use_shared_torso:
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

        # For separate-base-frame arms (e.g. YAM bimanual), shift sphere
        # centres into a common world frame before distance computation.
        if base_pos_right is not None:
            c_r = c_r + base_pos_right.to(c_r.device)
        if base_pos_left is not None:
            c_l = c_l + base_pos_left.to(c_l.device)

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
            best_r_idx = int(map_r[global_row // Nl].item())
            best_l_idx = int(map_l[global_row % Nl].item())

    closest_r_link: str | None = None
    closest_l_link: str | None = None
    if debug_links and n_colliding > 0 and best_r_idx >= 0 and best_l_idx >= 0:
        q_r = joints_r[best_r_idx : best_r_idx + 1].to(device=dev)
        q_l = joints_l[best_l_idx : best_l_idx + 1].clone().to(device=dev)
        if n_shared_joints > 0 and use_shared_torso:
            q_l[:, :n_shared_joints] = q_r[:, :n_shared_joints]
        state_r = kin_right.get_state(q_r)
        state_l = kin_left.get_state(q_l)
        sph_r = state_r.link_spheres_tensor.view(1, -1, 4)
        sph_l = state_l.link_spheres_tensor.view(1, -1, 4)
        c_r, r_r = sph_r[..., :3], sph_r[..., 3]
        c_l, r_l = sph_l[..., :3], sph_l[..., 3]
        if base_pos_right is not None:
            c_r = c_r + base_pos_right.to(c_r.device)
        if base_pos_left is not None:
            c_l = c_l + base_pos_left.to(c_l.device)
        valid_r = r_r > 0
        valid_l = r_l > 0
        aa = (c_r * c_r).sum(dim=-1, keepdim=True)
        bb = (c_l * c_l).sum(dim=-1).unsqueeze(1)
        ab = torch.bmm(c_r, c_l.transpose(1, 2))
        dist2 = torch.clamp(aa + bb - 2.0 * ab, min=0.0)
        radii = r_r.unsqueeze(-1) + r_l.unsqueeze(-2) + collision_margin
        valid_pairs = valid_r.unsqueeze(-1) & valid_l.unsqueeze(-2)
        penetration = torch.sqrt(dist2) - radii
        penetration = torch.where(
            valid_pairs, penetration, torch.tensor(float("inf"), device=dev)
        )
        pen_flat = penetration.view(-1)
        min_flat_idx = int(pen_flat.argmin().item())
        nr = sph_r.shape[1]
        sphere_r_idx = min_flat_idx // sph_l.shape[1]
        sphere_l_idx = min_flat_idx % sph_l.shape[1]
        closest_r_link = _sphere_index_to_link_name(kin_right, sphere_r_idx)
        closest_l_link = _sphere_index_to_link_name(kin_left, sphere_l_idx)

    return BlockCollisionResult(
        collides=n_colliding > 0,
        n_colliding_pairs=n_colliding,
        total_pairs=total,
        min_penetration=global_min_pen,
        closest_r_idx=best_r_idx,
        closest_l_idx=best_l_idx,
        closest_r_link=closest_r_link,
        closest_l_link=closest_l_link,
    )


# ---------------------------------------------------------------------------
# MILP construction and solve
# ---------------------------------------------------------------------------

def _solve_block_milp(
    blocks_r: list[SubtaskBlock],
    blocks_l: list[SubtaskBlock],
    mutex_pairs: list[tuple[SubtaskBlock, SubtaskBlock]],
    hold_orderings: list[tuple[SubtaskBlock, SubtaskBlock]],
    task_orderings: list[_TaskOrdering],
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

    # Build index maps so we can look up the next block on each arm
    r_idx_map = {b: i for i, b in enumerate(blocks_r)}
    l_idx_map = {b: i for i, b in enumerate(blocks_l)}

    # Check if a mutex pair is directed by a task ordering
    def _is_directed(br: SubtaskBlock, bl: SubtaskBlock) -> str | None:
        """Return 'left_leads' or 'right_leads' if task ordering applies, else None."""
        for to in task_orderings:
            if (to.yielding_arm == "right" and br.subtask_index == to.yielding_subtask
                    and to.leading_arm == "left" and bl.subtask_index == to.leading_subtask):
                return "left_leads"
            if (to.yielding_arm == "left" and bl.subtask_index == to.yielding_subtask
                    and to.leading_arm == "right" and br.subtask_index == to.leading_subtask):
                return "right_leads"
        return None

    z_vars: dict[tuple[SubtaskBlock, SubtaskBlock], Variable] = {}
    n_directed = 0
    for br, bl in mutex_pairs:
        i_r = r_idx_map[br]
        i_l = l_idx_map[bl]
        next_r = blocks_r[i_r + 1] if i_r + 1 < len(blocks_r) else None
        next_l = blocks_l[i_l + 1] if i_l + 1 < len(blocks_l) else None

        direction = _is_directed(br, bl)

        if direction is not None:
            # Directed: force the yielding arm to wait (no binary variable)
            n_directed += 1
            if direction == "left_leads":
                # Right yields: t[br] >= t[next_l] (or t[bl] + dur_l if last)
                if next_l is not None:
                    constraints.append(Constraint(
                        lower=0.0,
                        coefficients={block_vars[br].name: +1, block_vars[next_l].name: -1},
                    ))
                else:
                    constraints.append(Constraint(
                        lower=bl.duration * base_dt,
                        coefficients={block_vars[br].name: +1, block_vars[bl].name: -1},
                    ))
            else:
                # Left yields: t[bl] >= t[next_r] (or t[br] + dur_r if last)
                if next_r is not None:
                    constraints.append(Constraint(
                        lower=0.0,
                        coefficients={block_vars[bl].name: +1, block_vars[next_r].name: -1},
                    ))
                else:
                    constraints.append(Constraint(
                        lower=br.duration * base_dt,
                        coefficients={block_vars[bl].name: +1, block_vars[br].name: -1},
                    ))
            continue

        # Undirected: MILP chooses via binary variable
        z = Variable(
            name=f"z_{br.var_name}__{bl.var_name}",
            integer=True,
            lower=0.0,
            upper=1.0,
        )
        z_vars[(br, bl)] = z
        variables.append(z)

        if next_r is not None:
            constraints.append(Constraint(
                lower=0.0,
                coefficients={block_vars[bl].name: +1, block_vars[next_r].name: -1, z.name: +M},
            ))
        else:
            constraints.append(Constraint(
                lower=br.duration * base_dt,
                coefficients={block_vars[bl].name: +1, block_vars[br].name: -1, z.name: +M},
            ))

        if next_l is not None:
            constraints.append(Constraint(
                lower=0.0 - M,
                coefficients={block_vars[br].name: +1, block_vars[next_l].name: -1, z.name: -M},
            ))
        else:
            constraints.append(Constraint(
                lower=bl.duration * base_dt - M,
                coefficients={block_vars[br].name: +1, block_vars[bl].name: -1, z.name: -M},
            ))

    print(
        f"[DAG MILP] blocks_r={len(blocks_r)}, blocks_l={len(blocks_l)}, "
        f"mutexes={len(mutex_pairs)} ({n_directed} directed, {len(z_vars)} binary), "
        f"hold_edges={len(hold_orderings)}"
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
    densify_factor: int = 4,
    max_block_size: int = 0,
    pair_batch: int = 4096,
    n_shared_joints: int = 6,
    min_dt: float | None = None,
    milp_max_time: float = 5.0,
    hold_latter_entire_subtask: bool = False,
    hold_former_part: str = "skill",
    debug_collision_links: bool = False,
    use_shared_torso: bool = True,
    base_pos_right: torch.Tensor | None = None,
    base_pos_left: torch.Tensor | None = None,
) -> DiscreteSchedule:
    """Build a discrete schedule using DAG-based segment-level MILP ordering.

    Each subtask block is treated as an atomic unit with fixed duration.  Hold
    constraints split blocks at the hold point so the pre-hold portion can run
    concurrently while only the post-hold portion waits for the other arm.

    Large blocks are optionally chopped into chunks of at most
    ``max_block_size`` waypoints so the MILP can serialise only the chunks
    that actually collide, allowing the rest to overlap.

    The MILP decides the *start time* of each (sub-)block to minimise makespan
    while avoiding inter-arm collisions and respecting hold orderings.

    Args:
        arm_right / arm_left: Arm trajectories with ``subtask_boundaries`` populated.
        planner_right / planner_left: Arm planners (for FK kinematics).
        step_dt: Simulator step duration.
        hold_constraints: Sequential ordering constraints.
        collision_margin: Sphere collision margin (m).
        densify_factor: Linear interpolation factor for densifying paths before
            collision checking (e.g. 4 inserts 3 intermediate samples between
            each pair of consecutive waypoints).
        max_block_size: Maximum waypoints per block. Blocks exceeding this are
            split into consecutive chunks. 0 disables splitting.
        pair_batch: FK batch size for collision checking.
        n_shared_joints: Number of leading shared (torso) joints to synchronise.
        min_dt: Minimum time delta between consecutive waypoints inside a block.
        milp_max_time: MILP solver timeout (seconds).
        hold_latter_entire_subtask: If True, hold edge targets the latter's first block
            (MP) so the latter does not start the subtask until the former finishes.
            If False, hold edge targets the latter's skill block (latter MP can overlap).
        hold_former_part: Which part of the former's subtask the latter waits for:
            "mp" (former MP only), "skill" (former skill only), "entire" (former full subtask).
            Default "skill".
    """
    blocks_r = _extract_blocks(arm_right)
    blocks_l = _extract_blocks(arm_left)
    base_dt = min_dt if min_dt is not None else max(step_dt, 1e-3)

    # Process hold constraints: split blocks at hold points and extract task orderings
    blocks_r, blocks_l, hold_orderings, task_orderings = _process_hold_constraints(
        blocks_r, blocks_l, hold_constraints, arm_right, arm_left,
        hold_latter_entire_subtask=hold_latter_entire_subtask,
        hold_former_part=hold_former_part,
    )

    # Split large blocks into smaller chunks for finer MILP granularity
    if max_block_size > 0:
        blocks_r = _split_large_blocks(blocks_r, arm_right, max_block_size)
        blocks_l = _split_large_blocks(blocks_l, arm_left, max_block_size)

        # Remap hold orderings: the original block objects may have been
        # replaced by chunks.  The former block's intent is "must finish" so
        # we point to the LAST chunk covering its range.  The latter block's
        # intent is "can't start" so we point to the FIRST chunk.
        all_new = blocks_r + blocks_l
        updated_orderings: list[tuple[SubtaskBlock, SubtaskBlock]] = []
        for former, latter in hold_orderings:
            new_former = former
            if former not in all_new:
                candidates = [b for b in all_new
                              if b.arm_name == former.arm_name and b.end_idx == former.end_idx]
                if candidates:
                    new_former = candidates[0]
            new_latter = latter
            if latter not in all_new:
                candidates = [b for b in all_new
                              if b.arm_name == latter.arm_name and b.start_idx == latter.start_idx]
                if candidates:
                    new_latter = candidates[0]
            updated_orderings.append((new_former, new_latter))
            if new_former is not former or new_latter is not latter:
                print(f"[DAG] Remapped hold ordering: {new_former} -> {new_latter}")
        hold_orderings = updated_orderings

    print(f"[DAG Schedule] Right blocks: {blocks_r}")
    print(f"[DAG Schedule] Left blocks:  {blocks_l}")

    # Check collisions between cross-arm block pairs
    kin_right = planner_right.motion_gen.kinematics
    kin_left = planner_left.motion_gen.kinematics

    mutex_pairs: list[tuple[SubtaskBlock, SubtaskBlock]] = []
    print(
        f"[DAG Collision] Checking {len(blocks_r)} x {len(blocks_l)} block pairs, "
        f"margin={collision_margin}m, densify={densify_factor}x, use_shared_torso={use_shared_torso}"
    )
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
                densify_factor=densify_factor,
                debug_links=debug_collision_links,
                use_shared_torso=use_shared_torso,
                base_pos_right=base_pos_right,
                base_pos_left=base_pos_left,
            )
            status = "COLLIDE" if result.collides else "clear"
            msg = (
                f"[DAG Collision] {br} x {bl}: {status} | "
                f"pairs={result.n_colliding_pairs}/{result.total_pairs} | "
                f"min_pen={result.min_penetration:.4f}m | "
                f"closest=R[{br.start_idx + result.closest_r_idx}] vs L[{bl.start_idx + result.closest_l_idx}]"
            )
            if result.collides and result.closest_r_link and result.closest_l_link:
                msg += f" | links: {result.closest_r_link} vs {result.closest_l_link}"
            print(msg)
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
        task_orderings,
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

    # Debug: show time gaps (holds) in each arm's time array
    for arm_name, times in [("right", times_r), ("left", times_l)]:
        for i in range(1, len(times)):
            gap = times[i] - times[i - 1]
            if gap > base_dt * 2:
                print(
                    f"[DAG Schedule] {arm_name} time gap at wp {i-1}->{i}: "
                    f"{times[i-1]:.3f} -> {times[i]:.3f} (gap={gap:.3f}s, "
                    f"{int(gap / step_dt)} ticks of hold)"
                )

    dev = arm_right.joint_positions.device
    idx_r = _discretize(times_r, total_time, step_dt, len_r, dev, gripper_actions=arm_right.gripper_actions)
    idx_l = _discretize(times_l, total_time, step_dt, len_l, dev, gripper_actions=arm_left.gripper_actions)

    print(f"[DAG Schedule] Discretized: idx_r[:10]={idx_r[:10].tolist()}, idx_l[:10]={idx_l[:10].tolist()}")
    print(f"[DAG Schedule] idx_r[-10:]={idx_r[-10:].tolist()}, idx_l[-10:]={idx_l[-10:].tolist()}")

    # Debug: detect holds (consecutive ticks with the same index)
    for arm_name, idx in [("right", idx_r), ("left", idx_l)]:
        holds = []
        run_start = 0
        for t in range(1, len(idx)):
            if idx[t] != idx[t - 1]:
                run_len = t - run_start
                if run_len > 5:
                    holds.append((run_start, t - 1, int(idx[run_start].item()), run_len))
                run_start = t
        run_len = len(idx) - run_start
        if run_len > 5:
            holds.append((run_start, len(idx) - 1, int(idx[run_start].item()), run_len))
        if holds:
            for tick_start, tick_end, wp_idx, length in holds:
                print(
                    f"[DAG Schedule] {arm_name} HOLD: ticks {tick_start}-{tick_end} "
                    f"({length} ticks, {length * step_dt:.2f}s) at waypoint {wp_idx}"
                )

    return DiscreteSchedule(left_indices=idx_l, right_indices=idx_r, total_time=total_time, step_dt=step_dt)
