"""Deterministic daily optimizer for the v2 agent.

Objective is lexicographic:
  1. fit every required TileJob,
  2. use the fewest workers (handled by the caller trying worker counts upward),
  3. minimize the longest worker route,
  4. minimize total route length.

There are no task priorities. Jobs are atomic during the main search. A single
safe two-part split may be tried only when the atomic packing cannot fit the
current worker count.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import inf

from agents.v2_core import TileJob

SHED_ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))
MAX_HANDS = 16
FULL_DAY_BUDGET = 23


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def route(start, target):
    x, y = start
    tx, ty = target
    return (
        [["EAST"]] * max(0, tx - x)
        + [["WEST"]] * max(0, x - tx)
        + [["SOUTH"]] * max(0, ty - y)
        + [["NORTH"]] * max(0, y - ty)
    )


def nearest_shed(position, shed_access=SHED_ACCESS):
    return min(shed_access, key=lambda p: (manhattan(position, p), SHED_ACCESS.index(p)))


def spawn_positions(post_move_positions, count, shed_access=SHED_ACCESS):
    """Exact least-occupied NWSE spawn rule, evaluated after current movement."""
    occupied = {position: 0 for position in shed_access}
    for position in map(tuple, post_move_positions):
        if position in occupied:
            occupied[position] += 1
    starts = []
    for _ in range(count):
        start = min(shed_access, key=lambda p: (occupied[p], SHED_ACCESS.index(p)))
        starts.append(start)
        occupied[start] += 1
    return starts


@dataclass(frozen=True)
class WorkerSpec:
    start: tuple[int, int]
    budget: int
    start_hour: int = 1


@dataclass
class WorkerPlan:
    spec: WorkerSpec
    jobs: list[TileJob]
    queue: list
    pickups: Counter
    job_windows: dict[str, tuple[int, int]]

    @property
    def length(self):
        return len(self.queue)


def required_initial_inventory(jobs):
    """Minimum pickup after simulating production/consumption along the route."""
    balance = Counter()
    required = Counter()
    for job in jobs:
        for step in job.actions:
            for item, amount in step.consume.items():
                balance[item] -= amount
                if balance[item] < 0:
                    required[item] = max(required[item], -balance[item])
            for item, amount in step.produce.items():
                balance[item] += amount
    return +required


def build_worker_plan(spec, jobs, shed_access=SHED_ACCESS):
    queue = []
    current = spec.start
    pickups = required_initial_inventory(jobs)
    if pickups:
        shed = nearest_shed(current, shed_access)
        queue.extend(route(current, shed))
        queue.extend(
            [["PICKUP", item, amount] for item, amount in sorted(pickups.items()) if amount > 0]
        )
        current = shed

    windows = {}
    for job in jobs:
        queue.extend(route(current, job.position))
        action_start = spec.start_hour + len(queue)
        queue.extend(job.raw_actions)
        action_end = spec.start_hour + len(queue)
        windows[job.key] = (action_start, action_end)
        current = job.position
    return WorkerPlan(spec, list(jobs), queue, pickups, windows)


def _plans_for(specs, buckets, shed_access, validate_dependencies=True):
    plans = [build_worker_plan(spec, bucket, shed_access) for spec, bucket in zip(specs, buckets)]
    if any(plan.length > plan.spec.budget for plan in plans):
        return None
    windows = {}
    dependencies = []
    for plan in plans:
        windows.update(plan.job_windows)
        for job in plan.jobs:
            if job.release_from:
                dependencies.append((job.key, job.release_from))
    if validate_dependencies:
        for suffix, prefix in dependencies:
            if suffix not in windows or prefix not in windows:
                return None
            if windows[suffix][0] < windows[prefix][1]:
                return None
    return plans


def _objective(plans):
    if plans is None:
        return (inf, inf, inf)
    lengths = [plan.length for plan in plans]
    return (
        max(lengths, default=0),
        sum(lengths),
        sum(len(plan.pickups) for plan in plans),
    )


def _ordered_jobs(jobs, specs, mode):
    if mode == "long":
        return sorted(
            jobs,
            key=lambda job: (
                -job.tile_action_count,
                min(manhattan(spec.start, job.position) for spec in specs),
                job.position[1], job.position[0], job.key,
            ),
        )
    return sorted(
        jobs,
        key=lambda job: (
            min(manhattan(spec.start, job.position) for spec in specs),
            job.position[1], job.position[0], -job.tile_action_count, job.key,
        ),
    )


def _construct(jobs, specs, shed_access, mode):
    buckets = [[] for _ in specs]
    plans = [build_worker_plan(spec, [], shed_access) for spec in specs]
    for job in _ordered_jobs(jobs, specs, mode):
        best = None
        for worker in range(len(specs)):
            bucket = buckets[worker]
            for insertion in range(len(bucket) + 1):
                candidate_bucket = bucket[:insertion] + [job] + bucket[insertion:]
                candidate_plan = build_worker_plan(specs[worker], candidate_bucket, shed_access)
                if candidate_plan.length > specs[worker].budget:
                    continue
                lengths = [
                    candidate_plan.length if i == worker else plans[i].length
                    for i in range(len(plans))
                ]
                pickup_count = (
                    sum(len(plan.pickups) for plan in plans)
                    - len(plans[worker].pickups)
                    + len(candidate_plan.pickups)
                )
                score = (max(lengths, default=0), sum(lengths), pickup_count, worker, insertion)
                if best is None or score < best[0]:
                    best = (score, worker, candidate_bucket, candidate_plan)
        if best is None:
            return None
        _, worker, candidate_bucket, candidate_plan = best
        buckets[worker] = candidate_bucket
        plans[worker] = candidate_plan
    return buckets


def _reorder_once(specs, buckets, shed_access):
    current = _plans_for(specs, buckets, shed_access)
    current_score = _objective(current)
    best_score, best_buckets = current_score, buckets
    for worker, bucket in enumerate(buckets):
        if len(bucket) < 2:
            continue
        for source in range(len(bucket)):
            job = bucket[source]
            shortened = bucket[:source] + bucket[source + 1:]
            for dest in range(len(bucket)):
                candidate_bucket = shortened[:dest] + [job] + shortened[dest:]
                if candidate_bucket == bucket:
                    continue
                candidate = [list(items) for items in buckets]
                candidate[worker] = candidate_bucket
                plans = _plans_for(specs, candidate, shed_access)
                score = _objective(plans)
                if score < best_score:
                    best_score, best_buckets = score, candidate
    return best_buckets, best_score < current_score


def _move_once(specs, buckets, shed_access):
    current = _plans_for(specs, buckets, shed_access)
    current_score = _objective(current)
    if current is None:
        return buckets, False
    longest = max(range(len(current)), key=lambda i: (current[i].length, -i))
    best_score, best_buckets = current_score, buckets
    for source_index, job in enumerate(buckets[longest]):
        source_bucket = buckets[longest][:source_index] + buckets[longest][source_index + 1:]
        for worker in range(len(buckets)):
            if worker == longest:
                continue
            for insertion in range(len(buckets[worker]) + 1):
                candidate = [list(items) for items in buckets]
                candidate[longest] = source_bucket
                candidate[worker].insert(insertion, job)
                plans = _plans_for(specs, candidate, shed_access)
                score = _objective(plans)
                if score < best_score:
                    best_score, best_buckets = score, candidate
    return best_buckets, best_score < current_score


def _swap_once(specs, buckets, shed_access):
    current = _plans_for(specs, buckets, shed_access)
    current_score = _objective(current)
    if current is None:
        return buckets, False
    longest = max(range(len(current)), key=lambda i: (current[i].length, -i))
    best_score, best_buckets = current_score, buckets
    for other in range(len(buckets)):
        if other == longest:
            continue
        for i, left in enumerate(buckets[longest]):
            for j, right in enumerate(buckets[other]):
                candidate = [list(items) for items in buckets]
                candidate[longest][i] = right
                candidate[other][j] = left
                plans = _plans_for(specs, candidate, shed_access)
                score = _objective(plans)
                if score < best_score:
                    best_score, best_buckets = score, candidate
    return best_buckets, best_score < current_score


def _improve(specs, buckets, shed_access, iterations=3):
    for _ in range(iterations):
        changed = False
        for improve in (_reorder_once, _move_once, _swap_once):
            buckets, did_change = improve(specs, buckets, shed_access)
            changed = changed or did_change
        if not changed:
            break
    return buckets


def _atomic_optimize(jobs, specs, shed_access=SHED_ACCESS, iterations=3):
    best_plans = None
    best_score = (inf, inf, inf)
    for mode in ("local", "long"):
        buckets = _construct(jobs, specs, shed_access, mode)
        if buckets is None:
            continue
        buckets = _improve(specs, buckets, shed_access, iterations)
        plans = _plans_for(specs, buckets, shed_access)
        score = _objective(plans)
        if score < best_score:
            best_plans, best_score = plans, score
    return best_plans


def optimize_routes(jobs, specs, shed_access=SHED_ACCESS, iterations=3, allow_tail_split=True):
    jobs = list(jobs)
    specs = list(specs)
    plans = _atomic_optimize(jobs, specs, shed_access, iterations)
    if plans is not None or not allow_tail_split:
        return plans
    candidates = [job for job in jobs if job.split() is not None]
    candidates.sort(key=lambda job: (-job.tile_action_count, job.position[1], job.position[0]))
    for job in candidates:
        prefix, suffix = job.split()
        split_jobs = [item for item in jobs if item is not job] + [prefix, suffix]
        plans = _atomic_optimize(split_jobs, specs, shed_access, iterations)
        if plans is not None:
            return plans
    return None


def minimum_stationary_hands(jobs, farmer_start, max_hands=MAX_HANDS, shed_access=SHED_ACCESS, iterations=2):
    """Hour-0 estimate when the farmer intentionally PASSes before HIRE."""
    farmer_start = tuple(farmer_start)
    raw_actions = sum(job.tile_action_count for job in jobs)
    lower_workers = max(1, (raw_actions + FULL_DAY_BUDGET - 1) // FULL_DAY_BUDGET)
    lower_hands = max(0, lower_workers - 1)
    for hand_count in range(lower_hands, max_hands + 1):
        hand_starts = spawn_positions([farmer_start], hand_count, shed_access)
        specs = [WorkerSpec(farmer_start, FULL_DAY_BUDGET, 1)]
        specs.extend(WorkerSpec(start, FULL_DAY_BUDGET, 1) for start in hand_starts)
        plans = optimize_routes(jobs, specs, shed_access, iterations)
        if plans is not None:
            return hand_count, plans
    return max_hands, None


def moved_position(position, op, board_size=10):
    x, y = position
    name = op[0] if op else "PASS"
    if name == "EAST":
        x += 1
    elif name == "WEST":
        x -= 1
    elif name == "SOUTH":
        y += 1
    elif name == "NORTH":
        y -= 1
    return (max(0, min(board_size - 1, x)), max(0, min(board_size - 1, y)))


def optimize_with_pending_hires(
    jobs,
    existing_positions,
    pending_hires,
    current_hour=1,
    shed_access=SHED_ACCESS,
    iterations=2,
    fixed_point_rounds=4,
):
    """Plan current work plus hands hired on this turn using exact spawn occupancy."""
    existing_positions = [tuple(p) for p in existing_positions]
    pending_starts = spawn_positions(existing_positions, pending_hires, shed_access)

    plans = None
    for _ in range(fixed_point_rounds):
        specs = [WorkerSpec(position, 24 - current_hour, current_hour) for position in existing_positions]
        pending_hour = current_hour + 1
        specs.extend(WorkerSpec(start, 24 - pending_hour, pending_hour) for start in pending_starts)
        plans = optimize_routes(jobs, specs, shed_access, iterations)
        if plans is None:
            return None, pending_starts

        first_ops = [(plan.queue[0] if plan.queue else ["PASS"]) for plan in plans[:len(existing_positions)]]
        post_positions = [moved_position(position, op) for position, op in zip(existing_positions, first_ops)]
        exact_starts = spawn_positions(post_positions, pending_hires, shed_access)
        if exact_starts == pending_starts:
            return plans, exact_starts
        pending_starts = exact_starts

    specs = [WorkerSpec(position, 24 - current_hour, current_hour) for position in existing_positions]
    pending_hour = current_hour + 1
    specs.extend(WorkerSpec(start, 24 - pending_hour, pending_hour) for start in pending_starts)
    plans = optimize_routes(jobs, specs, shed_access, iterations)
    if plans is None:
        return None, pending_starts
    first_ops = [(plan.queue[0] if plan.queue else ["PASS"]) for plan in plans[:len(existing_positions)]]
    post_positions = [moved_position(position, op) for position, op in zip(existing_positions, first_ops)]
    if spawn_positions(post_positions, pending_hires, shed_access) != pending_starts:
        return None, pending_starts
    return plans, pending_starts
