"""Priority-free daily scheduler for the v2 agent.

Objectives are lexicographic:
  1. schedule every required TileJob,
  2. use the fewest hands,
  3. minimize the longest worker route,
  4. minimize total route length.

Main scheduling keeps TileJobs atomic. A bounded tail-fill pass may use the
single safe split carried by a job, with one release-step constraint; there is
no global DAG/timeline solver.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

from agents.v2_model import SupplyPlan, TileJob, WorkerPlan

FARMER_BUDGET = 23
HAND_BUDGET = 23
MAX_HANDS = 16
SHED_ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))
LOCAL_SEARCH_PASSES = 4


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
    return min(
        shed_access,
        key=lambda p: (manhattan(position, p), shed_access.index(p)),
    )


def predicted_spawn_starts(worker_positions, new_count, shed_access=SHED_ACCESS):
    """Engine-style least-occupied NWSE spawn rule.

    `worker_positions` must be positions after the current turn's movement.
    """
    occupied = Counter(tuple(p) for p in worker_positions)
    starts = []
    for _ in range(new_count):
        start = min(shed_access, key=lambda p: (occupied[p], shed_access.index(p)))
        starts.append(start)
        occupied[start] += 1
    return starts


def _pickup_requirement(jobs):
    """Minimum initial pickup after simulating inventory along this route.

    Output from an earlier job may cover a later job. The route itself is never
    reordered just to exploit that inventory.
    """
    balance = Counter()
    pickup = Counter()
    for job in jobs:
        for item, amount in job.needs.items():
            if amount <= 0:
                continue
            balance[item] -= amount
            if balance[item] < 0:
                pickup[item] = max(pickup[item], -balance[item])
        balance.update(job.produces)
    return +pickup


def _queue(start, jobs, shed_access=SHED_ACCESS):
    jobs = list(jobs)
    pickup = _pickup_requirement(jobs)
    queue = []
    current = tuple(start)
    if pickup:
        shed = nearest_shed(current, shed_access)
        queue.extend(route(current, shed))
        for item in sorted(pickup):
            amount = int(pickup[item])
            if amount > 0:
                queue.append(["PICKUP", item, amount])
        current = shed
    for job in jobs:
        queue.extend(route(current, job.position))
        queue.extend([list(action) for action in job.actions])
        current = job.position
    return queue, pickup


def route_length(start, jobs, shed_access=SHED_ACCESS):
    return len(_queue(start, jobs, shed_access)[0])


def _job_times(start, jobs, index, shed_access):
    """Return (first_action_step, step_after_last_action) for one queued job."""
    pickup = _pickup_requirement(jobs)
    current = tuple(start)
    step = 0
    if pickup:
        shed = nearest_shed(current, shed_access)
        step += manhattan(current, shed) + len(pickup)
        current = shed
    for job_index, job in enumerate(jobs):
        step += manhattan(current, job.position)
        action_start = step
        step += len(job.actions)
        if job_index == index:
            return action_start, step
        current = job.position
    raise IndexError(index)


def _best_insertion(start, bucket, job, budget, shed_access):
    best = None
    for index in range(len(bucket) + 1):
        candidate = bucket[:index] + [job] + bucket[index:]
        length = route_length(start, candidate, shed_access)
        if length > budget:
            continue
        score = (length, index)
        if best is None or score < best[0]:
            best = (score, candidate)
    return None if best is None else best[1]


def _initial_pack(jobs, starts, budgets, shed_access):
    buckets = [[] for _ in starts]
    unassigned = []
    ordered = sorted(
        jobs,
        key=lambda job: (
            min(manhattan(start, job.position) for start in starts),
            job.position[1],
            job.position[0],
            len(job.actions),
        ),
    )
    for job in ordered:
        candidates = []
        for worker, (start, budget) in enumerate(zip(starts, budgets)):
            inserted = _best_insertion(start, buckets[worker], job, budget, shed_access)
            if inserted is None:
                continue
            length = route_length(start, inserted, shed_access)
            old = route_length(start, buckets[worker], shed_access)
            candidates.append((length - old, length, worker, inserted))
        if not candidates:
            unassigned.append(job)
            continue
        _, _, worker, candidate = min(candidates, key=lambda row: row[:3])
        buckets[worker] = candidate
    return buckets, unassigned


def _score(buckets, starts, shed_access):
    lengths = [route_length(start, bucket, shed_access) for start, bucket in zip(starts, buckets)]
    return (max(lengths, default=0), sum(lengths), tuple(lengths))


def _reorder_bucket(start, bucket, budget, shed_access):
    if len(bucket) < 2:
        return bucket
    ordered = sorted(bucket, key=lambda j: (manhattan(start, j.position), j.position[1], j.position[0]))
    result = []
    for job in ordered:
        best = _best_insertion(start, result, job, budget, shed_access)
        if best is None:
            return bucket
        result = best
    return result if route_length(start, result, shed_access) <= route_length(start, bucket, shed_access) else bucket


def _improve(buckets, starts, budgets, shed_access):
    buckets = [list(bucket) for bucket in buckets]
    for _ in range(LOCAL_SEARCH_PASSES):
        changed = False
        current_score = _score(buckets, starts, shed_access)

        for worker in range(len(buckets)):
            candidate = _reorder_bucket(starts[worker], buckets[worker], budgets[worker], shed_access)
            if candidate != buckets[worker]:
                trial = [list(b) for b in buckets]
                trial[worker] = candidate
                if _score(trial, starts, shed_access) < current_score:
                    buckets = trial
                    current_score = _score(buckets, starts, shed_access)
                    changed = True

        moved = False
        for src in range(len(buckets)):
            for index, job in list(enumerate(buckets[src])):
                for dst in range(len(buckets)):
                    if src == dst:
                        continue
                    dst_candidate = _best_insertion(starts[dst], buckets[dst], job, budgets[dst], shed_access)
                    if dst_candidate is None:
                        continue
                    trial = [list(b) for b in buckets]
                    trial[src] = buckets[src][:index] + buckets[src][index + 1 :]
                    trial[dst] = dst_candidate
                    if _score(trial, starts, shed_access) < current_score:
                        buckets = trial
                        changed = moved = True
                        break
                if moved:
                    break
            if moved:
                break

        if not changed:
            swapped = False
            for a, b in combinations(range(len(buckets)), 2):
                for ia, ja in enumerate(buckets[a]):
                    for ib, jb in enumerate(buckets[b]):
                        ca = buckets[a][:ia] + [jb] + buckets[a][ia + 1 :]
                        cb = buckets[b][:ib] + [ja] + buckets[b][ib + 1 :]
                        if route_length(starts[a], ca, shed_access) > budgets[a]:
                            continue
                        if route_length(starts[b], cb, shed_access) > budgets[b]:
                            continue
                        trial = [list(x) for x in buckets]
                        trial[a], trial[b] = ca, cb
                        if _score(trial, starts, shed_access) < current_score:
                            buckets = trial
                            changed = swapped = True
                            break
                    if swapped:
                        break
                if swapped:
                    break
        if not changed:
            break
    return buckets


def _tail_fill(buckets, unassigned, starts, budgets, shed_access):
    """Use only approved two-part splits and enforce one release-step check."""
    buckets = [list(bucket) for bucket in buckets]
    remaining = []
    split_used = False
    for job in unassigned:
        parts = job.split()
        if parts is None:
            remaining.append(job)
            continue
        prefix, suffix = parts
        best = None
        for a in range(len(buckets)):
            for ia in range(len(buckets[a]) + 1):
                ba = buckets[a][:ia] + [prefix] + buckets[a][ia:]
                if route_length(starts[a], ba, shed_access) > budgets[a]:
                    continue
                _, prefix_end = _job_times(starts[a], ba, ia, shed_access)
                for b in range(len(buckets)):
                    if b == a:
                        continue
                    for ib in range(len(buckets[b]) + 1):
                        bb = buckets[b][:ib] + [suffix] + buckets[b][ib:]
                        if route_length(starts[b], bb, shed_access) > budgets[b]:
                            continue
                        suffix_start, _ = _job_times(starts[b], bb, ib, shed_access)
                        if suffix_start < prefix_end:
                            continue
                        trial = [list(x) for x in buckets]
                        trial[a], trial[b] = ba, bb
                        score = _score(trial, starts, shed_access)
                        candidate = (score, a, b, ba, bb)
                        if best is None or candidate[0] < best[0]:
                            best = candidate
        if best is None:
            remaining.append(job)
        else:
            _, a, b, ba, bb = best
            buckets[a], buckets[b] = ba, bb
            split_used = True
    return buckets, remaining, split_used


def pack_jobs(jobs, starts, budgets, shed_access=SHED_ACCESS):
    buckets, unassigned = _initial_pack(jobs, starts, budgets, shed_access)
    if not unassigned:
        return _improve(buckets, starts, budgets, shed_access), []
    buckets, unassigned, split_used = _tail_fill(buckets, unassigned, starts, budgets, shed_access)
    if unassigned:
        return buckets, unassigned
    # Do not reorder after a split: the prefix/suffix release relation was
    # checked against these exact queue positions.
    return buckets if split_used else _improve(buckets, starts, budgets, shed_access), []


def hands_needed(jobs, farmer_start, shed_access=SHED_ACCESS, max_hands=MAX_HANDS):
    for hand_count in range(max_hands + 1):
        hand_starts = predicted_spawn_starts([tuple(farmer_start)], hand_count, shed_access)
        starts = [tuple(farmer_start), *hand_starts]
        budgets = [FARMER_BUDGET] + [HAND_BUDGET] * hand_count
        _, unassigned = pack_jobs(jobs, starts, budgets, shed_access)
        if not unassigned:
            return hand_count
    return max_hands


def build_plans(jobs, starts, budgets, shed_access=SHED_ACCESS):
    buckets, unassigned = pack_jobs(jobs, starts, budgets, shed_access)
    plans = []
    for start, bucket in zip(starts, buckets):
        queue, pickup = _queue(start, bucket, shed_access)
        plans.append(WorkerPlan(tuple(start), list(bucket), queue, pickup))
    return plans, unassigned


def global_supply_plan(plans, shed):
    """Reserve real shed stock globally; Buyer covers only the shortfall."""
    remaining = Counter(shed)
    assigned = []
    shortfall = Counter()
    for plan in plans:
        pickup = Counter()
        for item, amount in plan.pickup.items():
            amount = int(amount)
            pickup[item] = amount
            use = min(max(0, remaining[item]), amount)
            remaining[item] -= use
            shortfall[item] += amount - use
        assigned.append(pickup)
    return SupplyPlan(tuple(assigned), +shortfall)
