"""Function 2: turn today's tasks into farmer/hand orders, minimizing labor cost.

Two costs compound here and both matter:
  - Hiring is Fibonacci-priced *per day* (hands don't persist -- every hand is
    rehired from scratch each morning), so hiring more than the day's actual
    workload requires is pure waste.
  - Queues are built at hour 1, after hour-0 purchases and hires have landed.
    Both the farmer and hands hired at hour 0 therefore have hours 1..23: 23
    executable steps. Counting 24 for the farmer leaves exactly one queued
    action unexecuted at day end, even when another hand has spare capacity.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

FARMER_BUDGET = 23
HAND_BUDGET = 23
SHED = (4, 4)
SHED_ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))
# Fibonacci hire cost grows fast (16 hands/day = $2,583 total), but a
# large farm's real per-day workload (see agents/planner.py's animal/crop
# mix) can still exceed what 12 hands + the farmer fit in a day -- verified
# directly (seed 1): capped at 12, several tiles were left as WEED and a
# ready SHEEP harvest sat uncollected past hour 8 on cash-rich late-game
# days where $2-3k for a couple more hands is trivial. Raised to 16;
# hands_needed/build_queues below already only hire as many as a day's
# real tasks and real cash justify, so this only matters on days that
# actually need it.
MAX_HANDS = 16


def route(start, target):
    x, y = start
    tx, ty = target
    return (
        [["EAST"]] * max(0, tx - x) + [["WEST"]] * max(0, x - tx)
        + [["SOUTH"]] * max(0, ty - y) + [["NORTH"]] * max(0, y - ty)
    )


def nearest_shed(position, shed_access=SHED_ACCESS):
    """Closest tile from which PICKUP/DROP can access the central shed."""
    return min(
        shed_access,
        key=lambda p: (abs(position[0] - p[0]) + abs(position[1] - p[1]), SHED_ACCESS.index(p)),
    )


def predicted_hand_starts(farmer_start, existing_hand_starts, hand_count):
    """Mirror the engine's least-occupied NWSE spawn rule for pending hires."""
    starts = [tuple(position) for position in existing_hand_starts[:hand_count]]
    while len(starts) < hand_count:
        occupants = {position: 0 for position in SHED_ACCESS}
        for position in [tuple(farmer_start), *starts]:
            if position in occupants:
                occupants[position] += 1
        starts.append(min(SHED_ACCESS, key=lambda p: (occupants[p], SHED_ACCESS.index(p))))
    return starts


@dataclass
class WorkerPlan:
    start: tuple[int, int]
    queue: list  # ops to pop one per turn


def _mandatory_actions(task):
    """Return the deadline-sensitive prefix of a task.

    A finished one-time crop may encode an entire producer transition in one
    ``Task``: WATER, HARVEST, then PLANT/PLACE the next target. Packing that
    atomically made a ripe crop's HARVEST depend on optional work after the
    tile becomes empty. With many WHEAT tiles maturing together, routes that
    had enough time to harvest were rejected simply because the replacement
    tail did not fit.

    For urgent cycle-ending crop work, HARVEST is the mandatory boundary.
    Replacement work is admitted only after every mandatory task assigned to
    that worker already fits its real turn budget.
    """
    if task.urgent and task.ends_cycle:
        for index, action in enumerate(task.actions):
            if action and action[0] == "HARVEST":
                return task.actions[:index + 1]
    return task.actions


def _has_optional_tail(task):
    return len(_mandatory_actions(task)) < len(task.actions)


def _task_actions(task, include_optional=False):
    if include_optional or not _has_optional_tail(task):
        return task.actions
    return _mandatory_actions(task)


def _task_needs(task, include_optional=False):
    """Inputs needed by the actions actually admitted to the worker queue."""
    if _has_optional_tail(task) and not include_optional:
        # WATER/HARVEST need no shed inputs. build_tasks' needs on a finished
        # crop belong to the post-HARVEST replacement (seed, animal, feed...).
        return Counter()
    return task.needs


def _task_queue(start, bucket, shed_access, full_task_ids=frozenset()):
    """Build one route for the selected mandatory work and safe optional tails.

    DROP empties the worker's inventory. Only collect supplies through the
    next DROP, so later supplies stay available in the shed in the meantime.
    ``full_task_ids`` contains cycle-ending tasks whose post-HARVEST tail has
    been proven to fit without displacing any mandatory work.
    """
    queue, current = [], start
    pickup_needed = True
    for task_index, task in enumerate(bucket):
        include_optional = id(task) in full_task_ids
        if pickup_needed:
            needs = Counter()
            for later in bucket[task_index:]:
                needs.update(
                    _task_needs(later, include_optional=id(later) in full_task_ids)
                )
                if later.immediate_drop:
                    break
            if needs:
                shed = nearest_shed(current, shed_access)
                queue += route(current, shed)
                queue += [["PICKUP", item, amount] for item, amount in needs.items() if amount > 0]
                current = shed
            pickup_needed = False
        queue += route(current, task.position) + _task_actions(task, include_optional)
        current = task.position
        if task.immediate_drop:
            shed = nearest_shed(current, shed_access)
            queue += route(current, shed) + [["DROP"]]
            current = shed
            if task.refinance_feed:
                queue += [["PASS"], ["PICKUP", "WHEAT", 1]]
                queue += route(current, task.position) + [["FEED"], ["CARE"]]
                current = task.position
            pickup_needed = True
    return queue, current


def _admit_optional_tails(start, bucket, budget, shed_access):
    """Greedily restore post-HARVEST tails that fit after mandatory packing.

    Mandatory prefixes are fixed first. We then add the cheapest replacement
    tail one at a time, recomputing the exact queue (including shed pickups)
    after each admission. Therefore every admitted PLANT/PLACE may delay later
    harvests only within a queue whose *entire* length still fits the worker's
    real budget; a replacement can never make a ripe crop miss day end.
    """
    full_task_ids = set()
    optional = [task for task in bucket if _has_optional_tail(task)]
    while optional:
        base_length = len(
            _task_queue(start, bucket, shed_access, full_task_ids)[0]
        )
        choices = []
        for task in optional:
            candidate_ids = full_task_ids | {id(task)}
            projected = len(
                _task_queue(start, bucket, shed_access, candidate_ids)[0]
            )
            if projected <= budget:
                choices.append(
                    (
                        projected - base_length,
                        projected,
                        task.position[1],
                        task.position[0],
                        task,
                    )
                )
        if not choices:
            break
        _, _, _, _, chosen = min(choices, key=lambda item: item[:4])
        full_task_ids.add(id(chosen))
        optional.remove(chosen)
    return full_task_ids


def _pack(tasks, worker_starts, budgets, shed_access=SHED_ACCESS):
    """Greedy bin-pack using each candidate worker's exact mandatory queue.

    The first pass includes travel from the real spawn, required PICKUPs and
    every deadline-sensitive action, but treats post-HARVEST replacement tails
    as optional. build_queues restores as many tails as safely fit only after
    all mandatory work has been assigned.
    """
    buckets = [[] for _ in worker_starts]

    def work_length(start, bucket):
        return len(_task_queue(start, bucket, shed_access)[0])

    # Visit nearby work first within each priority class. The previous
    # action-length ordering put long/far tasks at the front (notably the
    # top rows of NW), so a worker crossed the farm and then doubled back to
    # perform work beside the shed. Distance is measured from the closest
    # real worker start; insertion then optimizes each worker's own route.
    ordered = sorted(
        tasks,
        key=lambda task: (
            not task.urgent,
            not task.animal_harvest,
            not (
                task.actions
                and task.actions[0][0] in ("FEED", "CARE", "COLLECT_FERTILIZER")
            ),
            task.deadline if task.deadline is not None else float("inf"),
            min(
                abs(start[0] - task.position[0]) + abs(start[1] - task.position[1])
                for start in worker_starts
            ),
            -len(_mandatory_actions(task)),
            task.position[1],
            task.position[0],
        ),
    )
    unassigned = []
    lengths = [0] * len(worker_starts)
    for task in ordered:
        candidates = []
        for worker, bucket in enumerate(buckets):
            for insertion in range(len(bucket) + 1):
                candidate = bucket[:insertion] + [task] + bucket[insertion:]
                # Animal survival work remains ahead of ordinary crop work,
                # but tasks within the same priority class may be inserted
                # wherever the route is shortest. Appending in global task
                # order was the source of workers crossing the farm first
                # and then doubling back to tiles beside their own spawn.
                priority = lambda queued: (
                    not queued.urgent,
                    not queued.animal_harvest,
                    not (
                        queued.actions
                        and queued.actions[0][0] in ("FEED", "CARE", "COLLECT_FERTILIZER")
                    ),
                    queued.deadline if queued.deadline is not None else float("inf"),
                    not queued.immediate_drop,
                )
                if any(
                    priority(candidate[index]) > priority(candidate[index + 1])
                    for index in range(len(candidate) - 1)
                ):
                    continue
                projected = work_length(worker_starts[worker], candidate)
                if projected <= budgets[worker]:
                    # Time-sensitive work must finish early for survival,
                    # crop expiry and same-day sales. For ordinary work,
                    # minimize extra travel and pickups instead.
                    time_sensitive = (
                        task.urgent or task.animal_harvest
                        or task.deadline is not None or task.immediate_drop
                    )
                    cost = projected if time_sensitive else projected - lengths[worker]
                    candidates.append((cost, projected, worker, insertion))
        candidates.sort()
        for _, projected, worker, insertion in candidates:
            buckets[worker].insert(insertion, task)
            lengths[worker] = projected
            break
        else:
            unassigned.append(task)
    return buckets, unassigned


def hands_needed(
    tasks,
    farmer_start,
    existing_hand_starts=(),
    shed_access=SHED_ACCESS,
    pending_hand_budget=HAND_BUDGET,
    max_hands=MAX_HANDS,
):
    """Fewest hands (0..max_hands) that fit today's mandatory work.

    Urgent animal care and ripe-crop harvests are never sacrificed to optional
    post-HARVEST replacement work. Replacement tails use spare turns after the
    minimum safe workforce has been established.

    `max_hands` defaults to the module ceiling but accepts a lower override --
    agents/planner.py's plan_targets uses this to test a candidate portfolio
    against a safety-margined hand count before committing to it.
    """
    minimum = len(existing_hand_starts)
    for count in range(minimum, max_hands + 1):
        starts = [tuple(farmer_start)] + predicted_hand_starts(
            farmer_start, existing_hand_starts, count
        )
        budgets = [FARMER_BUDGET]
        budgets += [HAND_BUDGET] * min(count, len(existing_hand_starts))
        budgets += [pending_hand_budget] * max(0, count - len(existing_hand_starts))
        _, unassigned = _pack(tasks, starts, budgets, shed_access)
        if not unassigned:
            return count, []
    starts = [tuple(farmer_start)] + predicted_hand_starts(
        farmer_start, existing_hand_starts, max_hands
    )
    budgets = [FARMER_BUDGET]
    budgets += [HAND_BUDGET] * min(max_hands, len(existing_hand_starts))
    budgets += [pending_hand_budget] * max(0, max_hands - len(existing_hand_starts))
    _, unassigned = _pack(tasks, starts, budgets, shed_access)
    return max_hands, unassigned


def build_queues(
    tasks,
    farmer_start,
    hand_count,
    hand_starts=(),
    shed_access=SHED_ACCESS,
    pending_hand_budget=HAND_BUDGET,
    existing_hand_budget=None,
    worker_budgets=None,
):
    """Assign tasks to workers, then spend spare turns on safe replacements.

    `worker_budgets` overrides all budgets when appending work after existing
    queues: each worker can have a different number of turns left.

    A worker returns to the shed to DROP mid-day only if it both carries a
    sellable item and has spare turns left after its assigned work. Inventories
    are dropped to the shed automatically at day end, so an early return must
    never displace mandatory work.

    `existing_hand_budget`, when given, limits already-real workers during
    intraday repacking; left None, the normal hour-1 full-day budgets apply.
    """
    starts = [tuple(farmer_start)] + predicted_hand_starts(
        farmer_start, hand_starts, hand_count
    )
    farmer_budget = FARMER_BUDGET if existing_hand_budget is None else existing_hand_budget
    existing_budget = HAND_BUDGET if existing_hand_budget is None else existing_hand_budget
    budgets = [farmer_budget]
    budgets += [existing_budget] * min(hand_count, len(hand_starts))
    budgets += [pending_hand_budget] * max(0, hand_count - len(hand_starts))
    if worker_budgets is not None:
        if len(worker_budgets) != len(starts):
            raise ValueError("worker_budgets must contain one budget per worker")
        budgets = list(worker_budgets)
    buckets, unassigned = _pack(tasks, starts, budgets, shed_access)

    plans = []
    for start, budget, bucket in zip(starts, budgets, buckets):
        if not bucket:
            plans.append(WorkerPlan(start, []))
            continue

        full_task_ids = _admit_optional_tails(start, bucket, budget, shed_access)
        queue, current = _task_queue(start, bucket, shed_access, full_task_ids)
        carries_sellable = any(task.sells and not task.immediate_drop for task in bucket)
        if carries_sellable:
            shed = nearest_shed(current, shed_access)
            trip_home = route(current, shed) + [["DROP"]]
            if len(queue) + len(trip_home) <= budget:
                queue += trip_home
        plans.append(WorkerPlan(start, queue))
    return plans, unassigned