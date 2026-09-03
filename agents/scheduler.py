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


def _pack(tasks, worker_starts, budgets, shed_access=SHED_ACCESS):
    """Greedy bin-pack using each candidate worker's exact queue length.

    The projection includes travel from the real spawn, one operation for
    every distinct PICKUP, travel between tasks, and all tile actions. The
    optional DROP trip is added later only when it fits.
    """
    buckets = [[] for _ in worker_starts]

    def work_length(start, bucket):
        """Exact queue length before the optional end-of-day DROP trip."""
        needs = sum((task.needs for task in bucket), Counter())
        current = start
        length = 0
        if needs:
            shed = nearest_shed(current, shed_access)
            length += abs(current[0] - shed[0]) + abs(current[1] - shed[1])
            # build_queues emits one PICKUP operation per distinct item.
            length += sum(1 for amount in needs.values() if amount)
            current = shed
        for task_index, task in enumerate(bucket):
            length += abs(current[0] - task.position[0]) + abs(current[1] - task.position[1])
            length += len(task.actions)
            current = task.position
            if task.immediate_drop:
                shed = nearest_shed(current, shed_access)
                length += abs(current[0] - shed[0]) + abs(current[1] - shed[1]) + 1
                current = shed
                if task.refinance_feed:
                    length += 2  # PASS, PICKUP
                    length += abs(current[0] - task.position[0]) + abs(current[1] - task.position[1])
                    length += 2  # FEED, CARE
                    current = task.position
                remaining_needs = sum(
                    (later.needs for later in bucket[task_index + 1:]), Counter()
                )
                if remaining_needs:
                    shed = nearest_shed(current, shed_access)
                    length += abs(current[0] - shed[0]) + abs(current[1] - shed[1])
                    length += sum(1 for amount in remaining_needs.values() if amount)
                    current = shed
        return length

    # Visit nearby work first within each priority class. The previous
    # action-length ordering put long/far tasks at the front (notably the
    # top rows of NW), so a worker crossed the farm and then doubled back to
    # perform work beside the shed. Distance is measured from the closest
    # real worker start; `_pack` then preserves this order in each bucket.
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
            -len(task.actions),
            task.position[1],
            task.position[0],
        ),
    )
    unassigned = []
    for task in ordered:
        candidates = []
        for worker, bucket in enumerate(buckets):
            for insertion in range(len(bucket) + 1):
                candidate = bucket[:insertion] + [task] + bucket[insertion:]
                # Animal survival work remains ahead of ordinary crop work,
                # but tasks within the same priority class may be inserted
                # wherever the route is shortest.  Appending in global task
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
                candidates.append(
                    (work_length(worker_starts[worker], candidate), worker, insertion)
                )
        candidates.sort()
        for projected, worker, insertion in candidates:
            if projected <= budgets[worker]:
                buckets[worker].insert(insertion, task)
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
    """Fewest hands (0..max_hands) that fit today's tasks within budget.

    Urgent (animal-care) tasks are never dropped for lack of hands -- a
    missed feed is a lasting loss (the animal escapes), not just a delayed
    harvest -- so this also reports which tasks had to be dropped if even
    max_hands can't fit everything.

    `max_hands` defaults to the module ceiling but accepts a lower override
    -- agents/planner.py's plan_targets uses this to test a candidate
    portfolio against a *safety-margined* hand count (a few below the real
    MAX_HANDS) before committing to it, so routine day-to-day task-mix
    variation never actually reaches the true ceiling in practice.
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
):
    """Assign tasks to farmer + hand_count hands and lay out each one's route.

    A worker returns to the shed to DROP mid-day only if it both carries a
    sellable item *and* has spare turns left after its assigned tasks --
    inventories are dropped to the shed automatically at day end regardless
    (see kaggriculture's _end_of_day), so an early return only pays for
    itself when it enables a same-day sale (selling.py) or frees carry
    capacity; it should never be bought at the cost of a task that would
    otherwise go undone.

    `pending_hand_budget` only ever reaches a hand beyond `len(hand_starts)`
    -- every slot up to that point is budgeted FARMER_BUDGET/HAND_BUDGET
    (a full day), because the normal caller (agents/expansion_agent.py's
    hour-1 queue build) really does hand each of those a full day. That
    assumption breaks for a caller re-packing *already-real* workers
    partway through the day (agents/expansion_agent.py's
    _schedule_late_placements, calling with `farmer_start`/`hand_starts`
    covering every currently-idle worker, real positions and all) -- there,
    every slot is "existing" by this function's accounting, so
    pending_hand_budget never applies to any of them and they were each
    silently budgeted a full day's worth of turns instead of what's
    actually left. Confirmed directly (seed 1, day 23): two mid-day
    workers built routes 17-18 steps long against only 16 real turns left,
    for tasks including a SHEEP already a day overdue on its feed.
    `existing_hand_budget`, when given, overrides both FARMER_BUDGET and
    HAND_BUDGET for exactly this case; left None, behavior is unchanged.
    """
    starts = [tuple(farmer_start)] + predicted_hand_starts(
        farmer_start, hand_starts, hand_count
    )
    farmer_budget = FARMER_BUDGET if existing_hand_budget is None else existing_hand_budget
    existing_budget = HAND_BUDGET if existing_hand_budget is None else existing_hand_budget
    budgets = [farmer_budget]
    budgets += [existing_budget] * min(hand_count, len(hand_starts))
    budgets += [pending_hand_budget] * max(0, hand_count - len(hand_starts))
    buckets, unassigned = _pack(tasks, starts, budgets, shed_access)

    plans = []
    for start, budget, bucket in zip(starts, budgets, buckets):
        if not bucket:
            plans.append(WorkerPlan(start, []))
            continue
        needs = sum((task.needs for task in bucket), Counter())
        carries_sellable = any(task.sells and not task.immediate_drop for task in bucket)
        queue, current = [], start
        if needs:
            shed = nearest_shed(current, shed_access)
            queue += route(current, shed)
            queue += [["PICKUP", item, amount] for item, amount in needs.items() if amount]
            current = shed
        for task_index, task in enumerate(bucket):
            queue += route(current, task.position) + task.actions
            current = task.position
            if task.immediate_drop:
                shed = nearest_shed(current, shed_access)
                queue += route(current, shed) + [["DROP"]]
                current = shed
                if task.refinance_feed:
                    queue += [["PASS"], ["PICKUP", "WHEAT", 1]]
                    queue += route(current, task.position) + [["FEED"], ["CARE"]]
                    current = task.position
                remaining_needs = sum(
                    (later.needs for later in bucket[task_index + 1:]), Counter()
                )
                if remaining_needs:
                    shed = nearest_shed(current, shed_access)
                    queue += route(current, shed)
                    queue += [
                        ["PICKUP", item, amount]
                        for item, amount in remaining_needs.items()
                        if amount
                    ]
                    current = shed
        if carries_sellable:
            shed = nearest_shed(current, shed_access)
            trip_home = route(current, shed) + [["DROP"]]
            if len(queue) + len(trip_home) <= budget:
                queue += trip_home
        plans.append(WorkerPlan(start, queue))
    return plans, unassigned
