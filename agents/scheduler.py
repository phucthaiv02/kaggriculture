"""Function 2: turn today's tasks into farmer/hand orders, minimizing labor cost.

Two costs compound here and both matter:
  - Hiring is Fibonacci-priced *per day* (hands don't persist -- every hand is
    rehired from scratch each morning), so hiring more than the day's actual
    workload requires is pure waste.
  - Queues are built at hour 1, after hour-0 purchases and hires have landed.
    Both the farmer and hands hired at hour 0 therefore have hours 1..23: 23
    executable steps.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

FARMER_BUDGET = 23
HAND_BUDGET = 23
SHED = (4, 4)
SHED_ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))
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
    queue: list


def _mandatory_actions(task):
    """Return the deadline-sensitive prefix of a finished crop transition.

    A ripe one-time crop can encode WATER, HARVEST and the next PLANT/PLACE in
    one Task. The replacement is not allowed to decide whether the harvest
    itself fits. This matters especially for a synchronized WHEAT harvest wall.
    """
    if task.urgent and task.ends_cycle:
        for index, action in enumerate(task.actions):
            if action and action[0] == "HARVEST":
                return task.actions[:index + 1]
    return task.actions


def _optional_actions(task):
    mandatory = _mandatory_actions(task)
    if len(mandatory) == len(task.actions):
        return []
    return task.actions[len(mandatory):]


def _mandatory_needs(task):
    """Inputs required before the mandatory prefix finishes."""
    if _optional_actions(task):
        # Finished-crop transition inputs belong to work after HARVEST.
        return Counter()
    return task.needs


def _task_queue(start, bucket, shed_access):
    """Build only mandatory work for a worker.

    Keeping this queue free of post-HARVEST PLANT/PLACE is stronger than a
    static length check: the agent intentionally retries an unavailable PICKUP
    or PLANT at the front of its queue. A retrying optional action must never
    sit in front of another ripe crop's HARVEST.
    """
    queue, current = [], start
    pickup_needed = True
    for task_index, task in enumerate(bucket):
        if pickup_needed:
            needs = Counter()
            for later in bucket[task_index:]:
                needs.update(_mandatory_needs(later))
                if later.immediate_drop:
                    break
            if needs:
                shed = nearest_shed(current, shed_access)
                queue += route(current, shed)
                queue += [["PICKUP", item, amount] for item, amount in needs.items() if amount > 0]
                current = shed
            pickup_needed = False
        queue += route(current, task.position) + _mandatory_actions(task)
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


def _tail_queue(start, task, shed_access):
    """Route one optional replacement after all mandatory work is complete."""
    actions = _optional_actions(task)
    if not actions:
        return [], start
    queue, current = [], start
    if task.needs:
        shed = nearest_shed(current, shed_access)
        queue += route(current, shed)
        queue += [["PICKUP", item, amount] for item, amount in task.needs.items() if amount > 0]
        current = shed
    queue += route(current, task.position) + actions
    return queue, task.position


def _append_optional_tails(queue, current, bucket, budget, shed_access):
    """Spend spare turns on replacements, strictly after every harvest.

    Retried PICKUP/PLANT operations can overrun their nominal duration, so no
    optional tail is ever inserted into the mandatory route. Tails are appended
    only as a postlude; if one stalls, it can only delay other optional work.
    """
    remaining = [task for task in bucket if _optional_actions(task)]
    while remaining:
        choices = []
        for task in remaining:
            extension, endpoint = _tail_queue(current, task, shed_access)
            if len(queue) + len(extension) <= budget:
                choices.append((len(extension), task.position[1], task.position[0], task, extension, endpoint))
        if not choices:
            break
        _, _, _, chosen, extension, endpoint = min(choices, key=lambda item: item[:3])
        queue += extension
        current = endpoint
        remaining.remove(chosen)
    return queue, current


def _pack(tasks, worker_starts, budgets, shed_access=SHED_ACCESS):
    """Greedy bin-pack using exact mandatory queue length."""
    buckets = [[] for _ in worker_starts]

    def work_length(start, bucket):
        return len(_task_queue(start, bucket, shed_access)[0])

    ordered = sorted(
        tasks,
        key=lambda task: (
            not task.urgent,
            not (task.urgent and any(action and action[0] == "WATER" for action in _mandatory_actions(task))),
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
                priority = lambda queued: (
                    not queued.urgent,
                    not (queued.urgent and any(action and action[0] == "WATER" for action in _mandatory_actions(queued))),
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
    """Fewest hands that fit all mandatory work within the day."""
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
    """Assign mandatory work, then append safe post-HARVEST replacement tails."""
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
        queue, current = _task_queue(start, bucket, shed_access)
        queue, current = _append_optional_tails(
            queue, current, bucket, budget, shed_access
        )
        carries_sellable = any(task.sells and not task.immediate_drop for task in bucket)
        if carries_sellable:
            shed = nearest_shed(current, shed_access)
            trip_home = route(current, shed) + [["DROP"]]
            if len(queue) + len(trip_home) <= budget:
                queue += trip_home
        plans.append(WorkerPlan(start, queue))
    return plans, unassigned