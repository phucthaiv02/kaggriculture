"""Turn today's tasks into farmer/hand orders, minimizing labor cost.

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
# Cap daily hiring; actual headcount is bounded by workload and cash.
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


def _cashout_needed(bucket):
    if not any(task.cashout for task in bucket):
        return False
    for task in reversed(bucket):
        if task.immediate_drop:
            return False
        if task.sells:
            return True
    return False


def _cashout_length(current, bucket, shed_for):
    if not _cashout_needed(bucket):
        return 0
    shed = shed_for(current)
    return abs(current[0]-shed[0]) + abs(current[1]-shed[1]) + 1


def _task_queue(start, bucket, shed_access):
    """Build the mandatory route, sharing execution and packing accounting.

    DROP empties the worker's inventory. Only collect supplies through the
    next DROP, so later supplies stay available in the shed in the meantime.
    """
    queue, current = [], start
    pickup_needed = True
    for task_index, task in enumerate(bucket):
        if pickup_needed:
            needs = Counter()
            for later in bucket[task_index:]:
                needs.update(later.needs)
                if later.immediate_drop:
                    break
            if needs:
                shed = nearest_shed(current, shed_access)
                queue += route(current, shed)
                queue += [["PICKUP", item, amount] for item, amount in needs.items() if amount > 0]
                current = shed
            pickup_needed = False
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
            pickup_needed = True
    if _cashout_needed(bucket):
        shed = nearest_shed(current, shed_access)
        queue += route(current, shed) + [["DROP"]]
        current = shed
    return queue, current


def _task_length(start, bucket, shed_for):
    """Count mandatory steps without allocating speculative action queues."""
    # Almost every production route has just one initial supply pickup.
    # Avoid Counter allocation/update for every speculative insertion.
    if not any(task.immediate_drop for task in bucket):
        needs = {}
        for task in bucket:
            for item, amount in task.needs.items():
                needs[item] = needs.get(item, 0) + amount
        current, length = start, 0
        if needs:
            current = shed_for(start)
            length = abs(start[0]-current[0]) + abs(start[1]-current[1])
            length += sum(amount > 0 for amount in needs.values())
        for task in bucket:
            position = task.position
            length += abs(current[0]-position[0]) + abs(current[1]-position[1]) + len(task.actions)
            current = position
        return length + _cashout_length(current, bucket, shed_for)
    length, current, pickup_needed = 0, start, True
    for index, task in enumerate(bucket):
        if pickup_needed:
            needs = Counter()
            for later in bucket[index:]:
                needs.update(later.needs)
                if later.immediate_drop:
                    break
            if needs:
                shed = shed_for(current)
                length += abs(current[0]-shed[0]) + abs(current[1]-shed[1])
                length += sum(amount > 0 for amount in needs.values())
                current = shed
            pickup_needed = False
        position = task.position
        length += abs(current[0]-position[0]) + abs(current[1]-position[1]) + len(task.actions)
        current = position
        if task.immediate_drop:
            shed = shed_for(current)
            distance = abs(current[0]-shed[0]) + abs(current[1]-shed[1])
            length += distance + 1
            current = shed
            if task.refinance_feed:
                length += distance + 4
                current = position
            pickup_needed = True
    return length + _cashout_length(current, bucket, shed_for)


def _priority(task):
    return (
        not task.urgent,
        not task.animal_harvest,
        not (task.actions and task.actions[0][0] in ("FEED", "CARE", "COLLECT_FERTILIZER")),
        task.deadline if task.deadline is not None else float("inf"),
        not task.immediate_drop,
    )


def _bucket_budget(base_budget, bucket, terminal_day):
    """Return the executable worker budget for one terminal-day bucket.

    The terminal observation removes one normally executable worker turn from
    every worker. Only a bucket that still needs a final DROP must finish one
    turn earlier again, leaving the following observation for the shared SELL.
    """
    if not terminal_day:
        return base_budget
    return max(0, base_budget - 1 - int(_cashout_needed(bucket)))


def _pack_greedy(tasks, worker_starts, budgets, shed_access=SHED_ACCESS, variant=0):
    """Greedy bin-pack using each candidate worker's exact queue length.

    The projection includes travel from the real spawn, one operation for
    every distinct PICKUP, travel between tasks, and all tile actions. The
    optional DROP trip is added later only when it fits.
    """
    buckets = [[] for _ in worker_starts]

    sheds = {}
    def shed_for(position):
        if position not in sheds:
            sheds[position] = nearest_shed(position, shed_access)
        return sheds[position]

    priorities = {id(task): _priority(task) for task in tasks}
    reserve_turn = any(task.animal_harvest for task in tasks)
    terminal_day = any(task.cashout for task in tasks)

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
            -len(task.actions),
            task.position[1],
            task.position[0],
        ),
    )
    if variant:
        # Keep the safety classes, but try difficult / spatially grouped work
        # first so easy nearby jobs do not strand capacity at the farm edges.
        def alternative(task):
            x, y = task.position
            distance = min(abs(x-s[0]) + abs(y-s[1]) for s in worker_starts)
            geometry = ((-distance - len(task.actions), y, x) if variant == 1
                        else (x, y) if variant == 2 else (y, x))
            return priorities[id(task)], geometry
        ordered = sorted(tasks, key=alternative)
    unassigned = []
    lengths = [0] * len(worker_starts)
    for task in ordered:
        best = None
        priority = priorities[id(task)]
        time_sensitive = (task.urgent or task.animal_harvest
                          or task.deadline is not None or task.immediate_drop)
        for worker, bucket in enumerate(buckets):
            for insertion in range(len(bucket) + 1):
                # Buckets are already priority-sorted; only the two new
                # neighbours can violate the invariant after insertion.
                if insertion and priorities[id(bucket[insertion - 1])] > priority:
                    continue
                if insertion < len(bucket) and priority > priorities[id(bucket[insertion])]:
                    continue
                candidate = bucket[:insertion] + [task] + bucket[insertion:]
                projected = _task_length(worker_starts[worker], candidate, shed_for)
                # Keep one turn for an opportunistic animal harvest or DROP;
                # runtime guards may insert HARVEST while crossing a ready pen.
                projected += int(bool(variant) and reserve_turn)
                if projected <= _bucket_budget(budgets[worker], candidate, terminal_day):
                    # Time-sensitive work must finish early for survival,
                    # crop expiry and same-day sales. For ordinary work,
                    # minimize extra travel and pickups instead.
                    cost = projected if time_sensitive and not variant else projected - lengths[worker]
                    candidate = (cost, projected, worker, insertion)
                    if best is None or candidate < best:
                        best = candidate
        if best is not None:
            _, projected, worker, insertion = best
            buckets[worker].insert(insertion, task)
            lengths[worker] = projected
        else:
            unassigned.append(task)
    return buckets, unassigned


def _pack(tasks, worker_starts, budgets, shed_access=SHED_ACCESS):
    """Try bounded deterministic alternatives before paying for another hand.

    Preserve the original early-completion routes whenever they fit. A rescue
    uses identical tasks and priority constraints, only changing assignment and
    visit order. No search over targets, care dates or opening actions occurs.
    """
    best = _pack_greedy(tasks, worker_starts, budgets, shed_access)
    # The initial NW-only farm is governed by the opening's cash/refinancing
    # sequence. Preserve its worker assignments until land expansion.
    if len(shed_access) == 1 or not tasks or sum(len(t.actions) for t in tasks) > sum(budgets):
        return best
    # Keep investment tasks on their established route: intraday buyers use
    # these assignments to commit animals/seeds and pending hires together.
    if any(op[0] in ("PLANT", "PLACE", "DIG", "BUILD_COOP", "BUILD_PASTURE")
           for task in tasks for op in task.actions):
        return best
    terminal_day = any(task.cashout for task in tasks)
    def score(result):
        if result[1]:
            return (float("inf"), float("inf"))
        stranded, travel = 0, 0
        for start, budget, bucket in zip(worker_starts, budgets, result[0]):
            if not bucket:
                continue
            length = _task_length(start, bucket, lambda p: nearest_shed(p, shed_access))
            end = bucket[-1].position
            shed = nearest_shed(end, shed_access)
            home = abs(end[0]-shed[0]) + abs(end[1]-shed[1]) + 1
            effective_budget = _bucket_budget(budget, bucket, terminal_day)
            if not _cashout_needed(bucket) and length + home > effective_budget:
                stranded += sum(sum(task.sells.values()) for task in bucket)
            travel += length
        return stranded, travel

    best_score = score(best)
    preserve_complete = not best[1]
    for variant in (1, 2, 3):
        candidate = _pack_greedy(tasks, worker_starts, budgets, shed_access, variant)
        candidate_score = score(candidate)
        # Do not trade away tasks. Among complete schedules prefer routes
        # that can return more goods before the shed's end-of-day capacity cap.
        if candidate_score < best_score and (
            not preserve_complete or candidate_score[0] < best_score[0]
        ):
            best, best_score = candidate, candidate_score
    return best


def hands_needed(
    tasks,
    farmer_start,
    existing_hand_starts=(),
    shed_access=SHED_ACCESS,
    pending_hand_budget=HAND_BUDGET,
    max_hands=MAX_HANDS,
):
    """Find the fewest hands that fit today's tasks, up to max_hands.

    Return unassigned tasks when even the maximum headcount cannot fit them."""
    minimum = min(len(existing_hand_starts), max_hands)
    # Every action and each distinct positive supply pickup is unavoidable,
    # even with zero travel. Skip headcounts below this admissible bound.
    mandatory_steps = sum(len(task.actions) for task in tasks)
    mandatory_steps += len({item for task in tasks for item, amount in task.needs.items() if amount > 0})
    for count in range(minimum, max_hands + 1):
        starts = [tuple(farmer_start)] + predicted_hand_starts(
            farmer_start, existing_hand_starts, count
        )
        budgets = [FARMER_BUDGET]
        budgets += [HAND_BUDGET] * min(count, len(existing_hand_starts))
        budgets += [pending_hand_budget] * max(0, count - len(existing_hand_starts))
        if count < max_hands and sum(budgets) < mandatory_steps:
            continue
        _, unassigned = _pack(tasks, starts, budgets, shed_access)
        if not unassigned:
            return count, []
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
    """Assign tasks to farmer + hand_count hands and build their routes.

    pending_hand_budget applies only to hands not yet in hand_starts.
    existing_hand_budget overrides both the farmer and already-spawned hands
    for mid-day scheduling. worker_budgets overrides every worker individually.

    Append a return/DROP for sellable goods only when spare turns remain;
    never displace assigned work just to return to the shed."""
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
        carries_sellable = any(task.sells and not task.immediate_drop for task in bucket)
        if carries_sellable and not _cashout_needed(bucket):
            shed = nearest_shed(current, shed_access)
            trip_home = route(current, shed) + [["DROP"]]
            if len(queue) + len(trip_home) <= budget:
                queue += trip_home
        plans.append(WorkerPlan(start, queue))
    return plans, unassigned