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


_ANIMAL_SERVICE_OPS = ("FEED", "CARE", "COLLECT_FERTILIZER")


def _is_animal_service(task):
    return task.animal_harvest or bool(
        task.urgent and task.actions and task.actions[0][0] in _ANIMAL_SERVICE_OPS
    )


def _priority(task):
    return (
        not task.rescue,
        task.mandatory is False,
        not task.urgent,
        not task.animal_harvest,
        not (task.actions and task.actions[0][0] in _ANIMAL_SERVICE_OPS),
        task.deadline if task.deadline is not None else float("inf"),
        not task.immediate_drop,
    )


def _rescue_priority(task):
    """Relax only the split between urgent animal harvest and maintenance.

    This priority is never used for a route that already fits. It is a bounded
    fallback for a smaller headcount that the strict safety ordering could not
    pack, allowing one worker to service both tasks at the same pen in one visit.
    """
    if _is_animal_service(task):
        return (
            not task.rescue,
            task.mandatory is False,
            not task.urgent,
            False,
            False,
            task.deadline if task.deadline is not None else float("inf"),
            not task.immediate_drop,
        )
    return _priority(task)


def _bucket_budget(base_budget, bucket, terminal_day):
    """Return the executable worker budget for one terminal-day bucket.

    The terminal observation removes one normally executable worker turn from
    every worker. Only a bucket that still needs a final DROP must finish one
    turn earlier again, leaving the following observation for the shared SELL.
    """
    if not terminal_day:
        return base_budget
    return max(0, base_budget - 1 - int(_cashout_needed(bucket)))


def _pack_greedy(
    tasks,
    worker_starts,
    budgets,
    shed_access=SHED_ACCESS,
    variant=0,
    group_animal=False,
):
    """Greedy bin-pack using each candidate worker's exact queue length.

    The projection includes travel from the real spawn, one operation for
    every distinct PICKUP, travel between tasks, and all tile actions. The
    optional DROP trip is added later only when it fits.

    group_animal is an emergency packing mode used only after the strict route
    fails at a candidate headcount. Normal callers retain the original ordering.
    """
    buckets = [[] for _ in worker_starts]

    sheds = {}
    def shed_for(position):
        if position not in sheds:
            sheds[position] = nearest_shed(position, shed_access)
        return sheds[position]

    priority_for = _rescue_priority if group_animal else _priority
    priorities = {id(task): priority_for(task) for task in tasks}
    reserve_turn = any(task.animal_harvest for task in tasks)
    terminal_day = any(task.terminal_day or task.cashout for task in tasks)

    # Keep the production greedy ordering exactly unchanged unless this is the
    # explicit hand-count rescue. Opening and already-valid schedules therefore
    # cannot move because of the relaxed animal grouping.
    if group_animal:
        ordered = sorted(
            tasks,
            key=lambda task: (
                priorities[id(task)],
                min(
                    abs(start[0] - task.position[0]) + abs(start[1] - task.position[1])
                    for start in worker_starts
                ),
                task.position[1],
                task.position[0],
                not task.animal_harvest,
                -len(task.actions),
            ),
        )
    else:
        ordered = sorted(
            tasks,
            key=lambda task: (
                not task.rescue,
                task.mandatory is False,
                not task.urgent,
                not task.animal_harvest,
                not (
                    task.actions
                    and task.actions[0][0] in _ANIMAL_SERVICE_OPS
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
        task_is_service = group_animal and _is_animal_service(task)
        for worker, bucket in enumerate(buckets):
            paired_harvest = None
            paired_service = None
            if task_is_service:
                for index, queued in enumerate(bucket):
                    if queued.position != task.position:
                        continue
                    if queued.animal_harvest:
                        paired_harvest = index
                    elif _is_animal_service(queued):
                        paired_service = index
            for insertion in range(len(bucket) + 1):
                # Buckets are already priority-sorted; only the two new
                # neighbours can violate the invariant after insertion.
                if insertion and priorities[id(bucket[insertion - 1])] > priority:
                    continue
                if insertion < len(bucket) and priority > priorities[id(bucket[insertion])]:
                    continue
                # In rescue mode, tasks at the same pen may share a visit but
                # explicit HARVEST stays before that pen's FEED/CARE work.
                if task_is_service:
                    if task.animal_harvest and paired_service is not None and insertion > paired_service:
                        continue
                    if not task.animal_harvest and paired_harvest is not None and insertion <= paired_harvest:
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

    # The strict route remains authoritative whenever it fits. Only when it
    # leaves work behind may we relax harvest-vs-maintenance grouping, and then
    # only if the relaxed candidate completes the whole workload. This makes
    # the optimization capable of saving a hand without changing valid routes.
    if best[1] and any(_is_animal_service(task) for task in tasks):
        grouped = _pack_greedy(
            tasks, worker_starts, budgets, shed_access, group_animal=True
        )
        if not grouped[1]:
            return grouped

    terminal_day = any(task.terminal_day or task.cashout for task in tasks)
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
    marginal_hire_costs=None,
):
    """Find the fewest hands that fit today's tasks, up to max_hands.

    With marginal_hire_costs, cover mandatory work first, then buy capacity
    only when its additional optional cash exceeds the next hire price.
    Existing hands are sunk cost. Omitted prices retain capacity-only sizing.
    Return work excluded by capacity or the economic stopping rule."""
    minimum = min(len(existing_hand_starts), max_hands)
    # Every action and each distinct positive supply pickup is unavoidable,
    # even with zero travel. Skip headcounts below this admissible bound.
    required = tasks if marginal_hire_costs is None else [
        task for task in tasks if task.mandatory is not False
        and len(task.actions) <= max(FARMER_BUDGET, HAND_BUDGET, pending_hand_budget)
    ]
    mandatory_steps = sum(len(task.actions) for task in required)
    mandatory_steps += len({item for task in required for item, amount in task.needs.items() if amount > 0})
    candidates = []
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
        if marginal_hire_costs is None and not unassigned:
            return count, []
        candidates.append((count, unassigned))
    if marginal_hire_costs is None:
        return max_hands, unassigned

    # A single added hand can leave the same number of required tasks behind
    # while unlocking a better packing at the following headcount. Inspect the
    # bounded search as a whole instead of stopping on that local plateau.
    required_missing = [
        sum(task.mandatory is not False for task in missing)
        for _, missing in candidates
    ]
    best_required = min(required_missing)
    base_index = required_missing.index(best_required)
    chosen_count, chosen_missing = candidates[base_index]

    # Required capacity is paid for regardless of price. Beyond it, every
    # extra hand must recover enough optional value to cover its own Fibonacci
    # increment; existing hands remain sunk cost.
    prior_value = sum(
        task.value for task in chosen_missing if task.mandatory is False
    )
    for count, missing in candidates[base_index + 1:]:
        lost_value = sum(task.value for task in missing if task.mandatory is False)
        price = marginal_hire_costs[count - minimum - 1]
        if prior_value - lost_value <= price:
            break
        chosen_count, chosen_missing, prior_value = count, missing, lost_value
    return chosen_count, chosen_missing


def _rebalance_feed(buckets, starts, budgets, shed_access, tasks):
    """Consolidate stocked feed in two bounded passes without delaying other work.

    Prefer fewer feed carriers (four turns per carrier in the search score),
    then shorter mandatory routes. A receiving route must also fit its return
    and one spare turn. Keep the caller's headcount and unassigned work intact.
    """
    terminal = any(t.terminal_day or t.cashout for t in tasks)
    def service(t):
        return t.needs.get("WHEAT", 0) > 0 and not t.immediate_drop and all(
            op[0] in ("FEED", "CARE", "COLLECT_FERTILIZER") for op in t.actions)
    def metrics(worker, bucket):
        length = _task_length(starts[worker], bucket, lambda p: nearest_shed(p, shed_access))
        if not bucket:
            return length, 0, 0
        end = bucket[-1].position
        shed = nearest_shed(end, shed_access)
        home = abs(end[0]-shed[0]) + abs(end[1]-shed[1]) + 1
        late = (sum(sum(t.sells.values()) for t in bucket)
                if not _cashout_needed(bucket) and length + home > _bucket_budget(budgets[worker], bucket, terminal) else 0)
        feeds = int(any(t.needs.get("WHEAT", 0) for t in bucket))
        return length, late, feeds
    for _ in range(2):
        changed = False
        for source in range(len(buckets)-1, -1, -1):
            for task in list(buckets[source]):
                if not service(task):
                    continue
                index = buckets[source].index(task)
                remaining = buckets[source][:index] + buckets[source][index+1:]
                old_source = metrics(source, buckets[source])
                new_source = metrics(source, remaining)
                best = None
                for target in range(len(buckets)):
                    if source == target or not any(service(t) for t in buckets[target]):
                        continue
                    bucket = buckets[target]
                    old_target = metrics(target, bucket)
                    for insertion in range(len(bucket)+1):
                        # Insert after fixed work so planting and harvest are not delayed.
                        if any(not service(t) for t in bucket[insertion:]):
                            continue
                        # Do not delay expiry-sensitive crop work.
                        if any(t.deadline is not None for t in bucket[insertion:]):
                            continue
                        merged = bucket[:insertion] + [task] + bucket[insertion:]
                        new_target = metrics(target, merged)
                        end = merged[-1].position
                        shed = nearest_shed(end, shed_access)
                        home = 0 if _cashout_needed(merged) else abs(end[0]-shed[0]) + abs(end[1]-shed[1]) + 1
                        if new_target[0] + home + 1 > _bucket_budget(budgets[target], merged, terminal):
                            continue
                        if new_source[1] + new_target[1] > old_source[1] + old_target[1]:
                            continue
                        delta_steps = new_source[0] + new_target[0] - old_source[0] - old_target[0]
                        delta_feeds = new_source[2] + new_target[2] - old_source[2] - old_target[2]
                        score = delta_steps + 4 * delta_feeds
                        if score < 0 and (best is None or score < best[0]):
                            best = score, target, merged
                if best:
                    _, target, merged = best
                    buckets[source], buckets[target] = remaining, merged
                    changed = True
        if not changed:
            break
    return buckets


def build_queues(
    tasks,
    farmer_start,
    hand_count,
    hand_starts=(),
    shed_access=SHED_ACCESS,
    pending_hand_budget=HAND_BUDGET,
    existing_hand_budget=None,
    worker_budgets=None,
    available_wheat=None,
):
    """Assign tasks to farmer + hand_count hands and build their routes.

    pending_hand_budget applies only to hands not yet in hand_starts.
    existing_hand_budget overrides both the farmer and already-spawned hands
    for mid-day scheduling. worker_budgets overrides every worker individually.

    Append a return/DROP for sellable goods only when spare turns remain;
    never displace assigned work just to return to the shed. When available_wheat
    is supplied, consolidate stocked feed deliveries after assignment."""
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

    demand = sum(t.needs.get("WHEAT", 0) for t in tasks)
    if available_wheat is not None and 1 < demand <= available_wheat and len(shed_access) > 1:
        buckets = _rebalance_feed(buckets, starts, budgets, shed_access, tasks)

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
