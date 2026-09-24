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
# Cap daily hiring; overflow HIRE orders are dispatched in morning batches
# before routes freeze, so admission may use the full supported workforce.
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


def _uncovered_needs(needs, inventory):
    """Return supplies which are not already carried by this worker."""
    available = Counter(inventory or {})
    missing = Counter()
    for item, amount in needs.items():
        carried = min(amount, available[item])
        available[item] -= carried
        if amount > carried:
            missing[item] = amount - carried
    return missing


def _task_queue(start, bucket, shed_access, initial_inventory=None):
    """Build the mandatory route, sharing execution and packing accounting.

    DROP empties the worker's inventory. Only collect supplies through the
    next DROP, so later supplies stay available in the shed in the meantime.
    """
    queue, current = [], start
    carried = initial_inventory
    pickup_needed = True
    for task_index, task in enumerate(bucket):
        if pickup_needed and (task.needs or not task.ends_cycle):
            needs = Counter()
            for later in bucket[task_index:]:
                needs.update(later.needs)
                if later.immediate_drop:
                    break
            needs = _uncovered_needs(needs, carried)
            carried = None
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


def _task_length(start, bucket, shed_for, initial_inventory=None):
    """Count mandatory steps without allocating speculative action queues."""
    # Almost every production route has just one initial supply pickup.
    # Avoid Counter allocation/update for every speculative insertion.
    if not any(task.immediate_drop for task in bucket):
        needs = {}
        for task in bucket:
            for item, amount in task.needs.items():
                needs[item] = needs.get(item, 0) + amount
        needs = _uncovered_needs(needs, initial_inventory)
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
    carried = initial_inventory
    for index, task in enumerate(bucket):
        if pickup_needed and (task.needs or not task.ends_cycle):
            needs = Counter()
            for later in bucket[index:]:
                needs.update(later.needs)
                if later.immediate_drop:
                    break
            needs = _uncovered_needs(needs, carried)
            carried = None
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
    if task.mandatory is False:
        return (True, not task.immediate_drop)
    return (
        False,
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


def _pack_greedy(
    tasks,
    worker_starts,
    budgets,
    shed_access=SHED_ACCESS,
    variant=0,
    group_animal=False,
    worker_inventories=None,
):
    """Greedy bin-pack using exact route length with an O(1) common fast path.

    Normal production buckets have no forced mid-route DROP/cashout. For those
    buckets, inserting one task only changes two Manhattan edges, action count
    and (at most) the initial shed pickup set, so recomputing the whole route
    for every candidate insertion is unnecessary. Opening refinance/cashout
    tasks retain the exact legacy projection as a bounded slow path.
    """
    buckets = [[] for _ in worker_starts]
    worker_inventories = list(worker_inventories or [{} for _ in worker_starts])

    sheds = {}
    def shed_for(position):
        if position not in sheds:
            sheds[position] = nearest_shed(position, shed_access)
        return sheds[position]

    def distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    priorities = {id(task): _priority(task) for task in tasks}
    terminal_day = any(task.terminal_day or task.cashout for task in tasks)

    if group_animal:
        ordered = sorted(
            tasks,
            key=lambda task: (
                priorities[id(task)],
                min(distance(start, task.position) for start in worker_starts),
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
                priorities[id(task)],
                min(distance(start, task.position) for start in worker_starts),
                -len(task.actions),
                task.position[1],
                task.position[0],
            ),
        )
    if variant:
        def alternative(task):
            x, y = task.position
            dist = min(distance((x, y), start) for start in worker_starts)
            geometry = ((-dist - len(task.actions), y, x) if variant == 1
                        else (x, y) if variant == 2 else (y, x))
            return priorities[id(task)], geometry
        ordered = sorted(tasks, key=alternative)

    def admission_key(task):
        dist = min(distance(start, task.position) for start in worker_starts)
        capacity = dist + len(task.actions) + len(task.needs)
        if task.cashout:
            shed = shed_for(task.position)
            capacity += distance(task.position, shed) + 1
        return priorities[id(task)], -task.value / max(1, capacity) if task.mandatory is False else 0

    ordered.sort(key=admission_key)
    unassigned = []
    lengths = [0] * len(worker_starts)

    # Metadata for the common no-mid-route-DROP case.
    # path_lengths excludes the initial worker->shed leg and PICKUP ops.
    fast_bucket = [True] * len(worker_starts)
    path_lengths = [0] * len(worker_starts)
    action_steps = [0] * len(worker_starts)
    need_items = [set() for _ in worker_starts]

    for task in ordered:
        best = None
        best_fast = None
        priority = priorities[id(task)]
        time_sensitive = task.immediate_drop or task.cashout
        task_is_service = group_animal and _is_animal_service(task)
        task_need_items = {
            item for item, amount in task.needs.items() if amount > 0
        }
        task_fast = not task.immediate_drop and not task.cashout

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

            deferred_cashout = any(
                queued.ends_cycle and not queued.needs for queued in [*bucket, task]
            )
            use_fast = (
                fast_bucket[worker] and task_fast and not worker_inventories[worker]
                and not ((need_items[worker] or task_need_items) and deferred_cashout)
            )
            if use_fast:
                old_items = need_items[worker]
                new_pickups = len(old_items | task_need_items)
                old_origin = shed_for(worker_starts[worker]) if old_items else worker_starts[worker]
                new_origin = shed_for(worker_starts[worker]) if new_pickups else worker_starts[worker]
                base_path = path_lengths[worker]
                if bucket and old_origin != new_origin:
                    first = bucket[0].position
                    base_path += distance(new_origin, first) - distance(old_origin, first)
                prefix = distance(worker_starts[worker], new_origin) if new_pickups else 0
                effective_budget = max(0, budgets[worker] - int(terminal_day))

            for insertion in range(len(bucket) + 1):
                if insertion and priorities[id(bucket[insertion - 1])] > priority:
                    continue
                if insertion < len(bucket) and priority > priorities[id(bucket[insertion])]:
                    continue
                if task_is_service:
                    if task.animal_harvest and paired_service is not None and insertion > paired_service:
                        continue
                    if not task.animal_harvest and paired_harvest is not None and insertion <= paired_harvest:
                        continue

                fast_detail = None
                if use_fast:
                    previous = new_origin if insertion == 0 else bucket[insertion - 1].position
                    candidate_path = base_path
                    if insertion < len(bucket):
                        following = bucket[insertion].position
                        candidate_path += (
                            distance(previous, task.position)
                            + distance(task.position, following)
                            - distance(previous, following)
                        )
                    else:
                        candidate_path += distance(previous, task.position)
                    projected = (
                        prefix + new_pickups + candidate_path
                        + action_steps[worker] + len(task.actions)
                    )
                    fits = projected <= effective_budget
                    fast_detail = candidate_path
                else:
                    candidate_bucket = bucket[:insertion] + [task] + bucket[insertion:]
                    projected = _task_length(
                        worker_starts[worker], candidate_bucket, shed_for,
                        worker_inventories[worker],
                    )
                    fits = projected <= _bucket_budget(
                        budgets[worker], candidate_bucket, terminal_day
                    )

                if fits:
                    cost = projected if time_sensitive and not variant else projected - lengths[worker]
                    rank = (cost, projected, worker, insertion)
                    if best is None or rank < best:
                        best = rank
                        best_fast = fast_detail if use_fast else None

        if best is not None:
            _, projected, worker, insertion = best
            buckets[worker].insert(insertion, task)
            lengths[worker] = projected
            if best_fast is not None:
                path_lengths[worker] = best_fast
                action_steps[worker] += len(task.actions)
                need_items[worker].update(task_need_items)
            else:
                fast_bucket[worker] = False
        else:
            unassigned.append(task)
    return buckets, unassigned

def _full_assignment_step_lower_bound(tasks, worker_starts):
    """Necessary total-step bound for assigning every task.

    Any complete schedule must execute every task action and at least one
    PICKUP for each distinct required item.  Its worker routes must also
    connect every distinct task position to at least one worker start.  The
    multi-source Manhattan MST is a lower bound on that route length: treating
    every worker start as connected to a virtual root for free can only make
    the network cheaper than real worker routes.

    If this bound already exceeds aggregate worker budgets, no packing variant
    can possibly assign every task.  In that case _pack historically returns
    the default greedy result after spending time on variants that all fail,
    so callers may skip those variants without changing the result.
    """
    if not tasks:
        return 0
    actions = sum(len(task.actions) for task in tasks)
    pickups = len({
        item for task in tasks for item, amount in task.needs.items()
        if amount > 0
    })
    remaining = set(task.position for task in tasks)
    if not remaining:
        return actions + pickups

    starts = tuple(worker_starts)
    # _pack always has at least the farmer, but keep the helper total for
    # isolated callers/tests.  With no starts, no non-empty assignment exists.
    if not starts:
        return float('inf')

    def distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    best = {
        position: min(distance(position, start) for start in starts)
        for position in remaining
    }
    travel = 0
    while best:
        position, edge = min(
            best.items(), key=lambda item: (item[1], item[0][1], item[0][0])
        )
        travel += edge
        del best[position]
        for other in tuple(best):
            step = distance(position, other)
            if step < best[other]:
                best[other] = step
    return actions + pickups + travel


def _pack(
    tasks,
    worker_starts,
    budgets,
    shed_access=SHED_ACCESS,
    optimize_routes=True,
    worker_inventories=None,
):
    """Pack tasks, separating feasibility search from route polishing.

    hands_needed only needs a feasible assignment; spending three extra full
    packing passes to improve an already-feasible route made morning planning
    scale badly with farm size. Final queue construction may still request
    route polishing, but stops as soon as no sellable output is stranded.
    """
    best = _pack_greedy(
        tasks, worker_starts, budgets, shed_access,
        worker_inventories=worker_inventories,
    )
    if not tasks or _full_assignment_step_lower_bound(tasks, worker_starts) > sum(budgets):
        return best

    if best[1] and any(_is_animal_service(task) for task in tasks):
        grouped = _pack_greedy(
            tasks, worker_starts, budgets, shed_access, group_animal=True,
            worker_inventories=worker_inventories,
        )
        if not grouped[1]:
            return grouped

    if not optimize_routes:
        if not best[1]:
            return best
        for variant in (1, 2, 3):
            candidate = _pack_greedy(
                tasks, worker_starts, budgets, shed_access, variant,
                worker_inventories=worker_inventories,
            )
            if not candidate[1]:
                return candidate
        return best

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
    # Once all assigned workers can return sellable output, extra full repacks
    # only optimize travel distance and are not worth risking the 1s act budget.
    if not best[1] and best_score[0] == 0:
        return best

    for variant in (1, 2, 3):
        candidate = _pack_greedy(
            tasks, worker_starts, budgets, shed_access, variant,
            worker_inventories=worker_inventories,
        )
        candidate_score = score(candidate)
        if candidate_score < best_score:
            best, best_score = candidate, candidate_score
            if best_score[0] == 0:
                break
    return best

def hands_needed(
    tasks,
    farmer_start,
    existing_hand_starts=(),
    shed_access=SHED_ACCESS,
    pending_hand_budget=HAND_BUDGET,
    existing_hand_budget=None,
    max_hands=MAX_HANDS,
    marginal_hire_costs=None,
):
    """Find the fewest/economically useful hands without polishing each route.

    Feasibility sizing is cheaper than final route construction. With economic
    sizing, larger Fibonacci-priced headcounts stop being evaluated once their
    cumulative extra hire cost exceeds all optional value that could possibly
    be recovered from the first fully-mandatory schedule.
    """
    minimum = min(len(existing_hand_starts), max_hands)
    existing_budget = HAND_BUDGET if existing_hand_budget is None else existing_hand_budget
    required = tasks if marginal_hire_costs is None else [
        task for task in tasks if task.mandatory is not False
        and len(task.actions) <= max(existing_budget, pending_hand_budget)
    ]
    mandatory_steps = sum(len(task.actions) for task in required)
    mandatory_steps += len({
        item for task in required for item, amount in task.needs.items() if amount > 0
    })
    candidates = []
    first_required_count = None
    max_optional_recovery = 0.0

    for count in range(minimum, max_hands + 1):
        if marginal_hire_costs is not None and first_required_count is not None:
            extra_cost = sum(
                marginal_hire_costs[
                    first_required_count - minimum:count - minimum
                ]
            )
            if extra_cost > max_optional_recovery:
                break

        starts = [tuple(farmer_start)] + predicted_hand_starts(
            farmer_start, existing_hand_starts, count
        )
        budgets = [existing_budget]
        budgets += [existing_budget] * min(count, len(existing_hand_starts))
        budgets += [pending_hand_budget] * max(0, count - len(existing_hand_starts))
        if count < max_hands and sum(budgets) < mandatory_steps:
            continue

        _, unassigned = _pack(
            tasks, starts, budgets, shed_access, optimize_routes=False
        )
        if marginal_hire_costs is None and not unassigned:
            return count, []

        candidates.append((count, unassigned))
        if marginal_hire_costs is not None and first_required_count is None:
            missing_required = sum(
                task.mandatory is not False for task in unassigned
            )
            if missing_required == 0:
                first_required_count = count
                max_optional_recovery = sum(
                    max(0.0, task.value)
                    for task in unassigned if task.mandatory is False
                )

    if marginal_hire_costs is None:
        return max_hands, unassigned

    required_missing = [
        sum(task.mandatory is not False for task in missing)
        for _, missing in candidates
    ]
    best_required = min(required_missing)
    base_index = required_missing.index(best_required)
    chosen_count, chosen_missing = candidates[base_index]

    base_count = chosen_count
    best_net = -sum(task.value for task in chosen_missing if task.mandatory is False)
    for index in range(base_index + 1, len(candidates)):
        count, missing = candidates[index]
        if required_missing[index] != best_required:
            continue
        lost_value = sum(task.value for task in missing if task.mandatory is False)
        price = sum(marginal_hire_costs[base_count - minimum:count - minimum])
        net = -lost_value - price
        if net > best_net:
            chosen_count, chosen_missing, best_net = count, missing, net
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
    worker_inventories=None,
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
    worker_inventories = list(worker_inventories or [{} for _ in starts])
    if len(worker_inventories) != len(starts):
        raise ValueError("worker_inventories must contain one inventory per worker")
    buckets, unassigned = _pack(
        tasks, starts, budgets, shed_access,
        worker_inventories=worker_inventories,
    )

    demand = sum(t.needs.get("WHEAT", 0) for t in tasks)
    if available_wheat is not None and 1 < demand <= available_wheat and len(shed_access) > 1:
        admitted_buckets = buckets
        rebalanced = _rebalance_feed(
            [list(bucket) for bucket in buckets], starts, budgets, shed_access, tasks
        )
        # Rebalancing is only a route optimization. It may never invalidate
        # admission by producing a concrete queue which cannot finish inside
        # that worker remaining turns. Validate exact queues, including carried
        # supplies, rather than relying on another route-length estimate.
        terminal = any(t.terminal_day or t.cashout for t in tasks)
        rebalanced_fits = True
        for start, budget, bucket, inventory in zip(
            starts, budgets, rebalanced, worker_inventories
        ):
            queue, _ = _task_queue(start, bucket, shed_access, inventory)
            if len(queue) > _bucket_budget(budget, bucket, terminal):
                rebalanced_fits = False
                break
        buckets = rebalanced if rebalanced_fits else admitted_buckets

    plans = []
    for start, budget, bucket, inventory in zip(starts, budgets, buckets, worker_inventories):
        if not bucket:
            plans.append(WorkerPlan(start, []))
            continue
        queue, current = _task_queue(start, bucket, shed_access, inventory)
        carries_sellable = any(task.sells and not task.immediate_drop for task in bucket)
        if carries_sellable and not _cashout_needed(bucket):
            shed = nearest_shed(current, shed_access)
            trip_home = route(current, shed) + [["DROP"]]
            if len(queue) + len(trip_home) <= budget:
                queue += trip_home
        plans.append(WorkerPlan(start, queue))
    return plans, unassigned
