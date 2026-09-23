"""Rebuild worker routes from the live observation on every executable turn.

The daily planner still decides which positions/tasks are admitted. This
module only re-optimizes execution order and worker assignment from the current
worker positions and inventories. Routes are intentionally ephemeral: the
agent executes one operation, observes the result, then rebuilds them again.
"""

from __future__ import annotations

from collections import Counter

from agents.scheduler import WorkerPlan, build_queues, nearest_shed, route


def planning_observation(obs):
    """Expose carried inputs to task generation without mutating the real obs.

    build_tasks historically looked only at the shed when deciding whether an
    animal/feed/fertilizer continuation could be scheduled. Under rolling
    execution an input may already be on a worker from the previous turn, so
    treating carried stock as planning-available prevents a valid continuation
    from disappearing after a PICKUP or HARVEST.
    """
    private = dict(obs["private"])
    available = Counter(private["shed"])
    for inventory in private.get("inventories", ()):
        available.update(inventory)
    private["shed"] = dict(available)
    return dict(obs, private=private)


def _positive(counter):
    return Counter({item: amount for item, amount in counter.items() if amount > 0})


def _missing(needs, inventory):
    return Counter({
        item: amount - inventory.get(item, 0)
        for item, amount in needs.items()
        if amount > inventory.get(item, 0)
    })


def _consume(inventory, needs):
    for item, amount in needs.items():
        if amount <= 0:
            continue
        used = min(inventory.get(item, 0), amount)
        if used:
            inventory[item] -= used
            if inventory[item] <= 0:
                inventory.pop(item, None)


def _bucket_queue(start, bucket, shed_access, initial_inventory):
    """Build one route while consuming stock already carried by this worker."""
    queue, current = [], tuple(start)
    carried = _positive(Counter(initial_inventory))
    pickup_needed = True

    for task_index, task in enumerate(bucket):
        if pickup_needed:
            needs = Counter()
            for later in bucket[task_index:]:
                needs.update(later.needs)
                if later.immediate_drop:
                    break
            missing = _missing(needs, carried)
            if missing:
                shed = nearest_shed(current, shed_access)
                queue += route(current, shed)
                queue += [["PICKUP", item, amount]
                          for item, amount in missing.items() if amount > 0]
                carried.update(missing)
                current = shed
            pickup_needed = False

        queue += route(current, task.position) + task.actions
        current = task.position
        _consume(carried, task.needs)

        # Outputs become carried stock after the task. This lets a same-route
        # continuation consume freshly harvested WHEAT without a shed detour.
        for item, amount in task.sells.items():
            if amount > 0:
                carried[item] += amount

        if task.immediate_drop:
            shed = nearest_shed(current, shed_access)
            queue += route(current, shed) + [["DROP"]]
            current = shed
            carried.clear()
            if task.refinance_feed:
                queue += [["PASS"], ["PICKUP", "WHEAT", 1]]
                queue += route(current, task.position) + [["FEED"], ["CARE"]]
                current = task.position
            pickup_needed = True

    if any(task.cashout for task in bucket):
        carries = False
        for task in reversed(bucket):
            if task.immediate_drop:
                break
            if task.sells:
                carries = True
                break
        if carries:
            shed = nearest_shed(current, shed_access)
            queue += route(current, shed) + [["DROP"]]
            current = shed
            carried.clear()

    return queue, current, carried


def _task_demand(tasks):
    demand = Counter()
    for task in tasks:
        demand.update(task.needs)
    return _positive(demand)


def _choose_carried_assignment(tasks, starts, inventories, budgets, shed_access, shed):
    """Pin work that can consume stock already carried by a worker.

    The ordinary scheduler assumes task inputs are in the shed. Rolling
    execution invalidates that assumption immediately after a PICKUP or a
    harvest. Pinning such tasks to the carrier keeps input ownership explicit;
    all other tasks remain free for normal spatial optimization.
    """
    remaining = list(tasks)
    buckets = [[] for _ in starts]
    carried = [_positive(Counter(inv)) for inv in inventories]
    shed_left = _positive(Counter(shed))

    while remaining:
        best = None
        for task_index, task in enumerate(remaining):
            if not task.needs:
                continue
            for worker, inventory in enumerate(carried):
                covered = sum(
                    min(inventory.get(item, 0), amount)
                    for item, amount in task.needs.items()
                )
                if covered <= 0:
                    continue
                missing = _missing(task.needs, inventory)
                if any(missing[item] > shed_left.get(item, 0) for item in missing):
                    continue

                candidate_bucket = buckets[worker] + [task]
                candidate_queue, _end, _left = _bucket_queue(
                    starts[worker], candidate_bucket, shed_access, inventories[worker]
                )
                if len(candidate_queue) > budgets[worker]:
                    continue

                current = starts[worker] if not buckets[worker] else buckets[worker][-1].position
                distance = abs(current[0] - task.position[0]) + abs(current[1] - task.position[1])
                key = (-covered, distance, len(candidate_queue), task.position[1],
                       task.position[0], worker, task_index)
                if best is None or key < best[0]:
                    best = (key, task_index, worker, missing)

        if best is None:
            break

        _key, task_index, worker, missing = best
        task = remaining.pop(task_index)
        buckets[worker].append(task)
        _consume(carried[worker], task.needs)
        for item, amount in missing.items():
            shed_left[item] -= amount
            if shed_left[item] <= 0:
                shed_left.pop(item, None)

    remaining_demand = _task_demand(remaining)
    stranded_workers = []
    for worker, inventory in enumerate(carried):
        if any(inventory.get(item, 0) and remaining_demand.get(item, 0)
               for item in inventory):
            stranded_workers.append(worker)

    if any(remaining_demand[item] > shed_left.get(item, 0) for item in remaining_demand):
        return buckets, remaining, shed_left, stranded_workers
    return buckets, remaining, shed_left, []


def build_rolling_queues(
    tasks,
    starts,
    inventories,
    budgets,
    shed_access,
    shed,
):
    """Return ephemeral routes optimized from the current turn's live state.

    The result has one WorkerPlan per current worker. Only the first operation
    is meant to execute; callers rebuild from the next observation.

    Carried inventory is a real dependency, not stale queue state. Work that
    can consume it stays with that carrier. Any inventory left after those
    continuations is routed back to the shed before unrelated work. This is
    what preserves old multi-turn flows such as fertilizer refinancing without
    preserving the old frozen queue itself.
    """
    starts = [tuple(position) for position in starts]
    budgets = list(budgets)
    inventories = list(inventories)
    if not (len(starts) == len(budgets) == len(inventories)):
        raise ValueError("starts, budgets and inventories must align one-per-worker")
    if not starts:
        return [], list(tasks)

    pinned, remaining, shed_left, stranded = _choose_carried_assignment(
        tasks, starts, inventories, budgets, shed_access, shed
    )

    prefixes, endpoints, remaining_budgets = [], [], []
    depositing = False
    for worker, (start, bucket, inventory, budget) in enumerate(
        zip(starts, pinned, inventories, budgets)
    ):
        queue, end, carried_after = _bucket_queue(start, bucket, shed_access, inventory)

        # Once all directly consumable carried inputs have been assigned, any
        # leftovers must become shared shed state again. Otherwise a per-turn
        # rebuild can strand harvested goods or a collected FERTILIZER forever
        # because the old queue continuation no longer exists.
        must_deposit = bool(carried_after) or worker in stranded
        if must_deposit:
            shed_position = nearest_shed(end, shed_access)
            drop = route(end, shed_position) + [["DROP"]]
            if len(queue) + len(drop) <= budget:
                queue += drop
                end = shed_position
                depositing = True

        prefixes.append(queue)
        endpoints.append(end)
        remaining_budgets.append(max(0, budget - len(queue)))

    remaining_demand = _task_demand(remaining)
    shortage_now = any(
        remaining_demand[item] > shed_left.get(item, 0)
        for item in remaining_demand
    )

    # A planned DROP is not shed stock until the next observation. If remaining
    # work currently lacks inputs, execute only the dependency-clearing prefixes
    # instead of issuing speculative PICKUPs against inventory that does not yet
    # exist in the shed. The next turn will rebuild with the deposited stock.
    if depositing and shortage_now:
        return [WorkerPlan(start, prefix) for start, prefix in zip(starts, prefixes)], remaining

    plans, unassigned = build_queues(
        remaining,
        endpoints[0],
        len(endpoints) - 1,
        endpoints[1:],
        shed_access,
        worker_budgets=remaining_budgets,
        available_wheat=shed_left.get("WHEAT", 0),
    )

    merged = []
    for start, prefix, plan in zip(starts, prefixes, plans):
        merged.append(WorkerPlan(start, prefix + plan.queue))
    return merged, unassigned
