"""Rebuild worker routes from the live observation on every executable turn.

The daily planner still decides which positions/tasks are admitted.  This
module only re-optimizes execution order and worker assignment from the current
worker positions and inventories.  Routes are intentionally ephemeral: the
agent executes one operation, observes the result, then rebuilds them again.
"""

from __future__ import annotations

from collections import Counter

from agents.scheduler import WorkerPlan, build_queues, nearest_shed, route


def planning_observation(obs):
    """Expose carried inputs to task generation without mutating the real obs.

    build_tasks historically looked only at the shed when deciding whether an
    animal/feed/fertilizer continuation could be scheduled.  Under rolling
    execution an input may already be on a worker from the previous turn, so
    treating carried stock as planning-available prevents a valid continuation
    from disappearing after a PICKUP or HARVEST.
    """
    private = dict(obs["private"])
    available = Counter(private["shed"])
    for inventory in private.get("inventories", ()):  # worker-carried inputs
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

        # Outputs become real carried stock after the task.  This matters for
        # WHEAT harvested immediately before an animal/feed continuation.
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
        # Mirror scheduler cashout semantics: only a suffix that still carries
        # task output needs the terminal DROP.
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
    """Pin only work that must/should consume already-carried inputs.

    A normal scheduler assumes every task input is still in the shed.  Rolling
    execution breaks that assumption after a previous PICKUP/HARVEST.  We pin
    such tasks to the carrier until no remaining task can use carried stock;
    the rest can then safely use the ordinary route packer from the remaining
    real shed stock.
    """
    remaining = list(tasks)
    buckets = [[] for _ in starts]
    carried = [_positive(Counter(inv)) for inv in inventories]
    shed_left = _positive(Counter(shed))

    while remaining:
        demand = _task_demand(remaining)
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
                # Consume carried stock first; among equal coverage, preserve
                # locality and then deterministic board order.
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

    # If carried inputs are still required by remaining work, stock is split
    # across workers in a way no task can directly consume.  Return those
    # carriers to the shed first; the next observation will expose a unified
    # stock pool and rolling planning resumes normally.
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

    The result has one WorkerPlan per current worker.  Only the first operation
    is meant to execute; callers must rebuild on the next observation.
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
    for worker, (start, bucket, inventory, budget) in enumerate(
        zip(starts, pinned, inventories, budgets)
    ):
        queue, end, _carried = _bucket_queue(start, bucket, shed_access, inventory)
        if worker in stranded:
            shed_position = nearest_shed(end, shed_access)
            drop = route(end, shed_position) + [["DROP"]]
            if len(queue) + len(drop) <= budget:
                queue += drop
                end = shed_position
        prefixes.append(queue)
        endpoints.append(end)
        remaining_budgets.append(max(0, budget - len(queue)))

    # If fragmented carried inputs still cannot be made available within this
    # turn's capacity, execute the safe prefixes only and retry from the next
    # observation rather than emitting a PICKUP that cannot succeed.
    if stranded and any(
        _task_demand(remaining)[item] > shed_left.get(item, 0)
        for item in _task_demand(remaining)
    ):
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
