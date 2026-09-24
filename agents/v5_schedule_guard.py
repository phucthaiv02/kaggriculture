"""Agent-v5 safety guard for concrete mandatory schedule completion.

Target economics deliberately excludes labor.  Labor is therefore a hard
feasibility constraint: after the scheduler admits mandatory work, every tile
action must actually appear inside a worker's executable turn budget.

The v3 scheduler uses a fast packing projection before materialising concrete
queues.  Under dense animal portfolios an intraday replan can redistribute
already-carried WHEAT and leave a mandatory FEED outside the executable queue
without asking for another hand.  This module keeps the existing packer and
routing policy, but verifies its concrete result and feeds any missing
mandatory task back into the normal emergency-hire path.

No action priority is introduced. WATER/FEED/CARE/HARVEST remain ordinary
mandatory scheduled work; the guard only enforces that a claimed feasible
schedule is actually complete.
"""
from __future__ import annotations

from collections import Counter
from functools import wraps

from agents.intraday import MOVES
from agents import scheduler


_INSTALLED = False


def _worker_budgets(
    farmer_start,
    hand_count,
    hand_starts=(),
    pending_hand_budget=scheduler.HAND_BUDGET,
    existing_hand_budget=None,
    worker_budgets=None,
):
    starts = [tuple(farmer_start)] + scheduler.predicted_hand_starts(
        farmer_start, hand_starts, hand_count
    )
    if worker_budgets is not None:
        return starts, list(worker_budgets)
    farmer_budget = (
        scheduler.FARMER_BUDGET
        if existing_hand_budget is None else existing_hand_budget
    )
    existing_budget = (
        scheduler.HAND_BUDGET
        if existing_hand_budget is None else existing_hand_budget
    )
    budgets = [farmer_budget]
    budgets += [existing_budget] * min(hand_count, len(hand_starts))
    budgets += [pending_hand_budget] * max(0, hand_count - len(hand_starts))
    return starts, budgets


def _executed_tile_actions(plans, budgets):
    """Count concrete non-movement actions reachable inside each turn budget."""
    executed = Counter()
    for plan, budget in zip(plans, budgets):
        position = tuple(plan.start)
        for operation in plan.queue[:max(0, int(budget))]:
            if operation[0] in MOVES:
                dx, dy = MOVES[operation[0]]
                position = (position[0] + dx, position[1] + dy)
                continue
            # PICKUP/DROP/PASS are route plumbing, not tile task actions.
            if operation[0] in ("PICKUP", "DROP", "PASS"):
                continue
            executed[(position, tuple(operation))] += 1
    return executed


def missing_mandatory_tasks(tasks, plans, budgets):
    """Return mandatory tasks not fully represented in executable queue prefixes."""
    available = _executed_tile_actions(plans, budgets)
    missing = []
    for task in tasks:
        if task.mandatory is False:
            continue
        required = Counter(
            (task.position, tuple(operation))
            for operation in task.actions
            if operation[0] not in ("PICKUP", "DROP", "PASS")
        )
        if any(available[key] < amount for key, amount in required.items()):
            missing.append(task)
            continue
        for key, amount in required.items():
            available[key] -= amount
    return missing


def _merge_unassigned(unassigned, extra):
    result = list(unassigned)
    seen = {id(task) for task in result}
    for task in extra:
        if id(task) not in seen:
            result.append(task)
            seen.add(id(task))
    return result


def install():
    """Install bounded queue and hand-sizing verification for v5."""
    global _INSTALLED
    if _INSTALLED:
        return

    original_build_queues = scheduler.build_queues
    original_hands_needed = scheduler.hands_needed

    @wraps(original_build_queues)
    def guarded_build_queues(
        tasks,
        farmer_start,
        hand_count,
        hand_starts=(),
        shed_access=scheduler.SHED_ACCESS,
        pending_hand_budget=scheduler.HAND_BUDGET,
        existing_hand_budget=None,
        worker_budgets=None,
        available_wheat=None,
        worker_inventories=None,
    ):
        plans, unassigned = original_build_queues(
            tasks,
            farmer_start,
            hand_count,
            hand_starts,
            shed_access,
            pending_hand_budget,
            existing_hand_budget,
            worker_budgets,
            available_wheat,
            worker_inventories,
        )
        # Dense animal service is where carried inputs and intraday replans
        # exposed optimistic admission. Avoid an extra scan for crop-only days.
        if not any(
            task.mandatory is not False and task.needs.get("WHEAT", 0) > 0
            for task in tasks
        ):
            return plans, unassigned
        _starts, budgets = _worker_budgets(
            farmer_start,
            hand_count,
            hand_starts,
            pending_hand_budget,
            existing_hand_budget,
            worker_budgets,
        )
        missing = missing_mandatory_tasks(tasks, plans, budgets)
        return plans, _merge_unassigned(unassigned, missing)

    @wraps(original_hands_needed)
    def guarded_hands_needed(
        tasks,
        farmer_start,
        existing_hand_starts=(),
        shed_access=scheduler.SHED_ACCESS,
        pending_hand_budget=scheduler.HAND_BUDGET,
        existing_hand_budget=None,
        max_hands=scheduler.MAX_HANDS,
        marginal_hire_costs=None,
    ):
        count, original_missing = original_hands_needed(
            tasks,
            farmer_start,
            existing_hand_starts,
            shed_access,
            pending_hand_budget,
            existing_hand_budget,
            max_hands,
            marginal_hire_costs,
        )
        if not any(
            task.mandatory is not False and task.needs.get("WHEAT", 0) > 0
            for task in tasks
        ):
            return count, original_missing

        # Verify the concrete queue at the packer's chosen headcount. Escalate
        # only for mandatory omissions; optional work retains the v3 economic
        # admission policy. This loop is bounded by MAX_HANDS (16).
        start = min(max_hands, max(len(existing_hand_starts), count))
        last_missing = list(original_missing)
        for candidate_count in range(start, max_hands + 1):
            plans, missing = guarded_build_queues(
                tasks,
                farmer_start,
                candidate_count,
                existing_hand_starts,
                shed_access,
                pending_hand_budget,
                existing_hand_budget,
            )
            mandatory_missing = [
                task for task in missing if task.mandatory is not False
            ]
            last_missing = missing
            if not mandatory_missing:
                return candidate_count, missing
        return max_hands, last_missing

    scheduler.build_queues = guarded_build_queues
    scheduler.hands_needed = guarded_hands_needed
    _INSTALLED = True
