"""Fill newly empty tiles using only uncommitted worker time and supplies."""

from collections import Counter
from agents.horizon import can_start_today

from kaggle_environments.envs.kaggriculture.kaggriculture import market_price

from agents.farm_tasks import ANIMAL_COST, CROPS, SEED_COST, build_tasks, purchase_orders
from agents.scheduler import WorkerPlan, build_queues, predicted_hand_starts

MOVES = {"EAST": (1, 0), "WEST": (-1, 0), "SOUTH": (0, 1), "NORTH": (0, -1)}
TILE_ACTIONS = {
    "DIG", "PLANT", "WATER", "FERTILIZE", "HARVEST", "PLACE",
    "BUILD_COOP", "BUILD_PASTURE", "FEED", "CARE", "COLLECT_FERTILIZER",
}
PROTECTIVE_OPS = {"WATER", "HARVEST", "FEED", "CARE"}


def _has_pending_protective_work(queue):
    """True when delaying this route can lose standing production or animals."""
    return any(operation and operation[0] in PROTECTIVE_OPS for operation in queue)


def queue_commitments(positions, plans):
    """Recover destinations and input reservations from remaining commands."""
    endpoints, occupied = [], set()
    seeds, supplies = Counter(), Counter()
    positions = [*positions, *(plan.start for plan in plans[len(positions):])]
    for index, position in enumerate(positions):
        x, y = position
        queue = plans[index].queue if index < len(plans) else []
        for operation in queue:
            op = operation[0]
            if op in MOVES:
                dx, dy = MOVES[op]
                x, y = x + dx, y + dy
            elif op in TILE_ACTIONS:
                occupied.add((x, y))
                if op == "PLANT":
                    seeds[operation[1]] += 1
            elif op == "PICKUP":
                supplies[operation[1]] += operation[2]
        endpoints.append((x, y))
    return endpoints, occupied, seeds, supplies


def schedule_open_tiles(obs, targets, plans, shed_access, hire_costs=()):
    """Append feasible work, replant in place first, and limit input shopping.

    Return (targets eligible for purchases, additional hire count). Occupied tiles keep
    their maintenance/replant demand; new empty tiles must fit after every
    worker's committed route, including one turn for market delivery.
    Actual orders remain cash-capped by purchase_orders in the caller.
    """
    farm = obs["farms"][obs["player"]]
    positions = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
    while len(plans) < len(positions):
        plans.append(WorkerPlan(positions[len(plans)], []))
    remaining = 24 - obs["hour"]
    endpoints, committed, seed_reserve, supply_reserve = queue_commitments(positions, plans)
    endpoints = endpoints[:len(positions)]
    available = dict(obs)
    available["private"] = dict(obs["private"])
    available["private"]["seeds"] = Counter(obs["private"]["seeds"]) - seed_reserve
    available["private"]["shed"] = Counter(obs["private"]["shed"]) - supply_reserve

    vacant, eligible = {}, {}
    for position, target in targets.items():
        if not target:
            continue
        x, y = position
        tile = farm["tiles"][y][x]
        if tile == "LOCKED":
            continue
        if position in committed or (isinstance(tile, dict) and (tile.get("crop") or tile.get("animal"))):
            eligible[position] = target
        elif can_start_today(target[0], obs):
            vacant[position] = target

    # The worker that just harvested need not leave and walk back later.
    # Prepending is safe only when the entire old queue still fits today.
    for index, position in enumerate(positions):
        target = vacant.get(position)
        if not target or target[0] not in CROPS:
            continue
        tasks = build_tasks(available, {position: target})
        if not tasks or len(tasks[0].actions) + len(plans[index].queue) > remaining:
            continue
        plans[index].queue[:0] = tasks[0].actions
        available["private"]["seeds"][target[0]] -= 1
        eligible[position] = vacant.pop(position)

    if not vacant:
        return eligible, 0
    endpoints, _, _, _ = queue_commitments(positions, plans)
    endpoints = endpoints[:len(positions)]
    budgets = [max(0, remaining - len(plans[i].queue)) for i in range(len(positions))]
    prospective = build_tasks(available, vacant, assume_crop_seeds=True, assume_animal_inputs=True)
    # Purchases land after worker actions; idle workers lose this turn when
    # waiting for inputs, while busy workers can finish their existing work.
    delivery_budgets = [min(budget, max(0, remaining - 1)) for budget in budgets]
    selected, hire_count, best_funded, funded_positions = {}, 0, -1, set()
    for count, hire_cost in enumerate((0, *hire_costs)):
        if hire_cost > farm["money"]:
            break
        new_starts = predicted_hand_starts(positions[0], positions[1:], len(positions) - 1 + count)
        starts = endpoints + new_starts[len(positions) - 1:]
        candidate_budgets = delivery_budgets + [max(0, remaining - 1)] * count
        _, rejected = build_queues(
            prospective, starts[0], len(starts) - 1, starts[1:], shed_access,
            worker_budgets=candidate_budgets,
        )
        rejected_ids = {id(task) for task in rejected}
        candidate = {task.position: vacant[task.position] for task in prospective if id(task) not in rejected_ids}
        shopping_targets = {**eligible, **candidate}
        orders = purchase_orders(
            obs, shopping_targets, list(shopping_targets),
            available_money=farm["money"] - hire_cost, replant_same_crop=True,
        )
        funded_obs = dict(available)
        funded_obs["private"] = dict(available["private"])
        for key in ("shed", "seeds"):
            funded_obs["private"][key] = Counter(available["private"][key])
        cash = farm["money"] - hire_cost
        market_inventory = dict(obs["market"]["inventory"])
        for op, item, quantity in orders:
            key = "seeds" if op == "BUY_SEED" else "shed"
            for _ in range(quantity):
                cost = (
                    SEED_COST[item] if op == "BUY_SEED" else
                    ANIMAL_COST[item] if op == "BUY_ANIMAL" else
                    market_price(item, market_inventory.get(item, 0) - 1, obs["market"].get("params"))
                )
                if cost > cash:
                    break
                cash -= cost
                funded_obs["private"][key][item] += 1
                if op == "BUY_PRODUCT":
                    market_inventory[item] = max(0, market_inventory.get(item, 0) - 1)
        # New purchases must first cover shortages in already-promised
        # work, rather than appearing as free stock for an extra hire.
        for key, reserve in (("seeds", seed_reserve), ("shed", supply_reserve)):
            shortage = reserve - Counter(obs["private"][key])
            funded_obs["private"][key] -= shortage
        funded = [
            task for task in build_tasks(funded_obs, candidate)
            if any(op[0] in ("PLANT", "PLACE") for op in task.actions)
        ]
        _, unfitted = build_queues(
            funded, starts[0], len(starts) - 1, starts[1:], shed_access,
            worker_budgets=candidate_budgets,
        )
        score = len(funded) - len(unfitted)
        if score > best_funded:
            selected, hire_count, best_funded = candidate, count, score
            rejected_funded = {id(task) for task in unfitted}
            funded_positions = {task.position for task in funded if id(task) not in rejected_funded}
        if not rejected:
            break
    eligible.update(selected)

    # `selected`/`funded_positions` above is purchase intent only. A seed
    # that is merely affordable is not executable inventory yet: market
    # orders settle after worker actions. Leave the worker's current queue
    # untouched; on the next observation the actual delivered seed will be
    # admitted by the `actual` block below. This keeps speculative PLANT out
    # of the front of WATER/HARVEST routes.

    # Already-owned inputs can be used this turn, including at hour 22.
    actual = [
        task for task in build_tasks(available, vacant)
        if any(op[0] in ("PLANT", "PLACE") for op in task.actions)
    ]
    additions, rejected = build_queues(
        actual, endpoints[0], len(positions) - 1, endpoints[1:], shed_access,
        worker_budgets=budgets,
    )
    rejected_ids = {id(task) for task in rejected}
    for task in actual:
        if id(task) not in rejected_ids:
            eligible[task.position] = vacant[task.position]
    for index, addition in enumerate(additions):
        plans[index].queue.extend(addition.queue)
    return eligible, hire_count
