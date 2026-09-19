"""Fill newly empty tiles using only uncommitted worker time and supplies."""

from collections import Counter
from agents.horizon import can_start_today
from agents.products import ANIMALS, ANIMAL_STRUCTURE

from kaggle_environments.envs.kaggriculture.kaggriculture import market_price

from agents.farm_tasks import ANIMAL_COST, CROPS, SEED_COST, build_tasks, purchase_orders, executable_targets, animal_output_at_risk
from agents.scheduler import WorkerPlan, build_queues, predicted_hand_starts

MOVES = {"EAST": (1, 0), "WEST": (-1, 0), "SOUTH": (0, 1), "NORTH": (0, -1)}
TILE_ACTIONS = {
    "DIG", "PLANT", "WATER", "FERTILIZE", "HARVEST", "PLACE",
    "BUILD_COOP", "BUILD_PASTURE", "FEED", "CARE", "COLLECT_FERTILIZER",
}


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
    targets = executable_targets(obs, targets)
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
        tasks = build_tasks(available, {position: target}, include_physical=False)
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
    prospective = build_tasks(available, vacant, assume_crop_seeds=True, assume_animal_inputs=True, include_physical=False)
    # Purchases land after worker actions; idle workers lose this turn when
    # waiting for inputs, while busy workers can finish their existing work.
    delivery_budgets = [min(budget, max(0, remaining - 1)) for budget in budgets]
    selected, hire_count, best_funded, funded_positions = {}, 0, -1, set()
    selected_funded_obs = available
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
            task for task in build_tasks(funded_obs, candidate, include_physical=False)
            if any(op[0] in ("PLANT", "PLACE") for op in task.actions)
        ]
        _, unfitted = build_queues(
            funded, starts[0], len(starts) - 1, starts[1:], shed_access,
            worker_budgets=candidate_budgets,
        )
        unfitted_ids = {id(task) for task in unfitted}
        score = sum(task.value for task in funded if id(task) not in unfitted_ids) - hire_cost
        if score > best_funded:
            selected, hire_count, best_funded = candidate, count, score
            selected_funded_obs = funded_obs
            rejected_funded = {id(task) for task in unfitted}
            funded_positions = {task.position for task in funded if id(task) not in rejected_funded}
        if not rejected:
            break
    eligible.update(selected)
    # Animal orders require a concrete route, including the delivery turn.
    animal_tasks = [task for task in build_tasks(selected_funded_obs, selected, include_physical=False)
                    if any(op[0] == "PLACE" for op in task.actions)]
    if animal_tasks:
        starts = endpoints + predicted_hand_starts(positions[0], positions[1:], len(positions)-1+hire_count)[len(positions)-1:]
        animal_plans, missing = build_queues(
            animal_tasks, starts[0], len(starts)-1, starts[1:], shed_access,
            worker_budgets=delivery_budgets + [max(0, remaining-1)] * hire_count)
        rejected_positions = {task.position for task in missing}
        for position, target in list(eligible.items()):
            if target[0] in ANIMALS and position in rejected_positions:
                eligible.pop(position)
        for index, addition in enumerate(animal_plans):
            if not addition.queue:
                continue
            while index >= len(plans):
                plans.append(WorkerPlan(starts[len(plans)], []))
            if index < len(positions) and not plans[index].queue:
                plans[index].queue.append(["PASS"])
            plans[index].queue.extend(addition.queue)
        for task in animal_tasks:
            if task.position not in rejected_positions:
                vacant.pop(task.position, None)
                available["private"]["shed"] -= task.needs
        endpoints, _, _, _ = queue_commitments(positions, plans)
        endpoints = endpoints[:len(positions)]
        budgets = [max(0, remaining-len(plans[i].queue)) for i in range(len(positions))]

    # If a replacement seed is affordable but not delivered yet, retain the
    # harvester on its tile. The agent's PLANT guard waits for the purchase
    # instead of letting the worker walk away and requiring a return trip.
    for index, position in enumerate(positions):
        target = vacant.get(position)
        if (
            position not in funded_positions or not target or target[0] not in CROPS
            or available["private"]["seeds"].get(target[0], 0) > 0
            or farm["tiles"][position[1]][position[0]] is not None
            or len(plans[index].queue) + 3 > remaining
        ):
            continue
        plans[index].queue[:0] = [["PLANT", target[0]], ["WATER"]]
        vacant.pop(position)
        budgets[index] -= 3

    # Already-owned inputs can be used this turn, including at hour 22.
    actual = [
        task for task in build_tasks(available, vacant, include_physical=False)
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


def rescue_survival(obs, plans):
    """Rescue WATER, FEED and capped output without displacing other physical risks."""
    from agents.scheduler import route, nearest_shed
    farm = obs['farms'][obs['player']]
    positions = [tuple(farm['farmer']), *map(tuple, farm['hands'])]
    remaining = 24 - obs['hour']
    endangered = {(x, y) for y, row in enumerate(farm['tiles']) for x, tile in enumerate(row)
                  if isinstance(tile, dict) and tile.get('kind') == 'PLANT'
                  and tile.get('consecutive_unwatered', 0) >= 1 and not tile.get('watered_today')}

    critical = {('WATER', p): remaining for p in endangered}
    for y, row in enumerate(farm['tiles']):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict):
                continue
            if tile.get('animal'):
                if tile.get('consecutive_unfed', 0) >= 1 and not tile.get('fed_today'):
                    critical[('FEED', (x, y))] = remaining
                if animal_output_at_risk(tile, obs['day']):
                    critical[('HARVEST', (x, y))] = remaining
            if tile.get('kind') == 'PLANT' and tile.get('yield_units', 0) and tile.get('max_lifespan_step') is not None:
                critical[('HARVEST', (x, y))] = min(remaining, tile['max_lifespan_step'] - obs.get('step', obs['day'] * 24 + obs['hour']) + 1)

    def coverage(worker_positions, queues):
        served = set()
        for index, (position, queue) in enumerate(zip(worker_positions, queues)):
            x, y = position
            inventory = Counter(obs['private']['inventories'][index]) if index < len(obs['private']['inventories']) else Counter()
            supplies = Counter(obs['private']['shed'])
            seeds = Counter(obs['private']['seeds'])
            for turn, op in enumerate(queue[:remaining], 1):
                if op[0] in MOVES:
                    dx, dy = MOVES[op[0]]
                    x, y = x + dx, y + dy
                elif op[0] == 'PICKUP':
                    if supplies[op[1]] < op[2]:
                        break  # Runtime waits here until the missing inputs arrive.
                    inventory[op[1]] += op[2]
                    supplies[op[1]] -= op[2]
                elif op[0] == 'PLANT':
                    if not seeds[op[1]]:
                        break
                    seeds[op[1]] -= 1
                elif op[0] == 'DROP':
                    supplies.update(inventory)
                    inventory.clear()
                elif op[0] == 'FEED':
                    if inventory['WHEAT'] > 0:
                        inventory['WHEAT'] -= 1
                        if turn <= critical.get(('FEED', (x, y)), -1):
                            served.add(('FEED', (x, y)))
                elif turn <= critical.get((op[0], (x, y)), -1):
                    served.add((op[0], (x, y)))
        return served

    while len(plans) < len(positions):
        plans.append(WorkerPlan(positions[len(plans)], []))
    queues = [plan.queue for plan in plans[:len(positions)]]
    protected = coverage(positions, queues)
    rescue_targets = sorted(
        (key for key in critical if key[0] in ('WATER', 'FEED')
         or farm['tiles'][key[1][1]][key[1][0]].get('animal')),
        key=lambda key: (key[0] == 'HARVEST', key[0] != 'WATER', key[1]),
    )
    for operation, target in rescue_targets:
        if (operation, target) in protected:
            continue
        best = None
        for index, start in enumerate(positions):
            outward = route(start, target) + [[operation]]
            if operation == 'FEED':
                inventory = obs['private']['inventories'][index] if index < len(obs['private']['inventories']) else {}
                if not inventory.get('WHEAT', 0):
                    if not obs['private']['shed'].get('WHEAT', 0):
                        continue
                    access = tuple(p for p in ((4,4), (5,4), (4,5), (5,5))
                                   if farm['tiles'][p[1]][p[0]] != 'LOCKED')
                    shed = nearest_shed(start, access)
                    outward = route(start, shed) + [['PICKUP', 'WHEAT', 1]] + route(shed, target) + [['FEED']]
            if len(outward) > remaining:
                continue
            candidate = outward + route(target, start) + queues[index]
            trial = list(queues)
            trial[index] = candidate
            covered = coverage(positions, trial)
            if not protected <= covered or (operation, target) not in covered:
                continue
            score = (bool(queues[index]), len(outward), index)
            if best is None or score < best[0]:
                best = score, index, candidate, covered
        if best:
            _, index, queues[index], protected = best
            plans[index].queue = queues[index]


def reconcile_animals(obs, targets):
    """Keep owned animals assigned to compatible vacant tiles until placement."""
    farm = obs['farms'][obs['player']]
    stock = Counter(obs['private']['shed'])
    for inventory in obs['private']['inventories']:
        stock.update(inventory)
    pinned = set()
    for animal in ANIMALS:
        if not stock[animal] or not can_start_today(animal, obs):
            continue
        candidates = []
        for y, row in enumerate(farm['tiles']):
            for x, tile in enumerate(row):
                position = (x, y)
                if position in pinned:
                    continue
                compatible = tile is None or (isinstance(tile, dict)
                    and tile.get('kind') == ANIMAL_STRUCTURE[animal] and not tile.get('animal'))
                if compatible:
                    candidates.append(position)
        candidates.sort(key=lambda p: (targets.get(p) != (animal, False),
                                        farm['tiles'][p[1]][p[0]] is None,
                                        abs(p[0]-4)+abs(p[1]-4), p))
        for position in candidates[:stock[animal]]:
            targets[position] = (animal, False)
            pinned.add(position)
    return pinned
