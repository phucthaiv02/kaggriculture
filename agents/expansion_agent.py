"""Coordinate opening, daily planning, intraday work and market orders.

Worker actions execute before market orders; newly purchased inputs and
hires become usable on the next observation. See RULES.md for the full flow."""

from __future__ import annotations

from collections import Counter

from kaggle_environments.envs.kaggriculture.kaggriculture import market_price

from agents.farm_tasks import (
    ANIMAL_COST,
    animal_output_at_risk,
    ANIMALS,
    build_tasks,
    feed_wheat_order,
    purchase_orders,
    reserved_items,
)
from agents.opening_book import make_opening_controller, should_buy_land_on_schedule
from agents.intraday import MOVES, queue_commitments, schedule_open_tiles, rescue_survival, reconcile_animals
from agents.planner import SEASON_END_DAY, plan_targets
from agents.horizon import can_start_today
from agents.scheduler import MAX_HANDS, build_queues, hands_needed
from agents.schedules import is_maintenance_day, should_care_animal, should_feed_animal, should_harvest_animal
from agents.selling import sell_orders
from agents.liquidation import liquidate_queues

def _prune_stale_harvests(farm, plans):
    """Remove invalid harvests at their routed tile before dispatching work."""
    positions = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
    for position, plan in zip(positions, plans):
        x, y = position
        queue = []
        for operation in plan.queue:
            if operation[0] in MOVES:
                dx, dy = MOVES[operation[0]]
                x, y = x + dx, y + dy
            if operation[0] == "HARVEST":
                tile = farm["tiles"][y][x]
                if not isinstance(tile, dict) or tile.get("yield_units", 0) <= 0:
                    continue
            queue.append(operation)
        plan.queue = queue


def _valid_harvest_operations(farm, operations):
    """Engine workers execute in order; each tile can pay out only once."""
    seen = set()
    result = list(operations)
    positions = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
    for index, position in enumerate(positions[:len(result)]):
        if result[index] != ["HARVEST"]:
            continue
        x, y = position
        tile = farm["tiles"][y][x]
        if position in seen or not isinstance(tile, dict) or tile.get("yield_units", 0) <= 0:
            result[index] = ["PASS"]
        else:
            seen.add(position)
    return result


def _protect_animal_structures(farm, operations):
    """Discard stale DIG/PLANT commands on permanent animal structures."""
    operations = list(operations)
    positions = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
    protected = {p for p in positions
                 if isinstance(farm["tiles"][p[1]][p[0]], dict)
                 and (farm["tiles"][p[1]][p[0]].get("animal")
                      or farm["tiles"][p[1]][p[0]].get("kind") in ("COOP", "PASTURE"))}
    for index, position in enumerate(positions[:len(operations)]):
        operation = operations[index]
        if operation and operation[0] in ("BUILD_COOP", "BUILD_PASTURE", "PLACE"):
            protected.add(position)
        if operation and operation[0] in ("DIG", "PLANT") and position in protected:
            operations[index] = ["PASS"]
    return operations


BOARD_SIZE = 10
# Bound expensive ROI work per action; continue the remaining tiles next turn.
TARGETS_PER_TURN = 2


def _active_positions(farm):
    tiles = farm["tiles"]
    return [(x, y) for y in range(BOARD_SIZE) for x in range(BOARD_SIZE) if tiles[y][x] != "LOCKED"]


def _open_shed_access(farm):
    half = BOARD_SIZE // 2
    positions = ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))
    return tuple((x, y) for x, y in positions if farm["tiles"][y][x] != "LOCKED")


def _hire_costs(farm, number):
    """Exact Fibonacci cost of `number` additional hires this day."""
    a, b = 1, 1
    costs = []
    for _ in range(farm.get("hires_today", 0) + number):
        costs.append(a)
        a, b = b, a + b
    return sum(costs[farm.get("hires_today", 0):])


def _affordable_hires(farm, desired, money):
    """How many of `desired` sequential HIRE orders can really execute."""
    affordable = 0
    for number in range(1, desired + 1):
        if _hire_costs(farm, number) > money:
            break
        affordable = number
    return affordable


def _sale_revenue(obs, sales):
    """Exact same-turn proceeds before subsequent BUY orders execute."""
    inventory = dict(obs["market"]["inventory"])
    params = obs["market"].get("params")
    revenue = 0
    for _op, item, quantity in sales:
        for _ in range(int(quantity)):
            stock = inventory.get(item, 0)
            price = market_price(item, stock, params)
            revenue += price
            inventory[item] = stock + int(price > 1)
    return revenue


def _maximum_cash_after_sales(obs, farm, sales, hires):
    """Spendable cash after every preceding SELL and HIRE is processed."""
    return max(
        0,
        farm["money"] + _sale_revenue(obs, sales) - _hire_costs(farm, hires),
    )


def _reserve_feed_for_affordable_animals(obs, farm, targets, reservations, sales, hires):
    """Keep shed WHEAT for animals the same market pass can actually buy."""
    reserve = dict(reservations)
    wheat_free = max(0, obs["private"]["shed"].get("WHEAT", 0) - reserve.get("WHEAT", 0))
    if not wheat_free:
        return reserve

    carried = {name: 0 for name in ANIMALS}
    for inventory in obs["private"]["inventories"]:
        for name in ANIMALS:
            carried[name] += inventory.get(name, 0)
    missing = {name: 0 for name in ANIMALS}
    tiles = farm["tiles"]
    for (x, y), target in targets.items():
        if not target or target[0] not in ANIMALS:
            continue
        name = target[0]
        tile = tiles[y][x]
        if not (isinstance(tile, dict) and tile.get("animal") == name):
            missing[name] += 1
    for name in ANIMALS:
        missing[name] = max(
            0,
            missing[name] - obs["private"]["shed"].get(name, 0) - carried[name],
        )

    money = _maximum_cash_after_sales(obs, farm, sales, hires)
    extra_feed = 0
    for name in sorted(ANIMALS, key=ANIMAL_COST.get):
        for _ in range(missing[name]):
            if extra_feed >= wheat_free or money < ANIMAL_COST[name]:
                break
            money -= ANIMAL_COST[name]
            extra_feed += 1
    reserve["WHEAT"] = reserve.get("WHEAT", 0) + extra_feed
    return reserve


def _hire_and_buy_orders(
    obs,
    farm,
    targets,
    hand_target,
    pending_sales=(),
    buy_inputs=True,
    replant_same_crop=False,
    reserve_hire_budget=True,
):
    hires = max(0, hand_target - len(farm["hands"]))
    hire_orders = [["HIRE"] for _ in range(hires)]
    purchase = []
    if buy_inputs:
        available_money = _maximum_cash_after_sales(
            obs, farm, pending_sales, hires if reserve_hire_budget else 0
        )
        wheat_sold = sum(
            int(order[2]) for order in pending_sales
            if order[0] == "SELL" and order[1] == "WHEAT"
        )
        purchase = purchase_orders(
            obs,
            targets,
            _active_positions(farm),
            available_money=available_money,
            available_wheat=max(0, obs["private"]["shed"].get("WHEAT", 0) - wheat_sold),
            replant_same_crop=replant_same_crop,
        )
    # Inputs that make today's precomputed tile tasks executable must land
    # before optional labor spending.  Reserving the cost of every desired
    # HIRE here could reduce seed affordability to zero, leaving those hands
    # with no PLANT work to execute.
    feed = [order for order in purchase if order[0] == "BUY_PRODUCT"]
    animals = [order for order in purchase if order[0] == "BUY_ANIMAL"]
    seeds = [order for order in purchase if order[0] == "BUY_SEED"]
    if reserve_hire_budget:
        # Outside the opening, protect survival capacity before committing
        # capital to new production.
        orders = feed + hire_orders + animals + seeds
    else:
        # The opening's fixed portfolio depends on same-day seeds/animals;
        # buying these before optional labor prevents a queued PLANT no-op.
        orders = feed + animals + seeds + hire_orders
    if should_buy_land_on_schedule(obs, farm):
        # This debug purchase must claim its cash before optional hires and
        # inputs; putting it last can submit the order yet leave it unfunded.
        orders.insert(0, ["BUY_LAND"])
    return orders


def make_agent(end_day=SEASON_END_DAY, seed=0):
    del seed  # every decision reacts to live prices/shed state; nothing to seed
    targets = {}
    state = {
        "day": -1, "hand_target": 0, "plans": [], "reserved": {},
        "opening_active": False, "scheduled_placements": set(),
        "unverified_hand_indices": set(),
        "deferred_expansion_positions": set(),
        "pending_targets": set(),
    }
    opening_governs = make_opening_controller()

    def _validate_predicted_hands(farm, hour):
        """Clear routes whose predicted hire spawn differs from the observed position.

        Idle workers are then rescheduled from their real positions."""
        hands = tuple(map(tuple, farm["hands"]))
        still_unverified = set()
        for index in state["unverified_hand_indices"]:
            hand_index = index - 1
            if hand_index >= len(hands):
                if hour == 1:
                    still_unverified.add(index)
                continue
            plan = state["plans"][index] if index < len(state["plans"]) else None
            if plan is not None and plan.start != hands[hand_index]:
                plan.queue = []
        state["unverified_hand_indices"] = still_unverified
        if hour >= 2:
            # Morning HIRE orders have resolved. Failed/truncated purchases
            # must release their tasks and inputs for intraday scheduling.
            del state["plans"][len(hands) + 1:]

    def _animal_still_needs_attention(tile, name, day, last_day):
        """Check outstanding harvest, scheduled feed and care for a placed animal."""
        age = day - tile["placed_day"]
        harvest_ready = should_harvest_animal(
            name, age, tile.get("yield_units", 0), force=day == last_day or animal_output_at_risk(tile, day),
        )
        needs_feed = should_feed_animal(name, age, last_day - tile["placed_day"]) and not tile.get("fed_today")
        needs_care = (
            should_care_animal(name, age, last_day - tile["placed_day"])
            and tile.get("fed_today")
            and not tile.get("cared_today")
        )
        return harvest_ready or needs_feed or needs_care

    def _crop_still_needs_water(tile, name, fertilize_commit, day):
        """Rescue scheduled watering without duplicating planting or placement."""
        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
            return False
        if tile.get("watered_today"):
            return False
        crop = tile["crop"]
        age = day - tile["planted_day"]
        current_fertilized = (
            fertilize_commit
            if crop == name
            else tile.get("fertilized_until_day", -1) >= tile["planted_day"]
        )
        return tile.get("consecutive_unwatered", 0) >= 1 or is_maintenance_day(crop, age, current_fertilized)

    def _schedule_late_placements(obs, farm, day, hour):
        """Assign idle workers pending placements and unfinished maintenance."""
        if hour < 2 or not state["plans"]:
            return
        worker_positions = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
        idle_indices = [
            index for index, plan in enumerate(state["plans"])
            if not plan.queue and index < len(worker_positions)
        ]
        pending_targets = {}
        state["scheduled_placements"] = {
            position for position in state["scheduled_placements"]
            if not (
                isinstance(farm["tiles"][position[1]][position[0]], dict)
                and farm["tiles"][position[1]][position[0]].get("animal")
                == (targets.get(position) or (None, False))[0]
            )
        }
        if not idle_indices:
            return
        committed = queue_commitments(worker_positions, state["plans"])[1]
        for position in _active_positions(farm):
            tile = farm["tiles"][position[1]][position[0]]
            target = targets.get(position)
            if isinstance(tile, dict) and tile.get("animal"):
                target = (tile["animal"], False)
            if not target:
                continue
            if position in committed:
                continue
            x, y = position
            tile = farm["tiles"][y][x]
            name, fertilize_commit = target
            if name in ANIMALS:
                placed = isinstance(tile, dict) and tile.get("animal") == name
                if not placed:
                    if position not in state["scheduled_placements"]:
                        pending_targets[position] = target
                elif _animal_still_needs_attention(tile, name, day, obs.get("_planning_end_day", end_day)):
                    pending_targets[position] = target
            elif (
                position in state["deferred_expansion_positions"]
                and not isinstance(tile, dict)
                and position not in state["scheduled_placements"]
            ):
                pending_targets[position] = target
            elif _crop_still_needs_water(tile, name, fertilize_commit, day):
                pending_targets[position] = target
        if not pending_targets:
            return
        tasks = build_tasks(
            obs,
            pending_targets,
            prioritize_fertilizer_drop=state["opening_active"],
            include_physical=False,
        )
        placement_tasks = [
            task for task in tasks
            if any(
                action[0] in ("PLANT", "PLACE", "FEED", "CARE", "HARVEST", "COLLECT_FERTILIZER", "WATER")
                for action in task.actions
            )
        ]
        if not placement_tasks:
            return
        shed_access = _open_shed_access(farm)
        remaining = 24 - hour
        # Existing workers have only the remaining turns, not a full day.
        plans, unassigned = build_queues(
            placement_tasks,
            worker_positions[idle_indices[0]],
            len(idle_indices) - 1,
            tuple(worker_positions[index] for index in idle_indices[1:]),
            shed_access,
            pending_hand_budget=remaining,
            existing_hand_budget=remaining,
        )
        # Apply tasks that fit; unassigned work can retry on later turns.
        unassigned_positions = {task.position for task in unassigned}
        for index, plan in zip(idle_indices, plans):
            state["plans"][index] = plan
        state["scheduled_placements"].update(
            task.position for task in placement_tasks
            if task.position not in unassigned_positions
            and any(action[0] in ("PLANT", "PLACE") for action in task.actions)
        )
        state["reserved"] = reserved_items(placement_tasks)

    def agent(obs, configuration=None):
        effective_end = end_day
        if configuration is not None:
            # The final observation is terminal; no action can run from it.
            last_action_day = (int(configuration['episodeSteps']) - 2) // int(configuration.get('turnsPerDay', 24))
            effective_end = min(effective_end, last_action_day)
        obs = dict(obs, _planning_end_day=effective_end)
        day, hour = obs["day"], obs["hour"]
        farm = obs["farms"][obs["player"]]

        positions = _active_positions(farm)
        new_positions = [position for position in positions if position not in targets]
        if hour > 0 and new_positions:
            state["deferred_expansion_positions"].update(new_positions)
        if hour == 0 or new_positions:
            state["opening_active"] = opening_governs(obs, targets, positions)
            if not state["opening_active"]:
                state["pending_targets"].update(positions if hour == 0 else new_positions)
        pinned_animals = reconcile_animals(obs, targets)
        state["pending_targets"].difference_update(pinned_animals)
        if not state["opening_active"] and state["pending_targets"]:
            committed = (
                queue_commitments(
                    [tuple(farm["farmer"]), *map(tuple, farm["hands"])], state["plans"],
                )[1] if hour > 0 else set()
            )
            waiting = state["pending_targets"] & committed
            state["pending_targets"] = waiting | set(plan_targets(
                obs, targets, positions, effective_end,
                max_positions=TARGETS_PER_TURN,
                replan_positions=state["pending_targets"] - committed,
            ))
            # Record discovery once; unpriced tiles remain idle until their
            # batch is evaluated, while existing worker queues keep running.
            for position in new_positions:
                targets.setdefault(position, None)

        obs["_pending_targets"] = state["pending_targets"]
        if hour > 0:
            if state["unverified_hand_indices"]:
                _validate_predicted_hands(farm, hour)
            _prune_stale_harvests(farm, state["plans"])

        if hour == 0:
            tasks = build_tasks(
                obs, targets,
                assume_crop_seeds=day > 0,
                assume_animal_inputs=state["opening_active"],
                prioritize_fertilizer_drop=state["opening_active"],
            )  # include the PLANT + WATER enabled by today's seed purchases
            tasks = [
                task for task in tasks
                if task.position not in state["deferred_expansion_positions"]
                or isinstance(farm["tiles"][task.position[1]][task.position[0]], dict)
            ]
            hand_target, _dropped = hands_needed(
                tasks,
                tuple(farm["farmer"]),
                tuple(map(tuple, farm["hands"])),
                _open_shed_access(farm),
                marginal_hire_costs=(None if state["opening_active"] and day < effective_end else
                    [_hire_costs(farm, n + 1) - _hire_costs(farm, n)
                     for n in range(MAX_HANDS - len(farm["hands"]))]),
            )
            state.update(
                day=day, hand_target=hand_target, plans=[], reserved={},
                scheduled_placements=set(), unverified_hand_indices=set(),
            )
            reservations = reserved_items(tasks)
            # First value every non-WHEAT sale, then decide how much WHEAT
            # must stay in the shed for animals that cash can fund today.
            non_wheat_reservations = dict(reservations)
            non_wheat_reservations["WHEAT"] = obs["private"]["shed"].get("WHEAT", 0)
            non_wheat_sales = sell_orders(obs, non_wheat_reservations)
            hires = max(0, hand_target - len(farm["hands"]))
            reservations = _reserve_feed_for_affordable_animals(
                obs, farm, targets, reservations, non_wheat_sales, hires
            )
            sales = sell_orders(obs, reservations)
            return {
                "farmer": ["PASS"], "hands": [["PASS"] for _ in farm["hands"]],
                "market": (
                    sales + _hire_and_buy_orders(
                        obs, farm, (targets if state["opening_active"] else {
                            p: t for p, t in targets.items() if not t or t[0] not in ANIMALS
                        }), hand_target, pending_sales=sales,
                        replant_same_crop=True,
                        reserve_hire_budget=not state["opening_active"],
                    )
                )[:10],
            }

        if state["day"] == day and not state["plans"]:
            tasks = build_tasks(
                obs,
                targets,
                assume_crop_seeds=True,
                prioritize_fertilizer_drop=state["opening_active"],
            )  # preserve same-day HARVEST -> PLANT while seed arrives mid-day
            state["deferred_expansion_positions"] = {
                position for position in state["deferred_expansion_positions"]
                if not isinstance(farm["tiles"][position[1]][position[0]], dict)
            }
            tasks = [
                task for task in tasks
                if task.position not in state["deferred_expansion_positions"]
            ]
            existing_hands = tuple(map(tuple, farm["hands"]))
            shed_access = _open_shed_access(farm)
            hand_count, _dropped = hands_needed(
                tasks,
                tuple(farm["farmer"]),
                existing_hands,
                shed_access,
                pending_hand_budget=22,
                marginal_hire_costs=(None if state["opening_active"] and day < effective_end else
                    [_hire_costs(farm, n + 1) - _hire_costs(farm, n)
                     for n in range(MAX_HANDS - len(existing_hands))]),
            )
            missing_hands = max(0, hand_count - len(existing_hands))
            affordable_hands = _affordable_hires(
                farm, missing_hands, farm["money"]
            )
            hand_count = len(existing_hands) + affordable_hands
            state["hand_target"] = hand_count
            plans, _unassigned = build_queues(
                tasks,
                tuple(farm["farmer"]),
                hand_count,
                existing_hands,
                shed_access,
                pending_hand_budget=22,
                available_wheat=obs["private"]["shed"].get("WHEAT", 0),
            )
            state["plans"], state["reserved"] = plans, reserved_items(tasks)
            # Indices beyond today's already-real hands got a route built
            # from predicted_hand_starts' *guess* at where they'll spawn --
            # see _validate_predicted_hands below for why that guess can be
            # wrong, and what happens once it's checked against reality.
            state["unverified_hand_indices"] = set(
                range(len(existing_hands) + 1, hand_count + 1)
            )
            state["scheduled_placements"] = {
                task.position
                for task in tasks
                if any(action[0] in ("PLANT", "PLACE") for action in task.actions)
            }

        if state["unverified_hand_indices"]:
            _validate_predicted_hands(farm, hour)
        # Morning hires already have pending orders/plans. Extra hires here
        # are only for newly available work after that pass has settled.
        hire_costs = (
            [_hire_costs(farm, count) for count in range(1, MAX_HANDS - len(farm["hands"]) + 1)]
            if hour >= 2 and not state["unverified_hand_indices"] else []
        )
        if day < effective_end:
            rescue_survival(obs, state["plans"])
        purchase_targets, expansion_hires = schedule_open_tiles(
            obs, targets, state["plans"], _open_shed_access(farm), hire_costs,
        )
        state["unverified_hand_indices"].update(
            range(len(farm["hands"]) + 1, len(state["plans"]))
        )
        _schedule_late_placements(obs, farm, day, hour)
        # Remaining PICKUPs, including newly appended work, must be protected
        # from the same turn's sales.
        positions = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
        state["reserved"] = queue_commitments(positions, state["plans"])[3]

        placement_reserve = Counter(state["reserved"])
        plans = state["plans"]
        if day == effective_end:
            last_action_step = (effective_end + 1) * 24 - 2
            if configuration is not None:
                last_action_step = min(last_action_step, int(configuration['episodeSteps']) - 2)
            liquidate_queues(obs, plans, last_action_step, _open_shed_access(farm))
        farmer_plan = plans[0] if plans else None
        farmer_op = farmer_plan.queue.pop(0) if farmer_plan and farmer_plan.queue else ["PASS"]
        hand_ops = []
        for index in range(len(farm["hands"])):
            plan = plans[index + 1] if index + 1 < len(plans) else None
            hand_ops.append(plan.queue.pop(0) if plan and plan.queue else ["PASS"])

        # A planned seed purchase may land later in the day after fertilizer
        # is sold. Never consume a queued PLANT as a no-op before its real
        # seed exists; hold it at the front of that worker's queue instead.
        worker_ops = [farmer_op, *hand_ops]
        seeds_available = dict(obs["private"]["seeds"])
        shed_available = Counter(obs["private"]["shed"])
        for index, operation in enumerate(worker_ops):
            if operation and operation[0] in ('PLANT', 'PLACE') and not can_start_today(operation[1], obs):
                worker_ops[index] = ['PASS']
                if index < len(plans):
                    queue = plans[index].queue
                    # Cancel care for the rejected new producer, preserving
                    # the worker's route and other tiles' harvest tasks.
                    while queue and queue[0][0] in ('WATER', 'FERTILIZE', 'FEED', 'CARE'):
                        queue.pop(0)
                continue
            if operation and operation[0] == "PICKUP":
                item = operation[1]
                requested = operation[2] if len(operation) > 2 else 1
                taken = min(requested, shed_available[item])
                shed_available[item] -= taken
                if taken < requested:
                    plan = plans[index] if index < len(plans) else None
                    if plan is not None:
                        plan.queue.insert(0, ["PICKUP", item, requested - taken])
                    worker_ops[index] = ["PICKUP", item, taken] if taken else ["PASS"]
                continue
            if not operation or operation[0] != "PLANT":
                continue
            crop = operation[1]
            if seeds_available.get(crop, 0) > 0:
                seeds_available[crop] -= 1
                continue
            plan = plans[index] if index < len(plans) else None
            if plan is not None:
                plan.queue.insert(0, operation)
            worker_ops[index] = ["PASS"]
        # Hard real-time rule: if a worker is already standing on ready
        # animal output, HARVEST now. Put its queued operation back instead
        # of discarding it, so FEED/CARE/WATER still executes next turn.
        worker_positions = positions
        harvested_positions = {
            position for position, operation in zip(worker_positions, worker_ops)
            if operation == ["HARVEST"]
        }
        for index, position in enumerate(worker_positions[:len(worker_ops)]):
            x, y = position
            tile = farm["tiles"][y][x]
            ready_animal = (
                isinstance(tile, dict)
                and tile.get("animal")
                and tile.get("yield_units", 0) > 0
            )
            if not ready_animal or position in harvested_positions:
                continue
            displaced = worker_ops[index]
            # Opportunistic output must not push the last planned action past rollover.
            if (displaced != ["HARVEST"] and index < len(plans)
                    and len(plans[index].queue) + 1 >= 24 - hour):
                continue
            if displaced != ["HARVEST"]:
                plan = plans[index] if index < len(plans) else None
                if plan is not None and displaced != ["PASS"]:
                    plan.queue.insert(0, displaced)
                worker_ops[index] = ["HARVEST"]
            harvested_positions.add(position)

        worker_ops = _valid_harvest_operations(farm, worker_ops)
        worker_ops = _protect_animal_structures(farm, worker_ops)

        farmer_op, hand_ops = worker_ops[0], worker_ops[1:]

        # Sell and buy every turn so harvests and fertilizer can fund new
        # production immediately. Vacant-tile purchases are restricted to
        # the capacity admitted by the intraday scheduler.
        market = sell_orders(obs, state["reserved"])
        if hour == 1:
            # Morning top-up hires retain their existing sizing. Input
            # shopping below also covers feasible newly unlocked tiles.
            market += _hire_and_buy_orders(
                obs,
                farm,
                targets,
                state["hand_target"],
                buy_inputs=False,
            )
        else:
            market += feed_wheat_order(obs, targets, _active_positions(farm))

        # Fertilizer dropped on the previous turn is sold first. Its exact
        # proceeds are therefore available to BUY_ANIMAL later in this same
        # market pass; the animal is observed and scheduled for PLACE on a
        # subsequent turn once the morning work queues are idle.
        wheat_sold = sum(
            int(order[2]) for order in market
            if order[0] == "SELL" and order[1] == "WHEAT"
        )
        # Only SELL orders add cash.  ``market`` can also contain a
        # BUY_PRODUCT feed order; treating that as sale revenue would
        # make an animal look affordable before it really is.
        sales = [order for order in market if order[0] == "SELL"]
        available_money = _maximum_cash_after_sales(
            obs, farm, sales, expansion_hires + sum(order[0] == "HIRE" for order in market)
        )
        late_orders = purchase_orders(
            obs,
            purchase_targets,
            _active_positions(farm),
            available_money=available_money,
            available_wheat=max(
                0, obs["private"]["shed"].get("WHEAT", 0) - wheat_sold
            ),
            replant_same_crop=True,
        )
        existing_feed_buy = any(
            order[0] == "BUY_PRODUCT" and order[1] == "WHEAT"
            for order in market
        )
        for order in late_orders:
            if order[0] == "BUY_ANIMAL":
                item = order[1]
                shortage = max(0, placement_reserve[item] - obs["private"]["shed"].get(item, 0))
                if state["opening_active"]:
                    # The opening book commits its conversion/placement sequence.
                    market.append(order)
                elif shortage:
                    market.append(["BUY_ANIMAL", item, min(order[2], shortage)])
            elif order[0] == "BUY_SEED":
                market.append(order)
            elif order[0] == "BUY_PRODUCT" and not existing_feed_buy:
                market.append(order)
                existing_feed_buy = True

        # Inputs precede optional expansion hires. If the ten-order cap cuts
        # off some hires, the next observation schedules only real workers.
        market += [["HIRE"] for _ in range(expansion_hires)]
        return {"farmer": farmer_op, "hands": hand_ops, "market": market[:10]}

    return agent
