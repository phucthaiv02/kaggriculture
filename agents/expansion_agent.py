"""Coordinate opening, daily planning, intraday work and market orders.

Worker actions execute before market orders; newly purchased inputs and
hires become usable on the next observation. See RULES.md for the full flow."""

from __future__ import annotations

from collections import Counter

from kaggle_environments.envs.kaggriculture.kaggriculture import market_price

from agents.farm_tasks import (
    ANIMAL_COST,
    ANIMALS,
    build_tasks,
    feed_wheat_order,
    feed_wheat_reserve,
    purchase_orders,
    reserved_items,
)
from agents.opening_book import make_opening_controller, should_buy_land_on_schedule
from agents.intraday import MOVES, queue_commitments, reconcile_animals
from agents.planner import SEASON_END_DAY, plan_targets
from agents.horizon import can_start_today
from agents.scheduler import MAX_HANDS, build_queues, hands_needed, nearest_shed, route
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
MARKET_ORDER_CAP = 10
# Bound expensive ROI work per morning while a persistent backlog guarantees
# that every eligible tile is visited before the cycle restarts.
TARGETS_PER_DAY = 10
INVESTMENTS_PER_DAY = 20


def _active_positions(farm):
    tiles = farm["tiles"]
    return [(x, y) for y in range(BOARD_SIZE) for x in range(BOARD_SIZE) if tiles[y][x] != "LOCKED"]


def _open_shed_access(farm):
    half = BOARD_SIZE // 2
    positions = ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))
    return tuple((x, y) for x, y in positions if farm["tiles"][y][x] != "LOCKED")


def _pop_market_batch(queue, cap=MARKET_ORDER_CAP):
    """Remove and return the next planned market batch without dropping its tail."""
    batch = list(queue[:cap])
    del queue[:cap]
    return batch


def _strip_partial_animal_builds(tasks, opening_active=False):
    """Post-opening, never BUILD an animal structure without same-plan PLACE.

    Opening deliberately keeps its historical bootstrap behavior.  After the
    hand-off, a failed/missing animal purchase may remove PLACE when tasks are
    rebuilt from observed inventory; in that case retain physical work such as
    HARVEST but remove the standalone BUILD instead of leaving an empty pen.
    """
    if opening_active:
        return tasks
    cleaned = []
    for task in tasks:
        operations = [operation[0] for operation in task.actions]
        has_build = any(operation.startswith("BUILD_") for operation in operations)
        if has_build and "PLACE" not in operations:
            task.actions = [
                operation for operation in task.actions
                if not operation[0].startswith("BUILD_")
            ]
        if task.actions:
            cleaned.append(task)
    return cleaned


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
    mandatory_hand_target=None,
):
    hires = max(0, hand_target - len(farm["hands"]))
    mandatory_hires = (
        max(
            0,
            (mandatory_hand_target if mandatory_hand_target is not None else len(farm["hands"]))
            - len(farm["hands"]),
        )
        if reserve_hire_budget else 0
    )
    hire_orders = [["HIRE"] for _ in range(hires)]
    purchase = []
    if buy_inputs:
        # Optional hires execute after admitted inputs, so only mandatory
        # survival capacity may reduce the money those inputs can spend.
        available_money = _maximum_cash_after_sales(
            obs, farm, pending_sales, mandatory_hires
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
    feed = [order for order in purchase if order[0] == "BUY_PRODUCT"]
    animals = [order for order in purchase if order[0] == "BUY_ANIMAL"]
    seeds = [order for order in purchase if order[0] == "BUY_SEED"]
    if reserve_hire_budget:
        orders = (
            hire_orders[:mandatory_hires] + feed + animals + seeds
            + hire_orders[mandatory_hires:]
        )
    else:
        # The opening's fixed portfolio depends on same-day seeds/animals;
        # buying these before optional labor prevents a queued PLANT no-op.
        orders = feed + animals + seeds + hire_orders
    if should_buy_land_on_schedule(obs, farm):
        if reserve_hire_budget:
            orders.insert(mandatory_hires, ["BUY_LAND"])
        else:
            orders.insert(0, ["BUY_LAND"])
    return orders


def make_agent(end_day=SEASON_END_DAY, seed=0):
    del seed  # every decision reacts to live prices/shed state; nothing to seed
    targets = {}
    state = {
        "day": -1, "hand_target": 0, "mandatory_hand_target": 0,
        "plans": [], "reserved": {},
        "opening_active": False,
        "deferred_expansion_positions": set(),
        "pending_targets": set(),
        "frozen_positions": set(), "unassigned": [], "plan_frozen": False,
        "purchase_positions": set(), "committed_targets": set(),
        "investment_backlog": set(), "daily_targets": {},
        "emergency_hires": 0,
        "morning_market_queue": [],
    }
    opening_governs = make_opening_controller()

    def _dispatch_morning_market(farm):
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in farm["hands"]],
            "market": _pop_market_batch(state["morning_market_queue"]),
        }

    def _global_replan(obs, farm, hour):
        """Rebuild every remaining route from observed positions and time.

        The observation is the completion ledger: WATER/FEED/CARE flags,
        removed output and newly created producers make completed operations
        disappear from build_tasks. Only positions admitted by the morning
        plan may start new investment; physical producers are always scanned.
        """
        frozen_targets = {
            position: targets.get(position)
            for position in state["frozen_positions"]
            if position in targets
        }
        replanned = build_tasks(
            obs, frozen_targets,
            prioritize_fertilizer_drop=state["opening_active"],
        )
        replanned = _strip_partial_animal_builds(
            replanned, opening_active=state["opening_active"]
        )
        starts = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
        budget = max(0, 24 - hour)
        prefixes, planning_starts, budgets = [], [], []
        shed_access = _open_shed_access(farm)
        for start, inventory in zip(starts, obs["private"]["inventories"]):
            prefix = []
            planning_start = start
            if any(inventory.values()):
                planning_start = nearest_shed(start, shed_access)
                prefix = route(start, planning_start) + [["DROP"]]
            prefixes.append(prefix)
            planning_starts.append(planning_start)
            budgets.append(max(0, budget - len(prefix)))
        plans, unassigned = build_queues(
            replanned, planning_starts[0], len(starts) - 1, planning_starts[1:],
            shed_access, worker_budgets=budgets,
            available_wheat=obs["private"]["shed"].get("WHEAT", 0),
        )
        for plan, start, prefix in zip(plans, starts, prefixes):
            plan.start = start
            plan.queue[:0] = prefix
        state["plans"], state["unassigned"] = plans, unassigned
        assigned = {id(task) for task in replanned} - {id(task) for task in unassigned}
        state["reserved"] = reserved_items(
            [task for task in replanned if id(task) in assigned]
        )

    def _plan_invalid(obs, farm):
        """Return true when a queued front action contradicts observed state."""
        positions = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
        if len(state["plans"]) != len(positions):
            return True
        shed = Counter(obs["private"]["shed"])
        seeds = Counter(obs["private"]["seeds"])
        for index, position in enumerate(positions):
            queue = state["plans"][index].queue
            if not queue:
                continue
            op = queue[0]
            tile = farm["tiles"][position[1]][position[0]]
            inventory = obs["private"]["inventories"][index]
            if op[0] == "PICKUP" and shed[op[1]] < op[2]:
                return True
            if op[0] == "PICKUP":
                shed[op[1]] -= op[2]
            if op[0] == "PLANT" and (
                seeds[op[1]] <= 0 or not can_start_today(op[1], obs)
                or (isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE"))
            ):
                return True
            if op[0] == "PLANT":
                seeds[op[1]] -= 1
            if op[0] == "PLACE" and (
                inventory.get(op[1], 0) <= 0 or not can_start_today(op[1], obs)
            ):
                return True
            if op[0] == "DIG" and isinstance(tile, dict) and (
                tile.get("animal") or tile.get("kind") in ("COOP", "PASTURE")
            ):
                return True
            if op[0].startswith("BUILD_") and tile is not None:
                return True
            if op[0] == "WATER" and not (
                isinstance(tile, dict) and tile.get("kind") == "PLANT"
            ):
                return True
            if op[0] == "FEED" and (
                not isinstance(tile, dict) or not tile.get("animal")
                or inventory.get("WHEAT", 0) <= 0
            ):
                return True
            if op[0] == "CARE" and (
                not isinstance(tile, dict) or not tile.get("animal")
            ):
                return True
            if op[0] == "HARVEST" and (
                not isinstance(tile, dict) or tile.get("yield_units", 0) <= 0
            ):
                return True
        return False

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
        active_position_set = set(positions)
        new_positions = [position for position in positions if position not in targets]
        if hour == 0:
            state["deferred_expansion_positions"].clear()
            state["morning_market_queue"].clear()
        physical_positions = {
            (x, y)
            for y, row in enumerate(farm["tiles"])
            for x, tile in enumerate(row)
            if isinstance(tile, dict) and (tile.get("crop") or tile.get("animal"))
        }
        if hour == 0:
            stale_commitments = {
                position for position in state["committed_targets"]
                if position not in active_position_set or not targets.get(position)
            }
            expired_commitments = {
                position for position in state["committed_targets"] - physical_positions
                if position in active_position_set and targets.get(position)
                and not can_start_today(targets[position][0], obs)
            }
            state["committed_targets"].difference_update(
                stale_commitments | expired_commitments
            )
            state["pending_targets"].update(expired_commitments)
        if hour > 0 and new_positions:
            state["deferred_expansion_positions"].update(new_positions)
        if hour == 0 or new_positions:
            state["opening_active"] = opening_governs(obs, targets, positions)
            if not state["opening_active"]:
                state["pending_targets"].update(new_positions)
                if hour == 0 and not state["pending_targets"]:
                    unrealized_committed = (
                        state["committed_targets"] - physical_positions
                    )
                    state["pending_targets"].update(
                        active_position_set - unrealized_committed
                    )
        pinned_animals = reconcile_animals(obs, targets)
        state["committed_targets"].update(physical_positions | pinned_animals)
        state["investment_backlog"].difference_update(
            physical_positions | state["committed_targets"]
        )
        state["pending_targets"].difference_update(pinned_animals)
        state["pending_targets"].difference_update(
            state["committed_targets"] - physical_positions
        )
        obs["_committed_targets"] = set(state["committed_targets"])
        if hour == 0 and not state["opening_active"] and state["pending_targets"]:
            pending_before = set(state["pending_targets"])
            state["pending_targets"] = set(plan_targets(
                obs, targets, positions, effective_end,
                max_positions=TARGETS_PER_DAY,
                replan_positions=state["pending_targets"],
            ))
            evaluated = pending_before - state["pending_targets"]
            state["investment_backlog"].update(
                position for position in evaluated
                if targets.get(position)
                and not isinstance(farm["tiles"][position[1]][position[0]], dict)
            )
            for position in new_positions:
                targets.setdefault(position, None)

        obs["_pending_targets"] = state["pending_targets"]
        if hour == 0:
            if state["opening_active"]:
                daily_targets = dict(targets)
            else:
                def admission_key(position):
                    return (
                        min(abs(position[0] - x) + abs(position[1] - y)
                            for x, y in _open_shed_access(farm)),
                        position[1], position[0],
                    )

                investment_positions = sorted(
                    state["investment_backlog"], key=admission_key,
                )[:INVESTMENTS_PER_DAY]
                included = (
                    physical_positions | state["committed_targets"]
                    | set(investment_positions)
                )
                daily_targets = {
                    position: targets.get(position) for position in included
                    if position in targets
                }
            state["daily_targets"] = daily_targets
            tasks = build_tasks(
                obs, daily_targets,
                assume_crop_seeds=True,
                assume_animal_inputs=True,
                prioritize_fertilizer_drop=state["opening_active"],
            )
            tasks = _strip_partial_animal_builds(
                tasks, opening_active=state["opening_active"]
            )
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
            mandatory_tasks = [task for task in tasks if task.mandatory is not False]
            mandatory_hand_target, _mandatory_missing = hands_needed(
                mandatory_tasks,
                tuple(farm["farmer"]), tuple(map(tuple, farm["hands"])),
                _open_shed_access(farm),
            )
            state.update(
                day=day, hand_target=hand_target,
                mandatory_hand_target=mandatory_hand_target,
                plans=[], reserved={}, emergency_hires=0,
                frozen_positions=set(), unassigned=[], plan_frozen=False,
            )
            preliminary, rejected = build_queues(
                tasks, tuple(farm["farmer"]), hand_target,
                tuple(map(tuple, farm["hands"])), _open_shed_access(farm),
            )
            rejected_ids = {id(task) for task in rejected}
            investment_task_positions = {
                task.position for task in tasks
                if any(op[0] in ("PLANT", "PLACE") for op in task.actions)
            }
            state["purchase_positions"] = {
                task.position for task in tasks
                if id(task) not in rejected_ids
                and task.position in investment_task_positions
            }
            rejected_investments = {
                task.position for task in rejected
                if task.position in investment_task_positions
            }
            state["investment_backlog"].update(rejected_investments)

            # Purchases are narrower than maintenance. A standing future COW
            # target on a still-growing crop is not permission to buy a COW;
            # only a PLANT/PLACE task actually admitted today may buy input.
            purchase_targets = {
                p: t for p, t in daily_targets.items()
                if p in state["purchase_positions"]
            }
            assigned_tasks = [task for task in tasks if id(task) not in rejected_ids]
            reservations = reserved_items(assigned_tasks)
            reservations["WHEAT"] = max(
                reservations.get("WHEAT", 0),
                feed_wheat_reserve(obs, purchase_targets, positions),
            )
            non_wheat_reservations = dict(reservations)
            non_wheat_reservations["WHEAT"] = obs["private"]["shed"].get("WHEAT", 0)
            non_wheat_sales = sell_orders(obs, non_wheat_reservations)
            protected_hires = (
                0 if state["opening_active"] else
                max(0, mandatory_hand_target - len(farm["hands"]))
            )
            reservations = _reserve_feed_for_affordable_animals(
                obs, farm, purchase_targets, reservations,
                non_wheat_sales, protected_hires,
            )
            sales = sell_orders(obs, reservations)
            orders = _hire_and_buy_orders(
                obs, farm, purchase_targets, hand_target, pending_sales=sales,
                replant_same_crop=True,
                reserve_hire_budget=not state["opening_active"],
                mandatory_hand_target=mandatory_hand_target,
            )
            if state["opening_active"]:
                # Preserve the opening book's historical bootstrap semantics.
                return {
                    "farmer": ["PASS"], "hands": [["PASS"] for _ in farm["hands"]],
                    "market": (sales + orders)[:MARKET_ORDER_CAP],
                }
            state["morning_market_queue"] = list(sales) + list(orders)
            return _dispatch_morning_market(farm)

        # Post-opening market work is itself part of the daily plan.  Never
        # discard orders beyond the ten-order engine cap; workers remain idle
        # until every planned batch has been observed on a later turn.
        if not state["opening_active"] and state["day"] == day and state["morning_market_queue"]:
            return _dispatch_morning_market(farm)

        if state["day"] == day and not state["plan_frozen"]:
            tasks = build_tasks(
                obs,
                state["daily_targets"],
                prioritize_fertilizer_drop=state["opening_active"],
            )
            tasks = _strip_partial_animal_builds(
                tasks, opening_active=state["opening_active"]
            )
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
            remaining_budget = max(0, 24 - hour)
            pending_hand_budget = max(0, 23 - hour)
            desired_hands, _dropped = hands_needed(
                tasks,
                tuple(farm["farmer"]),
                existing_hands,
                shed_access,
                pending_hand_budget=pending_hand_budget,
                marginal_hire_costs=(None if state["opening_active"] and day < effective_end else
                    [_hire_costs(farm, n + 1) - _hire_costs(farm, n)
                     for n in range(MAX_HANDS - len(existing_hands))]),
            )
            hand_count = len(existing_hands)
            state["hand_target"] = desired_hands

            # If the complete economic schedule still wants more workers after
            # the morning purchases landed, acquire affordable capacity first
            # and only freeze queues after the next observation confirms it.
            if not state["opening_active"] and desired_hands > hand_count:
                affordable = _affordable_hires(
                    farm, desired_hands - hand_count, farm["money"]
                )
                if affordable:
                    state["morning_market_queue"] = [["HIRE"] for _ in range(affordable)]
                    return _dispatch_morning_market(farm)

            plans, unassigned = build_queues(
                tasks,
                tuple(farm["farmer"]),
                hand_count,
                existing_hands,
                shed_access,
                pending_hand_budget=pending_hand_budget,
                existing_hand_budget=remaining_budget,
                available_wheat=obs["private"]["shed"].get("WHEAT", 0),
            )

            mandatory_tasks = [task for task in tasks if task.mandatory is not False]
            mandatory_unassigned = [
                task for task in unassigned if task.mandatory is not False
            ]
            if mandatory_unassigned:
                plans, mandatory_unassigned = build_queues(
                    mandatory_tasks,
                    tuple(farm["farmer"]),
                    hand_count,
                    existing_hands,
                    shed_access,
                    pending_hand_budget=pending_hand_budget,
                    existing_hand_budget=remaining_budget,
                    available_wheat=obs["private"]["shed"].get("WHEAT", 0),
                )
                unassigned = mandatory_unassigned + [
                    task for task in tasks if task.mandatory is False
                ]

            if mandatory_unassigned:
                required_hands, _ = hands_needed(
                    mandatory_tasks,
                    tuple(farm["farmer"]),
                    existing_hands,
                    shed_access,
                    pending_hand_budget=pending_hand_budget,
                )
                needed = max(0, required_hands - hand_count)
                affordable = _affordable_hires(farm, needed, farm["money"])
                if not state["opening_active"] and affordable:
                    state["morning_market_queue"] = [["HIRE"] for _ in range(affordable)]
                    state["plans"] = []
                    state["unassigned"] = tasks
                    state["reserved"] = {}
                    return _dispatch_morning_market(farm)
                # No partial rescue execution.  If the complete mandatory
                # schedule is infeasible with observed resources, expose the
                # planning failure by idling and retrying from the next state.
                state["plans"] = []
                state["unassigned"] = tasks
                state["reserved"] = {}
                state["plan_frozen"] = False
                return {
                    "farmer": ["PASS"],
                    "hands": [["PASS"] for _ in farm["hands"]],
                    "market": [],
                }

            state["emergency_hires"] = 0
            state["plans"] = plans
            state["unassigned"] = unassigned
            assigned_ids = {id(task) for task in tasks} - {id(task) for task in unassigned}
            state["reserved"] = reserved_items(
                [task for task in tasks if id(task) in assigned_ids]
            )
            state["frozen_positions"] = {task.position for task in tasks}
            investment_task_positions = {
                task.position for task in tasks
                if any(op[0] in ("PLANT", "PLACE") for op in task.actions)
            }
            admitted_investments = {
                task.position for task in tasks
                if id(task) in assigned_ids
                and task.position in investment_task_positions
            }
            state["committed_targets"].difference_update(investment_task_positions)
            state["committed_targets"].update(admitted_investments)
            state["investment_backlog"].difference_update(admitted_investments)
            state["investment_backlog"].update(
                task.position for task in unassigned
                if task.position in investment_task_positions
            )
            state["plan_frozen"] = True

        if _plan_invalid(obs, farm):
            _global_replan(obs, farm, hour)
        blocked = _plan_invalid(obs, farm)

        positions = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
        state["reserved"] = queue_commitments(positions, state["plans"])[3]

        plans = state["plans"]
        if day == effective_end:
            last_action_step = (effective_end + 1) * 24 - 2
            if configuration is not None:
                last_action_step = min(last_action_step, int(configuration['episodeSteps']) - 2)
            liquidate_queues(obs, plans, last_action_step, _open_shed_access(farm))
        farmer_plan = plans[0] if plans else None
        farmer_op = (
            farmer_plan.queue.pop(0)
            if not blocked and farmer_plan and farmer_plan.queue else ["PASS"]
        )
        hand_ops = []
        for index in range(len(farm["hands"])):
            plan = plans[index + 1] if index + 1 < len(plans) else None
            hand_ops.append(
                plan.queue.pop(0) if not blocked and plan and plan.queue else ["PASS"]
            )

        worker_ops = [farmer_op, *hand_ops]
        seeds_available = dict(obs["private"]["seeds"])
        shed_available = Counter(obs["private"]["shed"])
        for index, operation in enumerate(worker_ops):
            if operation and operation[0] in ('PLANT', 'PLACE') and not can_start_today(operation[1], obs):
                worker_ops[index] = ['PASS']
                if index < len(plans):
                    queue = plans[index].queue
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
        worker_ops = _valid_harvest_operations(farm, worker_ops)
        worker_ops = _protect_animal_structures(farm, worker_ops)

        farmer_op, hand_ops = worker_ops[0], worker_ops[1:]

        committed_feed_targets = {
            position: target for position, target in targets.items()
            if position in state["committed_targets"]
        }
        sale_reserve = Counter(state["reserved"])
        sale_reserve["WHEAT"] = max(
            sale_reserve["WHEAT"],
            feed_wheat_reserve(obs, committed_feed_targets, _active_positions(farm)),
        )
        market = sell_orders(obs, sale_reserve)
        if day < effective_end:
            market += feed_wheat_order(
                obs, committed_feed_targets, _active_positions(farm)
            )
        return {"farmer": farmer_op, "hands": hand_ops, "market": market[:MARKET_ORDER_CAP]}

    return agent
