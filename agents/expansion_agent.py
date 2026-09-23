"""Coordinate opening, daily planning, rolling execution and market orders.

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
from agents.intraday import queue_commitments, reconcile_animals
from agents.planner import SEASON_END_DAY, plan_targets
from agents.horizon import can_start_today
from agents.scheduler import MAX_HANDS, build_queues, hands_needed
from agents.rolling_scheduler import build_rolling_queues, planning_observation
from agents.selling import sell_orders
from agents.liquidation import liquidate_queues


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


def _pop_market_batch(queue, cap=MARKET_ORDER_CAP):
    batch = list(queue[:cap])
    del queue[:cap]
    return batch


def _strip_partial_animal_builds(tasks, opening_active=False):
    """Post-opening, BUILD and PLACE are one atomic investment task."""
    if opening_active:
        return tasks
    cleaned = []
    for task in tasks:
        operations = [operation[0] for operation in task.actions]
        if any(op.startswith("BUILD_") for op in operations) and "PLACE" not in operations:
            task.actions = [
                operation for operation in task.actions
                if not operation[0].startswith("BUILD_")
            ]
        if task.actions:
            cleaned.append(task)
    return cleaned


# A rolling rebuild is useful only when the executable dependency graph changes.
# WATER/FEED/CARE/FERTILIZE are already encoded inside the current Task route and
# do not unlock a different tile. Repacking every MOVE (or every maintenance op)
# made workers repeatedly trade destinations and burn the day on WEST/EAST
# oscillations. These operations change inventory, occupancy, or the set of live
# tile tasks, so the next observation should rebuild from the new state.
ROLLING_REPLAN_OPS = {
    "PICKUP", "DROP", "HARVEST", "DIG", "PLANT", "PLACE",
    "BUILD_COOP", "BUILD_PASTURE", "COLLECT_FERTILIZER",
}


def _needs_route_rebuild(operations):
    return any(operation and operation[0] in ROLLING_REPLAN_OPS for operation in operations)


def _plans_have_work(plans):
    return any(plan.queue for plan in plans)


# One land purchase unlocks one 5x5 quadrant. Price and admit up to the whole
# quadrant in the following morning instead of intentionally carrying 5-15
# already-selected empty tiles for extra days. Target scoring itself is unchanged.
TARGETS_PER_DAY = 25
INVESTMENTS_PER_DAY = 25


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


def _opening_retry_orders(obs, farm, purchase_targets, pending_sales=()):
    """Retry already-admitted opening investments when intraday cash arrives.

    The fixed opening intentionally finances later animal conversions from
    fertilizer/harvest cash generated during the day. A morning-only purchase
    pass can therefore admit PLACE work but miss its animal by a few dollars.
    Re-evaluate the same admitted targets after observed sales; this does not
    select new targets or reopen schedule admission, and normal play keeps its
    morning-only investment purchases.

    purchase_orders still prices the feed needed to make an animal affordable,
    but intraday WHEAT itself is owned by feed_wheat_order. Filtering PRODUCT
    orders here prevents the investment retry from duplicating that feed path.
    """
    wheat_sold = sum(
        int(order[2]) for order in pending_sales
        if order[0] == "SELL" and order[1] == "WHEAT"
    )
    purchase = purchase_orders(
        obs,
        purchase_targets,
        _active_positions(farm),
        available_money=_maximum_cash_after_sales(obs, farm, pending_sales, 0),
        available_wheat=max(
            0, obs["private"]["shed"].get("WHEAT", 0) - wheat_sold
        ),
        replant_same_crop=True,
    )
    return [
        order for order in purchase
        if order[0] in ("BUY_ANIMAL", "BUY_SEED")
    ]


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
        orders = feed + animals + seeds + hire_orders
    if should_buy_land_on_schedule(obs, farm):
        if reserve_hire_budget:
            orders.insert(mandatory_hires, ["BUY_LAND"])
        else:
            orders.insert(0, ["BUY_LAND"])
    return orders


def make_agent(end_day=SEASON_END_DAY, seed=0, decision_log=None):
    del seed
    targets = {}
    state = {
        "day": -1, "hand_target": 0, "mandatory_hand_target": 0,
        "plans": [], "reserved": {},
        "opening_active": False,
        "deferred_expansion_positions": set(),
        "pending_targets": set(),
        "frozen_positions": set(), "unassigned": [], "schedule_admitted": False,
        "purchase_positions": set(), "committed_targets": set(),
        "investment_backlog": set(), "daily_targets": {},
        "morning_market_queue": [], "replan_needed": True,
    }
    opening_governs = make_opening_controller()

    def _dispatch_morning_market(farm):
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in farm["hands"]],
            "market": _pop_market_batch(state["morning_market_queue"]),
        }

    def _rebuild_rolling_routes(obs, farm, hour):
        """Re-optimize the admitted remainder from the live observation.

        `frozen_positions` is the daily schedule boundary. Rebuilding may change
        worker assignment and route order inside that set, but it cannot pull in
        a task the daily admission did not accept. There is deliberately no
        stale-queue fallback: an exhausted WorkerPlan is not evidence that its
        old schedule is still feasible.
        """
        frozen_targets = {
            position: targets.get(position)
            for position in state["frozen_positions"]
        }
        task_obs = planning_observation(obs)
        replanned = build_tasks(
            task_obs,
            frozen_targets,
            prioritize_fertilizer_drop=state["opening_active"],
            include_physical=False,
        )
        replanned = _strip_partial_animal_builds(
            replanned, opening_active=state["opening_active"]
        )
        # Every task here comes from a position admitted by the morning
        # schedule. Rebuilding after HARVEST/PLANT may change the physical
        # tile enough for build_tasks to classify the successor as an
        # "optional investment" again, but rolling execution must not reopen
        # admission or drop the remainder of an already-admitted chain.
        # This is schedule commitment, not an action-kind runtime priority.
        for task in replanned:
            task.mandatory = True
        starts = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
        budget = max(0, 24 - hour)
        plans, unassigned = build_rolling_queues(
            replanned,
            starts,
            obs["private"]["inventories"],
            [budget] * len(starts),
            _open_shed_access(farm),
            obs["private"]["shed"],
        )
        state["plans"], state["unassigned"] = plans, unassigned
        assigned = {id(task) for task in replanned} - {id(task) for task in unassigned}
        state["reserved"] = reserved_items(
            [task for task in replanned if id(task) in assigned]
        )
        state["replan_needed"] = False

    def agent(obs, configuration=None):
        effective_end = end_day
        if configuration is not None:
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
                decision_log=decision_log,
            ))
            evaluated = pending_before - state["pending_targets"]
            state["investment_backlog"].update(
                position for position in evaluated
                if targets.get(position) and position not in physical_positions
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
            tasks = [
                task for task in tasks
                if task.position not in state["deferred_expansion_positions"]
                or isinstance(farm["tiles"][task.position[1]][task.position[0]], dict)
            ]
            hand_target, rejected = hands_needed(
                tasks,
                tuple(farm["farmer"]),
                tuple(map(tuple, farm["hands"])),
                _open_shed_access(farm),
                marginal_hire_costs=(None if state["opening_active"] and day < effective_end else
                    [_hire_costs(farm, n + 1) - _hire_costs(farm, n)
                     for n in range(MAX_HANDS - len(farm["hands"]))]),
            )
            mandatory_tasks = [task for task in tasks if task.mandatory is not False]
            if len(mandatory_tasks) == len(tasks):
                mandatory_hand_target = hand_target
            else:
                mandatory_hand_target, _mandatory_missing = hands_needed(
                    mandatory_tasks,
                    tuple(farm["farmer"]), tuple(map(tuple, farm["hands"])),
                    _open_shed_access(farm),
                )
            state.update(
                day=day, hand_target=hand_target,
                mandatory_hand_target=mandatory_hand_target,
                plans=[], reserved={},
                frozen_positions=set(), unassigned=[], schedule_admitted=False,
                morning_market_queue=[], replan_needed=True,
            )
            # hands_needed already ran the exact feasibility pack at the chosen
            # headcount. Repacking the same tasks here was a duplicate
            # superlinear morning pass used only to recover the same rejected set.
            rejected_ids = {id(task) for task in rejected}
            mandatory_ids = {
                id(task) for task in tasks if task.mandatory is not False
            }
            pre_admitted_ids = (
                {id(task) for task in tasks} - rejected_ids
            ) | mandatory_ids
            investment_task_positions = {
                task.position for task in tasks
                if any(op[0] in ("PLANT", "PLACE") for op in task.actions)
            }
            state["purchase_positions"] = {
                task.position for task in tasks
                if id(task) in pre_admitted_ids
                and task.position in investment_task_positions
            }
            rejected_investments = {
                task.position for task in rejected
                if id(task) not in mandatory_ids
                and task.position in investment_task_positions
            }
            state["investment_backlog"].update(rejected_investments)

            # Hour-0 funding must mirror the admission rule used after market
            # orders land. Otherwise a mandatory HARVEST->PLANT successor can
            # be frozen into today's schedule at hour 1 without its seed ever
            # being purchased because preliminary packing happened to reject it.
            assigned_tasks = [
                task for task in tasks if id(task) in pre_admitted_ids
            ]
            fertilizer_purchase_positions = {
                task.position for task in assigned_tasks
                if task.needs.get("FERTILIZER", 0) > 0
            }
            purchase_targets = {
                p: t for p, t in daily_targets.items()
                if (p in state["purchase_positions"]
                    or p in fertilizer_purchase_positions)
            }
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
            if not state["opening_active"]:
                sales = sales[:max(0, 10 - protected_hires)]
            orders = _hire_and_buy_orders(
                obs, farm, purchase_targets, hand_target, pending_sales=sales,
                replant_same_crop=True,
                reserve_hire_budget=not state["opening_active"],
                mandatory_hand_target=mandatory_hand_target,
            )
            full_market = list(sales) + list(orders)
            if not state["opening_active"]:
                state["morning_market_queue"] = [
                    order for order in full_market[MARKET_ORDER_CAP:]
                    if order[0] in ("BUY_ANIMAL", "BUY_PRODUCT")
                ]
            return {
                "farmer": ["PASS"], "hands": [["PASS"] for _ in farm["hands"]],
                "market": full_market[:MARKET_ORDER_CAP],
            }

        if (not state["opening_active"] and state["day"] == day
                and state["morning_market_queue"]):
            return _dispatch_morning_market(farm)

        if state["day"] == day and not state["schedule_admitted"]:
            task_obs = planning_observation(obs)
            tasks = build_tasks(
                task_obs,
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
            pending_budget = max(0, 23 - hour)
            # Optional labor was already economically sized and ordered at
            # hour 0. Re-running the full economic hand search after the
            # morning market queue cannot add an optional worker here; this
            # phase only schedules workers that actually arrived. If mandatory
            # work still does not fit, the emergency path below sizes the exact
            # missing survival capacity.
            hand_count = len(existing_hands)
            state["hand_target"] = hand_count
            plans, unassigned = build_queues(
                tasks,
                tuple(farm["farmer"]),
                hand_count,
                existing_hands,
                shed_access,
                pending_hand_budget=pending_budget,
                existing_hand_budget=remaining_budget,
                available_wheat=task_obs["private"]["shed"].get("WHEAT", 0),
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
                    pending_hand_budget=pending_budget,
                    existing_hand_budget=remaining_budget,
                    available_wheat=task_obs["private"]["shed"].get("WHEAT", 0),
                )
                unassigned = mandatory_unassigned + [
                    task for task in tasks if task.mandatory is False
                ]

            state["plans"] = plans
            assigned_ids = {id(task) for task in tasks} - {id(task) for task in unassigned}
            mandatory_ids = {
                id(task) for task in tasks if task.mandatory is not False
            }
            admitted_ids = assigned_ids | mandatory_ids
            deferred_tasks = [
                task for task in unassigned if id(task) not in mandatory_ids
            ]
            state["unassigned"] = deferred_tasks
            state["reserved"] = reserved_items(
                [task for task in tasks if id(task) in admitted_ids]
            )
            state["frozen_positions"] = {
                task.position for task in tasks if id(task) in admitted_ids
            }
            investment_task_positions = {
                task.position for task in tasks
                if any(op[0] in ("PLANT", "PLACE") for op in task.actions)
            }
            admitted_investments = {
                task.position for task in tasks
                if id(task) in admitted_ids
                and task.position in investment_task_positions
            }
            state["committed_targets"].difference_update(investment_task_positions)
            state["committed_targets"].update(admitted_investments)
            state["investment_backlog"].difference_update(admitted_investments)
            state["investment_backlog"].update(
                task.position for task in deferred_tasks
                if task.position in investment_task_positions
            )
            # Admission is a once-per-day decision. Rolling execution may
            # reassign/reorder the admitted set but never reopens headcount or
            # task admission later in the day.
            state["schedule_admitted"] = True
            # The admission pass just produced a route from this exact live
            # observation. Do not immediately throw it away with a second pack.
            state["replan_needed"] = False

        if state["replan_needed"] or not _plans_have_work(state["plans"]):
            _rebuild_rolling_routes(obs, farm, hour)

        positions = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
        state["reserved"] = queue_commitments(positions, state["plans"])[3]

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

        worker_ops = [farmer_op, *hand_ops]
        planned_worker_ops = [list(operation) for operation in worker_ops]
        seeds_available = dict(obs["private"]["seeds"])
        shed_available = Counter(obs["private"]["shed"])
        inventories = obs["private"]["inventories"]
        for index, operation in enumerate(worker_ops):
            if not operation:
                worker_ops[index] = ["PASS"]
                continue
            op = operation[0]
            if op in ("PLANT", "PLACE") and not can_start_today(operation[1], obs):
                worker_ops[index] = ["PASS"]
                continue
            inventory = inventories[index] if index < len(inventories) else {}
            if op == "PLACE" and inventory.get(operation[1], 0) <= 0:
                worker_ops[index] = ["PASS"]
                continue
            if op == "FEED" and inventory.get("WHEAT", 0) <= 0:
                worker_ops[index] = ["PASS"]
                continue
            if op == "FERTILIZE" and inventory.get("FERTILIZER", 0) <= 0:
                worker_ops[index] = ["PASS"]
                continue
            if op == "PICKUP":
                item = operation[1]
                requested = operation[2] if len(operation) > 2 else 1
                taken = min(requested, shed_available[item])
                shed_available[item] -= taken
                worker_ops[index] = ["PICKUP", item, taken] if taken else ["PASS"]
                continue
            if op != "PLANT":
                continue
            crop = operation[1]
            if seeds_available.get(crop, 0) > 0:
                seeds_available[crop] -= 1
            else:
                worker_ops[index] = ["PASS"]

        worker_ops = _valid_harvest_operations(farm, worker_ops)
        worker_ops = _protect_animal_structures(farm, worker_ops)
        invalidated = any(
            planned != actual and actual == ["PASS"] and planned != ["PASS"]
            for planned, actual in zip(planned_worker_ops, worker_ops)
        )
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
        sales = sell_orders(obs, sale_reserve)
        market = list(sales)
        if day < effective_end:
            market += feed_wheat_order(
                obs, committed_feed_targets, _active_positions(farm)
            )
            if state["opening_active"]:
                opening_purchase_targets = {
                    position: target
                    for position, target in state["daily_targets"].items()
                    if position in state["purchase_positions"]
                }
                market += _opening_retry_orders(
                    obs, farm, opening_purchase_targets, sales
                )
        market = market[:10]

        # Continue a worker's current route through ordinary movement and
        # maintenance. Rebuild only after a dependency/topology change, an
        # invalidated queued op, or a market purchase that changes available
        # inputs on the next observation.
        state["replan_needed"] = bool(
            invalidated
            or _needs_route_rebuild(worker_ops)
            or any(order and order[0].startswith("BUY_") for order in market)
        )
        return {"farmer": farmer_op, "hands": hand_ops, "market": market}

    return agent
