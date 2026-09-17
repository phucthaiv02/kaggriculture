"""V2 optimizer agent.

This agent is intentionally separate from ``agents.expansion_agent`` so replay
comparison stays easy. Strategy still reuses the opening book and target
economics, while execution is rebuilt around locked lifecycle schedules,
neutral atomic TileJobs, deterministic route optimization, batched supplies,
and a fail-closed executor with no runtime rescue rules.
"""

from __future__ import annotations

from agents.farm_tasks import build_tasks
from agents.horizon import SEASON_END_DAY
from agents.opening_book import make_opening_controller, should_buy_land_on_schedule
from agents.planner import plan_targets, should_buy_land
from agents.v2_core import LifecycleBook, build_jobs, convert_legacy_tasks
from agents.v2_market import (
    MAX_MARKET_ORDERS,
    end_of_day_sell_orders,
    market_batches,
    plan_input_orders,
)
from agents.v2_scheduler import (
    MAX_HANDS,
    minimum_stationary_hands,
    optimize_with_pending_hires,
)

SHED_ACCESS = ((4, 4), (5, 4), (4, 5), (5, 5))


def _active_positions(farm):
    tiles = farm["tiles"]
    return [
        (x, y)
        for y in range(len(tiles))
        for x in range(len(tiles[y]))
        if tiles[y][x] != "LOCKED"
    ]


def _open_shed_access(farm):
    return tuple(
        position
        for position in SHED_ACCESS
        if farm["tiles"][position[1]][position[0]] != "LOCKED"
    )


def _worker_positions(farm):
    return [tuple(farm["farmer"]), *map(tuple, farm["hands"])]


def _pass_action(farm, market=None):
    return {
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in farm["hands"]],
        "market": list(market or ()),
    }


def _effective_end_day(configuration, configured_end):
    if configuration is None:
        return configured_end
    episode_steps = int(configuration["episodeSteps"])
    turns = int(configuration.get("turnsPerDay", 24))
    last_action_day = (episode_steps - 2) // turns
    return min(configured_end, last_action_day)


def _build_day_jobs(obs, targets, book, opening_active):
    if opening_active:
        legacy = build_tasks(
            obs,
            targets,
            assume_crop_seeds=True,
            assume_animal_inputs=True,
            prioritize_fertilizer_drop=False,
        )
        return convert_legacy_tasks(obs, legacy)
    return build_jobs(obs, targets, book)


def _refresh_strategy(obs, targets, opening_governs, end_day):
    farm = obs["farms"][obs["player"]]
    positions = _active_positions(farm)
    opening_active = opening_governs(obs, targets, positions)
    if not opening_active:
        plan_targets(obs, targets, positions, end_day)
    return opening_active, positions


def _land_order_needed(obs, farm, positions, opening_active):
    if opening_active:
        return should_buy_land_on_schedule(obs, farm)
    return len(farm["unlocked_quadrants"]) < 3 and should_buy_land(farm, positions)


def _inventory_for_worker(obs, index):
    inventories = obs["private"].get("inventories", [])
    return inventories[index] if index < len(inventories) else {}


def _valid_op(obs, worker_index, op):
    """Validate only; never replace a plan with a different useful action."""
    if not op:
        return False
    name = op[0]
    if name in ("PASS", "NORTH", "SOUTH", "EAST", "WEST"):
        return True

    farm = obs["farms"][obs["player"]]
    positions = _worker_positions(farm)
    if worker_index >= len(positions):
        return False
    x, y = positions[worker_index]
    tile = farm["tiles"][y][x]
    inventory = _inventory_for_worker(obs, worker_index)

    if name == "PICKUP":
        item = op[1]
        amount = int(op[2]) if len(op) > 2 else 1
        return (
            (x, y) in _open_shed_access(farm)
            and int(obs["private"]["shed"].get(item, 0)) >= amount
        )
    if name == "PLANT":
        crop = op[1]
        return tile is None and int(obs["private"]["seeds"].get(crop, 0)) > 0
    if name == "WATER":
        return isinstance(tile, dict) and tile.get("kind") == "PLANT"
    if name == "HARVEST":
        return isinstance(tile, dict) and int(tile.get("yield_units", 0)) > 0
    if name == "FERTILIZE":
        return (
            isinstance(tile, dict)
            and tile.get("kind") == "PLANT"
            and int(inventory.get("FERTILIZER", 0)) > 0
        )
    if name in ("BUILD_COOP", "BUILD_PASTURE"):
        return tile is None
    if name == "PLACE":
        animal = op[1] if len(op) > 1 else None
        expected = "COOP" if animal == "GOOSE" else "PASTURE"
        return (
            animal is not None
            and isinstance(tile, dict)
            and tile.get("kind") == expected
            and not tile.get("animal")
            and int(inventory.get(animal, 0)) > 0
        )
    if name == "FEED":
        return (
            isinstance(tile, dict)
            and tile.get("animal")
            and not tile.get("fed_today")
            and int(inventory.get("WHEAT", 0)) > 0
        )
    if name == "CARE":
        return (
            isinstance(tile, dict)
            and tile.get("animal")
            and tile.get("fed_today")
            and not tile.get("cared_today")
        )
    if name == "COLLECT_FERTILIZER":
        return (
            isinstance(tile, dict)
            and tile.get("animal")
            and bool(tile.get("fertilizer_available"))
        )
    if name == "DIG":
        return isinstance(tile, dict) and not tile.get("animal")
    return False


def make_agent(end_day=SEASON_END_DAY, seed=0):
    del seed

    targets = {}
    book = LifecycleBook()
    opening_governs = make_opening_controller()
    state = {
        "day": -1,
        "plans": None,
        "desired_hands": 0,
        "deferred_orders": [],
        "opening_active": False,
        "planned_hour": None,
        "expected_pending_starts": {},
        "checked_pending": set(),
        "invariant_failures": 0,
    }

    def record_failure():
        state["invariant_failures"] += 1

    def execute_plans(obs):
        farm = obs["farms"][obs["player"]]
        hour = obs["hour"]
        positions = _worker_positions(farm)
        plans = state["plans"] or []

        for index, expected in state["expected_pending_starts"].items():
            if index in state["checked_pending"] or index >= len(positions):
                continue
            state["checked_pending"].add(index)
            if positions[index] != expected:
                record_failure()

        ops = []
        for index in range(len(positions)):
            if index >= len(plans):
                ops.append(["PASS"])
                continue
            plan = plans[index]
            if hour < plan.spec.start_hour or not plan.queue:
                ops.append(["PASS"])
                continue
            op = plan.queue.pop(0)
            if not _valid_op(obs, index, op):
                record_failure()
                op = ["PASS"]
            ops.append(op)

        return {
            "farmer": ops[0] if ops else ["PASS"],
            "hands": ops[1:],
        }

    def start_day(obs, effective_end):
        farm = obs["farms"][obs["player"]]
        state.update(
            day=obs["day"],
            plans=None,
            desired_hands=0,
            deferred_orders=[],
            planned_hour=None,
            expected_pending_starts={},
            checked_pending=set(),
        )

        opening_active, positions = _refresh_strategy(
            obs, targets, opening_governs, effective_end
        )
        state["opening_active"] = opening_active
        jobs = _build_day_jobs(obs, targets, book, opening_active)

        hand_count, estimate = minimum_stationary_hands(
            jobs,
            tuple(farm["farmer"]),
            max_hands=MAX_HANDS,
            shed_access=_open_shed_access(farm),
        )
        state["desired_hands"] = hand_count

        input_orders = plan_input_orders(obs, estimate or [])
        buy_land = _land_order_needed(obs, farm, positions, opening_active)
        batches = market_batches(
            input_orders,
            hire_count=hand_count,
            buy_land=buy_land,
            cap=MAX_MARKET_ORDERS,
        )
        first = batches[0]
        state["deferred_orders"] = [order for batch in batches[1:] for order in batch]
        return _pass_action(farm, first)

    def plan_from_current_hour(obs, effective_end):
        farm = obs["farms"][obs["player"]]
        hour = obs["hour"]

        opening_active, _positions = _refresh_strategy(
            obs, targets, opening_governs, effective_end
        )
        state["opening_active"] = opening_active
        jobs = _build_day_jobs(obs, targets, book, opening_active)
        existing = _worker_positions(farm)

        max_pending = min(
            MAX_MARKET_ORDERS,
            max(0, MAX_HANDS - len(farm["hands"])),
        )
        selected = None
        selected_pending = 0
        selected_starts = []

        for pending in range(max_pending + 1):
            plans, starts = optimize_with_pending_hires(
                jobs,
                existing,
                pending,
                current_hour=hour,
                shed_access=_open_shed_access(farm),
            )
            if plans is not None:
                selected = plans
                selected_pending = pending
                selected_starts = starts
                break

        if selected is None:
            if hour < 23 and len(farm["hands"]) < MAX_HANDS:
                hires = min(MAX_MARKET_ORDERS, MAX_HANDS - len(farm["hands"]))
                return _pass_action(farm, [["HIRE"] for _ in range(hires)])
            record_failure()
            return _pass_action(farm)

        missing_inputs = plan_input_orders(obs, selected)
        if missing_inputs:
            batch = list(missing_inputs[:MAX_MARKET_ORDERS])
            free_slots = MAX_MARKET_ORDERS - len(batch)
            hires_now = min(selected_pending, free_slots)
            batch.extend([["HIRE"] for _ in range(hires_now)])
            return _pass_action(farm, batch)

        state["plans"] = selected
        state["planned_hour"] = hour
        first_pending_index = len(existing)
        state["expected_pending_starts"] = {
            first_pending_index + offset: start
            for offset, start in enumerate(selected_starts)
        }
        state["checked_pending"] = set()

        worker = execute_plans(obs)
        worker["market"] = [["HIRE"] for _ in range(selected_pending)]
        return worker

    def agent(obs, configuration=None):
        effective_end = _effective_end_day(configuration, end_day)
        obs = dict(obs, _planning_end_day=effective_end)
        day, hour = obs["day"], obs["hour"]
        farm = obs["farms"][obs["player"]]

        if day != state["day"]:
            if hour == 0:
                return start_day(obs, effective_end)
            state.update(day=day, plans=None, deferred_orders=[])

        if hour == 0:
            return start_day(obs, effective_end)

        if state["deferred_orders"] and any(
            order[0] != "HIRE" for order in state["deferred_orders"]
        ):
            batch = state["deferred_orders"][:MAX_MARKET_ORDERS]
            state["deferred_orders"] = state["deferred_orders"][MAX_MARKET_ORDERS:]
            return _pass_action(farm, batch)
        if state["deferred_orders"]:
            state["deferred_orders"] = []

        if state["plans"] is None:
            action = plan_from_current_hour(obs, effective_end)
        else:
            worker = execute_plans(obs)
            worker["market"] = []
            action = worker

        if hour == 23:
            action["market"] = end_of_day_sell_orders(obs)

        return action

    agent.debug_state = state
    agent.targets = targets
    agent.lifecycle_book = book
    return agent


agent = make_agent()
