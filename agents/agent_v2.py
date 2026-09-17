"""V2 Kaggriculture agent built around locked lifecycle schedules.

Strategy decides targets. Experiment-backed lifecycle schedules decide the
required tile actions. The daily scheduler only solves labor/routes. The
executor has no runtime harvest interrupt, no rescue scheduler, and no hidden
priority classes.
"""

from __future__ import annotations

from collections import Counter

from agents.farm_tasks import purchase_orders
from agents.opening_book import make_opening_controller, should_buy_land_on_schedule
from agents.planner import SEASON_END_DAY, plan_targets, should_buy_land
from agents.selling import SELLABLE
from agents.v2_lifecycle import ScheduleBook
from agents.v2_scheduler import (
    MAX_HANDS,
    build_plans,
    global_supply_plan,
    hands_needed,
    predicted_spawn_starts,
)
from agents.v2_tasks import build_v2_jobs

BOARD_SIZE = 10
MARKET_CAP = 10
SHED_CAPACITY = 100


def _active_positions(farm):
    tiles = farm["tiles"]
    return [
        (x, y)
        for y in range(BOARD_SIZE)
        for x in range(BOARD_SIZE)
        if tiles[y][x] != "LOCKED"
    ]


def _open_shed_access(farm):
    half = BOARD_SIZE // 2
    candidates = ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))
    return tuple((x, y) for x, y in candidates if farm["tiles"][y][x] != "LOCKED")


def _dedupe_market_orders(orders):
    result = []
    quantities = Counter()
    index = {}
    for order in orders:
        if not order:
            continue
        name = order[0]
        if name in {"BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "SELL"} and len(order) >= 3:
            key = (name, order[1])
            if key not in index:
                index[key] = len(result)
                result.append([name, order[1], 0])
            quantities[key] += int(order[2])
        else:
            result.append(list(order))
    for key, row in index.items():
        result[row][2] = quantities[key]
    return [order for order in result if len(order) < 3 or order[2] > 0]


def _capacity_sell_orders(obs):
    """Cash out at the last step and guarantee room for carried output."""
    shed = obs["private"]["shed"]
    carried = Counter()
    for inventory in obs["private"].get("inventories", []):
        carried.update(inventory)
    incoming = sum(max(0, int(n)) for n in carried.values())
    used = sum(max(0, int(n)) for n in shed.values())
    excess = max(0, used + incoming - SHED_CAPACITY)

    orders = []
    sold = 0
    items = [item for item in SELLABLE if item != "WHEAT"] + ["WHEAT"]
    for item in items:
        quantity = int(shed.get(item, 0))
        if quantity <= 0:
            continue
        if item == "WHEAT" and sold >= excess:
            break
        amount = quantity if item != "WHEAT" else min(quantity, max(0, excess - sold))
        if amount:
            orders.append(["SELL", item, amount])
            sold += amount
    return orders[:MARKET_CAP]


def _supply_buy_orders(shortfall):
    orders = []
    for item, quantity in sorted(shortfall.items()):
        quantity = int(quantity)
        if quantity <= 0:
            continue
        if item in {"WHEAT", "FERTILIZER"}:
            orders.append(["BUY_PRODUCT", item, quantity])
        elif item in {"GOOSE", "COW", "SHEEP"}:
            orders.append(["BUY_ANIMAL", item, quantity])
    return orders


def _extra_hires_to_fit(jobs, starts, hour, shed_access):
    """Minimum extra hands that make the real jobs fit after one HIRE turn.

    While an unplanned day is still being solved every real worker PASSes, so
    their post-movement positions are exactly ``starts``. HIRE is processed
    after those PASSes. That makes the spawn rule deterministic here: predict
    from the real current occupancy, give every worker one fewer remaining
    step, and retry the exact same packer used for execution.
    """
    existing_hands = max(0, len(starts) - 1)
    room = max(0, MAX_HANDS - existing_hands)
    next_budget = max(0, 24 - (hour + 1))
    if not room or not next_budget:
        return 0

    for extra in range(1, room + 1):
        spawned = predicted_spawn_starts(starts, extra, shed_access)
        candidate_starts = [*starts, *spawned]
        budgets = [next_budget] * len(candidate_starts)
        _, unassigned = build_plans(
            jobs,
            candidate_starts,
            budgets,
            shed_access=shed_access,
        )
        if not unassigned:
            return extra
    return 0


def make_agent(end_day=SEASON_END_DAY, seed=0):
    del seed
    targets = {}
    opening_governs = make_opening_controller()
    schedules = ScheduleBook(end_day=end_day)
    state = {
        "day": -1,
        "plans": [],
        "planned": False,
        "hand_target": 0,
        "pending_market": [],
        "last_market_submission_hour": -1,
        "invariant_failures": 0,
    }

    def begin_day(obs, farm):
        active = _active_positions(farm)
        opening = opening_governs(obs, targets, active)
        if not opening:
            plan_targets(obs, targets, active, end_day)

        # Daily labor sizing is independent of what happens to be in the shed
        # right now. Required PLANT/PLACE work is part of the schedule first;
        # Buyer is responsible for making those inputs exist before execution.
        assumed_jobs = build_v2_jobs(
            obs,
            targets,
            schedules,
            assume_crop_seeds=True,
            assume_animal_inputs=True,
        )
        shed_access = _open_shed_access(farm)
        desired_hands = hands_needed(
            assumed_jobs,
            tuple(farm["farmer"]),
            shed_access=shed_access,
            max_hands=MAX_HANDS,
        )
        state["hand_target"] = desired_hands

        market = []
        if should_buy_land_on_schedule(obs, farm):
            market.append(["BUY_LAND"])
        elif (
            not opening
            and len(farm.get("unlocked_quadrants", ())) < 3
            and should_buy_land(farm, active)
        ):
            market.append(["BUY_LAND"])

        # Scheduler owns the count; Buyer only places HIRE orders.
        market.extend([["HIRE"] for _ in range(max(0, desired_hands - len(farm["hands"])))])
        # Same-crop replacement seed is part of HARVEST -> PLANT -> WATER,
        # rather than being delayed until the tile is observed empty tomorrow.
        market.extend(
            purchase_orders(
                obs,
                targets,
                active,
                replant_same_crop=True,
            )
        )
        state["pending_market"] = _dedupe_market_orders(market)
        state["plans"] = []
        state["planned"] = False
        state["last_market_submission_hour"] = -1

    def build_day_plan(obs, farm):
        # Use the same inventory-independent job set as morning hand sizing.
        # The old code switched back to real current inputs here, which could
        # change atomic chains/split points and make a hand count that fitted at
        # dawn fail once the real queue was constructed.
        jobs = build_v2_jobs(
            obs,
            targets,
            schedules,
            assume_crop_seeds=True,
            assume_animal_inputs=True,
        )
        starts = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
        remaining = max(0, 24 - obs["hour"])
        budgets = [remaining] * len(starts)
        shed_access = _open_shed_access(farm)
        plans, unassigned = build_plans(
            jobs,
            starts,
            budgets,
            shed_access=shed_access,
        )

        if unassigned and len(farm["hands"]) < MAX_HANDS:
            extra = _extra_hires_to_fit(
                jobs,
                starts,
                obs["hour"],
                shed_access,
            )
            if extra:
                # This is still the scheduler solving the day, not a runtime
                # rescue. Nothing has started yet; hire the missing capacity,
                # observe the exact spawn next turn, then solve once more.
                state["hand_target"] = len(farm["hands"]) + extra
                state["pending_market"] = [["HIRE"] for _ in range(extra)]
                state["plans"] = []
                state["planned"] = False
                return False

        if unassigned:
            # At the hard worker/time ceiling keep the feasible queues but make
            # the invariant visible. There is deliberately no task priority or
            # hidden rescue ordering here.
            state["invariant_failures"] += len(unassigned)

        state["plans"] = plans
        state["planned"] = True

        supply = global_supply_plan(plans, obs["private"]["shed"])
        state["pending_market"] = _dedupe_market_orders(
            [*state["pending_market"], *_supply_buy_orders(supply.buy_shortfall)]
        )
        return True

    def agent(obs):
        farm = obs["farms"][obs["player"]]
        day, hour = obs["day"], obs["hour"]

        if day != state["day"]:
            state["day"] = day
            begin_day(obs, farm)

        positions = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
        operations = [["PASS"] for _ in positions]

        # Market scheduler starts at step 0 and respects the 10-order cap.
        market = state["pending_market"][:MARKET_CAP]
        state["pending_market"] = state["pending_market"][MARKET_CAP:]
        if market:
            state["last_market_submission_hour"] = hour

        # Market processing follows worker actions. Never construct a route
        # from the stale observation in the same turn a required input/HIRE was
        # submitted; use the next observation where it has actually landed.
        settled = state["last_market_submission_hour"] < hour
        if (
            not state["planned"]
            and not state["pending_market"]
            and hour > 0
            and settled
        ):
            build_day_plan(obs, farm)
            if state["pending_market"]:
                room = max(0, MARKET_CAP - len(market))
                extra = state["pending_market"][:room]
                state["pending_market"] = state["pending_market"][len(extra):]
                market.extend(extra)
                if extra:
                    state["last_market_submission_hour"] = hour

        can_execute = (
            state["planned"]
            and not state["pending_market"]
            and state["last_market_submission_hour"] < hour
        )
        if can_execute:
            for index in range(min(len(operations), len(state["plans"]))):
                queue = state["plans"][index].queue
                if queue:
                    operations[index] = queue.pop(0)

        # No forced HARVEST interrupt, no displaced-op reinsertion, no late
        # rescue scheduler and no mid-day DROP. Workers carry output to EOD.
        if hour == 23:
            market = _capacity_sell_orders(obs)

        return {
            "farmer": operations[0] if operations else ["PASS"],
            "hands": operations[1:],
            "market": market[:MARKET_CAP],
        }

    return agent


agent = make_agent()
