"""Choose the highest marginal net profit/day after forecast market slippage."""
from collections import Counter
from copy import copy
from dataclasses import dataclass

from kaggle_environments.envs.kaggriculture import kaggriculture as official_game
from kaggle_environments.envs.kaggriculture.kaggriculture import market_price
from agents.forecast import MarketForecast, Production, production
from agents.labor import LaborForecast
from agents.schedules import CROP_LAST_AGE, ONGOING_CROPS, cycle_finished
from agents.horizon import (
    SEASON_END_DAY, cycle_end as _cycle_end,
    planner_cycle_end as _planner_cycle_end,
    first_yield_age as _first_yield_age, can_start,
)

CROPS = ("WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY")
ANIMALS = ("GOOSE", "COW", "SHEEP")
SEED_COST = {name: official_game.CROPS[name]["seed"] for name in CROPS}
ANIMAL_COST = {name: official_game.ANIMALS[name]["cost"] for name in ANIMALS}
LAND_ORDER = official_game.LAND_ORDER
LAND_BUY_UTILIZATION = 0.75

# A standard 720-turn season has 30 days, indexed 0 through 29 by the engine.


def _expected_price(product, inventory, committed_units, town_demand=None):
    stock = inventory.get(product, 0) + committed_units.get(product, 0)
    return market_price(product, stock - (town_demand or {}).get(product, 0))


def _rotation(name, fertilize, day, end_day, tile=None):
    """Include crop replants only when a scheduled harvest fits the window."""
    end_day = _cycle_end(day, end_day)
    if tile is None and day + _first_yield_age(name) > end_day:
        return Production(), 0
    if name in ANIMALS:
        return production(name, fertilize, day, end_day, tile), (
            0 if tile is not None else ANIMAL_COST[name]
        )
    output = production(name, fertilize, day, end_day, tile)
    cost = 0 if tile is not None else SEED_COST[name]
    planted = tile["planted_day"] if tile is not None else day
    start = max(day, planted + CROP_LAST_AGE[name])
    first_harvest = (
        _first_yield_age(name)
        if name in ONGOING_CROPS
        else CROP_LAST_AGE[name]
    )
    while start + first_harvest <= end_day:
        output.add(production(name, fertilize, start, end_day))
        cost += SEED_COST[name]
        start += CROP_LAST_AGE[name]
    return output, cost


def _candidates(day, end_day):
    """Legacy shared-window candidates retained for experiments/tests."""
    end_day = _cycle_end(day, end_day)
    for name in sorted((*CROPS, *ANIMALS)):
        first = _first_yield_age(name)
        if day + first > end_day:
            continue
        for fertilize in ((False, True) if name in CROPS else (False,)):
            output, cost = _rotation(name, fertilize, day, end_day)
            yield (name, fertilize), output, cost


def _daily_candidates(day, end_day):
    """Fresh targets projected only through their own comparison horizon."""
    for name in sorted((*CROPS, *ANIMALS)):
        target_end = _planner_cycle_end(name, day, end_day)
        first = _first_yield_age(name)
        if day + first > target_end:
            continue
        for fertilize in ((False, True) if name in CROPS else (False,)):
            output, cost = _rotation(name, fertilize, day, target_end)
            yield (name, fertilize), output, cost


@dataclass(frozen=True)
class TargetProfit:
    choice: tuple
    output: Production
    market_cash: float
    capital_cost: float
    labor_cost: float
    days: int = 1

    @property
    def profit(self):
        return self.market_cash - self.capital_cost - self.labor_cost

    @property
    def profit_per_day(self):
        return self.profit / max(1, self.days)


def evaluate_targets(market, baseline, candidates, labor=None, position=(4, 4)):
    """Compare every target over the legacy shared min(16, remaining) window.

    This path remains available to the analysis helpers and regression tests.
    Live planner allocation uses ``evaluate_daily_targets`` below.
    """
    labor = labor or LaborForecast()
    results = []
    end = _cycle_end(market.day, market.end_day)
    scoped_market = copy(market)
    scoped_market.end_day = end

    def scoped(flow):
        result = Production()
        for field in ("sales", "inputs", "visits"):
            getattr(result, field).update(
                (d, value)
                for d, value in getattr(flow, field).items()
                if market.day <= d <= end
            )
        return result

    scoped_baseline = scoped(baseline)
    baseline_cost = labor.cost(scoped_baseline.visits)
    baseline_value = scoped_market.value(scoped_baseline)
    for choice, output, cost in candidates:
        if market.day + _first_yield_age(choice[0]) > end:
            continue
        output = scoped(output)
        results.append(
            TargetProfit(
                choice,
                output,
                scoped_market.marginal_profit(
                    scoped_baseline, output, 0, baseline_value
                ),
                cost,
                labor.marginal_cost(
                    scoped_baseline, output, position, baseline_cost
                ),
                max(1, end - market.day),
            )
        )
    return results


def evaluate_daily_targets(market, baseline, candidates, labor=None, position=(4, 4)):
    """Evaluate fresh targets on per-type horizons and normalize by elapsed day.

    Crops are scoped to their max-yield age. Animal windows are GOOSE=15,
    COW=14 and SHEEP=12 elapsed days, corresponding to first yield plus the
    requested six/three/two additional harvests. Near season end the window is
    truncated to the last playable day, but a target still must reach at least
    its first yield.
    """
    labor = labor or LaborForecast()
    results = []

    def scoped(flow, end):
        result = Production()
        for field in ("sales", "inputs", "visits"):
            getattr(result, field).update(
                (d, value)
                for d, value in getattr(flow, field).items()
                if market.day <= d <= end
            )
        return result

    for choice, output, cost in candidates:
        name = choice[0]
        end = _planner_cycle_end(name, market.day, market.end_day)
        if market.day + _first_yield_age(name) > end:
            continue

        scoped_market = copy(market)
        scoped_market.end_day = end
        scoped_baseline = scoped(baseline, end)
        scoped_output = scoped(output, end)
        baseline_cost = labor.cost(scoped_baseline.visits)
        baseline_value = scoped_market.value(scoped_baseline)
        results.append(
            TargetProfit(
                choice,
                scoped_output,
                scoped_market.marginal_profit(
                    scoped_baseline, scoped_output, 0, baseline_value
                ),
                cost,
                labor.marginal_cost(
                    scoped_baseline, scoped_output, position, baseline_cost
                ),
                max(1, end - market.day),
            )
        )
    return results


def _choose(market, baseline, candidates, counts, labor=None, position=(4, 4)):
    """Legacy absolute-profit selector retained for experiments/tests."""
    profitable = [
        result
        for result in evaluate_targets(
            market, baseline, candidates, labor, position
        )
        if result.profit > 0
    ]
    if not profitable:
        return None, None
    result = max(
        profitable,
        key=lambda r: (r.profit, -counts[r.choice[0]], r.choice),
    )
    return result.choice, result.output


def _choose_daily(market, baseline, candidates, counts, labor=None, position=(4, 4)):
    """Choose the highest positive marginal net profit per elapsed day."""
    profitable = [
        result
        for result in evaluate_daily_targets(
            market, baseline, candidates, labor, position
        )
        if result.profit > 0
    ]
    if not profitable:
        return None, None
    result = max(
        profitable,
        key=lambda r: (
            r.profit_per_day,
            r.profit,
            -counts[r.choice[0]],
            r.choice,
        ),
    )
    return result.choice, result.output


def _score(name, end_day, day, inventory, wheat_price, committed_units, unlocked_shops=()):
    market = MarketForecast(inventory, unlocked_shops, day, end_day)
    baseline = Production()
    baseline.sales[day].update(committed_units)
    candidates = [c for c in _candidates(day, end_day) if c[0][0] == name]
    results = evaluate_targets(market, baseline, candidates)
    if not results:
        return None, False
    best = max(results, key=lambda r: r.profit)
    return best.profit, best.choice[1]


def best_target(end_day, day, inventory, wheat_price, committed_units,
                category_capital=None, total_capital=0, unlocked_shops=()):
    # Legacy arguments retained for callers; capital share no longer affects rank.
    market = MarketForecast(inventory, unlocked_shops, day, end_day)
    baseline = Production()
    baseline.sales[day].update(committed_units)
    return _choose_daily(
        market, baseline, list(_daily_candidates(day, end_day)), Counter()
    )[0]


def should_buy_land(farm, active_positions):
    """Buy the next quadrant only once real production -- not just a target
    decision -- already fills most of the land currently owned. A target is
    assigned to (almost) every open tile the instant it unlocks (see
    plan_targets), so gating on "has a target" would trigger on day 0 before
    a single seed is planted; gating on what's actually growing/placed
    reflects real land pressure instead."""
    n_extra = len(farm["unlocked_quadrants"]) - 1
    if n_extra >= len(LAND_ORDER):
        return False
    if not active_positions:
        return False
    tiles = farm["tiles"]
    occupied = 0
    for x, y in active_positions:
        tile = tiles[y][x]
        if isinstance(tile, dict) and (
            tile.get("kind") == "PLANT" or "animal" in tile
        ):
            occupied += 1
    return occupied / len(active_positions) >= LAND_BUY_UTILIZATION


def plan_targets(obs, targets, active_positions, end_day):
    """Reprice targets against actual production and route-feasible commitments.

    Existing crops are forecast from their real age, held yield and watering
    state, including visible opponent crops. Unsown choices are reconsidered
    each day. Fresh choices use target-specific horizons and are ranked by
    marginal net profit/day, while shared baseline saturation and route load
    are updated after every accepted tile.

    Capacity comes from LaborForecast itself. Every accepted target enters the
    shared baseline, and later candidates are rejected naturally if the
    combined visits cannot fit the farmer plus maximum hands.
    """
    day = obs["day"]
    end_day = _cycle_end(day, end_day)
    farm = obs["farms"][obs["player"]]
    baseline = Production()
    counts = Counter()
    replanning = []
    external = Production()

    for position in active_positions:
        x, y = position
        tile = farm["tiles"][y][x]
        current = targets.get(position)
        if isinstance(tile, dict) and tile.get("animal"):
            continue
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if not cycle_finished(tile["crop"], day - tile["planted_day"], tile):
                continue
            # Preserve a conversion already scheduled by the opening.
            if (
                current
                and current[0] != tile["crop"]
                and can_start(current[0], day, end_day)
            ):
                continue
        replanning.append(position)

    for player, other_farm in enumerate(obs["farms"]):
        # Hand-built test observations may alias the same farm twice.
        if player != obs["player"] and other_farm is farm:
            continue
        for y, row in enumerate(other_farm["tiles"]):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict):
                    continue
                name = tile.get("animal") or (
                    tile.get("crop") if tile.get("kind") == "PLANT" else None
                )
                if not name:
                    continue
                target = targets.get((x, y)) if player == obs["player"] else None
                fertilize = bool(target and target[0] == name and target[1])
                # Opponent future fertilizer decisions are not observable.
                destination = baseline if player == obs["player"] else external
                if player == obs["player"] and (x, y) not in replanning:
                    future, _ = _rotation(name, fertilize, day, end_day, tile)
                else:
                    future = production(name, fertilize, day, end_day, tile)
                destination.add(future, (x, y))
                if player == obs["player"] and (x, y) not in replanning:
                    counts[name] += 1

    # Include unsold goods exactly once, separately from remaining tile output.
    baseline.sales[day].update({
        p: n
        for p, n in obs["private"]["shed"].items()
        if p in official_game.PRODUCTS
    })
    for carried in obs["private"].get("inventories", []):
        baseline.sales[day].update({
            p: n
            for p, n in carried.items()
            if p in official_game.PRODUCTS
        })
    for position, target in targets.items():
        if not target or position in replanning:
            continue
        x, y = position
        tile = farm["tiles"][y][x]
        actual = (
            tile.get("animal") or tile.get("crop")
            if isinstance(tile, dict)
            else None
        )
        if actual != target[0] and not actual:
            baseline.add(_rotation(*target, day, end_day)[0], position)
            counts[target[0]] += 1

    market = MarketForecast(
        obs["market"]["inventory"],
        obs["town"]["unlocked_shops"],
        day,
        end_day,
        obs.get("hour", 0),
        obs["market"].get("params"),
        external,
    )
    candidates = list(_daily_candidates(day, end_day))
    shed_access = ((4, 4), (5, 4), (4, 5), (5, 5))

    def distance(p):
        return min(
            abs(p[0] - s[0]) + abs(p[1] - s[1])
            for s in shed_access
        )

    labor = LaborForecast(tuple(
        p
        for p in shed_access
        if farm["tiles"][p[1]][p[0]] != "LOCKED"
    ))

    # Assign near the shed first so greedy marginal allocation spends route
    # capacity on cheaper-to-service positions before distant ones. Every
    # accepted target changes market saturation and labor load for the next
    # position; the score itself is marginal net profit/day on that target's
    # own horizon.
    for position in sorted(
        replanning, key=lambda p: (distance(p), p[1], p[0])
    ):
        x, y = position
        tile = farm["tiles"][y][x]
        allowed = candidates
        if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE"):
            allowed = [
                c
                for c in candidates
                if c[0][0] in ANIMALS
                and official_game.ANIMALS[c[0][0]]["structure"] == tile["kind"]
            ]

        choice, output = _choose_daily(
            market, baseline, allowed, counts, labor, position
        )
        targets[position] = choice
        if not choice:
            continue

        baseline.add(output, position)
        counts[choice[0]] += 1
