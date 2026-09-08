"""Choose the highest marginal net profit after forecast market slippage."""
from collections import Counter
from copy import copy
from dataclasses import dataclass

from kaggle_environments.envs.kaggriculture import kaggriculture as official_game
from kaggle_environments.envs.kaggriculture.kaggriculture import market_price
from agents.forecast import MarketForecast, Production, production
from agents.labor import LaborForecast
from agents.scheduler import MAX_HANDS
from agents.schedules import CROP_LAST_AGE, ONGOING_CROPS, cycle_finished
from agents.horizon import (
    SEASON_END_DAY, TARGET_HORIZON_DAYS, cycle_end as _cycle_end,
    first_yield_age as _first_yield_age, can_start,
)

CROPS = ("WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY")
ANIMALS = ("GOOSE", "COW", "SHEEP")
SEED_COST = {name: official_game.CROPS[name]["seed"] for name in CROPS}
ANIMAL_COST = {name: official_game.ANIMALS[name]["cost"] for name in ANIMALS}
LAND_ORDER = official_game.LAND_ORDER
LAND_BUY_UTILIZATION = 0.75

# At most one brand-new producer per currently possible worker is allowed to
# influence the shared market/labor baseline during one planning pass. The
# executor may start more than this, but those extra desired targets are not
# treated as committed production until a later observation makes them real.
FRESH_COMMIT_SLOTS = MAX_HANDS + 1  # farmer + maximum hired hands

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
    end_day = _cycle_end(day, end_day)
    for name in sorted((*CROPS, *ANIMALS)):
        first = _first_yield_age(name)
        if day + first > end_day:
            continue
        for fertilize in ((False, True) if name in CROPS else (False,)):
            output, cost = _rotation(name, fertilize, day, end_day)
            yield (name, fertilize), output, cost


@dataclass(frozen=True)
class TargetProfit:
    choice: tuple
    output: Production
    market_cash: float
    capital_cost: float
    labor_cost: float

    @property
    def profit(self):
        """Single-cycle added cash, never divided by days, cost or tile share."""
        return self.market_cash - self.capital_cost - self.labor_cost


def evaluate_targets(market, baseline, candidates, labor=None, position=(4, 4)):
    """Compare every target over the same min(16, remaining days) horizon.

    market_cash = change in all sales less feed/fertilizer purchases, after
    per-unit slippage, existing production and SHOP/TOWN demand.
    capital_cost = all seeds in the window or one animal purchase.
    labor_cost = additional daily worker cost over the same horizon.
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
            )
        )
    return results


def _choose(market, baseline, candidates, counts, labor=None, position=(4, 4)):
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
    return _choose(market, baseline, list(_candidates(day, end_day)), Counter())[0]


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


def _is_fresh_target_tile(tile):
    """True when a choice would create a producer on currently idle land."""
    return not (
        isinstance(tile, dict)
        and (tile.get("kind") == "PLANT" or tile.get("animal"))
    )


def plan_targets(obs, targets, active_positions, end_day):
    """Reprice targets against actual production plus a bounded commitment set.

    Existing crops are forecast from their real age, held yield and watering
    state, including visible opponent crops. Unsown choices are reconsidered
    each day. No profitable candidate means no planting, with harvest-only
    cleanup handled by build_tasks.

    A target decision is not the same thing as a producer that will actually
    start this pass. Fresh empty/structure tiles are still assigned desired
    targets so the executor can use any spare capacity, but only the first
    ``FRESH_COMMIT_SLOTS`` (nearest the shed, matching the planning order) are
    allowed to add future supply/labor to the shared baseline. That prevents
    a whole unlocked quadrant from depressing later candidate scores before
    the scheduler has had a chance to materialize it. Finished/replanting
    producers remain committed normally.
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
    candidates = list(_candidates(day, end_day))
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
    fresh_commits = 0

    # Assign near the shed first. Only a bounded number of brand-new choices
    # enter the shared baseline; all desired targets are still recorded.
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

        choice, output = _choose(
            market, baseline, allowed, counts, labor, position
        )
        targets[position] = choice
        if not choice:
            continue

        fresh = _is_fresh_target_tile(tile)
        if fresh and fresh_commits >= FRESH_COMMIT_SLOTS:
            # Desired but not trusted as immediate committed production.
            continue

        baseline.add(output, position)
        counts[choice[0]] += 1
        if fresh:
            fresh_commits += 1
