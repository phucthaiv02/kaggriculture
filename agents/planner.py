"""Choose the highest marginal net profit after forecast market slippage."""
from collections import Counter
from copy import copy
from dataclasses import dataclass

from kaggle_environments.envs.kaggriculture import kaggriculture as official_game
from agents.forecast import MarketForecast, Production, production
from agents.labor import LaborForecast
from agents.schedules import CROP_LAST_AGE, ONGOING_CROPS, cycle_finished
from agents.horizon import (
    SEASON_END_DAY, cycle_end as _cycle_end,
    first_yield_age as _first_yield_age, can_start,
)

from agents.products import ANIMALS, ANIMAL_COST, CROPS, SEED_COST


def _rotation(name, fertilize, day, end_day, tile=None):
    """Include crop replants only when a scheduled harvest fits the window."""
    end_day = _cycle_end(day, end_day)
    if tile is None and day + _first_yield_age(name) > end_day:
        return Production(), 0
    if name in ANIMALS:
        return production(name, fertilize, day, end_day, tile), (0 if tile is not None else ANIMAL_COST[name])
    output = production(name, fertilize, day, end_day, tile)
    cost = 0 if tile is not None else SEED_COST[name]
    planted = tile['planted_day'] if tile is not None else day
    start = max(day, planted + CROP_LAST_AGE[name])
    first_harvest = _first_yield_age(name) if name in ONGOING_CROPS else CROP_LAST_AGE[name]
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
                (d, value) for d, value in getattr(flow, field).items()
                if market.day <= d <= end)
        return result
    scoped_baseline = scoped(baseline)
    prepared_labor = labor.prepare(scoped_baseline.visits)
    baseline_cost = labor.prepared_cost(prepared_labor)
    baseline_value = scoped_market.value(scoped_baseline)
    for choice, output, cost in candidates:
        if market.day + _first_yield_age(choice[0]) > end:
            continue
        output = scoped(output)
        results.append(TargetProfit(
            choice, output,
            scoped_market.marginal_profit(scoped_baseline, output, 0, baseline_value),
            cost, labor.marginal_cost(scoped_baseline, output, position, baseline_cost, prepared=prepared_labor),
        ))
    return results


def _choose(market, baseline, candidates, counts, labor=None, position=(4, 4)):
    profitable = [result for result in evaluate_targets(market, baseline, candidates, labor, position)
                  if result.profit > 0]
    if not profitable:
        return None, None
    result = max(profitable, key=lambda r: (r.profit, -counts[r.choice[0]], r.choice))
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


def plan_targets(obs, targets, active_positions, end_day, *, max_positions=None, replan_positions=None):
    """Reprice each new commitment against actual remaining production.

    Existing crops are forecast from their real age, held yield and watering
    state, including visible opponent crops. Unsown choices are reconsidered
    each day. No profitable candidate means no planting, with harvest-only
    cleanup handled by build_tasks.

    With max_positions, price only that many eligible tiles and return the
    remainder for later turns. replan_positions restricts this pass to pending
    work; all other existing targets still contribute to the forecast.
    """
    day = obs["day"]
    end_day = _cycle_end(day, end_day)
    farm = obs["farms"][obs["player"]]
    baseline = Production()
    counts = Counter()
    replanning = []
    external = Production()
    for position in active_positions:
        if replan_positions is not None and position not in replan_positions:
            continue
        x, y = position
        tile = farm["tiles"][y][x]
        current = targets.get(position)
        if isinstance(tile, dict) and tile.get("animal"):
            continue
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if not cycle_finished(tile["crop"], day - tile["planted_day"], tile):
                continue
            # Preserve a conversion already scheduled by the opening.
            if current and current[0] != tile["crop"] and can_start(current[0], day, end_day):
                continue
        replanning.append(position)

    shed_access = ((4, 4), (5, 4), (4, 5), (5, 5))
    def distance(p):
        return min(abs(p[0] - s[0]) + abs(p[1] - s[1]) for s in shed_access)
    replanning.sort(key=lambda p: (distance(p), p[1], p[0]))
    pending = replanning[max_positions:] if max_positions is not None else []
    if max_positions is not None:
        replanning = replanning[:max_positions]
    if not replanning:
        return pending
    replanning_set = set(replanning)

    for player, other_farm in enumerate(obs["farms"]):
        # Hand-built test observations may alias the same farm twice.
        if player != obs["player"] and other_farm is farm:
            continue
        for y, row in enumerate(other_farm["tiles"]):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict):
                    continue
                name = tile.get("animal") or (tile.get("crop") if tile.get("kind") == "PLANT" else None)
                if not name:
                    continue
                target = targets.get((x, y)) if player == obs["player"] else None
                fertilize = bool(target and target[0] == name and target[1])
                # Opponent future fertilizer decisions are not observable.
                destination = baseline if player == obs["player"] else external
                if player == obs["player"] and (x, y) not in replanning_set:
                    future, _ = _rotation(name, fertilize, day, end_day, tile)
                else:
                    future = production(name, fertilize, day, end_day, tile)
                destination.add(future, (x, y))
                if player == obs["player"] and (x, y) not in replanning_set:
                    counts[name] += 1

    # Include unsold goods exactly once, separately from remaining tile output.
    baseline.sales[day].update({p: n for p, n in obs["private"]["shed"].items() if p in official_game.PRODUCTS})
    for carried in obs["private"].get("inventories", []):
        baseline.sales[day].update({p: n for p, n in carried.items() if p in official_game.PRODUCTS})
    for position, target in targets.items():
        if not target or position in replanning_set:
            continue
        x, y = position
        tile = farm["tiles"][y][x]
        actual = tile.get("animal") or tile.get("crop") if isinstance(tile, dict) else None
        if actual != target[0] and not actual:
            baseline.add(_rotation(*target, day, end_day)[0], position)
            counts[target[0]] += 1

    market = MarketForecast(obs["market"]["inventory"], obs["town"]["unlocked_shops"],
                            day, end_day, obs.get("hour", 0), obs["market"].get("params"), external)
    candidates = list(_candidates(day, end_day))
    labor = LaborForecast(tuple(p for p in shed_access if farm["tiles"][p[1]][p[0]] != "LOCKED"))
    # Assign in a stable order near the shed, updating supply and labor after every pick.
    for position in replanning:
        x, y = position
        tile = farm["tiles"][y][x]
        allowed = candidates
        if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE"):
            allowed = [c for c in candidates if c[0][0] in ANIMALS
                       and official_game.ANIMALS[c[0][0]]["structure"] == tile["kind"]]
        choice, output = _choose(market, baseline, allowed, counts, labor, position)
        targets[position] = choice
        if choice:
            # Include this commitment's supply and labor in the shared window.
            baseline.add(output, position)
            counts[choice[0]] += 1
    return pending
