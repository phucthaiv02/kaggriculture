"""Choose production targets from marginal market profit and capital cost."""
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


TARGET_OPTIONS = (1, 2, 3, 4)
TARGET_OPTION = 1
TARGET_DISCOUNT = 0.97
TARGET_ROI_CAPITAL_FLOOR = 100.0
TARGET_LABOR_SHORTLIST_FRACTION = 0.10
TARGET_LABOR_SHORTLIST_FLOOR = 150.0


def set_target_option(option):
    """Select the target ranking formula used by plan_targets."""
    global TARGET_OPTION
    option = int(option)
    if option not in TARGET_OPTIONS:
        raise ValueError(f"target option must be one of {TARGET_OPTIONS}, got {option}")
    TARGET_OPTION = option


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
    labor_cost: float = 0.0
    discounted_market_cash: float | None = None

    @property
    def profit(self):
        """Marginal market cash minus target capital; labor is scheduler-owned."""
        return self.market_cash - self.capital_cost

    def score(self, option=None):
        """Return one of the benchmark target-ranking formulas."""
        option = TARGET_OPTION if option is None else int(option)
        if option == 1:
            return self.profit
        if option in (2, 4):
            market_cash = (
                self.market_cash if self.discounted_market_cash is None
                else self.discounted_market_cash
            )
            return market_cash - self.capital_cost
        if option == 3:
            return self.profit / max(self.capital_cost, TARGET_ROI_CAPITAL_FLOOR)
        raise ValueError(f"target option must be one of {TARGET_OPTIONS}, got {option}")


def evaluate_targets(
    market, baseline, candidates, labor=None, position=(4, 4), *, target_option=None
):
    """Compare targets over the same min(16, remaining days) horizon.

    market_cash is the change in all sales less feed/fertilizer purchases after
    per-unit slippage, existing production and SHOP/TOWN demand. capital_cost
    is seed/replant or animal capital. Hire/labor cost is deliberately excluded
    from target economics and remains the scheduler's responsibility.

    ``labor`` and ``position`` remain accepted for compatibility with analysis
    tools. Option 4 applies labor only after economic evaluation, inside _choose.
    """
    del labor, position
    option = TARGET_OPTION if target_option is None else int(target_option)
    if option not in TARGET_OPTIONS:
        raise ValueError(f"target option must be one of {TARGET_OPTIONS}, got {option}")

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
    baseline_value = scoped_market.value(scoped_baseline)
    discounted_baseline_value = (
        scoped_market.value(scoped_baseline, discount=TARGET_DISCOUNT)
        if option in (2, 4) else None
    )
    for choice, output, cost in candidates:
        if market.day + _first_yield_age(choice[0]) > end:
            continue
        output = scoped(output)
        combined = Production()
        combined.add(scoped_baseline)
        combined.add(output)
        market_cash = scoped_market.value(combined) - baseline_value
        discounted_market_cash = None
        if option in (2, 4):
            discounted_market_cash = (
                scoped_market.value(combined, discount=TARGET_DISCOUNT)
                - discounted_baseline_value
            )
        results.append(TargetProfit(
            choice,
            output,
            market_cash,
            cost,
            0.0,
            discounted_market_cash,
        ))
    return results


def _choose(
    market, baseline, candidates, counts, labor=None, position=(4, 4), *, target_option=None
):
    option = TARGET_OPTION if target_option is None else int(target_option)
    scored = []
    for result in evaluate_targets(
        market, baseline, candidates, labor, position, target_option=option
    ):
        score = result.score(option)
        if score > 0:
            scored.append((score, result))
    if not scored:
        return None, None

    if option == 4:
        # Economics remains authoritative. Labor only breaks ties among targets
        # close enough to the best discounted profit that route cost can matter.
        best_economic = max(score for score, _ in scored)
        margin = max(
            TARGET_LABOR_SHORTLIST_FLOOR,
            TARGET_LABOR_SHORTLIST_FRACTION * abs(best_economic),
        )
        shortlist = [
            (score, result) for score, result in scored
            if score >= best_economic - margin
        ]
        labor = LaborForecast() if labor is None else labor
        prepared = labor.prepare(baseline.visits)
        baseline_cost = labor.prepared_cost(prepared)

        def option4_key(item):
            score, result = item
            labor_cost = labor.marginal_cost(
                baseline,
                result.output,
                position,
                baseline_cost=baseline_cost,
                prepared=prepared,
            )
            return (
                labor_cost,
                _first_yield_age(result.choice[0]),
                result.capital_cost,
                -score,
                counts[result.choice[0]],
                result.choice,
            )

        _, result = min(shortlist, key=option4_key)
        return result.choice, result.output

    _, result = max(
        scored,
        key=lambda item: (item[0], -counts[item[1].choice[0]], item[1].choice),
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
    best = max(results, key=lambda r: r.score())
    return best.score(), best.choice[1]


def best_target(end_day, day, inventory, wheat_price, committed_units,
                category_capital=None, total_capital=0, unlocked_shops=(),
                target_option=None):
    # Legacy arguments retained for callers; capital share no longer affects rank.
    market = MarketForecast(inventory, unlocked_shops, day, end_day)
    baseline = Production()
    baseline.sales[day].update(committed_units)
    return _choose(
        market, baseline, list(_candidates(day, end_day)), Counter(),
        target_option=target_option,
    )[0]


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
    committed_positions = obs.get("_committed_targets")
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
        if (actual != target[0] and not actual
                and (committed_positions is None or position in committed_positions)):
            baseline.add(_rotation(*target, day, end_day)[0], position)
            counts[target[0]] += 1

    market = MarketForecast(obs["market"]["inventory"], obs["town"]["unlocked_shops"],
                            day, end_day, obs.get("hour", 0), obs["market"].get("params"), external)
    candidates = list(_candidates(day, end_day))
    labor = LaborForecast(shed_access) if TARGET_OPTION == 4 else None
    # Assign in a stable order near the shed, updating supply after every pick.
    for position in replanning:
        x, y = position
        tile = farm["tiles"][y][x]
        allowed = candidates
        if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE"):
            allowed = [c for c in candidates if c[0][0] in ANIMALS
                       and official_game.ANIMALS[c[0][0]]["structure"] == tile["kind"]]
        choice, output = _choose(
            market, baseline, allowed, counts, labor=labor, position=position
        )
        targets[position] = choice
        if choice:
            # Include this commitment's supply in the shared window.
            baseline.add(output, position)
            counts[choice[0]] += 1
    return pending
