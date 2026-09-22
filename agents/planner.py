"""Choose production targets from marginal market profit and capital cost."""
from collections import Counter
from copy import copy
from dataclasses import dataclass

from kaggle_environments.envs.kaggriculture import kaggriculture as official_game
from agents.fertilizer import FertilizerPlan, cycle_plan, plan_ages, plan_json
from agents.forecast import MarketForecast, Production, production
from agents.schedules import (
    CROP_FERTILIZE_DAYS, CROP_LAST_AGE, ONGOING_CROPS, cycle_finished,
)
from agents.horizon import (
    SEASON_END_DAY, cycle_end as _cycle_end,
    first_yield_age as _first_yield_age, can_start,
)

from agents.products import ANIMALS, ANIMAL_COST, CROPS, SEED_COST


TARGET_SWITCH_MARGIN = 1.0


def _crop_cycle_starts(name, day, end_day, tile=None):
    """Cycle starts represented by the existing rotation forecast."""
    end_day = _cycle_end(day, end_day)
    first_yield = _first_yield_age(name)
    if tile is None:
        if day + first_yield > end_day:
            return []
        starts = [day]
        next_start = day + CROP_LAST_AGE[name]
    else:
        planted = tile["planted_day"]
        starts = [planted]
        next_start = max(day, planted + CROP_LAST_AGE[name])

    # A later rotation only matters when it can produce at least once inside
    # the same planner horizon. Same-day harvest -> replant remains allowed.
    while next_start + first_yield <= end_day:
        starts.append(next_start)
        next_start += CROP_LAST_AGE[name]
    return starts


def _fertilizer_plans(name, day, end_day):
    """Enumerate every legal dated fertilizer event combination in horizon."""
    starts = _crop_cycle_starts(name, day, end_day)
    events = []
    for cycle_index, start in enumerate(starts):
        for age in sorted(CROP_FERTILIZE_DAYS[name]):
            when = start + age
            if day <= when <= _cycle_end(day, end_day):
                events.append((cycle_index, age))

    # Current crops have at most five fertilizer events in a 16-day window, so
    # exhaustive enumeration is tiny (<= 32 plans) and avoids a heuristic.
    for mask in range(1 << len(events)):
        cycles = [[] for _ in starts]
        for bit, (cycle_index, age) in enumerate(events):
            if mask & (1 << bit):
                cycles[cycle_index].append(age)
        yield FertilizerPlan(tuple(tuple(values) for values in cycles))


def _rotation(name, fertilize, day, end_day, tile=None):
    """Include replants only when a scheduled harvest fits the window.

    For crops, ``fertilize`` may be a legacy boolean or a FertilizerPlan whose
    cycles independently select fertilizer events. Future-cycle events are
    forecast assumptions only; the real tile is repriced when each cycle ends.
    """
    end_day = _cycle_end(day, end_day)
    if tile is None and day + _first_yield_age(name) > end_day:
        return Production(), 0
    if name in ANIMALS:
        return production(name, False, day, end_day, tile), (0 if tile is not None else ANIMAL_COST[name])

    starts = _crop_cycle_starts(name, day, end_day, tile)
    if not starts:
        return Production(), 0

    output = Production()
    cost = 0
    for index, start in enumerate(starts):
        per_cycle = cycle_plan(fertilize, index)
        if index == 0 and tile is not None:
            output.add(production(name, per_cycle, day, end_day, tile))
            continue
        output.add(production(name, per_cycle, start, end_day))
        cost += SEED_COST[name]
    return output, cost


def _candidates(day, end_day):
    end_day = _cycle_end(day, end_day)
    for name in sorted((*CROPS, *ANIMALS)):
        first = _first_yield_age(name)
        if day + first > end_day:
            continue
        if name in ANIMALS:
            output, cost = _rotation(name, False, day, end_day)
            yield (name, False), output, cost
            continue
        for fertilize in _fertilizer_plans(name, day, end_day):
            output, cost = _rotation(name, fertilize, day, end_day)
            yield (name, fertilize), output, cost


@dataclass(frozen=True)
class TargetProfit:
    choice: tuple
    output: Production
    market_cash: float
    capital_cost: float
    labor_cost: float = 0.0

    @property
    def profit(self):
        """Marginal market cash minus target capital; labor is scheduler-owned."""
        return self.market_cash - self.capital_cost


def evaluate_targets(
    market, baseline, candidates, labor=None, position=(4, 4)
):
    """Compare targets over the same min(16, remaining days) horizon.

    market_cash is the change in all sales less feed/fertilizer purchases after
    per-unit slippage, existing production and SHOP/TOWN demand. capital_cost
    is seed/replant or animal capital. Hire/labor cost is deliberately excluded
    from target economics and remains the scheduler's responsibility.

    ``labor`` and ``position`` remain accepted for compatibility with analysis
    tools, but no labor forecast contributes to the score.
    """
    del labor, position
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
    for choice, output, cost in candidates:
        if market.day + _first_yield_age(choice[0]) > end:
            continue
        output = scoped(output)
        combined = Production()
        combined.add(scoped_baseline)
        combined.add(output)
        market_cash = scoped_market.value(combined) - baseline_value
        results.append(TargetProfit(choice, output, market_cash, cost, 0.0))
    return results


def _choice_current_key(choice):
    """Compare standing commitment only; later forecast cycles are revisited."""
    if choice is None:
        return None
    name, fertilize = choice
    if name in ANIMALS:
        return name, ()
    ages = plan_ages(fertilize)
    if ages is None:  # legacy True means every verified event this cycle
        ages = tuple(sorted(CROP_FERTILIZE_DAYS[name]))
    return name, tuple(ages)


def _choose(
    market, baseline, candidates, counts, labor=None, position=(4, 4), *, current=None,
    audit=None,
):
    rows = evaluate_targets(market, baseline, candidates, labor, position)
    if audit is not None:
        audit.extend(rows)
    scored = []
    for result in rows:
        score = result.profit
        if score > 0:
            scored.append((score, result))
    if not scored:
        return None, None
    best_score, result = max(
        scored,
        key=lambda item: (item[0], -counts[item[1].choice[0]], item[1].choice),
    )
    if current is not None:
        current_key = _choice_current_key(current)
        current_rows = [
            (score, row) for score, row in scored
            if _choice_current_key(row.choice) == current_key
        ]
        if current_rows:
            current_score, current_result = max(current_rows, key=lambda item: item[0])
            if current_score >= best_score - TARGET_SWITCH_MARGIN:
                result = current_result
    return result.choice, result.output


def _legacy_choice(choice):
    """Keep analysis helpers' historic ``(name, bool)`` return contract."""
    if choice is None:
        return None
    name, fertilize = choice
    return (name, bool(fertilize)) if name in CROPS else choice


def _score(name, end_day, day, inventory, wheat_price, committed_units, unlocked_shops=()):
    market = MarketForecast(inventory, unlocked_shops, day, end_day)
    baseline = Production()
    baseline.sales[day].update(committed_units)
    candidates = [c for c in _candidates(day, end_day) if c[0][0] == name]
    results = evaluate_targets(market, baseline, candidates)
    if not results:
        return None, False
    best = max(results, key=lambda r: r.profit)
    return best.profit, bool(best.choice[1])


def best_target(end_day, day, inventory, wheat_price, committed_units,
                category_capital=None, total_capital=0, unlocked_shops=()):
    # Legacy arguments retained for callers; capital share no longer affects rank.
    market = MarketForecast(inventory, unlocked_shops, day, end_day)
    baseline = Production()
    baseline.sales[day].update(committed_units)
    return _legacy_choice(_choose(
        market, baseline, list(_candidates(day, end_day)), Counter()
    )[0])


def _log_decision(decision_log, day, position, rows, selected):
    if decision_log is None:
        return
    decision_log.append({
        "day": day,
        "position": list(position),
        "selected": None if selected is None else [selected[0], plan_json(selected[1])],
        "candidates": [
            {
                "target": [row.choice[0], plan_json(row.choice[1])],
                "market_cash": row.market_cash,
                "capital_cost": row.capital_cost,
                "labor_cost": row.labor_cost,
                "score": row.profit,
                "profitable": row.profit > 0,
                "selected": row.choice == selected,
            }
            for row in rows
        ],
    })


def plan_targets(obs, targets, active_positions, end_day, *, max_positions=None,
                 replan_positions=None, decision_log=None):
    """Reprice each new commitment against actual remaining production.

    Existing crops are forecast from their real age, held yield and watering
    state, including visible opponent crops. Unsown choices are reconsidered
    each day. No profitable candidate means no planting, with harvest-only
    cleanup handled by build_tasks.

    Crop candidates score every legal fertilizer event combination in the
    horizon. The selected target stores the current cycle plus later forecast
    cycles; only the current cycle is operationally committed, because the tile
    is repriced again when that cycle finishes.

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
                fertilizer = target[1] if target and target[0] == name else False
                # Opponent future fertilizer decisions are not observable.
                destination = baseline if player == obs["player"] else external
                if player == obs["player"] and (x, y) not in replanning_set:
                    future, _ = _rotation(name, fertilizer, day, end_day, tile)
                else:
                    future = production(
                        name,
                        cycle_plan(fertilizer, 0) if player == obs["player"] else False,
                        day,
                        end_day,
                        tile,
                    )
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
    # Assign in a stable order near the shed, updating supply after every pick.
    for position in replanning:
        x, y = position
        tile = farm["tiles"][y][x]
        allowed = candidates
        if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE"):
            allowed = [c for c in candidates if c[0][0] in ANIMALS
                       and official_game.ANIMALS[c[0][0]]["structure"] == tile["kind"]]
        current = targets.get(position)
        audit = [] if decision_log is not None else None
        choice, output = _choose(
            market, baseline, allowed, counts, position=position, current=current,
            audit=audit,
        )
        targets[position] = choice
        _log_decision(decision_log, day, position, audit or (), choice)
        if choice:
            # Include this commitment's supply in the shared window.
            baseline.add(output, position)
            counts[choice[0]] += 1
    return pending
