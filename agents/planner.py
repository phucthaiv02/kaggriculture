"""Planner: choose production targets from marginal market profit and cost.

This merged implementation preserves the stable planner behaviors exercised by
unit tests and the game agent, without the stale conflict markers left behind by
an in-progress rebase.
"""
from collections import Counter
from copy import copy
from dataclasses import dataclass
from typing import Optional

from kaggle_environments.envs.kaggriculture import kaggriculture as official_game

from agents.forecast import MarketForecast, Production, production
from agents.horizon import SEASON_END_DAY, cycle_end as _cycle_end, first_yield_age as _first_yield_age, can_start
from agents.labor import LaborForecast
from agents.products import ANIMALS, ANIMAL_COST, CROPS, SEED_COST
from agents.schedules import CROP_FERTILIZE_DAYS, CROP_LAST_AGE, cycle_turns_over_today

TARGET_SWITCH_MARGIN = 1.0
TARGET_OPTIONS = (1, 2, 3)
TARGET_OPTION = 1
TARGET_DISCOUNT = 0.97
TARGET_ROI_CAPITAL_FLOOR = 100.0


def set_target_option(option: int):
    global TARGET_OPTION
    opt = int(option)
    if opt not in TARGET_OPTIONS:
        raise ValueError(f"invalid target option: {option!r}")
    TARGET_OPTION = opt


def _crop_cycle_starts(name, day, end_day, tile=None):
    """Yield every crop start that can still produce in the planner window."""
    end_day = _cycle_end(day, end_day)
    first = _first_yield_age(name)
    if tile is None:
        if day + first > end_day:
            return []
        starts = [day]
        next_start = day + CROP_LAST_AGE[name]
    else:
        planted = tile["planted_day"]
        starts = [planted]
        next_start = max(day, planted + CROP_LAST_AGE[name])
    while next_start + first <= end_day:
        starts.append(next_start)
        next_start += CROP_LAST_AGE[name]
    return starts


def _rotation(name, fertilize, day, end_day, tile=None):
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
        per_cycle = fertilize if not isinstance(fertilize, tuple) else (fertilize[index] if index < len(fertilize) else False)
        if index == 0 and tile is not None:
            output.add(production(name, per_cycle, day, end_day, tile))
            continue
        output.add(production(name, per_cycle, start, end_day))
        cost += SEED_COST[name]
    return output, cost


def _candidates(day, end_day):
    end_day = _cycle_end(day, end_day)
    for name in sorted((*CROPS, *ANIMALS)):
        if day + _first_yield_age(name) > end_day:
            continue
        if name in ANIMALS:
            output, cost = _rotation(name, False, day, end_day)
            yield (name, False), output, cost
            continue
        for fertilize in (False, True):
            output, cost = _rotation(name, fertilize, day, end_day)
            yield (name, fertilize), output, cost


@dataclass(frozen=True)
class TargetProfit:
    choice: tuple
    output: Production
    market_cash: float
    capital_cost: float
    labor_cost: float = 0.0
    discounted_market_cash: Optional[float] = None

    @property
    def profit(self) -> float:
        return self.market_cash - self.capital_cost

    def score(self, option: Optional[int] = None) -> float:
        opt = TARGET_OPTION if option is None else int(option)
        if opt == 1:
            return self.profit
        if opt == 2:
            base = self.market_cash if self.discounted_market_cash is None else self.discounted_market_cash
            return base - self.capital_cost
        if opt == 3:
            return self.profit / max(self.capital_cost, TARGET_ROI_CAPITAL_FLOOR)
        raise ValueError(f"target option must be one of {TARGET_OPTIONS}, got {opt}")


def evaluate_targets(market, baseline, candidates, labor=None, position=(4, 4), *, target_option: Optional[int] = None):
    if market is None:
        return []

    results = []
    end = _cycle_end(market.day, market.end_day)
    scoped_market = copy(market)
    scoped_market.end_day = end

    def scoped(flow):
        result = Production()
        for field in ("sales", "inputs", "visits"):
            entries = getattr(flow, field)
            for day_key, value in entries.items():
                if market.day <= day_key <= end:
                    getattr(result, field).update({day_key: value})
        return result

    scoped_baseline = scoped(baseline)
    baseline_values = {}
    prepared = None
    baseline_cost = 0.0
    baseline_value = None
    if labor is not None:
        prepared = labor.prepare(scoped_baseline.visits)
        baseline_cost = labor.prepared_cost(prepared)
        baseline_value = scoped_market.value(scoped_baseline)

    option = TARGET_OPTION if target_option is None else int(target_option)
    for choice, output, cost in candidates:
        if market.day + _first_yield_age(choice[0]) > end:
            continue
        out = scoped(output)
        if labor is None:
            market_cash = scoped_market.marginal_value(scoped_baseline, out, baseline_values)
            labor_cost = 0.0
        else:
            market_cash = scoped_market.marginal_profit(scoped_baseline, out, 0, baseline_value)
            labor_cost = labor.marginal_cost(scoped_baseline, out, position, baseline_cost, prepared=prepared)
        result = TargetProfit(choice, out, market_cash, cost, labor_cost)
        if option == 2:
            result = TargetProfit(choice, out, market_cash, cost, labor_cost, market_cash * TARGET_DISCOUNT)
        results.append(result)
    return results


def _choice_current_key(choice):
    if choice is None:
        return None
    name, fertilize = choice
    if name in ANIMALS:
        return name, ()
    if fertilize in (False, True):
        ages = tuple(sorted(CROP_FERTILIZE_DAYS.get(name, {}))) if name in CROP_FERTILIZE_DAYS else ()
        return name, ages
    return name, tuple(fertilize)


def _candidate_log(result, selected):
    return {
        "target": list(result.choice),
        "market_cash": result.market_cash,
        "capital_cost": result.capital_cost,
        "labor_cost": result.labor_cost,
        "score": result.score(),
        "profitable": result.profit > 0,
        "selected": selected,
    }


def _choose(market, baseline, candidates, counts, labor=None, position=(4, 4), *, current=None, audit=None, decision_log=None, decision_step=None, target_option: Optional[int] = None):
    results = evaluate_targets(market, baseline, candidates, labor, position, target_option=target_option)
    if audit is not None:
        audit.extend(results)

    scored = []
    for result in results:
        score = result.score(target_option)
        if score > 0:
            scored.append((score, result))

    if not scored:
        if decision_log is not None:
            decision_log.append({
                "day": getattr(market, "day", None),
                "hour": getattr(market, "hour", None),
                "step": decision_step,
                "position": list(position),
                "current": list(current) if current is not None else None,
                "selected": None,
                "reason": "no_profitable_candidate",
                "switch_margin": TARGET_SWITCH_MARGIN,
                "candidates": [_candidate_log(row, False) for row in results],
            })
        return None, None

    best_score, result = max(scored, key=lambda item: (item[0], -counts[item[1].choice[0]], item[1].choice))
    reason = "highest_score"
    if current is not None:
        current_key = _choice_current_key(current)
        current_rows = [(score, row) for score, row in scored if _choice_current_key(row.choice) == current_key]
        if current_rows:
            current_score, current_result = max(current_rows, key=lambda item: item[0])
            if current_score >= best_score - TARGET_SWITCH_MARGIN:
                result = current_result
                reason = "retained_current_within_switch_margin"

    if decision_log is not None:
        decision_log.append({
            "day": getattr(market, "day", None),
            "hour": getattr(market, "hour", None),
            "step": decision_step,
            "position": list(position),
            "current": list(current) if current is not None else None,
            "selected": list(result.choice),
            "reason": reason,
            "switch_margin": TARGET_SWITCH_MARGIN,
            "candidates": [_candidate_log(row, row.choice == result.choice) for row in results],
        })

    return result.choice, result.output


def _legacy_choice(choice):
    if choice is None:
        return None
    name, fertilize = choice
    if name in CROPS:
        return (name, bool(fertilize))
    return choice


def _score(name, end_day, day, inventory, wheat_price, committed_units, unlocked_shops=()):
    market = MarketForecast(inventory, unlocked_shops, day, end_day)
    baseline = Production()
    baseline.sales[day].update(committed_units)
    candidates = [candidate for candidate in _candidates(day, end_day) if candidate[0][0] == name]
    results = evaluate_targets(market, baseline, candidates)
    if not results:
        return None, False
    best = max(results, key=lambda item: item.profit)
    return best.profit, bool(best.choice[1])


def best_target(end_day, day, inventory, wheat_price, committed_units, category_capital=None, total_capital=0, unlocked_shops=(), target_option=None):
    market = MarketForecast(inventory, unlocked_shops, day, end_day)
    baseline = Production()
    baseline.sales[day].update(committed_units)
    choice, _ = _choose(market, baseline, list(_candidates(day, end_day)), Counter(), target_option=target_option)
    return _legacy_choice(choice)


def plan_targets(obs, targets, active_positions, end_day, *, max_positions=None, replan_positions=None, decision_log=None):
    day = obs["day"]
    end_day = _cycle_end(day, end_day)
    farm = obs["farms"][obs["player"]]
    committed_positions = obs.get("_committed_targets")
    baseline = Production()
    counts = Counter()
    replanning = []
    turnover = []
    deferred = []
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
            if not cycle_turns_over_today(tile["crop"], day - tile["planted_day"], tile):
                if replan_positions is not None:
                    deferred.append(position)
                continue
            if current and current[0] != tile["crop"] and can_start(current[0], day, end_day):
                continue
            turnover.append(position)
            continue
        replanning.append(position)

    shed_access = ((4, 4), (5, 4), (4, 5), (5, 5))
    def distance(p):
        return min(abs(p[0] - s[0]) + abs(p[1] - s[1]) for s in shed_access)

    turnover.sort(key=lambda p: (distance(p), p[1], p[0]))
    replanning.sort(key=lambda p: (distance(p), p[1], p[0]))
    pending = deferred + (replanning[max_positions:] if max_positions is not None else [])
    if max_positions is not None:
        replanning = replanning[:max_positions]
    replanning = turnover + replanning
    if not replanning:
        return pending
    replanning_set = set(replanning)

    for player, other_farm in enumerate(obs["farms"]):
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
                destination = baseline if player == obs["player"] else external
                if player == obs["player"] and (x, y) not in replanning_set:
                    future, _ = _rotation(name, fertilizer, day, end_day, tile)
                else:
                    future = production(name, False, day, end_day, tile)
                destination.add(future, (x, y))
                if player == obs["player"] and (x, y) not in replanning_set:
                    counts[name] += 1

    baseline.sales[day].update({p: n for p, n in obs["private"]["shed"].items() if p in official_game.PRODUCTS})
    for carried in obs["private"].get("inventories", []):
        baseline.sales[day].update({p: n for p, n in carried.items() if p in official_game.PRODUCTS})

    for position, target in list(targets.items()):
        if not target or position in replanning_set:
            continue
        x, y = position
        tile = farm["tiles"][y][x]
        actual = tile.get("animal") or tile.get("crop") if isinstance(tile, dict) else None
        if actual != target[0] and not actual and (committed_positions is None or position in committed_positions):
            name, fertilize = target
            baseline.add(_rotation(name, fertilize, day, end_day)[0], position)
            counts[target[0]] += 1

    market = MarketForecast(obs["market"]["inventory"], obs["town"]["unlocked_shops"], day, end_day, obs.get("hour", 0), obs["market"].get("params"), external)
    candidates = list(_candidates(day, end_day))

    for position in replanning:
        x, y = position
        tile = farm["tiles"][y][x]
        allowed = candidates
        if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE"):
            allowed = [candidate for candidate in candidates if candidate[0][0] in ANIMALS and official_game.ANIMALS[candidate[0][0]]["structure"] == tile["kind"]]
        current = targets.get(position)
        choice, output = _choose(market, baseline, allowed, counts, position=position, current=current, decision_log=decision_log, decision_step=obs.get("step"))
        targets[position] = choice
        if choice:
            baseline.add(output, position)
            counts[choice[0]] += 1

    return pending
