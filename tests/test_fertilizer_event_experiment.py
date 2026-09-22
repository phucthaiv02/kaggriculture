"""Experiment: score WHEAT fertilizer as dated events instead of a boolean.

This is deliberately test-only.  It does not change planner/scheduler behavior.
The goal is to verify that the existing product-local MarketForecast can price
FERTILIZER opportunity cost and that a partial late-only fertilizer schedule
can beat both current boolean choices.
"""

from itertools import combinations
import json
from pathlib import Path

from kaggle_environments.envs.kaggriculture import kaggriculture as game

from agents.forecast import MarketForecast, Production, production


DAY = 8
END_DAY = 24
WHEAT_CYCLE = 4
FERTILIZE_AGE = 2
EVENTS = tuple(range(DAY + FERTILIZE_AGE, END_DAY, WHEAT_CYCLE))


def _plans():
    for size in range(len(EVENTS) + 1):
        for picked in combinations(EVENTS, size):
            yield picked


def _wheat_rotation(fertilize_days):
    """Exact current WHEAT rotations, choosing fertilizer independently/cycle.

    WHEAT is the clean experiment because its verified WATER schedule is the
    same with and without fertilizer and it has exactly one fertilizer event
    per cycle. Therefore mixing fertilized/unfertilized cycles does not need
    any new crop-maintenance assumptions.
    """
    fertilize_days = set(fertilize_days)
    result = Production()
    start = DAY
    while start + WHEAT_CYCLE <= END_DAY:
        result.add(production(
            "WHEAT",
            start + FERTILIZE_AGE in fertilize_days,
            start,
            END_DAY,
        ))
        start += WHEAT_CYCLE
    return result


def _combined(*flows):
    result = Production()
    for flow in flows:
        result.add(flow)
    return result


def _score_case(initial_fertilizer_stock, fertilizer_sales_per_day):
    inventory = {product: game.MARKET_I0 for product in game.PRODUCTS}
    inventory["FERTILIZER"] = initial_fertilizer_stock

    # Model a growing animal portfolio producing fertilizer from the day after
    # placement. There is no shop/town demand for FERTILIZER, so these sales
    # monotonically soften its future spot price unless the candidate consumes it.
    baseline = Production()
    for when in range(DAY + 1, END_DAY + 1):
        baseline.sales[when]["FERTILIZER"] = fertilizer_sales_per_day

    market = MarketForecast(inventory, (), DAY, END_DAY)
    baseline_value = market.value(baseline)

    rows = []
    for plan in _plans():
        candidate = _wheat_rotation(plan)
        market_cash = market.value(_combined(baseline, candidate)) - baseline_value
        rows.append({
            "fertilize_days": list(plan),
            "market_cash": market_cash,
            "fertilizer_uses": sum(candidate.inputs[d]["FERTILIZER"] for d in candidate.inputs),
            "wheat_sales": sum(candidate.sales[d]["WHEAT"] for d in candidate.sales),
        })

    # Spot sale quote before that day's baseline fertilizer settlement.
    # This captures the user's observation that early fertilizer can sell for
    # more while accumulating animal supply depresses later prices.
    spot = {}
    stocks = dict(inventory)
    for when in range(DAY, END_DAY + 1):
        if when in EVENTS:
            spot[str(when)] = market.price("FERTILIZER", stocks["FERTILIZER"])
        market._settle_day(stocks, baseline, when)

    # Full-horizon opportunity cost of consuming one fertilizer on each event
    # day against the same baseline supply path. This is intentionally reported
    # separately from the spot quote: foregoing an early sale also leaves market
    # stock lower, which can raise the value of later fertilizer sales.
    opportunity = {}
    for when in EVENTS:
        consume = Production()
        consume.inputs[when]["FERTILIZER"] = 1
        opportunity[str(when)] = baseline_value - market.value(_combined(baseline, consume))

    rows.sort(
        key=lambda row: (
            row["market_cash"],
            -len(row["fertilize_days"]),
            row["fertilize_days"],
        ),
        reverse=True,
    )
    best = rows[0]
    none = next(row for row in rows if not row["fertilize_days"])
    all_events = next(row for row in rows if tuple(row["fertilize_days"]) == EVENTS)
    boolean_best = max((none, all_events), key=lambda row: row["market_cash"])
    return {
        "initial_fertilizer_stock": initial_fertilizer_stock,
        "fertilizer_sales_per_day": fertilizer_sales_per_day,
        "events": list(EVENTS),
        "fertilizer_spot_sale_price": spot,
        "fertilizer_horizon_opportunity_cost": opportunity,
        "best_event_plan": best,
        "best_boolean_plan": boolean_best,
        "gain_over_boolean": best["market_cash"] - boolean_best["market_cash"],
        "plans": rows,
    }


def test_late_only_fertilizer_plan_can_beat_boolean_commitment():
    # Search only market state/supply, not crop rules. We want an existence
    # proof of the economic case discussed in planner review: fertilizer spot
    # price falls over time, and independently-scored WHEAT cycles can prefer
    # using only later fertilizer events instead of the current none/all choice.
    suffixes = {tuple(EVENTS[index:]) for index in range(1, len(EVENTS))}
    found = None
    searched = []
    for supply in (1, 2, 3, 4, 5, 6):
        supply_best = None
        for stock in range(game.MARKET_I0 - 500, game.MARKET_I0 + 501, 5):
            case = _score_case(stock, supply)
            plan = tuple(case["best_event_plan"]["fertilize_days"])
            if plan not in suffixes or case["gain_over_boolean"] <= 0:
                continue
            if supply_best is None or case["gain_over_boolean"] > supply_best["gain_over_boolean"]:
                supply_best = case
            if found is None or case["gain_over_boolean"] > found["gain_over_boolean"]:
                found = case
        searched.append({
            "supply": supply,
            "found": supply_best is not None,
            "best_gain": None if supply_best is None else supply_best["gain_over_boolean"],
        })

    report = {
        "purpose": "dated fertilizer opportunity-cost experiment",
        "day": DAY,
        "end_day": END_DAY,
        "searched": searched,
        "case": found,
    }
    out = Path("ci-artifacts")
    out.mkdir(exist_ok=True)
    (out / "fertilizer-event-experiment.json").write_text(json.dumps(report, indent=2))

    assert found is not None, "no strict late-only optimum found in scanned market states"
    selected = tuple(found["best_event_plan"]["fertilize_days"])
    assert selected in suffixes
    assert found["gain_over_boolean"] > 0
    spot = found["fertilizer_spot_sale_price"]
    assert spot[str(EVENTS[-1])] < spot[str(EVENTS[0])]
