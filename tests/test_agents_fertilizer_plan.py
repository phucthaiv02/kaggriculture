"""Regression tests for production per-event fertilizer planning."""
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from agents.fertilizer import FertilizerPlan
from agents.forecast import MarketForecast, Production
from agents.planner import _candidates, evaluate_targets
from agents.schedules import is_maintenance_day, should_fertilize_today


def _combined(*flows):
    result = Production()
    for flow in flows:
        result.add(flow)
    return result


def test_scheduler_helpers_execute_only_selected_current_cycle_events():
    plan = FertilizerPlan(((10,), (7, 10)))
    assert not should_fertilize_today("TOMATO", 7, plan)
    assert should_fertilize_today("TOMATO", 10, plan)
    # Any fertilizer event in this cycle uses the verified fertilized water
    # cadence; later-cycle events do not leak into today's crop.
    assert is_maintenance_day("TOMATO", 5, plan)

    future_only = FertilizerPlan(((), (7, 10)))
    assert not should_fertilize_today("TOMATO", 10, future_only)
    assert not is_maintenance_day("TOMATO", 5, future_only)


def test_wheat_candidates_include_late_only_multi_cycle_plan():
    rows = [row for row in _candidates(8, 24) if row[0][0] == "WHEAT"]
    late = next(
        row for row in rows
        if isinstance(row[0][1], FertilizerPlan)
        and row[0][1].cycles == ((), (), (2,), (2,))
    )
    _choice, flow, _cost = late
    fertilizer_days = sorted(
        day for day, inputs in flow.inputs.items() if inputs.get("FERTILIZER", 0)
    )
    assert fertilizer_days == [18, 22]


def test_market_score_can_select_late_only_wheat_plan():
    day, end_day = 8, 24
    inventory = {product: game.MARKET_I0 for product in game.PRODUCTS}
    inventory["FERTILIZER"] = 10270

    baseline = Production()
    for when in range(day + 1, end_day + 1):
        baseline.sales[when]["FERTILIZER"] = 1

    market = MarketForecast(inventory, (), day, end_day)
    candidates = [row for row in _candidates(day, end_day) if row[0][0] == "WHEAT"]
    scored = evaluate_targets(market, baseline, candidates)
    best = max(scored, key=lambda row: row.profit)

    assert best.choice[1].cycles == ((), (), (2,), (2,))
    none = next(row for row in scored if not any(row.choice[1].cycles))
    all_events = next(
        row for row in scored
        if row.choice[1].cycles == ((2,), (2,), (2,), (2,))
    )
    assert best.profit > max(none.profit, all_events.profit)
