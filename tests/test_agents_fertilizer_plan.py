"""Regression tests for production per-event fertilizer planning."""
import pytest
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from agents.farm_tasks import build_tasks, purchase_orders
from agents.fertilizer import FertilizerPlan
from agents.forecast import MarketForecast, Production
from agents.planner import _candidates, evaluate_targets
from agents.schedules import is_maintenance_day, should_fertilize_today


def _combined(*flows):
    result = Production()
    for flow in flows:
        result.add(flow)
    return result


def _wheat_obs(*, fertilizer=0, fertilized_until_day=-1):
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "yield_units": 1,
        "watered_today": False,
        "consecutive_unwatered": 0,
        "fertilized_until_day": fertilized_until_day,
    }
    inventory = {product: game.MARKET_I0 for product in game.PRODUCTS}
    farm = {
        "money": 3000,
        "tiles": board,
        "farmer": [4, 4],
        "hands": [],
        "hires_today": 0,
        "unlocked_quadrants": ["NW"],
    }
    return {
        "day": 2,
        "hour": 0,
        "player": 0,
        "farms": [farm, farm],
        "market": {
            "inventory": inventory,
            "prices": {product: game.market_price(product, stock)
                       for product, stock in inventory.items()},
        },
        "town": {"unlocked_shops": []},
        "private": {
            "shed": {"FERTILIZER": fertilizer},
            "seeds": {},
            "inventories": [{}],
        },
        "_planning_end_day": 29,
    }


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


def test_product_local_marginal_value_matches_full_portfolio_subtraction():
    day, end_day = 8, 24
    inventory = {product: game.MARKET_I0 for product in game.PRODUCTS}
    inventory.update({"WHEAT": 10120, "CARROT": 9970, "FERTILIZER": 10270})

    baseline = Production()
    baseline.sales[8].update({"CARROT": 3, "FERTILIZER": 2})
    baseline.sales[12].update({"WHEAT": 5, "MILK": 2})
    baseline.inputs[14].update({"WHEAT": 1})
    baseline.inputs[18].update({"FERTILIZER": 1})

    external = Production()
    external.sales[10].update({"WHEAT": 4, "CARROT": 2})
    external.sales[18].update({"FERTILIZER": 3})
    external.inputs[22].update({"WHEAT": 1})

    market = MarketForecast(
        inventory, ("BAKERY",), day, end_day, external=external
    )
    baseline_value = market.value(baseline)
    cache = {}
    # Sample every crop/animal family plus several fertilizer-plan variants.
    # This also guards the allocation-free marginal path used by the planner.
    candidates = list(_candidates(day, end_day))
    sampled = candidates[::max(1, len(candidates) // 24)]
    for _choice, candidate, _cost in sampled:
        combined = _combined(baseline, candidate)
        expected = market.value(combined) - baseline_value
        actual = market.marginal_value(baseline, candidate, cache)
        assert actual == pytest.approx(expected)


def test_due_event_is_scheduled_even_when_shed_has_no_fertilizer():
    obs = _wheat_obs(fertilizer=0)
    targets = {(0, 0): ("WHEAT", FertilizerPlan(((2,),)))}
    tasks = build_tasks(obs, targets)
    task = next(task for task in tasks if task.position == (0, 0))
    assert ["FERTILIZE"] in task.actions
    assert task.needs["FERTILIZER"] == 1
    assert task.mandatory is True


def test_purchase_orders_fund_due_fertilizer_event():
    obs = _wheat_obs(fertilizer=0)
    targets = {(0, 0): ("WHEAT", FertilizerPlan(((2,),)))}
    orders = purchase_orders(obs, targets, [(0, 0)])
    assert ["BUY_PRODUCT", "FERTILIZER", 1] in orders


def test_opening_boolean_false_never_buys_fertilizer():
    obs = _wheat_obs(fertilizer=0)
    targets = {(0, 0): ("WHEAT", False)}
    tasks = build_tasks(obs, targets)
    assert not any(["FERTILIZE"] in task.actions for task in tasks)
    orders = purchase_orders(obs, targets, [(0, 0)])
    assert not any(order[:2] == ["BUY_PRODUCT", "FERTILIZER"] for order in orders)


def test_active_fertilizer_window_does_not_schedule_or_buy_duplicate():
    obs = _wheat_obs(fertilizer=0, fertilized_until_day=4)
    targets = {(0, 0): ("WHEAT", FertilizerPlan(((2,),)))}
    tasks = build_tasks(obs, targets)
    assert not any(["FERTILIZE"] in task.actions for task in tasks)
    orders = purchase_orders(obs, targets, [(0, 0)])
    assert not any(order[:2] == ["BUY_PRODUCT", "FERTILIZER"] for order in orders)
