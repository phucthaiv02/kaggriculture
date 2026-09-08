"""Check every cutoff against discrete, engine-verified harvest schedules."""
from collections import Counter

import pytest

from agents.forecast import MarketForecast, Production
from agents.planner import _candidates, _rotation, _score, evaluate_targets
from kaggle_environments.envs.kaggriculture import kaggriculture as game


HARVESTS = {
    'WHEAT': {4: 4, 8: 4, 12: 4, 16: 4},
    'CARROT': {3: 3, 6: 3, 9: 3, 12: 3, 15: 3},
    'MELON': {10: 6},
    'TOMATO': {8: 1, 9: 1, 10: 1, 11: 1},
    'STRAWBERRY': {10: 1, 12: 1, 14: 1, 16: 1},
    'GOOSE': {4: 4, **{d: 2 for d in range(5, 17)}},
    'COW': {8: 6, 10: 3, 12: 3, 14: 3, 16: 3},
    'SHEEP': {6: 6, 9: 4, 12: 4, 15: 4},
}


@pytest.mark.parametrize('start', [0, 7])
@pytest.mark.parametrize('remaining', range(18))
@pytest.mark.parametrize('name', HARVESTS)
def test_each_harvest_cutoff_and_first_yield_filter(name, remaining, start):
    rules = game.ANIMALS.get(name, game.CROPS.get(name))
    product = rules.get('product', name)
    horizon = min(16, remaining)
    candidates = {choice: (flow, cost) for choice, flow, cost
                  in _candidates(start, start + remaining)}
    for fertilize in ((False, True) if name in game.CROPS else (False,)):
        choice = (name, fertilize)
        if rules['first_yield_day'] > horizon:
            assert choice not in candidates
            continue
        assert choice in candidates
        flow, cost = candidates[choice]
        expected = {start + age: units for age, units in HARVESTS[name].items()
                    if age <= horizon}
        if fertilize:
            if name in ('TOMATO', 'STRAWBERRY'):
                expected = {d: units * 2 for d, units in expected.items()}
            elif name in ('WHEAT', 'CARROT'):
                expected = {d: rules['max_yield'] for d in expected}
        assert {d: units[product] for d, units in flow.sales.items()
                if units[product]} == expected
        plantings = max(1, len(expected)) if name in ('WHEAT', 'CARROT') else 1
        assert cost == plantings * rules.get('seed', rules.get('cost'))
        for field in (flow.sales, flow.inputs, flow.visits):
            assert all(start <= d <= start + horizon for d in field)


@pytest.mark.parametrize('name', HARVESTS)
def test_more_than_sixteen_remaining_days_cannot_increase_score(name):
    inventory = {p: game.MARKET_I0 for p in game.PRODUCTS}
    assert _score(name, 23, 7, inventory, 25, Counter()) == _score(
        name, 29, 7, inventory, 25, Counter())


def test_evaluation_caps_candidate_labor_and_sales_as_well_as_baseline():
    market = MarketForecast({p: game.MARKET_I0 for p in game.PRODUCTS}, (), 0, 29)
    output, cost = _rotation('GOOSE', False, 0, 29)
    baseline = Production()
    candidate = [(('GOOSE', False), output, cost)]
    before = evaluate_targets(market, baseline, candidate)[0]
    output.sales[17]['EGG'] = 1000
    output.inputs[17]['WHEAT'] = 1000
    output.visits[17] = [(None, 1000, (), True)]
    baseline.sales[17]['EGG'] = 1000
    baseline.visits[17] = [((0, 0), 1000, (), True)]
    after = evaluate_targets(market, baseline, candidate)[0]
    assert (after.market_cash, after.labor_cost) == (before.market_cash, before.labor_cost)
    assert 17 not in after.output.visits


def test_short_crop_still_affects_market_inside_shared_window():
    market = MarketForecast({p: game.MARKET_I0 for p in game.PRODUCTS}, (), 0, 29)
    output, cost = _rotation('WHEAT', False, 0, 29)
    candidate = [(('WHEAT', False), output, cost)]
    baseline = Production()
    before = evaluate_targets(market, baseline, candidate)[0]
    baseline.sales[10]['WHEAT'] = 100
    after = evaluate_targets(market, baseline, candidate)[0]
    assert after.market_cash < before.market_cash


def test_evaluation_rejects_unreachable_first_yield_even_with_supplied_output():
    market = MarketForecast({}, (), 0, 5)
    fake = Production()
    fake.sales[0]['MILK'] = 100
    assert evaluate_targets(market, Production(), [(('COW', False), fake, 0)]) == []


@pytest.mark.parametrize('end,expected,cost', [
    (7, {4: 4}, 10),
    (8, {4: 4, 8: 4}, 20),
    (9, {4: 4, 8: 4}, 20),
    (16, {4: 4, 8: 4, 12: 4, 16: 4}, 40),
])
def test_replant_requires_time_for_scheduled_harvest(end, expected, cost):
    flow, seed_cost = _rotation('WHEAT', False, 0, end)
    assert {d: u['WHEAT'] for d, u in flow.sales.items() if u['WHEAT']} == expected
    assert seed_cost == cost


@pytest.mark.parametrize('end,expected,cost', [
    (23, {16: 1}, 0),
    (25, {16: 1}, 0),
    (26, {16: 1, 26: 1}, 100),
    (27, {16: 1, 26: 1}, 100),
    (29, {16: 1, 26: 1, 28: 1}, 100),
])
def test_existing_ongoing_crop_replant_counts_only_reachable_ticks(end, expected, cost):
    tile = game._new_plant('STRAWBERRY', 0, 24)
    # Age 15: previous yields were harvested, with one final tick at age 16.
    tile['yield_units'] = 0
    tile['consecutive_unwatered'] = 0  # Watered on age 14.
    flow, seed_cost = _rotation('STRAWBERRY', False, 15, end, tile)
    assert {d: u['STRAWBERRY'] for d, u in flow.sales.items() if u['STRAWBERRY']} == expected
    assert seed_cost == cost
