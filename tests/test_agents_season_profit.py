from collections import Counter
import pytest

from agents.forecast import MarketForecast, Production
from agents.labor import LaborForecast
from agents.planner import SEASON_END_DAY, _candidates, _choose, _rotation, _score, evaluate_targets
from kaggle_environments.envs.kaggriculture import kaggriculture as game


def constant_market(end_day):
    market = MarketForecast({p: game.MARKET_I0 for p in game.PRODUCTS}, (), 0, end_day)
    market.price = lambda product, stock: {'WHEAT': 25, 'CARROT': 30}.get(product, 100)
    return market


def alternatives(end_day):
    return [((crop, False), *_rotation(crop, False, 0, end_day)) for crop in ('WHEAT', 'CARROT')]


def test_window_profit_includes_replants():
    # Three WHEAT harvests net 270; four CARROT harvests net 280.
    choice, _ = _choose(constant_market(12), Production(), alternatives(12), Counter())
    assert choice == ('CARROT', False)


def test_cycle_profit_is_not_profit_per_day():
    # With only 5 days available both crops finish once. CARROT's higher
    # profit/day must not override WHEAT's higher actual season cash.
    choice, _ = _choose(constant_market(5), Production(), alternatives(5), Counter())
    assert choice == ('WHEAT', False)


def test_replanting_from_actual_start_and_last_playable_day():
    flow, cost = _rotation('WHEAT', False, 21, SEASON_END_DAY)
    assert cost == 20
    assert {d: units['WHEAT'] for d, units in flow.sales.items() if units['WHEAT']} == {25: 4, 29: 4}
    last, _ = _rotation('WHEAT', False, 25, SEASON_END_DAY)
    assert last.sales[29]['WHEAT'] == 4
    assert not any(day > SEASON_END_DAY for day in flow.sales)
    too_late, cost = _rotation('WHEAT', False, 28, SEASON_END_DAY)
    assert cost == 0
    assert not too_late.sales


def test_animal_is_bought_once_and_only_projected_after_placement():
    flow, cost = _rotation('GOOSE', False, 7, SEASON_END_DAY)
    assert cost == game.ANIMALS['GOOSE']['cost']
    assert min(flow.inputs) == 7
    assert max(flow.inputs) == 23
    assert min(day for day, units in flow.sales.items() if units['EGG']) == 11


def test_score_and_selector_share_cycle_labor_cost(monkeypatch):
    inventory = {p: game.MARKET_I0 for p in game.PRODUCTS}
    before, _ = _score('GOOSE', SEASON_END_DAY, 7, inventory, 25, Counter())
    monkeypatch.setattr(LaborForecast, 'marginal_cost', lambda *args, **kwargs: 123)
    after, _ = _score('GOOSE', SEASON_END_DAY, 7, inventory, 25, Counter())
    assert after == before - 123


@pytest.mark.parametrize('name,last', [('GOOSE', 16), ('COW', 16), ('SHEEP', 15)])
def test_animal_cycle_includes_only_harvests_through_age_sixteen(name, last):
    flow, cost = _rotation(name, False, 3, SEASON_END_DAY)
    product = game.ANIMALS[name]['product']
    harvests = [d for d, units in flow.sales.items() if units[product]]
    assert min(harvests) == 3 + game.ANIMALS[name]['first_yield_day']
    assert max(harvests) == 3 + last
    assert max(flow.visits) == 3 + 16
    assert cost == game.ANIMALS[name]['cost']
    truncated, _ = _rotation(name, False, 3, 3 + last - 1)
    assert max(truncated.visits) <= 3 + last - 1


@pytest.mark.parametrize('name,last', [('TOMATO', 11), ('STRAWBERRY', 16), ('MELON', 10)])
def test_crop_cycle_includes_final_yield_without_replanting(name, last):
    for fertilize in (False, True):
        flow, cost = _rotation(name, fertilize, 2, SEASON_END_DAY)
        assert max(d for d, units in flow.sales.items() if units[name]) == 2 + last
        assert cost == game.CROPS[name]['seed']
        if name == 'MELON':
            assert flow.sales[2 + last][name] == game.CROPS[name]['max_yield']


def test_sales_and_labor_after_candidate_cycle_do_not_change_its_score():
    inventory = {p: game.MARKET_I0 for p in game.PRODUCTS}
    market = MarketForecast(inventory, (), 0, SEASON_END_DAY)
    baseline = Production()
    candidate = [(('WHEAT', False), *_rotation('WHEAT', False, 0, SEASON_END_DAY))]
    before = evaluate_targets(market, baseline, candidate)[0]
    baseline.sales[17]['WHEAT'] = 100
    baseline.visits[17] = [((0, 0), 100, (), True)]
    after = evaluate_targets(market, baseline, candidate)[0]
    assert after.profit == before.profit


def test_existing_crop_forecast_charges_only_future_replants():
    tile = game._new_plant('WHEAT', 0, 24)
    flow, cost = _rotation('WHEAT', False, 0, SEASON_END_DAY, tile)
    assert cost == 30
    assert [d for d, units in flow.sales.items() if units['WHEAT']] == [4, 8, 12, 16]


def test_route_reordering_cannot_credit_a_target_with_negative_labor_cost():
    baseline = Production()
    baseline.visits[0] = [
        ((6, 2), 5, (), True), ((2, 1), 6, (), True),
        ((3, 3), 6, (), True), ((4, 2), 5, (), True),
        ((3, 2), 6, (), True), ((8, 3), 2, (), True),
        ((8, 0), 3, (), True), ((0, 4), 6, (), True), ((5, 3), 6, (), True),
    ]
    candidate = Production()
    candidate.visits[0] = [(None, 2, (), True)]
    labor = LaborForecast()
    combined = Production()
    combined.add(baseline)
    combined.add(candidate, (6, 3))
    assert labor.cost(combined.visits) < labor.cost(baseline.visits)
    assert labor.marginal_cost(baseline, candidate, (6, 3)) == 0
