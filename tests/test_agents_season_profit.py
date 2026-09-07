from collections import Counter

from agents.forecast import MarketForecast, Production
from agents.labor import LaborForecast
from agents.planner import SEASON_END_DAY, _choose, _rotation, _score
from kaggle_environments.envs.kaggriculture import kaggriculture as game


def constant_market(end_day):
    market = MarketForecast({p: game.MARKET_I0 for p in game.PRODUCTS}, (), 0, end_day)
    market.price = lambda product, stock: {'WHEAT': 25, 'CARROT': 30}.get(product, 100)
    return market


def alternatives(end_day):
    return [((crop, False), *_rotation(crop, False, 0, end_day)) for crop in ('WHEAT', 'CARROT')]


def test_repeat_crop_beats_higher_single_harvest_profit():
    # One WHEAT harvest nets 90, one CARROT nets 70. Over 12 days,
    # WHEAT makes 3 harvests (270), CARROT makes 4 (280).
    choice, _ = _choose(constant_market(12), Production(), alternatives(12), Counter())
    assert choice == ('CARROT', False)


def test_remaining_season_profit_is_not_profit_per_day():
    # With only 5 days available both crops finish once. CARROT's higher
    # profit/day must not override WHEAT's higher actual season cash.
    choice, _ = _choose(constant_market(5), Production(), alternatives(5), Counter())
    assert choice == ('WHEAT', False)


def test_replant_from_actual_start_and_include_last_playable_day():
    flow, cost = _rotation('WHEAT', False, 21, SEASON_END_DAY)
    assert cost == 20
    assert {d: units['WHEAT'] for d, units in flow.sales.items() if units['WHEAT']} == {25: 4, 29: 4}
    assert not any(day > SEASON_END_DAY for day in flow.sales)
    too_late, cost = _rotation('WHEAT', False, 26, SEASON_END_DAY)
    assert cost == 0
    assert not too_late.sales


def test_animal_is_bought_once_and_only_projected_after_placement():
    flow, cost = _rotation('GOOSE', False, 7, SEASON_END_DAY)
    assert cost == game.ANIMALS['GOOSE']['cost']
    assert min(flow.inputs) == 7
    assert max(flow.inputs) <= SEASON_END_DAY
    assert min(day for day, units in flow.sales.items() if units['EGG']) == 11


def test_score_and_selector_share_season_labor_cost(monkeypatch):
    inventory = {p: game.MARKET_I0 for p in game.PRODUCTS}
    before, _ = _score('GOOSE', SEASON_END_DAY, 7, inventory, 25, Counter())
    monkeypatch.setattr(LaborForecast, 'marginal_cost', lambda *args, **kwargs: 123)
    after, _ = _score('GOOSE', SEASON_END_DAY, 7, inventory, 25, Counter())
    assert after == before - 123


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
