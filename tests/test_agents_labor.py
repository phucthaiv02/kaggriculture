from collections import Counter
from agents.forecast import Production, production
from agents.labor import LaborForecast
from agents.planner import _rotation, _choose
from agents.expansion_agent import _protect_animal_structures


def test_crop_window_accounts_for_every_seed_and_harvest():
    flow, cost = _rotation('WHEAT', False, 0, 12)
    assert cost == 30
    assert {day: units['WHEAT'] for day, units in flow.sales.items() if units['WHEAT']} == {4: 4, 8: 4, 12: 4}


def test_existing_rotation_does_not_charge_sunk_seed():
    from kaggle_environments.envs.kaggriculture import kaggriculture as game
    tile = game._new_plant('WHEAT', 0, 24)
    tile['yield_units'] = 3
    flow, cost = _rotation('WHEAT', False, 3, 8, tile)
    assert cost == 10
    assert [d for d, units in flow.sales.items() if units['WHEAT']] == [4, 8]


def test_daily_hires_are_not_free_or_paid_only_once():
    labor = LaborForecast(((4, 4),))
    busy = [((4, 4), 16, (), False), ((4, 3), 16, (), False)]
    one = labor.cost({0: busy})
    assert one > 0
    assert labor.cost({0: busy, 1: busy}) == 2 * one


def test_target_ranking_does_not_use_labor_cost():
    class Revenue:
        day, end_day = 0, 16
        def value(self, flows, discount=1.0):
            return sum(
                (discount ** day) * sum(units.values())
                for day, units in flows.sales.items()
            )

    busy, light = Production(), Production()
    busy.sales[8]['MILK'] = 100
    light.sales[10]['MELON'] = 90
    # Busy has deliberately awful worker demand, but target selection now
    # compares economics only. Scheduler admission owns this labor decision.
    for day in range(16):
        busy.visits[day] = [(None, 20, ('WHEAT',), True)]
    choice, _ = _choose(
        Revenue(), Production(),
        [(('COW', False), busy, 0), (('MELON', False), light, 0)],
        Counter(),
    )
    assert choice[0] == 'COW'


def test_option4_uses_labor_to_break_close_economic_choices():
    class Revenue:
        day, end_day = 0, 16
        def value(self, flows, discount=1.0):
            return sum(
                (discount ** day) * sum(units.values())
                for day, units in flows.sales.items()
            )

    busy, light = Production(), Production()
    busy.sales[0]['MILK'] = 100
    light.sales[0]['MELON'] = 95
    for day in range(16):
        busy.visits[day] = [(None, 20, ('WHEAT',), True)]
    choice, _ = _choose(
        Revenue(), Production(),
        [(('COW', False), busy, 0), (('MELON', False), light, 0)],
        Counter(), labor=LaborForecast(((4, 4),)), target_option=4,
    )
    assert choice[0] == 'MELON'


def test_option4_keeps_clear_economic_winner_despite_labor():
    class Revenue:
        day, end_day = 0, 16
        def value(self, flows, discount=1.0):
            return sum(
                (discount ** day) * sum(units.values())
                for day, units in flows.sales.items()
            )

    busy, light = Production(), Production()
    busy.sales[0]['MILK'] = 500
    light.sales[0]['MELON'] = 100
    for day in range(16):
        busy.visits[day] = [(None, 20, ('WHEAT',), True)]
    choice, _ = _choose(
        Revenue(), Production(),
        [(('COW', False), busy, 0), (('MELON', False), light, 0)],
        Counter(), labor=LaborForecast(((4, 4),)), target_option=4,
    )
    assert choice[0] == 'COW'


def test_never_dig_live_animals_or_empty_animal_structures():
    for tile in ({'kind':'COOP'}, {'kind':'PASTURE'}, {'kind':'COOP','animal':'GOOSE'}):
        farm = {'tiles': [[tile]], 'farmer':[0,0], 'hands':[]}
        assert _protect_animal_structures(farm, [['DIG']]) == [['PASS']]
        assert _protect_animal_structures(farm, [['PLANT','WHEAT']]) == [['PASS']]


def test_dig_guard_accounts_for_an_earlier_workers_construction():
    farm = {'tiles': [[None]], 'farmer':[0,0], 'hands':[[0,0]]}
    assert _protect_animal_structures(farm, [['BUILD_COOP'],['DIG']]) == [['BUILD_COOP'],['PASS']]


def test_digging_exhausted_crops_and_weeds_remains_available():
    for tile in ({'kind':'WEED'}, {'kind':'PLANT','crop':'TOMATO'}):
        farm = {'tiles': [[tile]], 'farmer':[0,0], 'hands':[]}
        assert _protect_animal_structures(farm, [['DIG']]) == [['DIG']]


def test_prepared_marginal_cost_matches_full_forecast_with_shared_tile_visits():
    from random import Random
    rng = Random(91)
    labor = LaborForecast()
    for _ in range(30):
        baseline, candidate = Production(), Production()
        for day in range(3):
            baseline.visits[day] = [
                ((rng.randrange(10), rng.randrange(10)), rng.randint(1, 8),
                 (rng.choice(('WHEAT', 'COW', 'FERTILIZER')),), bool(rng.randrange(2)))
                for _ in range(40)
            ]
            candidate.visits[day] = [(None, 2, ('WHEAT',), True), (None, 1, (), False)]
        position = baseline.visits[0][0][0]
        combined = Production()
        combined.add(baseline)
        combined.add(candidate, position)
        combined_cost = labor.cost(combined.visits)
        expected = (float('inf') if combined_cost == float('inf') else
                    max(0, combined_cost - labor.cost(baseline.visits)))
        assert labor.marginal_cost(baseline, candidate, position) == expected
        prepared = labor.prepare(baseline.visits)
        assert labor.marginal_cost(baseline, candidate, position, prepared=prepared) == expected
