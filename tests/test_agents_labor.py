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


def test_labor_cost_can_reverse_gross_profit_ranking():
    class Revenue:
        day, end_day = 0, 29
        def value(self, _): return 0
        def marginal_profit(self, baseline, output, cost, baseline_value):
            return 100 if output.visits else 90
    busy, light = Production(), Production()
    # The farm already needs many workers. A daily-work producer adds
    # costly marginal hires; the low-maintenance option fits existing labor.
    baseline = Production()
    for day in range(20):
        baseline.visits[day] = [((x, y), 18, (), False) for x, y in ((4,4),(4,3),(3,4),(3,3),(4,2),(2,4))]
        busy.visits[day] = [(None, 6, ('WHEAT',), True)]
    choice, _ = _choose(Revenue(), baseline, [(('GOOSE',False),busy,300), (('MELON',False),light,80)], Counter())
    assert choice[0] == 'MELON'


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
