"""Target-specific horizon and profit/day regression tests for live planning."""

from collections import Counter

import pytest

from agents.forecast import MarketForecast, Production
from agents.horizon import PLANNER_HORIZON_DAYS, planner_cycle_end
from agents.planner import _choose_daily, _daily_candidates
from kaggle_environments.envs.kaggriculture import kaggriculture as game


EXPECTED_HORIZONS = {
    "WHEAT": 4,
    "CARROT": 3,
    "MELON": 10,
    "TOMATO": 11,
    "STRAWBERRY": 16,
    "GOOSE": 15,
    "COW": 14,
    "SHEEP": 12,
}


class ZeroLabor:
    def cost(self, _visits):
        return 0

    def marginal_cost(self, *_args, **_kwargs):
        return 0


def test_requested_per_type_horizons_are_exact():
    assert PLANNER_HORIZON_DAYS == EXPECTED_HORIZONS
    for name, horizon in EXPECTED_HORIZONS.items():
        assert planner_cycle_end(name, 7, 29) == min(29, 7 + horizon)


@pytest.mark.parametrize("name", EXPECTED_HORIZONS)
def test_daily_candidates_do_not_project_past_their_own_horizon(name):
    candidates = {
        choice: flow
        for choice, flow, _cost in _daily_candidates(0, 29)
    }
    variants = ((False, True) if name in game.CROPS else (False,))
    product = game.ANIMALS[name]["product"] if name in game.ANIMALS else name
    for fertilize in variants:
        flow = candidates[(name, fertilize)]
        assert all(day <= EXPECTED_HORIZONS[name] for day in flow.sales)
        harvest_days = [day for day, units in flow.sales.items() if units[product]]
        assert harvest_days
        assert max(harvest_days) <= EXPECTED_HORIZONS[name]


def test_animal_windows_include_requested_number_of_followup_harvests():
    candidates = {
        choice: flow
        for choice, flow, _cost in _daily_candidates(0, 29)
    }
    expected_counts = {"GOOSE": 7, "COW": 4, "SHEEP": 3}
    for name, count in expected_counts.items():
        product = game.ANIMALS[name]["product"]
        harvest_days = [
            day
            for day, units in candidates[(name, False)].sales.items()
            if units[product]
        ]
        assert len(harvest_days) == count
        assert harvest_days[0] == game.ANIMALS[name]["first_yield_day"]
        assert harvest_days[-1] == EXPECTED_HORIZONS[name]


def test_profit_per_day_can_beat_higher_absolute_cycle_profit():
    inventory = {p: game.MARKET_I0 for p in game.PRODUCTS}
    market = MarketForecast(inventory, (), 0, 29)
    market.price = lambda product, stock: {"WHEAT": 25, "CARROT": 30}.get(product, 1)
    candidates = [
        candidate
        for candidate in _daily_candidates(0, 29)
        if candidate[0] in (("WHEAT", False), ("CARROT", False))
    ]
    choice, _output = _choose_daily(
        market,
        Production(),
        candidates,
        Counter(),
        ZeroLabor(),
    )
    # WHEAT: (4*25-10)/4 = 22.5/day.
    # CARROT: (3*30-20)/3 = 23.33/day.
    assert choice == ("CARROT", False)
