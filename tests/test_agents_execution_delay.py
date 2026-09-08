from collections import Counter

import pytest
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from agents import planner


BASE_INVENTORY = {product: game.MARKET_I0 for product in game.PRODUCTS}


def make_obs(day, hour, tile=None):
    board = [[None] * 10 for _ in range(10)]
    board[0][0] = tile
    farm = {
        "tiles": board,
        "unlocked_quadrants": ["NW"],
        "farmer": [4, 4],
        "hands": [],
        "money": 10000,
    }
    return {
        "day": day,
        "hour": hour,
        "player": 0,
        "farms": [farm, farm],
        "market": {
            "inventory": dict(BASE_INVENTORY),
            "prices": {
                product: game.market_price(product, game.MARKET_I0)
                for product in game.PRODUCTS
            },
        },
        "town": {"unlocked_shops": []},
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
    }


def _capture_first_candidate(monkeypatch):
    seen = []

    def choose(_market, _baseline, candidates, _counts, _labor=None, _position=(4, 4)):
        choice, output, _cost = list(candidates)[0]
        first_sale = min(
            day
            for day, units in output.sales.items()
            if sum(units.values()) > 0
        )
        seen.append((choice[0], first_sale))
        return choice, output

    monkeypatch.setattr(planner, "_choose", choose)
    return seen


def test_midday_empty_tile_is_valued_from_next_day(monkeypatch):
    seen = _capture_first_candidate(monkeypatch)
    obs = make_obs(day=7, hour=1)

    planner.plan_targets(obs, {}, [(0, 0)], end_day=29)

    # Sorted candidates start with CARROT. Its first yield is age 3, so a
    # conservative day-8 commitment first sells on day 11, not day 10.
    assert seen == [("CARROT", 11)]


def test_morning_empty_tile_can_start_today(monkeypatch):
    seen = _capture_first_candidate(monkeypatch)
    obs = make_obs(day=7, hour=0)

    planner.plan_targets(obs, {}, [(0, 0)], end_day=29)

    assert seen == [("CARROT", 10)]


def test_finished_live_crop_can_replant_same_day_midday(monkeypatch):
    seen = _capture_first_candidate(monkeypatch)
    tile = game._new_plant("WHEAT", 3, 24)
    tile["yield_units"] = 4
    obs = make_obs(day=7, hour=12, tile=tile)
    targets = {(0, 0): ("WHEAT", False)}

    planner.plan_targets(obs, targets, [(0, 0)], end_day=29)

    # A worker already servicing the finished crop can harvest/replant in
    # place, so this path keeps the existing same-day start assumption.
    assert seen == [("CARROT", 10)]


def test_midday_empty_structure_is_also_delayed(monkeypatch):
    seen = _capture_first_candidate(monkeypatch)
    obs = make_obs(day=7, hour=1, tile={"kind": "COOP"})

    planner.plan_targets(obs, {}, [(0, 0)], end_day=29)

    # COOP filters to GOOSE; first yield age 4 from the delayed day-8 start.
    assert seen == [("GOOSE", 12)]


@pytest.mark.parametrize(
    "hour,tile,expected",
    [
        (0, None, 7),
        (1, None, 8),
        (23, {"kind": "COOP"}, 8),
        (23, {"kind": "PLANT", "crop": "WHEAT"}, 7),
        (23, {"animal": "GOOSE", "kind": "COOP"}, 7),
    ],
)
def test_candidate_start_day_rule(hour, tile, expected):
    assert planner._candidate_start_day(7, hour, tile) == expected
