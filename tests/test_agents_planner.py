"""Isolated tests for agents/planner.py -- no environment, hand-built obs."""

from collections import Counter

from kaggle_environments.envs.kaggriculture.kaggriculture import MARKET_I0, PRODUCTS, market_price

from agents.planner import _expected_price, best_target, plan_targets, should_buy_land

BASE_INVENTORY = {p: MARKET_I0 for p in PRODUCTS}


def make_obs(day, tiles=(), unlocked_quadrants=("NW",), inventory=None):
    board = [[None] * 10 for _ in range(10)]
    for (x, y), tile in dict(tiles).items():
        board[y][x] = tile
    farm = {
        "tiles": board, "unlocked_quadrants": list(unlocked_quadrants),
        "farmer": [4, 4], "hands": [],
    }
    resolved_inventory = dict(inventory or BASE_INVENTORY)
    return {
        "day": day, "player": 0, "farms": [farm, farm],
        "market": {
            "inventory": resolved_inventory,
            "prices": {p: market_price(p, n) for p, n in resolved_inventory.items()},
        },
        "town": {"unlocked_shops": []},
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
    }


def test_expected_price_matches_engine_at_zero_commitment():
    for product in PRODUCTS:
        assert _expected_price(product, BASE_INVENTORY, {}) == market_price(product, MARKET_I0)


def test_expected_price_drops_as_committed_units_pile_up():
    """Selling more of the same product must never look more profitable than
    selling less of it -- the whole point of pricing the marginal unit."""
    light = _expected_price("MELON", BASE_INVENTORY, {"MELON": 12})
    heavy = _expected_price("MELON", BASE_INVENTORY, {"MELON": 120})
    assert heavy <= light


def test_best_target_none_when_nothing_fits_the_remaining_days():
    """1 day left: even WHEAT (fastest crop, 4 days) can't mature."""
    choice = best_target(end_day=1, day=0, inventory=BASE_INVENTORY, wheat_price=25, committed_units=Counter())
    assert choice is None


def test_best_target_returns_something_with_a_full_season_left():
    choice = best_target(end_day=30, day=0, inventory=BASE_INVENTORY, wheat_price=25, committed_units=Counter())
    assert choice is not None
    name, fertilize = choice
    assert isinstance(fertilize, bool)


def test_plan_targets_diversifies_across_many_tiles_in_one_pass():
    """Regression test for the concentration bug found via full-pipeline
    testing: a whole freshly-claimed quadrant (or a fresh 25-tile board)
    picked the same top-ROI crop for ~all of it because the discount wasn't
    tied to the engine's real price-impact curve. With real marginal pricing,
    committing many tiles to one product must eventually make something else
    the better choice."""
    positions = [(x, y) for y in range(5) for x in range(5)]
    obs = make_obs(day=0, unlocked_quadrants=("NW",))
    targets = {}
    plan_targets(obs, targets, positions, end_day=30)
    assert len(targets) == 25
    names = Counter(value[0] for value in targets.values() if value is not None)
    assert len(names) >= 2, f"collapsed onto a single type: {names}"
    # No single crop/animal should dominate essentially the whole board.
    assert max(names.values()) <= 20, names


def test_plan_targets_does_not_touch_a_tile_mid_growth():
    """A WHEAT tile at age 2 of 4 is nowhere near finished -- its target must
    be left exactly as it was, not reconsidered every day."""
    tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0, "yield_units": 1, "watered_today": False}
    obs = make_obs(day=2, tiles={(0, 0): tile})
    targets = {(0, 0): ("WHEAT", False)}
    plan_targets(obs, targets, [(0, 0)], end_day=30)
    assert targets[(0, 0)] == ("WHEAT", False)


def test_plan_targets_replans_a_finished_tile():
    tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0, "yield_units": 6, "watered_today": True}
    obs = make_obs(day=4, tiles={(0, 0): tile})
    targets = {(0, 0): ("WHEAT", False)}
    plan_targets(obs, targets, [(0, 0)], end_day=30)
    assert targets[(0, 0)] is not None  # some choice was (re-)made, not left stale by construction


def test_should_buy_land_false_below_utilization_threshold():
    positions = [(x, y) for y in range(5) for x in range(5)]
    tiles = {positions[i]: {"kind": "PLANT", "crop": "WHEAT"} for i in range(10)}  # 10/25 = 40%
    farm = {"unlocked_quadrants": ["NW"], "tiles": [[None] * 10 for _ in range(10)]}
    for (x, y), tile in tiles.items():
        farm["tiles"][y][x] = tile
    assert should_buy_land(farm, positions) is False


def test_should_buy_land_true_at_or_above_threshold():
    positions = [(x, y) for y in range(5) for x in range(5)]
    tiles = {positions[i]: {"kind": "PLANT", "crop": "WHEAT"} for i in range(19)}  # 19/25 = 76%
    farm = {"unlocked_quadrants": ["NW"], "tiles": [[None] * 10 for _ in range(10)]}
    for (x, y), tile in tiles.items():
        farm["tiles"][y][x] = tile
    assert should_buy_land(farm, positions) is True


def test_should_buy_land_false_once_all_quadrants_owned():
    positions = [(x, y) for y in range(5) for x in range(5)]
    farm = {"unlocked_quadrants": ["NW", "NE", "SW", "SE"], "tiles": [[None] * 10 for _ in range(10)]}
    for x, y in positions:
        farm["tiles"][y][x] = {"kind": "PLANT", "crop": "WHEAT"}
    assert should_buy_land(farm, positions) is False


def test_should_buy_land_ignores_a_target_decision_with_nothing_actually_planted():
    """Regression test: a target dict fully populated on day 0 (before a
    single seed is in the ground) must not look like 100% utilization."""
    positions = [(x, y) for y in range(5) for x in range(5)]
    farm = {"unlocked_quadrants": ["NW"], "tiles": [[None] * 10 for _ in range(10)]}
    assert should_buy_land(farm, positions) is False


if __name__ == "__main__":
    import sys
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"{'ALL PASSED' if not failures else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)
