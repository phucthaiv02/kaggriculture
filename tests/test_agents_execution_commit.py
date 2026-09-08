from collections import Counter

from kaggle_environments.envs.kaggriculture import kaggriculture as game

from agents import planner
from agents.forecast import Production
from agents.scheduler import MAX_HANDS


BASE_INVENTORY = {
    product: game.MARKET_I0 for product in game.PRODUCTS
}


def make_obs(day=7, hour=1):
    board = [[None] * 10 for _ in range(10)]
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
        # Alias the same farm so planner's test-fixture guard ignores player 1.
        "farms": [farm, farm],
        "market": {
            "inventory": dict(BASE_INVENTORY),
            "prices": {
                product: game.market_price(product, game.MARKET_I0)
                for product in game.PRODUCTS
            },
        },
        "town": {"unlocked_shops": []},
        "private": {
            "shed": {},
            "seeds": {},
            "inventories": [{}],
        },
    }


def test_fresh_commit_slots_track_max_parallel_workers():
    assert planner.FRESH_COMMIT_SLOTS == MAX_HANDS + 1


def test_empty_targets_keep_same_day_economics(monkeypatch):
    seen = []

    def choose(
        _market,
        _baseline,
        candidates,
        _counts,
        _labor=None,
        _position=(4, 4),
    ):
        choice, output, _cost = list(candidates)[0]
        first_sale = min(
            day
            for day, units in output.sales.items()
            if sum(units.values()) > 0
        )
        seen.append((choice[0], first_sale))
        return choice, output

    monkeypatch.setattr(planner, "_choose", choose)
    obs = make_obs(day=7, hour=1)

    planner.plan_targets(obs, {}, [(0, 0)], end_day=29)

    # CARROT first yields at age 3. V2 deliberately removes V1's blanket
    # D+1 valuation penalty; execution uncertainty is handled by the bounded
    # shared-baseline commitment envelope instead.
    assert seen == [("CARROT", 10)]


def test_only_execution_envelope_affects_shared_baseline(monkeypatch):
    obs = make_obs(day=7, hour=1)
    positions = [(x, y) for y in range(5) for x in range(5)]
    seen_wheat = []

    def choose(
        _market,
        baseline,
        _candidates,
        _counts,
        _labor=None,
        _position=(4, 4),
    ):
        committed = sum(
            baseline.sales.values(), Counter()
        )["WHEAT"]
        seen_wheat.append(committed)
        output = Production()
        output.sales[11]["WHEAT"] = 1
        return ("WHEAT", False), output

    monkeypatch.setattr(planner, "_choose", choose)
    targets = {}

    planner.plan_targets(obs, targets, positions, end_day=29)

    assert len(targets) == 25
    assert all(
        target == ("WHEAT", False)
        for target in targets.values()
    )

    # 17 possible simultaneous workers (farmer + MAX_HANDS) can provisionally
    # influence the portfolio. Later desired targets remain assigned, but do
    # not pretend to be committed production in this planning pass.
    expected = list(range(planner.FRESH_COMMIT_SLOTS + 1))
    expected += [planner.FRESH_COMMIT_SLOTS] * (
        len(positions) - len(expected)
    )
    assert seen_wheat == expected


def test_existing_producer_is_not_a_fresh_commitment():
    tile = game._new_plant("WHEAT", 3, 24)
    assert planner._is_fresh_target_tile(None)
    assert planner._is_fresh_target_tile({"kind": "COOP"})
    assert not planner._is_fresh_target_tile(tile)
    assert not planner._is_fresh_target_tile(
        {"kind": "COOP", "animal": "GOOSE"}
    )
