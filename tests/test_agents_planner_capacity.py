from kaggle_environments.envs.kaggriculture import kaggriculture as game

from agents import planner
from agents.forecast import Production, production


def _obs(day=7, active_positions=()):
    board = [["LOCKED"] * 10 for _ in range(10)]
    for x, y in active_positions:
        board[y][x] = None
    farm = {
        "tiles": board,
        "farmer": [4, 4],
        "hands": [],
        "money": 100000,
        "unlocked_quadrants": ["NW", "NE"],
    }
    return {
        "day": day,
        "hour": 1,
        "player": 0,
        "farms": [farm, farm],
        "private": {"shed": {}, "inventories": [], "seeds": {}},
        "market": {
            "inventory": {name: game.MARKET_I0 for name in game.PRODUCTS},
            "params": None,
        },
        "town": {"unlocked_shops": []},
    }


def test_fresh_crop_labor_visit_includes_batched_seed_pickup():
    flow = production("WHEAT", False, 7, 11)
    assert flow.visits[7]
    _position, _actions, pickups, _goods = flow.visits[7][0]

    assert "SEED:WHEAT" in pickups
    # The synthetic seed token is route-only. It must never become a market
    # BUY input or affect market cash forecasting.
    assert "SEED:WHEAT" not in flow.inputs[7]


def test_existing_crop_does_not_invent_seed_pickup():
    tile = game._new_plant("WHEAT", 7, 24)
    flow = production("WHEAT", False, 7, 11, tile)
    assert all(
        "SEED:WHEAT" not in pickups
        for _position, _actions, pickups, _goods in flow.visits.get(7, ())
    )


def test_plan_targets_has_no_fixed_seventeen_commit_plateau(monkeypatch):
    # A fake cheap output isolates the planner's commitment bookkeeping from
    # crop economics. The labor model is assumed to accept every item here.
    # V2/V3 plateaued after 17 baseline additions even though later desired
    # targets were still written. V4 must let the baseline grow with every
    # accepted target and leave capacity decisions to LaborForecast.
    positions = [(x, y) for y in range(4) for x in range(5)]  # 20 fresh tiles
    obs = _obs(active_positions=positions)
    baseline_sizes = []

    def fake_choose(market, baseline, candidates, counts, labor=None, position=(4, 4)):
        baseline_sizes.append(sum(len(entries) for entries in baseline.visits.values()))
        output = Production()
        output.visits[7].append((None, 1, (), False))
        return ("WHEAT", False), output

    monkeypatch.setattr(planner, "_choose", fake_choose)
    targets = {}
    planner.plan_targets(obs, targets, positions, 29)

    assert baseline_sizes == list(range(20))
    assert len(targets) == 20
    assert all(target == ("WHEAT", False) for target in targets.values())
