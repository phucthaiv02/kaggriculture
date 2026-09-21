"""Isolated tests for agents/planner.py -- no environment, hand-built obs."""

from collections import Counter

from kaggle_environments.envs.kaggriculture.kaggriculture import MARKET_I0, PRODUCTS, market_price

from agents.planner import best_target, plan_targets

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


def test_best_target_none_when_nothing_fits_the_remaining_days():
    """1 day left: even WHEAT (fastest crop, 4 days) can't mature."""
    choice = best_target(end_day=1, day=0, inventory=BASE_INVENTORY, wheat_price=25, committed_units=Counter())
    assert choice is None


def test_best_target_returns_something_with_a_full_season_left():
    choice = best_target(end_day=30, day=0, inventory=BASE_INVENTORY, wheat_price=25, committed_units=Counter())
    assert choice is not None
    name, fertilize = choice
    assert isinstance(fertilize, bool)


def test_profitable_alternatives_survive_category_concentration():
    # At equal crop/animal capital, the old hard cutoff erased MELON's
    # 182/day score while preserving WHEAT's 22.5/day score.
    choice = best_target(
        end_day=30, day=7, inventory=BASE_INVENTORY, wheat_price=25,
        committed_units=Counter(),
        category_capital={"CROP": 500, "ANIMAL": 500}, total_capital=1000,
    )
    assert choice[0] != "WHEAT", choice


def test_crop_score_adds_replants_inside_the_window():
    from agents.planner import _score

    committed = Counter({"WHEAT": 100})
    near = _score("WHEAT", 11, 7, BASE_INVENTORY, 25, committed)
    far = _score("WHEAT", 30, 7, BASE_INVENTORY, 25, committed)
    assert far[0] > near[0]


def test_unprofitable_candidates_are_rejected():
    choice = best_target(
        end_day=30, day=27, inventory={p: 100000 for p in PRODUCTS}, wheat_price=1,
        committed_units=Counter(), category_capital={"CROP": 1000},
        total_capital=1000,
    )
    assert choice is None


def test_choose_keeps_current_target_inside_switch_margin(monkeypatch):
    import agents.planner as planner
    from agents.forecast import Production

    best = planner.TargetProfit(("WHEAT", False), Production(), 110.0, 10.0)
    current = planner.TargetProfit(("CORN", False), Production(), 109.5, 10.0)
    monkeypatch.setattr(planner, "evaluate_targets", lambda *args, **kwargs: [best, current])

    choice, _ = planner._choose(None, None, None, Counter(), current=current.choice)
    assert choice == current.choice


def test_choose_switches_when_current_target_loses_by_more_than_margin(monkeypatch):
    import agents.planner as planner
    from agents.forecast import Production

    best = planner.TargetProfit(("WHEAT", False), Production(), 110.0, 10.0)
    current = planner.TargetProfit(("CORN", False), Production(), 108.9, 10.0)
    monkeypatch.setattr(planner, "evaluate_targets", lambda *args, **kwargs: [best, current])

    choice, _ = planner._choose(None, None, None, Counter(), current=current.choice)
    assert choice == best.choice


def test_choose_without_current_preserves_o1_best_choice(monkeypatch):
    import agents.planner as planner
    from agents.forecast import Production

    best = planner.TargetProfit(("WHEAT", False), Production(), 110.0, 10.0)
    runner_up = planner.TargetProfit(("CORN", False), Production(), 109.5, 10.0)
    monkeypatch.setattr(planner, "evaluate_targets", lambda *args, **kwargs: [runner_up, best])

    choice, _ = planner._choose(None, None, None, Counter())
    assert choice == best.choice


def test_choose_logs_candidate_scores_and_switch_reason(monkeypatch):
    import agents.planner as planner
    from agents.forecast import Production

    best = planner.TargetProfit(("WHEAT", False), Production(), 110.0, 10.0)
    current = planner.TargetProfit(("CORN", False), Production(), 109.5, 10.0)
    monkeypatch.setattr(planner, "evaluate_targets", lambda *args, **kwargs: [best, current])
    market = type("Market", (), {"day": 3, "hour": 0})()
    decisions = []

    choice, _ = planner._choose(
        market, None, None, Counter(), position=(2, 4),
        current=current.choice, decision_log=decisions,
    )

    assert choice == current.choice
    assert decisions[0]["position"] == [2, 4]
    assert decisions[0]["reason"] == "retained_current_within_switch_margin"
    assert [row["score"] for row in decisions[0]["candidates"]] == [100.0, 99.5]
    assert [row["selected"] for row in decisions[0]["candidates"]] == [False, True]


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
    # Do not impose a fixed quota when marginal profit still favors a crop.
    assert names["WHEAT"] < len(targets), names


def test_plan_targets_does_not_touch_a_tile_mid_growth():
    """A WHEAT tile at age 2 of 4 is nowhere near finished -- its target must
    be left exactly as it was, not reconsidered every day."""
    tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0, "yield_units": 1, "watered_today": False}
    obs = make_obs(day=2, tiles={(0, 0): tile})
    targets = {(0, 0): ("WHEAT", False)}
    plan_targets(obs, targets, [(0, 0)], end_day=30)
    assert targets[(0, 0)] == ("WHEAT", False)


def test_same_day_turnover_is_not_limited_by_target_batch():
    positions = [(x, y) for y in range(3) for x in range(4)]
    tiles = {
        position: {
            "kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
            "yield_units": 4, "watered_today": True,
        }
        for position in positions
    }
    obs = make_obs(day=4, tiles=tiles)
    targets = {position: ("WHEAT", False) for position in positions}
    pending = plan_targets(
        obs, targets, positions, end_day=20,
        max_positions=10, replan_positions=set(positions),
    )
    assert pending == []
    assert all(position in targets for position in positions)


def test_plan_targets_replans_a_finished_tile():
    tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0, "yield_units": 6, "watered_today": True}
    obs = make_obs(day=4, tiles={(0, 0): tile})
    targets = {(0, 0): ("WHEAT", False)}
    plan_targets(obs, targets, [(0, 0)], end_day=30)
    assert targets[(0, 0)] is not None  # some choice was (re-)made, not left stale by construction


def test_finished_wheat_has_no_fallback_when_no_profitable_cycle_fits():
    from agents.farm_tasks import build_tasks

    tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 25,
            "yield_units": 4, "watered_today": True}
    obs = make_obs(day=29, tiles={(0, 0): tile})
    targets = {(0, 0): ("WHEAT", False)}
    plan_targets(obs, targets, [(0, 0)], end_day=29)
    assert targets[(0, 0)] is None
    tasks = build_tasks(obs, targets)
    assert any(["HARVEST"] in task.actions for task in tasks)
    assert not any(action[0] == "PLANT" for task in tasks for action in task.actions)


def test_idle_tile_is_reconsidered_when_market_recovers():
    obs = make_obs(day=7)
    targets = {(0, 0): None}
    plan_targets(obs, targets, [(0, 0)], end_day=29)
    assert targets[(0, 0)] is not None


def test_empty_animal_structures_only_get_compatible_targets():
    for structure, names in (("COOP", {"GOOSE"}), ("PASTURE", {"COW", "SHEEP"})):
        obs = make_obs(day=7, tiles={(0, 0): {"kind": structure}})
        targets = {(0, 0): ("WHEAT", False)}
        plan_targets(obs, targets, [(0, 0)], end_day=29)
        assert targets[(0, 0)] is None or targets[(0, 0)][0] in names


def test_batched_planning_finishes_75_tiles_without_repricing_completed_batches(monkeypatch):
    import agents.planner as planner

    positions = [(x, y) for y in range(10) for x in range(10) if y < 5 or x < 5]
    obs = make_obs(day=10, unlocked_quadrants=("NW", "NE", "SW"))
    targets, visited = {}, []
    choose = planner._choose

    def record(*args, **kwargs):
        # Support both call conventions used in different commits of the repo.
        visited.append(kwargs.get("position", args[-1] if args else None))
        return choose(*args, **kwargs)

    monkeypatch.setattr(planner, "_choose", record)
    pending = positions
    while pending:
        before = dict(targets)
        count = len(visited)
        pending = plan_targets(obs, targets, positions, 29, max_positions=4, replan_positions=set(pending))
        assert 1 <= len(visited) - count <= 4
        assert all(targets[p] == choice for p, choice in before.items())
    assert len(visited) == len(set(visited)) == 75
    assert set(targets) == set(positions)


def test_uncommitted_empty_target_is_not_forecast_as_future_supply(monkeypatch):
    import agents.planner as planner
    from agents.forecast import Production

    obs = make_obs(day=7)
    obs['_committed_targets'] = set()
    targets = {(0, 0): ('WHEAT', False), (1, 0): None}
    seen = []

    def capture(*args, **kwargs):
        # Support multiple call conventions used across commits.
        # Prefer kwargs but fall back to positional args.
        counts = kwargs.get('counts')
        baseline = kwargs.get('baseline')
        if counts is None:
            counts = args[3] if len(args) > 3 else Counter()
        if baseline is None:
            baseline = args[1] if len(args) > 1 else Production()
        seen.append((Counter(counts), sum(baseline.sales.values(), Counter())))
        return None, Production()

    monkeypatch.setattr(planner, '_choose', capture)
    plan_targets(obs, targets, [(0, 0), (1, 0)], 29,
                 replan_positions={(1, 0)})
    assert seen[0][0]['WHEAT'] == 0

    obs['_committed_targets'] = {(0, 0)}
    plan_targets(obs, targets, [(0, 0), (1, 0)], 29,
                 replan_positions={(1, 0)})
    assert seen[1][0]['WHEAT'] == 1
