from collections import Counter

from kaggle_environments.envs.kaggriculture import kaggriculture as game

from agents.forecast import MarketForecast, Production
from agents.production_paths import (
    CONTINUATION_WIDTH, cycle_candidates, production_path_candidates,
)


BASE_INVENTORY = {product: game.MARKET_I0 for product in game.PRODUCTS}


def market(day=0, end_day=16):
    return MarketForecast(BASE_INVENTORY, (), day, end_day)


def test_first_cycle_candidate_count_is_bounded():
    rows = cycle_candidates(0, 16)
    # 2 variants each for WHEAT/CARROT/MELON, 4 each for TOMATO/STRAWBERRY,
    # plus three animals. Keep this explicit so a future combinatorial expansion
    # cannot silently reintroduce planner timeouts.
    assert len(rows) == 17


def test_animal_is_terminal_path_without_replacement():
    rows = production_path_candidates(
        market(), Production(), 0, 16, first_names={"COW"}
    )
    assert len(rows) == 1
    choice, _output, _cost, path = rows[0]
    assert choice == ("COW", False)
    assert len(path) == 1
    assert path[0][1] == ("COW", False)


def test_crop_path_reuses_slot_after_turnover():
    rows = production_path_candidates(
        market(), Production(), 0, 16, first_names={"WHEAT"}
    )
    assert len(rows) == 2
    # Both fertilizer variants represent a WHEAT first cycle. At normal market
    # inventory there is profitable remaining horizon after day 4, so the path
    # should value at least one successor instead of pretending the tile stops.
    assert all(len(path) >= 2 for _choice, _output, _cost, path in rows)
    assert all(path[1][0] >= 4 for _choice, _output, _cost, path in rows)


def test_path_length_is_finite_under_shortest_crop_cycle():
    rows = production_path_candidates(market(), Production(), 0, 16)
    # CARROT turns over every three days, so even the fastest all-crop path is
    # bounded by six starts in the inclusive 0..16 horizon.
    assert rows
    assert max(len(path) for _choice, _output, _cost, path in rows) <= 6
    assert CONTINUATION_WIDTH == 2


def test_v5_wiring_only_replaces_real_scoped_planner_candidates(monkeypatch):
    import agents.planner as planner
    import agents.v5_wiring as wiring

    seen = []
    replacement_output = Production()

    def fake_paths(_market, _baseline, day, end_day, *, first_names, **_kwargs):
        seen.append((day, end_day, set(first_names)))
        return [(("WHEAT", False), replacement_output, 0,
                 ((day, ("WHEAT", False)),))]

    captured = []

    def fake_evaluate(_market, _baseline, candidates, *args, **kwargs):
        captured.extend(candidates)
        return []

    monkeypatch.setattr(wiring, "production_path_candidates", fake_paths)
    monkeypatch.setattr(planner, "evaluate_targets", fake_evaluate)
    scoped_market = market(day=3, end_day=12)
    planner._choose(
        scoped_market,
        Production(),
        [(("MELON", False), Production(), 80)],
        Counter(),
        flows_scoped=True,
    )

    assert seen == [(3, 12, {"MELON"})]
    assert len(captured) == 1
    assert captured[0][0] == ("WHEAT", False)
    assert captured[0][1] is replacement_output
    assert captured[0][2] == 0


def test_direct_choose_compatibility_does_not_expand_paths(monkeypatch):
    import agents.planner as planner
    import agents.v5_wiring as wiring

    def fail(*args, **kwargs):
        raise AssertionError("direct _choose must not invoke v5 path expansion")

    monkeypatch.setattr(wiring, "production_path_candidates", fail)
    monkeypatch.setattr(planner, "evaluate_targets", lambda *args, **kwargs: [])
    planner._choose(None, None, [], Counter())


def test_schedule_guard_detects_mandatory_feed_beyond_turn_budget():
    from agents.farm_tasks import Task
    from agents.scheduler import WorkerPlan
    from agents.v5_schedule_guard import missing_mandatory_tasks

    feed = Task(
        (2, 0), [["FEED"]], needs=Counter({"WHEAT": 1}), mandatory=True,
    )
    plan = WorkerPlan((0, 0), [["EAST"], ["EAST"], ["FEED"]])

    assert missing_mandatory_tasks([feed], [plan], [2]) == [feed]
    assert missing_mandatory_tasks([feed], [plan], [3]) == []


def test_schedule_guard_ignores_optional_work_outside_budget():
    from agents.farm_tasks import Task
    from agents.scheduler import WorkerPlan
    from agents.v5_schedule_guard import missing_mandatory_tasks

    optional = Task((1, 0), [["HARVEST"]], mandatory=False)
    plan = WorkerPlan((0, 0), [["EAST"], ["HARVEST"]])

    assert missing_mandatory_tasks([optional], [plan], [1]) == []
