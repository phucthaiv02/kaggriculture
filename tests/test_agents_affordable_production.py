"""Capital must buy executable production instead of empty animal structures."""

import pytest

from agents import planner
from agents.farm_tasks import build_tasks, purchase_orders
from test_agents_planner import make_obs


def prefer_animals(monkeypatch, name="COW"):
    def choose(_market, _baseline, candidates, _counts, *_args):
        if not candidates:
            return None, None
        choice, output, _cost = max(
            candidates,
            key=lambda c: (c[0][0] == name, c[0][0] == "CARROT", not c[0][1]),
        )
        return choice, output
    monkeypatch.setattr(planner, "_choose_daily", choose)
    monkeypatch.setattr(planner, "_choose_daily_mirrored", choose)


@pytest.mark.parametrize("version", ["concentration", "mirror_v2"])
def test_two_hundred_selects_and_funds_crops_instead_of_pastures(monkeypatch, version):
    prefer_animals(monkeypatch)
    obs = make_obs(day=5, money=200)
    positions = [(4, 4), (3, 4)]
    targets = {position: ("COW", False) for position in positions}
    planner.plan_targets(obs, targets, positions, 29, version)
    assert all(target == ("CARROT", False) for target in targets.values())
    orders = purchase_orders(obs, targets, positions)
    assert orders == [["BUY_SEED", "CARROT", 2]]
    obs["private"]["seeds"]["CARROT"] = 2
    tasks = build_tasks(obs, targets)
    assert sum(["PLANT", "CARROT"] in task.actions for task in tasks) == 2
    assert not any(op[0].startswith("BUILD") for task in tasks for op in task.actions)


def test_multiple_plots_share_one_capital_budget(monkeypatch):
    prefer_animals(monkeypatch)
    obs = make_obs(day=5, money=420)
    positions = [(4, 4), (3, 4), (4, 3)]
    targets = {}
    planner.plan_targets(obs, targets, positions, 29)
    assert list(targets.values()).count(("COW", False)) == 1
    assert list(targets.values()).count(("CARROT", False)) == 1
    assert list(targets.values()).count(None) == 1


def test_animal_affordability_includes_placement_feed(monkeypatch):
    prefer_animals(monkeypatch, "SHEEP")
    obs = make_obs(day=5, money=500)
    targets = {}
    planner.plan_targets(obs, targets, [(4, 4)], 29)
    assert targets[(4, 4)] == ("CARROT", False)
    obs["private"]["shed"]["WHEAT"] = 1
    planner.plan_targets(obs, targets, [(4, 4)], 29)
    assert targets[(4, 4)] == ("SHEEP", False)


def test_owned_seeds_are_allocated_once_with_no_cash(monkeypatch):
    prefer_animals(monkeypatch)
    obs = make_obs(day=5, money=0)
    obs["private"]["seeds"]["CARROT"] = 1
    targets = {}
    planner.plan_targets(obs, targets, [(4, 4), (3, 4)], 29)
    assert list(targets.values()).count(("CARROT", False)) == 1
    assert list(targets.values()).count(None) == 1


def test_scoped_replanning_preserves_opening_and_reserves_its_inputs(monkeypatch):
    prefer_animals(monkeypatch)
    obs = make_obs(day=5, money=220)
    targets = {(4, 4): ("CARROT", False), (3, 4): ("COW", False)}
    planner.plan_targets(obs, targets, list(targets), 29, replan_positions={(3, 4)})
    assert targets == {(4, 4): ("CARROT", False), (3, 4): ("CARROT", False)}


def test_build_waits_for_real_animal_even_when_purchase_is_affordable():
    obs = make_obs(day=5, money=1000)
    target = {(4, 4): ("COW", False)}
    assert build_tasks(obs, target) == []
    obs["private"]["shed"]["COW"] = 1
    tasks = build_tasks(obs, target)
    assert tasks[0].actions == [["BUILD_PASTURE"], ["PLACE", "COW"]]


def test_melon_match_never_builds_without_a_carried_animal():
    from pathlib import Path
    from kaggle_environments import make
    from agents.expansion_agent import make_agent
    from experiments.play_match import configuration, notebook_agent

    config = configuration(1)
    config["episodeSteps"] = 169
    env = make("kaggriculture", configuration=config, debug=False)
    agent = make_agent(opening_version="melon_v2")
    opponent, _ = notebook_agent(Path(__file__).resolve().parents[1] / "public" / "A.ipynb")
    env.run([lambda obs: agent(obs), opponent])
    assert len(env.steps) == 169
    builds = 0
    for previous, current in zip(env.steps, env.steps[1:]):
        obs = previous[0].observation
        action = current[0].action or {}
        for index, operation in enumerate([action.get("farmer"), *action.get("hands", [])]):
            if operation not in (["BUILD_PASTURE"], ["BUILD_COOP"]):
                continue
            builds += 1
            inventory = obs.private.inventories[index]
            names = ("COW", "SHEEP") if operation == ["BUILD_PASTURE"] else ("GOOSE",)
            assert any(inventory.get(name, 0) for name in names), (obs.day, obs.hour, operation)
    assert builds > 0
    assert len(env.steps[-1][0].observation.farms[0].unlocked_quadrants) == 2


def test_finished_wheat_does_not_keep_an_unaffordable_animal_target(monkeypatch):
    prefer_animals(monkeypatch)
    tile = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": 1,
        "yield_units": 6, "watered_today": True,
    }
    obs = make_obs(day=5, money=200, tiles={(4, 4): tile})
    targets = {(4, 4): ("COW", False)}
    planner.plan_targets(obs, targets, list(targets), 29)
    assert targets[(4, 4)] == ("CARROT", False)
    tasks = build_tasks(obs, targets)
    assert any(["HARVEST"] in task.actions for task in tasks)
    assert not any(op[0].startswith("BUILD") for task in tasks for op in task.actions)


def test_unfunded_fixed_animal_does_not_consume_scoped_crop_budget(monkeypatch):
    prefer_animals(monkeypatch)
    obs = make_obs(day=5, money=200)
    targets = {(4, 4): ("SHEEP", False), (3, 4): None}
    planner.plan_targets(obs, targets, list(targets), 29, replan_positions={(3, 4)})
    assert targets[(4, 4)] == ("SHEEP", False)
    assert targets[(3, 4)] == ("CARROT", False)
