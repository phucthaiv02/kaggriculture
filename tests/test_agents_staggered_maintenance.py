"""Yield and execution checks for staggered WATER / FEED / CARE phases."""

from kaggle_environments.envs.kaggriculture import kaggriculture as game

from agents.farm_tasks import build_tasks, feed_wheat_order
from agents.maintenance import (
    COW_MAINTENANCE_VARIANTS,
    CROP_WATER_VARIANTS,
    should_care,
    should_feed,
    should_water,
)
from experiments.animal_yields import OPTIMAL_SCHEDULES
from experiments.crop_schedules import CASES


def _simulate_crop(crop, fertilized, waters):
    case = CASES[(crop, fertilized)]
    tile = game._new_plant(crop, 0, 24)
    farm = {"tiles": [[tile]], "farmer": [0, 0], "hands": []}
    private = {"inventories": [{}], "shed": {}, "seeds": {}}
    got = []
    last = max(case["harvest"])
    for day in range(last + 1):
        live = farm["tiles"][0][0]
        assert isinstance(live, dict) and live.get("kind") == "PLANT"
        inv = private["inventories"][0]
        inv.clear()
        if case["ongoing"] and day in case["harvest"] and live.get("yield_units", 0):
            got.append(live["yield_units"])
            game._apply_unit_action(farm, private, 0, ["HARVEST"], 1, day, 24)
        if day in case["fertilize"]:
            inv["FERTILIZER"] = 1
            game._apply_unit_action(farm, private, 0, ["FERTILIZE"], 1, day, 24)
        if day in waters:
            game._apply_unit_action(farm, private, 0, ["WATER"], 1, day, 24)
        if not case["ongoing"] and day in case["harvest"]:
            got.append(live.get("yield_units", 0))
            game._apply_unit_action(farm, private, 0, ["HARVEST"], 1, day, 24)
        if day < last:
            game._daily_refresh_plants(farm, day, 24)
    return got


def test_every_water_phase_keeps_the_validated_harvest_trace():
    for (crop, fertilized), variants in CROP_WATER_VARIANTS.items():
        case = CASES[(crop, fertilized)]
        baseline_count = len(case["water"])
        for waters in variants:
            assert len(waters) == baseline_count
            assert _simulate_crop(crop, fertilized, waters) == case["expected_yields"]


def _cow_trace(feed_days, care_days):
    schedule = OPTIMAL_SCHEDULES["COW"]
    tile = game._new_animal("COW", 0)
    farm = {"tiles": [[tile]], "farmer": [0, 0], "hands": []}
    private = {"inventories": [{}], "shed": {}, "seeds": {}}
    trace = []
    for day in range(30):
        live = farm["tiles"][0][0]
        inv = private["inventories"][0]
        inv.clear()
        if day in schedule["harvest"]:
            trace.append(live.get("yield_units", 0))
            if live.get("yield_units", 0):
                game._apply_unit_action(farm, private, 0, ["HARVEST"], 1, day, 24)
        if day in feed_days:
            inv["WHEAT"] = inv.get("WHEAT", 0) + 1
            game._apply_unit_action(farm, private, 0, ["FEED"], 1, day, 24)
        if day in care_days:
            game._apply_unit_action(farm, private, 0, ["CARE"], 1, day, 24)
        if day < 29:
            game._daily_refresh_animals(farm, day)
    return trace


def test_cow_phases_keep_every_harvest_and_action_count():
    expected = _cow_trace(*COW_MAINTENANCE_VARIANTS[0])
    assert sum(expected) == OPTIMAL_SCHEDULES["COW"]["max_yield"]
    for feed_days, care_days in COW_MAINTENANCE_VARIANTS[1:]:
        assert len(feed_days) == len(COW_MAINTENANCE_VARIANTS[0][0])
        assert len(care_days) == len(COW_MAINTENANCE_VARIANTS[0][1])
        assert _cow_trace(feed_days, care_days) == expected


def _obs_with_tiles(day, tiles, shed=None):
    return {
        "day": day,
        "hour": 1,
        "player": 0,
        "_planning_end_day": 29,
        "farms": [
            {
                "tiles": [tiles],
                "money": 10000,
                "farmer": [0, 0],
                "hands": [],
            }
        ],
        "private": {"shed": shed or {}, "seeds": {}, "inventories": [{}]},
        "market": {
            "prices": {"WHEAT": 25},
            "inventory": {"WHEAT": 100},
            "params": None,
        },
    }


def _plant(crop, planted_day=0):
    return {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": planted_day,
        "watered_today": False,
        "consecutive_unwatered": 0,
        "yield_units": 0,
        "max_lifespan_step": -1,
        "fertilized_until_day": -1,
    }


def _cow(placed_day=0):
    return {
        "kind": "PASTURE",
        "animal": "COW",
        "placed_day": placed_day,
        "yield_units": 0,
        "consecutive_unfed": 0,
        "fed_today": False,
        "cared_today": False,
        "fertilizer_available": False,
        "pending_care_bonus": 0,
    }


def test_same_age_tomatoes_split_water_by_position():
    obs = _obs_with_tiles(2, [_plant("TOMATO"), _plant("TOMATO")])
    targets = {(0, 0): ("TOMATO", False), (1, 0): ("TOMATO", False)}
    tasks = build_tasks(obs, targets)
    water_positions = {
        task.position
        for task in tasks
        if any(action == ["WATER"] for action in task.actions)
    }
    assert (0, 0) in water_positions
    assert (1, 0) not in water_positions
    assert should_water("TOMATO", 2, False, (0, 0))
    assert not should_water("TOMATO", 2, False, (1, 0))


def test_same_age_cows_split_feed_and_care_between_age_two_and_three():
    targets = {(0, 0): ("COW", False), (1, 0): ("COW", False)}

    day2 = _obs_with_tiles(2, [_cow(), _cow()], shed={"WHEAT": 2})
    tasks2 = build_tasks(day2, targets)
    ops2 = {task.position: [a[0] for a in task.actions] for task in tasks2}
    assert "FEED" not in ops2.get((0, 0), [])
    assert "FEED" in ops2.get((1, 0), [])
    assert "CARE" in ops2.get((1, 0), [])
    assert should_feed("COW", 2, (1, 0)) and should_care("COW", 2, (1, 0))

    day3 = _obs_with_tiles(3, [_cow(), _cow()], shed={"WHEAT": 2})
    tasks3 = build_tasks(day3, targets)
    ops3 = {task.position: [a[0] for a in task.actions] for task in tasks3}
    assert "FEED" in ops3.get((0, 0), [])
    assert "FEED" not in ops3.get((1, 0), [])


def test_staggered_cow_feed_is_funded_on_its_shifted_day():
    obs = _obs_with_tiles(2, [_cow(), _cow()], shed={})
    targets = {(0, 0): ("COW", False), (1, 0): ("COW", False)}
    assert feed_wheat_order(obs, targets, [(0, 0), (1, 0)]) == [
        ["BUY_PRODUCT", "WHEAT", 1]
    ]
