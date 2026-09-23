from agents.farm_tasks import build_tasks
from agents.planner import plan_targets
from agents.schedules import CROP_WATER_DAYS, cycle_turns_over_today, water_days


PRODUCTS = (
    "WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)


def crop_tile(crop, planted_day, yield_units=0, consecutive_unwatered=0):
    return {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": planted_day,
        "watered_today": False,
        "consecutive_unwatered": consecutive_unwatered,
        "yield_units": yield_units,
        "fertilized_until_day": -1,
    }


def obs(day, tiles, seeds=None):
    board = [[None] * 10 for _ in range(10)]
    for (x, y), tile in tiles.items():
        board[y][x] = tile
    farm = {
        "money": 3000.0,
        "tiles": board,
        "farmer": [4, 4],
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }
    return {
        "day": day,
        "hour": 0,
        "player": 0,
        "farms": [farm, farm],
        "private": {"shed": {}, "seeds": seeds or {}, "inventories": [{}]},
        "market": {
            "prices": {p: 25 for p in PRODUCTS},
            "inventory": {p: 10000 for p in PRODUCTS},
        },
        "town": {"unlocked_shops": []},
        "_planning_end_day": 29,
    }


def test_melon_same_cohort_uses_staggered_spatial_profiles():
    baseline = water_days("MELON", False, (0, 0), planted_day=0)
    shifted = water_days("MELON", False, (2, 0), planted_day=0)
    assert baseline == frozenset(CROP_WATER_DAYS[("MELON", False)])
    assert baseline != shifted
    assert 2 in baseline and 2 not in shifted
    assert 1 not in baseline and 1 in shifted


def test_opening_without_position_keeps_historical_calendar():
    assert water_days("MELON", False) == frozenset(
        CROP_WATER_DAYS[("MELON", False)]
    )


def test_ongoing_final_output_can_turn_over_today_before_becoming_finished():
    tile = crop_tile("STRAWBERRY", planted_day=0, yield_units=4)
    assert cycle_turns_over_today("STRAWBERRY", 16, tile)


def test_final_ongoing_crop_chains_harvest_dig_and_replant_same_day():
    state = obs(
        16,
        {(0, 0): crop_tile("STRAWBERRY", 0, yield_units=4)},
        seeds={"WHEAT": 1},
    )
    tasks = build_tasks(state, {(0, 0): ("WHEAT", False)})
    assert len(tasks) == 1
    assert tasks[0].actions == [
        ["HARVEST"],
        ["DIG"],
        ["PLANT", "WHEAT"],
        ["WATER"],
    ]
    assert tasks[0].ends_cycle


def test_final_ongoing_crop_with_no_successor_is_cleared_same_day():
    state = obs(16, {(0, 0): crop_tile("STRAWBERRY", 0, yield_units=4)})
    tasks = build_tasks(state, {(0, 0): None})
    assert len(tasks) == 1
    assert tasks[0].actions == [["HARVEST"], ["DIG"]]
    assert tasks[0].ends_cycle


def test_growing_pending_crop_stays_pending_until_turnover_day():
    position = (0, 0)
    state = obs(15, {position: crop_tile("STRAWBERRY", 0)})
    targets = {position: ("STRAWBERRY", False)}
    pending = plan_targets(
        state,
        targets,
        [position],
        29,
        max_positions=10,
        replan_positions={position},
    )
    assert pending == [position]
