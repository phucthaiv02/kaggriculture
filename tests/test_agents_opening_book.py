"""Isolated tests for the short hybrid opening."""

from collections import Counter

import pytest

from agents.opening_book import (
    LAND_BUY_DAYS,
    LAND_FIRST_DAY,
    LAND_INTERVAL_DAYS,
    LAND_MAX_EXTRA,
    OPENING_COUNTS,
    OPENING_HANDOFF_DAY,
    build_opening_targets,
    make_opening_controller,
    should_buy_land_on_schedule,
)


NW_POSITIONS = [(x, y) for y in range(5) for x in range(5)]


def make_obs(day, tiles=(), quadrants=("NW",)):
    board = [[None] * 10 for _ in range(10)]
    for (x, y), tile in dict(tiles).items():
        board[y][x] = tile
    farm = {"tiles": board, "unlocked_quadrants": list(quadrants)}
    return {"day": day, "player": 0, "farms": [farm, farm]}


def wheat_tile(planted_day=0, yield_units=1):
    return {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": planted_day,
        "yield_units": yield_units,
        "watered_today": True,
    }


def test_build_opening_targets_matches_fixed_portfolio():
    targets = build_opening_targets(NW_POSITIONS)
    assert len(targets) == 25
    assert Counter(value[0] for value in targets.values()) == Counter(OPENING_COUNTS)
    assert OPENING_COUNTS == {"MELON": 12, "WHEAT": 9, "COW": 2, "SHEEP": 2}
    assert all(fertilize is False for _name, fertilize in targets.values())


def test_build_opening_targets_rejects_wrong_board_size():
    with pytest.raises(ValueError, match="exactly 25"):
        build_opening_targets(NW_POSITIONS[:-1])


def test_fixed_opening_governs_days_zero_through_two():
    governs = make_opening_controller()
    targets = {}
    assert OPENING_HANDOFF_DAY == 3
    assert governs(make_obs(0), targets, NW_POSITIONS) is True
    assert governs(make_obs(1), targets, NW_POSITIONS) is True
    assert governs(make_obs(2), targets, NW_POSITIONS) is True


def test_opening_hands_off_to_planner_on_day_three_permanently():
    governs = make_opening_controller()
    targets = {}
    governs(make_obs(0), targets, NW_POSITIONS)
    assert governs(make_obs(3), targets, NW_POSITIONS) is False
    assert governs(make_obs(4), targets, NW_POSITIONS) is False


def test_day_two_conversion_still_creates_third_cow_and_sheep_before_handoff():
    governs = make_opening_controller()
    targets = {}
    governs(make_obs(0), targets, NW_POSITIONS)
    wheat_positions = [p for p, target in targets.items() if target[0] == "WHEAT"]
    closest = sorted(
        wheat_positions,
        key=lambda p: (abs(p[0] - 4) + abs(p[1] - 4), -p[1], -p[0]),
    )
    growing = {
        closest[0]: wheat_tile(),
        closest[1]: wheat_tile(),
    }
    assert governs(make_obs(2, growing), targets, NW_POSITIONS) is True
    assert targets[closest[0]][0] == "COW"
    assert targets[closest[1]][0] == "SHEEP"
    assert Counter(value[0] for value in targets.values()) == Counter(
        {"MELON": 12, "WHEAT": 7, "COW": 3, "SHEEP": 3}
    )


def test_successful_early_land_unlock_forces_handoff():
    governs = make_opening_controller()
    targets = {}
    governs(make_obs(0), targets, NW_POSITIONS)
    assert governs(make_obs(2, quadrants=("NW", "NE")), targets, NW_POSITIONS) is False


def test_land_policy_targets_days_six_and_nine_from_state():
    assert LAND_FIRST_DAY == 6
    assert LAND_INTERVAL_DAYS == 3
    assert LAND_MAX_EXTRA == 2
    assert LAND_BUY_DAYS == (6, 9)

    nw = {"unlocked_quadrants": ["NW"]}
    assert should_buy_land_on_schedule({"day": 5, "hour": 0}, nw) is False
    assert should_buy_land_on_schedule({"day": 6, "hour": 0}, nw) is True
    assert should_buy_land_on_schedule({"day": 6, "hour": 1}, nw) is False

    one_extra = {"unlocked_quadrants": ["NW", "NE"]}
    assert should_buy_land_on_schedule({"day": 8, "hour": 0}, one_extra) is False
    assert should_buy_land_on_schedule({"day": 9, "hour": 0}, one_extra) is True

    two_extra = {"unlocked_quadrants": ["NW", "NE", "SW"]}
    assert should_buy_land_on_schedule({"day": 9, "hour": 0}, two_extra) is False
    assert should_buy_land_on_schedule({"day": 12, "hour": 0}, two_extra) is False


def test_unfunded_land_attempt_retries_until_quadrant_count_changes():
    nw = {"unlocked_quadrants": ["NW"]}
    assert should_buy_land_on_schedule({"day": 6, "hour": 0}, nw) is True
    assert should_buy_land_on_schedule({"day": 7, "hour": 0}, nw) is True
    assert should_buy_land_on_schedule({"day": 8, "hour": 0}, nw) is True
