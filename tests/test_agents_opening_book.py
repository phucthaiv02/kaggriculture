"""Focused tests for fixed opening plus state-derived land cadence."""

from collections import Counter

from agents.opening_book import (
    LAND_BUY_DAYS,
    LAND_FIRST_DAY,
    LAND_INTERVAL_DAYS,
    LAND_MAX_EXTRA,
    OPENING_COUNTS,
    build_opening_targets,
    make_opening_controller,
    should_buy_land_on_schedule,
)

NW_POSITIONS = [(x, y) for y in range(5) for x in range(5)]


def make_obs(day, quadrants=("NW",)):
    board = [[None] * 10 for _ in range(10)]
    farm = {"tiles": board, "unlocked_quadrants": list(quadrants)}
    return {"day": day, "player": 0, "farms": [farm, farm]}


def test_fixed_opening_is_still_applied():
    governs = make_opening_controller()
    targets = {}
    assert governs(make_obs(0), targets, NW_POSITIONS) is True
    assert Counter(value[0] for value in targets.values()) == Counter(OPENING_COUNTS)
    assert OPENING_COUNTS == {"MELON": 12, "WHEAT": 9, "COW": 2, "SHEEP": 2}


def test_opening_keeps_governing_until_first_land_really_unlocks():
    governs = make_opening_controller()
    targets = {}
    governs(make_obs(0), targets, NW_POSITIONS)
    assert governs(make_obs(3), targets, NW_POSITIONS) is True
    assert governs(make_obs(5), targets, NW_POSITIONS) is True
    assert governs(make_obs(6, ("NW", "NE")), targets, NW_POSITIONS) is False
    assert governs(make_obs(7), targets, NW_POSITIONS) is False


def test_land_policy_is_days_six_and_nine_from_successful_state():
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


def test_unfunded_first_land_retries_without_handing_off_opening():
    farm = {"unlocked_quadrants": ["NW"]}
    assert should_buy_land_on_schedule({"day": 6, "hour": 0}, farm) is True
    assert should_buy_land_on_schedule({"day": 7, "hour": 0}, farm) is True
    assert should_buy_land_on_schedule({"day": 8, "hour": 0}, farm) is True
    governs = make_opening_controller()
    targets = {}
    governs(make_obs(0), targets, NW_POSITIONS)
    assert governs(make_obs(7), targets, NW_POSITIONS) is True


def test_build_opening_targets_fills_all_25_tiles():
    targets = build_opening_targets(NW_POSITIONS)
    assert len(targets) == 25
    assert all(not fertilize for _name, fertilize in targets.values())
