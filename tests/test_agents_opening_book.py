"""Isolated tests for agents/opening_book.py using hand-built observations."""

from collections import Counter

import pytest

from agents.opening_book import (
    LAND_BUY_DAYS,
    OPENING_COUNTS,
    build_opening_targets,
    make_opening_controller,
    should_buy_land_on_schedule,
)


NW_POSITIONS = [(x, y) for y in range(5) for x in range(5)]


def make_obs(day, tiles=()):
    board = [[None] * 10 for _ in range(10)]
    for (x, y), tile in dict(tiles).items():
        board[y][x] = tile
    farm = {"tiles": board, "unlocked_quadrants": ["NW"]}
    return {"day": day, "player": 0, "farms": [farm, farm]}


def wheat_tile(planted_day, yield_units=1):
    return {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": planted_day,
        "yield_units": yield_units,
        "watered_today": True,
    }


def melon_tile(planted_day, yield_units=0):
    return {
        "kind": "PLANT",
        "crop": "MELON",
        "planted_day": planted_day,
        "yield_units": yield_units,
        "watered_today": True,
    }


def test_build_opening_targets_matches_counts_and_fills_the_board():
    targets = build_opening_targets(NW_POSITIONS)
    assert len(targets) == 25
    counts = Counter(value[0] for value in targets.values())
    assert dict(counts) == OPENING_COUNTS
    assert all(fertilize is False for _name, fertilize in targets.values())


def test_opening_uses_requested_portfolio():
    assert OPENING_COUNTS == {"MELON": 12, "WHEAT": 9, "COW": 2, "SHEEP": 2}


def test_debug_land_purchase_is_only_on_day_seven():
    farm = {"unlocked_quadrants": ["NW"]}
    assert LAND_BUY_DAYS == (7,)
    assert should_buy_land_on_schedule({"day": 6, "hour": 0}, farm) is False
    assert should_buy_land_on_schedule({"day": 7, "hour": 0}, farm) is True
    assert should_buy_land_on_schedule({"day": 7, "hour": 1}, farm) is False
    assert should_buy_land_on_schedule({"day": 8, "hour": 0}, farm) is False


def test_build_opening_targets_rejects_wrong_board_size():
    with pytest.raises(ValueError, match="exactly 25"):
        build_opening_targets(NW_POSITIONS[:-1])


def test_first_call_applies_opening_book_and_keeps_governing():
    governs = make_opening_controller()
    targets = {}
    obs = make_obs(day=0)
    assert governs(obs, targets, NW_POSITIONS) is True
    assert len(targets) == 25


def test_still_governs_while_melon_is_growing():
    governs = make_opening_controller()
    targets = {}
    governs(make_obs(day=0), targets, NW_POSITIONS)
    melon_position = next(
        position for position, target in targets.items() if target[0] == "MELON"
    )
    obs = make_obs(
        day=5,
        tiles={melon_position: melon_tile(planted_day=0, yield_units=0)},
    )
    assert governs(obs, targets, NW_POSITIONS) is True


def test_hands_off_when_first_extra_quadrant_unlocks():
    governs = make_opening_controller()
    targets = {}
    governs(make_obs(day=0), targets, NW_POSITIONS)
    obs = make_obs(day=7)
    obs["farms"][0]["unlocked_quadrants"] = ["NW", "NE"]
    assert governs(obs, targets, NW_POSITIONS) is False


def test_land_purchase_handoff_is_permanent():
    governs = make_opening_controller()
    targets = {}

    governs(make_obs(day=0), targets, NW_POSITIONS)
    bought = make_obs(day=7)
    bought["farms"][0]["unlocked_quadrants"] = ["NW", "NE"]
    assert governs(bought, targets, NW_POSITIONS) is False
    assert governs(make_obs(day=8), targets, NW_POSITIONS) is False


def test_disappearing_melon_does_not_end_six_day_opening_early():
    governs = make_opening_controller()
    targets = {}
    governs(make_obs(day=0), targets, NW_POSITIONS)
    melon_position = next(
        position for position, target in targets.items() if target[0] == "MELON"
    )
    assert (
        governs(
            make_obs(day=1, tiles={melon_position: melon_tile(0, 1)}),
            targets,
            NW_POSITIONS,
        )
        is True
    )
    assert governs(make_obs(day=2), targets, NW_POSITIONS) is True


def test_unfunded_empty_melon_target_does_not_end_opening_early():
    governs = make_opening_controller()
    targets = {}
    governs(make_obs(day=0), targets, NW_POSITIONS)
    assert governs(make_obs(day=1), targets, NW_POSITIONS) is True


def test_converts_two_wheat_targets_to_one_cow_and_one_sheep_together_on_day_two():
    governs = make_opening_controller()
    targets = {}
    governs(make_obs(day=0), targets, NW_POSITIONS)
    wheat_positions = [
        position for position, target in targets.items() if target[0] == "WHEAT"
    ]
    melon_position = next(
        position for position, target in targets.items() if target[0] == "MELON"
    )
    closest_wheat = sorted(
        wheat_positions,
        key=lambda position: (
            abs(position[0] - 4) + abs(position[1] - 4),
            -position[1],
            -position[0],
        ),
    )
    growing = {
        melon_position: melon_tile(0, 1),
        closest_wheat[0]: wheat_tile(0),
        closest_wheat[1]: wheat_tile(0),
    }

    assert governs(make_obs(1, growing), targets, NW_POSITIONS) is True
    assert Counter(value[0] for value in targets.values()) == Counter(OPENING_COUNTS)

    growing[melon_position] = melon_tile(0, 1)
    assert governs(make_obs(2, growing), targets, NW_POSITIONS) is True
    assert targets[closest_wheat[0]][0] == "COW"
    assert targets[closest_wheat[1]][0] == "SHEEP"
    assert Counter(value[0] for value in targets.values()) == Counter(
        {"MELON": 12, "WHEAT": 7, "COW": 3, "SHEEP": 3}
    )

    growing[melon_position] = melon_tile(0, 1)
    assert governs(make_obs(3, growing), targets, NW_POSITIONS) is True
    assert targets[closest_wheat[0]][0] == "COW"
    assert targets[closest_wheat[1]][0] == "SHEEP"


def test_conversion_waits_until_a_real_wheat_crop_can_be_harvested():
    governs = make_opening_controller()
    targets = {}
    governs(make_obs(day=0), targets, NW_POSITIONS)
    melon_position = next(
        position for position, target in targets.items() if target[0] == "MELON"
    )

    assert (
        governs(
            make_obs(day=1, tiles={melon_position: melon_tile(0, 1)}),
            targets,
            NW_POSITIONS,
        )
        is True
    )
    assert Counter(value[0] for value in targets.values()) == Counter(OPENING_COUNTS)
