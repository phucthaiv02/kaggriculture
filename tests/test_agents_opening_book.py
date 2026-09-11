"""Isolated tests for agents/opening_book.py using hand-built observations."""

from collections import Counter

import pytest

from agents.farm_tasks import purchase_orders
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from agents.opening_book import (
    LAND_BUY_DAYS,
    MAX_SCHEDULED_LAND_PURCHASES,
    OPENING_COUNTS,
    OPENING_REFINANCE_DAY,
    PLANNER_HANDOFF_DAY,
    build_opening_targets,
    make_opening_controller,
    MELON_PLANNER_HANDOFF_DAY,
    should_buy_land_on_schedule,
)


NW_POSITIONS = [(x, y) for y in range(5) for x in range(5)]


def make_obs(day):
    board = [[None] * 10 for _ in range(10)]
    farm = {"tiles": board, "unlocked_quadrants": ["NW"]}
    return {"day": day, "player": 0, "farms": [farm, farm]}


def test_build_opening_targets_matches_counts_and_fills_the_board():
    targets = build_opening_targets(NW_POSITIONS)
    assert len(targets) == 25
    counts = Counter(value[0] for value in targets.values())
    assert dict(counts) == OPENING_COUNTS
    assert all(fertilize is False for _name, fertilize in targets.values())


def test_opening_uses_requested_portfolio():
    assert OPENING_COUNTS == {
        "WHEAT": 19,
        "COW": 2,
        "SHEEP": 2,
        "GOOSE": 2,
    }


def test_day_zero_purchase_plan_covers_every_required_feed_action():
    targets = build_opening_targets(NW_POSITIONS)
    board = [["LOCKED"] * 10 for _ in range(10)]
    for x, y in NW_POSITIONS:
        board[y][x] = None
    farm = {
        "money": 3000.0,
        "tiles": board,
        "farmer": [4, 4],
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }
    inventory = {name: game.MARKET_I0 for name in game.PRODUCTS}
    obs = {
        "day": 0,
        "hour": 0,
        "player": 0,
        "farms": [farm, farm],
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
        "market": {"inventory": inventory, "params": None},
    }
    orders = purchase_orders(obs, targets, NW_POSITIONS)
    feed = sum(
        int(order[2])
        for order in orders
        if order[0] == "BUY_PRODUCT" and order[1] == "WHEAT"
    )
    # At placement age 0, the two SHEEP and two GOOSE need FEED; COW does not.
    assert feed >= 4
    assert ["BUY_SEED", "WHEAT", 19] in orders
    assert ["BUY_ANIMAL", "GOOSE", 2] in orders
    assert ["BUY_ANIMAL", "COW", 2] in orders
    assert ["BUY_ANIMAL", "SHEEP", 2] in orders


def test_land_purchase_schedule_is_exactly_day_seven_and_day_ten():
    farm = {"unlocked_quadrants": ["NW"]}
    assert LAND_BUY_DAYS == (7, 10)
    assert MAX_SCHEDULED_LAND_PURCHASES == 2
    for day in (0, 6, 8, 9, 11, 20, 29):
        assert should_buy_land_on_schedule({"day": day, "hour": 0}, farm) is False
    assert should_buy_land_on_schedule({"day": 7, "hour": 0}, farm) is True
    assert should_buy_land_on_schedule({"day": 10, "hour": 0}, farm) is True
    assert should_buy_land_on_schedule({"day": 7, "hour": 1}, farm) is False
    assert should_buy_land_on_schedule({"day": 10, "hour": 1}, farm) is False


def test_day_ten_can_buy_second_quadrant_but_never_a_third():
    after_first_buy = {"unlocked_quadrants": ["NW", "NE"]}
    assert should_buy_land_on_schedule(
        {"day": 10, "hour": 0}, after_first_buy
    ) is True

    after_two_buys = {"unlocked_quadrants": ["NW", "NE", "SW"]}
    assert should_buy_land_on_schedule(
        {"day": 10, "hour": 0}, after_two_buys
    ) is False


def test_build_opening_targets_rejects_wrong_board_size():
    with pytest.raises(ValueError, match="exactly 25"):
        build_opening_targets(NW_POSITIONS[:-1])


def test_first_call_applies_opening_book():
    governs = make_opening_controller()
    targets = {}
    assert governs(make_obs(day=0), targets, NW_POSITIONS) is True
    assert len(targets) == 25


def test_opening_stays_active_through_fertilizer_refinance_day():
    governs = make_opening_controller()
    targets = {}
    assert OPENING_REFINANCE_DAY == 1
    assert governs(make_obs(day=0), targets, NW_POSITIONS) is True
    assert governs(make_obs(day=OPENING_REFINANCE_DAY), targets, NW_POSITIONS) is True
    assert Counter(value[0] for value in targets.values()) == Counter(OPENING_COUNTS)


def test_hands_off_to_planner_only_on_day_seven():
    governs = make_opening_controller()
    targets = {}
    governs(make_obs(day=0), targets, NW_POSITIONS)
    assert PLANNER_HANDOFF_DAY == 7
    assert governs(make_obs(day=PLANNER_HANDOFF_DAY), targets, NW_POSITIONS) is False


def test_extra_land_does_not_force_early_handoff():
    governs = make_opening_controller()
    targets = {}
    governs(make_obs(day=0), targets, NW_POSITIONS)
    obs = make_obs(day=1)
    obs["farms"][0]["unlocked_quadrants"] = ["NW", "NE"]
    assert governs(obs, targets, NW_POSITIONS) is True


def test_handoff_is_permanent():
    governs = make_opening_controller()
    targets = {}
    governs(make_obs(day=0), targets, NW_POSITIONS)
    assert governs(make_obs(day=PLANNER_HANDOFF_DAY), targets, NW_POSITIONS) is False
    assert governs(make_obs(day=1), targets, NW_POSITIONS) is False


def test_melon_v2_hands_off_before_day_five_seed_purchases():
    governs = make_opening_controller("melon_v2")
    targets = {}
    for day in range(MELON_PLANNER_HANDOFF_DAY):
        obs = make_obs(day)
        assert governs(obs, targets, NW_POSITIONS)
        counts = Counter(target[0] for target in targets.values() if target)
        assert counts == Counter({
            "WHEAT": 9 if day < 2 else 7, "MELON": 12,
            "COW": 2 if day < 2 else 3, "SHEEP": 3 if day >= 3 else 2,
        })
        if day >= 2:
            assert len(obs["_opening_early_harvest_positions"]) == 2
    assert not governs(make_obs(MELON_PLANNER_HANDOFF_DAY), targets, NW_POSITIONS)


def test_unknown_opening_version_is_rejected():
    with pytest.raises(ValueError, match="unknown opening version"):
        make_opening_controller("missing")
