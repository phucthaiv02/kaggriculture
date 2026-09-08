"""Tests for the no-opening cadence land experiment."""

from agents.opening_book import (
    LAND_BUY_DAYS,
    LAND_FIRST_DAY,
    LAND_INTERVAL_DAYS,
    LAND_MAX_EXTRA,
    make_opening_controller,
    should_buy_land_on_schedule,
)


NW_POSITIONS = [(x, y) for y in range(5) for x in range(5)]


def farm_with(*quadrants):
    return {"unlocked_quadrants": list(quadrants)}


def test_opening_controller_immediately_hands_off_without_mutating_targets():
    governs = make_opening_controller()
    targets = {}
    obs = {"day": 0}
    assert governs(obs, targets, NW_POSITIONS) is False
    assert targets == {}


def test_land_policy_is_state_derived_three_day_cadence():
    assert LAND_FIRST_DAY == 6
    assert LAND_INTERVAL_DAYS == 3
    assert LAND_MAX_EXTRA == 2
    assert LAND_BUY_DAYS == (6, 9)

    nw = farm_with("NW")
    assert should_buy_land_on_schedule({"day": 5, "hour": 0}, nw) is False
    assert should_buy_land_on_schedule({"day": 6, "hour": 0}, nw) is True
    assert should_buy_land_on_schedule({"day": 6, "hour": 1}, nw) is False

    one_extra = farm_with("NW", "NE")
    assert should_buy_land_on_schedule({"day": 8, "hour": 0}, one_extra) is False
    assert should_buy_land_on_schedule({"day": 9, "hour": 0}, one_extra) is True

    two_extra = farm_with("NW", "NE", "SW")
    assert should_buy_land_on_schedule({"day": 9, "hour": 0}, two_extra) is False
    assert should_buy_land_on_schedule({"day": 12, "hour": 0}, two_extra) is False


def test_unfunded_purchase_retries_until_land_count_changes():
    nw = farm_with("NW")
    assert should_buy_land_on_schedule({"day": 6, "hour": 0}, nw) is True
    assert should_buy_land_on_schedule({"day": 7, "hour": 0}, nw) is True
    assert should_buy_land_on_schedule({"day": 8, "hour": 0}, nw) is True

    one_extra = farm_with("NW", "NE")
    assert should_buy_land_on_schedule({"day": 8, "hour": 0}, one_extra) is False
    assert should_buy_land_on_schedule({"day": 9, "hour": 0}, one_extra) is True
