from agents.opening_book import (
    LAND_BUY_UTILIZATION,
    LAND_CASH_BUFFER,
    LAND_MIN_REMAINING_DAYS,
    LAND_PRICES,
    should_buy_land_on_schedule,
)


def make_farm(active_tiles, occupied, unlocked_quadrants, money):
    flat = [None] * active_tiles + ["LOCKED"] * (100 - active_tiles)
    for index in range(occupied):
        flat[index] = {
            "kind": "PLANT",
            "crop": "WHEAT",
            "planted_day": 0,
        }
    return {
        "tiles": [flat[row * 10:(row + 1) * 10] for row in range(10)],
        "unlocked_quadrants": list(unlocked_quadrants),
        "money": money,
    }


def obs(day, end_day=29, hour=0):
    return {
        "day": day,
        "hour": hour,
        "_planning_end_day": end_day,
    }


def test_first_land_keeps_verified_day_seven_schedule():
    farm = make_farm(25, 25, ["NW"], money=0)
    assert should_buy_land_on_schedule(obs(6), farm) is False
    assert should_buy_land_on_schedule(obs(7), farm) is True
    assert should_buy_land_on_schedule(obs(7, hour=1), farm) is False


def test_second_land_uses_real_utilization_and_cash_buffer():
    # 38 / 50 = 76%, just above the 75% production threshold.
    price = LAND_PRICES[1]
    farm = make_farm(
        50,
        38,
        ["NW", "NE"],
        money=price + LAND_CASH_BUFFER,
    )
    assert LAND_BUY_UTILIZATION == 0.75
    assert should_buy_land_on_schedule(obs(8), farm) is True


def test_targets_without_real_producers_do_not_unlock_more_land():
    # 37 / 50 = 74%; desired targets elsewhere are intentionally irrelevant.
    farm = make_farm(50, 37, ["NW", "NE"], money=100000)
    assert should_buy_land_on_schedule(obs(8), farm) is False


def test_dynamic_land_keeps_cash_buffer():
    price = LAND_PRICES[1]
    farm = make_farm(
        50,
        50,
        ["NW", "NE"],
        money=price + LAND_CASH_BUFFER - 1,
    )
    assert should_buy_land_on_schedule(obs(8), farm) is False


def test_dynamic_land_stops_when_too_little_season_remains():
    farm = make_farm(50, 50, ["NW", "NE"], money=100000)
    assert LAND_MIN_REMAINING_DAYS == 10
    assert should_buy_land_on_schedule(obs(20, end_day=29), farm) is False
    assert should_buy_land_on_schedule(obs(19, end_day=29), farm) is True


def test_third_land_can_unlock_after_second_field_fills():
    price = LAND_PRICES[2]
    farm = make_farm(
        75,
        57,
        ["NW", "NE", "SW"],
        money=price + LAND_CASH_BUFFER,
    )
    assert should_buy_land_on_schedule(obs(10), farm) is True


def test_no_order_after_every_quadrant_is_owned():
    farm = make_farm(
        100,
        100,
        ["NW", "NE", "SW", "SE"],
        money=100000,
    )
    assert should_buy_land_on_schedule(obs(10), farm) is False
