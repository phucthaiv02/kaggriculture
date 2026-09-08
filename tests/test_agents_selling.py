"""Isolated tests for agents/selling.py."""

from agents.selling import sell_orders


def make_obs(shed, hour=None):
    obs = {"private": {"shed": shed}}
    if hour is not None:
        obs["hour"] = hour
    return obs


def test_sells_surplus_wheat_and_other_inventory():
    obs = make_obs({"WHEAT": 5, "WOOL": 2})
    orders = sell_orders(obs, reserved={})
    assert ["SELL", "WHEAT", 5] in orders
    assert ["SELL", "WOOL", 2] in orders


def test_wheat_reserves_feed_and_sells_only_surplus():
    obs = make_obs({"WHEAT": 5})
    orders = sell_orders(obs, reserved={"WHEAT": 2})
    assert orders == [["SELL", "WHEAT", 3]]


def test_reservation_covering_the_whole_shed_sells_nothing():
    obs = make_obs({"WHEAT": 2})
    orders = sell_orders(obs, reserved={"WHEAT": 5})
    assert orders == []


def test_zero_shed_amount_produces_no_order():
    obs = make_obs({"WHEAT": 0, "WOOL": 3})
    orders = sell_orders(obs, reserved={})
    assert orders == [["SELL", "WOOL", 3]]


def test_ignores_non_sellable_items_like_seeds_or_animals():
    obs = make_obs({"SHEEP": 3})
    orders = sell_orders(obs, reserved={})
    assert orders == []


def test_surplus_sales_keep_two_days_of_animal_feed():
    obs = make_obs({"WHEAT": 10})
    obs.update(
        day=4,
        player=0,
        farms=[
            {
                "tiles": [
                    [
                        {
                            "animal": "SHEEP",
                            "placed_day": 0,
                            "fed_today": False,
                        }
                    ]
                ]
            }
        ],
    )
    assert sell_orders(obs, {}) == [["SELL", "WHEAT", 8]]


def test_holds_premium_animal_products_before_late_day_window():
    obs = make_obs({"WHEAT": 5, "EGG": 2, "MILK": 3, "WOOL": 4}, hour=8)
    assert sell_orders(obs, {}) == [["SELL", "WHEAT", 5]]


def test_sells_premium_animal_products_after_final_shop_tick():
    obs = make_obs({"EGG": 2, "MILK": 3, "WOOL": 4}, hour=20)
    assert sell_orders(obs, {}) == [
        ["SELL", "EGG", 2],
        ["SELL", "MILK", 3],
        ["SELL", "WOOL", 4],
    ]


def test_sells_overnight_premium_carry_at_hour_zero():
    obs = make_obs({"MILK": 3}, hour=0)
    assert sell_orders(obs, {}) == [["SELL", "MILK", 3]]
