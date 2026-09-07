"""Isolated tests for agents/selling.py."""

from agents.selling import sell_orders


def make_obs(shed):
    return {"private": {"shed": shed}}


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
