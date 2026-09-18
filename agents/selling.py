"""Sell surplus products while retaining scheduled feed and task inputs."""

from __future__ import annotations

from agents.products import CROPS
from agents.horizon import SEASON_END_DAY
from agents.schedules import should_feed_animal


SELLABLE = CROPS + ("EGG", "MILK", "WOOL", "FERTILIZER")


def sell_orders(obs, reserved):
    """Sell surplus while retaining feed due today and on the next day."""
    reserved = dict(reserved)
    if "farms" in obs:
        feed = 0
        for row in obs["farms"][obs["player"]]["tiles"]:
            for tile in row:
                if not isinstance(tile, dict) or not tile.get("animal"):
                    continue
                age = obs["day"] - tile["placed_day"]
                last_age = obs.get("_planning_end_day", SEASON_END_DAY) - tile["placed_day"]
                feed += int(
                    should_feed_animal(tile["animal"], age, last_age)
                    and not tile.get("fed_today")
                )
                feed += int(
                    obs["day"] < obs.get("_planning_end_day", SEASON_END_DAY)
                    and should_feed_animal(tile["animal"], age + 1, last_age)
                )
        reserved["WHEAT"] = max(reserved.get("WHEAT", 0), feed)

    shed = obs["private"]["shed"]
    orders = []
    for item in SELLABLE:
        quantity = shed.get(item, 0) - reserved.get(item, 0)
        if quantity > 0:
            orders.append(["SELL", item, quantity])
    return orders
