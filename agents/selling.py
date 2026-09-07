"""Sell surplus products while retaining scheduled feed and task inputs."""

from __future__ import annotations

CASH_CROPS = ("WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY")
SELLABLE = CASH_CROPS + ("EGG", "MILK", "WOOL", "FERTILIZER")


def sell_orders(obs, reserved):
    """Sell surplus, retaining two days of feed for established animals."""
    reserved = dict(reserved)
    if "farms" in obs:
        from agents.schedules import should_feed_animal
        feed = 0
        for row in obs["farms"][obs["player"]]["tiles"]:
            for tile in row:
                if not isinstance(tile, dict) or not tile.get("animal"):
                    continue
                age = obs["day"] - tile["placed_day"]
                feed += int(should_feed_animal(tile["animal"], age) and not tile.get("fed_today"))
                feed += int(should_feed_animal(tile["animal"], age + 1))
        reserved["WHEAT"] = max(reserved.get("WHEAT", 0), feed)
    shed = obs["private"]["shed"]
    orders = []
    for item in SELLABLE:
        quantity = shed.get(item, 0) - reserved.get(item, 0)
        if quantity > 0:
            orders.append(["SELL", item, quantity])
    return orders
