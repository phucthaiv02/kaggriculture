"""Sell surplus products while retaining scheduled feed and task inputs."""

from __future__ import annotations

from agents.schedules import should_feed_animal


CASH_CROPS = ("WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY")
PREMIUM_ANIMAL_PRODUCTS = ("EGG", "MILK", "WOOL")
SELLABLE = CASH_CROPS + PREMIUM_ANIMAL_PRODUCTS + ("FERTILIZER",)

# MarketForecast prices a day's output after the day's shop ticks. With the
# standard four-turn shop cadence, hour 20 is the final shop tick inside a
# 24-turn day. Holding premium animal output until then lets the execution
# policy match the economics used by the planner without predicting shops or
# adding any search/DP cost. Hour 0 liquidates anything harvested after the
# previous day's late-sale window so cash is not locked for another full day.
PREMIUM_SELL_HOUR = 20


def _sell_now(item, obs):
    if item not in PREMIUM_ANIMAL_PRODUCTS:
        return True
    hour = obs.get("hour")
    # Keep standalone helpers/tests and partial observations backward
    # compatible. Real game observations always provide hour.
    if hour is None:
        return True
    return hour == 0 or hour >= PREMIUM_SELL_HOUR


def sell_orders(obs, reserved):
    """Sell surplus while retaining feed due today and on the next day.

    EGG/MILK/WOOL are held through the intraday shop-demand ticks and sold in
    the late-day window. Other goods keep the old immediate-sale behavior so
    seed/feed/hire cash flow is otherwise unchanged.
    """
    reserved = dict(reserved)
    if "farms" in obs:
        feed = 0
        for row in obs["farms"][obs["player"]]["tiles"]:
            for tile in row:
                if not isinstance(tile, dict) or not tile.get("animal"):
                    continue
                age = obs["day"] - tile["placed_day"]
                feed += int(
                    should_feed_animal(tile["animal"], age)
                    and not tile.get("fed_today")
                )
                feed += int(should_feed_animal(tile["animal"], age + 1))
        reserved["WHEAT"] = max(reserved.get("WHEAT", 0), feed)

    shed = obs["private"]["shed"]
    orders = []
    for item in SELLABLE:
        if not _sell_now(item, obs):
            continue
        quantity = shed.get(item, 0) - reserved.get(item, 0)
        if quantity > 0:
            orders.append(["SELL", item, quantity])
    return orders
