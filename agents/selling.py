"""Sell surplus products while retaining scheduled feed and task inputs."""

from __future__ import annotations

from collections import Counter

from agents.schedules import animal_maintenance_can_still_pay, should_feed_animal


CASH_CROPS = ("WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY")
SELLABLE = CASH_CROPS + ("EGG", "MILK", "WOOL", "FERTILIZER")


def sell_orders(obs, reserved):
    """Sell surplus; on the final day also sell inventory being dropped now."""
    reserved = dict(reserved)
    end_day = obs.get("_planning_end_day", 29)
    final_day = obs.get("day", -1) >= end_day

    # Reserve only feed that can still contribute to a harvest before the
    # terminal boundary. This mirrors farm_tasks' end-game FEED/CARE pruning
    # instead of keeping dead WHEAT out of the market for another day.
    if "farms" in obs and not final_day:
        feed = 0
        day = obs["day"]
        for row in obs["farms"][obs["player"]]["tiles"]:
            for tile in row:
                if not isinstance(tile, dict) or not tile.get("animal"):
                    continue
                animal = tile["animal"]
                age = day - tile["placed_day"]
                feed += int(
                    animal_maintenance_can_still_pay(animal, age, day, end_day)
                    and should_feed_animal(animal, age)
                    and not tile.get("fed_today")
                )
                feed += int(
                    animal_maintenance_can_still_pay(
                        animal, age + 1, day + 1, end_day
                    )
                    and should_feed_animal(animal, age + 1)
                )
        reserved["WHEAT"] = max(reserved.get("WHEAT", 0), feed)

    # Unit actions execute before market orders. On the final day a worker can
    # therefore DROP carried harvest and have a SELL order from this same turn
    # consume it. Requesting shed + carried quantity is safe: the market stops
    # an order as soon as the shed runs out, and the next turn simply retries.
    carried = Counter()
    if final_day:
        for inventory in obs["private"].get("inventories", []):
            carried.update(inventory)

    shed = obs["private"]["shed"]
    orders = []
    for item in SELLABLE:
        quantity = shed.get(item, 0) - reserved.get(item, 0)
        if final_day:
            quantity += carried.get(item, 0)
        if quantity > 0:
            orders.append(["SELL", item, quantity])
    return orders
