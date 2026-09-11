"""Sell surplus products while retaining scheduled feed and task inputs."""

from __future__ import annotations

from collections import Counter
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

from agents.maintenance import should_feed
from agents.schedules import animal_maintenance_can_still_pay


CASH_CROPS = ("WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY")
SELLABLE = ("FERTILIZER",) + CASH_CROPS + ("EGG", "MILK", "WOOL")
# WHEAT is retained until the terminal liquidation.
TIMED_SALE_CROPS = ("CARROT", "MELON", "TOMATO", "STRAWBERRY")


def update_selling_state(obs, state):
    """Track opening stock separately from arrivals, using observed quantities."""
    shed = obs["private"]["shed"]
    if state.get("day") != obs["day"]:
        state.clear()
        state.update(day=obs["day"], opening=dict(shed) if obs["hour"] == 0 else {}, ready=set())
    else:
        for item in state["opening"]:
            state["opening"][item] = min(state["opening"][item], shed.get(item, 0))
    state["ready"].update(opponent_ready_crops(obs))


def opponent_ready_crops(obs):
    ready = set()
    for player, farm in enumerate(obs.get("farms", [])):
        if player == obs["player"]:
            continue
        for row in farm["tiles"]:
            for tile in row:
                if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                    continue
                crop = tile.get("crop")
                if (crop in CROPS and tile.get("yield_units", 0) > 0
                        and obs["day"] - tile["planted_day"] >= CROPS[crop]["first_yield_day"]):
                    ready.add(crop)
    return ready


def sell_orders(obs, reserved, *, selling_state=None, needs_investment=False):
    """Sell non-WHEAT output immediately, with fertilizer ordered first."""
    reserved = dict(reserved)
    end_day = obs.get("_planning_end_day", 29)
    final_day = obs.get("day", -1) >= end_day

    # Reserve only feed that can still contribute to a harvest before the
    # terminal boundary. Position-aware COW phases mirror build_tasks so the
    # seller never dumps wheat assigned to a staggered FEED day.
    if "farms" in obs and not final_day:
        feed = 0
        day = obs["day"]
        for y, row in enumerate(obs["farms"][obs["player"]]["tiles"]):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict) or not tile.get("animal"):
                    continue
                animal = tile["animal"]
                age = day - tile["placed_day"]
                position = (x, y)
                for offset in range(4):
                    feed += int(
                        animal_maintenance_can_still_pay(
                            animal, age + offset, day + offset, end_day
                        )
                        and should_feed(animal, age + offset, position)
                        and (offset > 0 or not tile.get("fed_today"))
                    )
        reserved["WHEAT"] = max(reserved.get("WHEAT", 0), feed)

    # Unit actions execute before market orders. On the final day a worker can
    # therefore DROP carried harvest and have a SELL order from this same turn
    # consume it. Requesting shed + carried quantity is safe: the market stops
    # an order as soon as the shed runs out, and the next turn simply retries.
    carried = Counter()
    if final_day or "farms" in obs:
        for inventory in obs["private"].get("inventories", []):
            carried.update(inventory)

    shed = obs["private"]["shed"]
    orders = []
    for item in SELLABLE:
        if item == "WHEAT" and not final_day:
            continue
        quantity = shed.get(item, 0) - reserved.get(item, 0)
        if final_day or item != "WHEAT":
            quantity += carried.get(item, 0)
        if quantity > 0:
            orders.append(["SELL", item, quantity])
    return orders
