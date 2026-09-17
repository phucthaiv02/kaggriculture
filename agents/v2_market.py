"""Market planning for the v2 optimizer agent.

Worker scheduling decides labor and HIRE demand. This module only translates
that plan into batched BUY orders, places HIRE orders into the per-turn market
budget, and sells from the shed on the final step of the day.
"""

from __future__ import annotations

from collections import Counter

from agents.v2_core import ANIMALS, CROPS

SELLABLE = CROPS + ("EGG", "MILK", "WOOL", "FERTILIZER")
MAX_MARKET_ORDERS = 10


def plan_input_orders(obs, plans):
    """Return aggregated BUY orders required by finalized worker routes."""
    pickup_demand = Counter()
    seed_demand = Counter()
    for plan in plans or ():
        pickup_demand.update(plan.pickups)
        for job in plan.jobs:
            for step in job.actions:
                if step.op and step.op[0] == "PLANT":
                    seed_demand[step.op[1]] += 1

    shed = obs["private"]["shed"]
    seeds = obs["private"]["seeds"]
    orders = []

    for crop in CROPS:
        missing = max(0, seed_demand[crop] - int(seeds.get(crop, 0)))
        if missing:
            orders.append(["BUY_SEED", crop, missing])

    for animal in ANIMALS:
        missing = max(0, pickup_demand[animal] - int(shed.get(animal, 0)))
        if missing:
            orders.append(["BUY_ANIMAL", animal, missing])

    for item in ("WHEAT", "FERTILIZER"):
        missing = max(0, pickup_demand[item] - int(shed.get(item, 0)))
        if missing:
            orders.append(["BUY_PRODUCT", item, missing])

    return orders


def market_batches(input_orders, hire_count=0, buy_land=False, cap=MAX_MARKET_ORDERS):
    """Split deterministic market work into turn-sized batches."""
    orders = []
    if buy_land:
        orders.append(["BUY_LAND"])
    orders.extend(input_orders)
    orders.extend([["HIRE"] for _ in range(max(0, int(hire_count)))])
    return [orders[i:i + cap] for i in range(0, len(orders), cap)] or [[]]


def end_of_day_sell_orders(obs, cap=MAX_MARKET_ORDERS):
    """Sell shed inventory on the final turn before carried inventory returns."""
    shed = obs["private"]["shed"]
    orders = [
        ["SELL", item, int(shed.get(item, 0))]
        for item in SELLABLE
        if int(shed.get(item, 0)) > 0
    ]
    return orders[:cap]
