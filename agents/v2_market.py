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


def end_of_day_sell_orders(obs, capacity=100, cap=MAX_MARKET_ORDERS):
    """Sell only enough shed stock to make room for carried end-of-day drops.

    Hands automatically return their carried inventory to the shed after the
    final turn, so routine mid-day DROP actions are unnecessary.  The seller
    therefore runs only on the last turn and creates exactly the currently
    visible amount of missing capacity. WHEAT and FERTILIZER are retained
    ahead of ordinary sale goods because they can be direct inputs tomorrow.
    """
    shed = obs["private"]["shed"]
    carried = Counter()
    for inventory in obs["private"].get("inventories", ()):
        carried.update(inventory)

    shed_total = sum(max(0, int(amount)) for amount in shed.values())
    carried_total = sum(max(0, int(amount)) for amount in carried.values())
    overflow = max(0, shed_total + carried_total - int(capacity))
    if not overflow:
        return []

    prices = obs.get("market", {}).get("prices", {})
    ordinary = [item for item in SELLABLE if item not in ("WHEAT", "FERTILIZER")]
    # When several goods can make the same room, realize the higher-priced
    # output first while keeping tomorrow's direct inputs until last.
    order = sorted(ordinary, key=lambda item: (-int(prices.get(item, 0)), item))
    order += ["FERTILIZER", "WHEAT"]

    orders = []
    remaining = overflow
    for item in order:
        if remaining <= 0 or len(orders) >= cap:
            break
        available = max(0, int(shed.get(item, 0)))
        if not available:
            continue
        quantity = min(available, remaining)
        orders.append(["SELL", item, quantity])
        remaining -= quantity
    return orders
