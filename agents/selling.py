"""Function 3: sell cash crops/products, but never animal-feed WHEAT.

Selling promptly is the cash-flow engine this whole pipeline is built
around (collect fertilizer -> sell it -> buy wheat -> feed, same day) --
WHEAT is a permanent strategic reserve: animals can need it before a same-day
market purchase becomes affordable or schedulable, so no harvested or bought
WHEAT is ever sold. Other inventory is held back only when a task needs it
today; that reservation is computed by the caller.
"""

from __future__ import annotations

CASH_CROPS = ("CARROT", "MELON", "TOMATO", "STRAWBERRY")
SELLABLE = CASH_CROPS + ("EGG", "MILK", "WOOL", "FERTILIZER")


def sell_orders(obs, reserved):
    """SELL non-WHEAT shed items beyond what `reserved` holds back."""
    shed = obs["private"]["shed"]
    orders = []
    for item in SELLABLE:
        quantity = shed.get(item, 0) - reserved.get(item, 0)
        if quantity > 0:
            orders.append(["SELL", item, quantity])
    return orders
