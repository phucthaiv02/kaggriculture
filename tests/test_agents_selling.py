"""Isolated tests for agents/selling.py."""

from agents.selling import sell_orders


def make_obs(shed):
    return {"private": {"shed": shed}}


def test_never_sells_wheat_but_sells_other_inventory_by_default():
    obs = make_obs({"WHEAT": 5, "WOOL": 2})
    orders = sell_orders(obs, reserved={})
    assert not any(order[1] == "WHEAT" for order in orders)
    assert ["SELL", "WOOL", 2] in orders


def test_wheat_is_never_sold_even_above_reserved_amount():
    obs = make_obs({"WHEAT": 5})
    orders = sell_orders(obs, reserved={"WHEAT": 2})
    assert orders == []


def test_reservation_covering_the_whole_shed_sells_nothing():
    obs = make_obs({"WHEAT": 2})
    orders = sell_orders(obs, reserved={"WHEAT": 5})
    assert orders == []


def test_zero_shed_amount_produces_no_order():
    obs = make_obs({"WHEAT": 0, "WOOL": 3})
    orders = sell_orders(obs, reserved={})
    assert orders == [["SELL", "WOOL", 3]]


def test_ignores_non_sellable_items_like_seeds_or_animals():
    obs = make_obs({"WHEAT": 1, "SHEEP": 3})
    orders = sell_orders(obs, reserved={})
    assert orders == []


if __name__ == "__main__":
    import sys
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"{'ALL PASSED' if not failures else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)
