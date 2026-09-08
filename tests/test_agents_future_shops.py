from copy import deepcopy

import pytest
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from agents.forecast import MarketForecast, Production


INVENTORY = {product: game.MARKET_I0 for product in game.PRODUCTS}


def _shop_increment(shop, product):
    products = game.SHOPS[shop]
    if product not in products:
        return 0
    return 2 if len(products) == 1 else 1


@pytest.mark.parametrize("product", ["EGG", "MILK", "WOOL", "STRAWBERRY"])
def test_one_future_shop_matches_bruteforce_expected_cash(product):
    """A hidden day-3 unlock is the uniform average of all 8 real shop draws."""
    inventory = deepcopy(INVENTORY)
    flow = Production()
    rival = Production()
    flow.sales[3][product] = 40
    rival.sales[3][product] = 3

    stochastic = MarketForecast(
        inventory, (), 2, 3, external=rival
    ).robust_value(flow)

    helper = MarketForecast(inventory, (), 2, 2, external=rival)
    outcomes = []
    for shop in game.SHOPS:
        stock = inventory[product]
        # day 2: flat town center buys one unit; no shop exists yet.
        if product in game.TOWN_CENTER_PRODUCTS:
            stock -= 1
        # shop unlocks for day 3 and consumes six times at the default
        # townShopSellInterval=4, then the center buys one more unit.
        stock -= 6 * _shop_increment(shop, product)
        if product in game.TOWN_CENTER_PRODUCTS:
            stock -= 1
        cash, _ = helper._settle_product_robust(
            stock, product, flow.sales[3][product], rival.sales[3][product]
        )
        outcomes.append(cash)

    assert stochastic == pytest.approx(sum(outcomes) / len(outcomes))


def test_future_unlocks_count_instances_and_stop_at_engine_cap():
    """Duplicates are real shop instances; seven active shops leave one draw."""
    market = MarketForecast(INVENTORY, ["BAKERY"] * 7, 2, 12)
    assert market.future_shop_days == (3,)

    full = MarketForecast(INVENTORY, ["BAKERY"] * 8, 2, 12)
    assert full.future_shop_days == ()


def test_exact_value_keeps_current_town_static_but_robust_value_prices_uncertainty():
    """Engine-equivalence path stays deterministic; planner path sees unknown shops."""
    inventory = deepcopy(INVENTORY)
    inventory["EGG"] = game.MARKET_I0 - game.MARKET_PARAMS["EGG"]["T"]
    flow = Production()
    flow.sales[3]["EGG"] = 40

    market = MarketForecast(inventory, (), 2, 3)
    assert market.robust_value(flow) > market.value(flow)
