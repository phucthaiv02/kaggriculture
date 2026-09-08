from kaggle_environments.envs.kaggriculture import kaggriculture as game

from agents.forecast import MarketForecast, Production


INVENTORY = {product: game.MARKET_I0 for product in game.PRODUCTS}


def flow(*, sales=None, inputs=None, day=0):
    result = Production()
    if sales:
        result.sales[day].update(sales)
    if inputs:
        result.inputs[day].update(inputs)
    return result


def marginal(candidate, *, baseline=None, external=None):
    market = MarketForecast(
        INVENTORY,
        (),
        0,
        0,
        external=external,
    )
    return market.marginal_profit(
        baseline or Production(),
        candidate,
        0,
    )


def test_unrelated_rival_products_do_not_change_candidate_marginal():
    baseline = flow(sales={"WHEAT": 80, "MILK": 30})
    candidate = flow(sales={"CARROT": 6})
    rival = flow(sales={"FERTILIZER": 100, "STRAWBERRY": 50, "WOOL": 40})

    assert marginal(candidate, baseline=baseline, external=rival) == marginal(
        candidate,
        baseline=baseline,
    )


def test_rival_future_inputs_are_not_assumed_to_be_market_buys():
    candidate = flow(sales={"WHEAT": 40})
    rival_needs_feed = flow(inputs={"WHEAT": 200})

    # Requiring future feed does not prove the rival will buy it from market;
    # it may already own WHEAT or harvest its own. Robust planning therefore
    # does not manufacture a price boost for our WHEAT.
    assert marginal(candidate, external=rival_needs_feed) == marginal(candidate)


def test_same_product_rival_supply_still_depresses_our_sale():
    candidate = flow(sales={"STRAWBERRY": 80})
    rival = flow(sales={"STRAWBERRY": 80})

    assert marginal(candidate, external=rival) < marginal(candidate)


def test_exact_queue_value_is_retained_as_diagnostic_model():
    own = flow(sales={"CARROT": 6, "WHEAT": 40})
    rival = flow(sales={"FERTILIZER": 100, "WHEAT": 40})
    market = MarketForecast(INVENTORY, (), 0, 0, external=rival)

    # The exact convention and robust planner convention are intentionally
    # separate. Exact queue pairing remains available for engine-validation;
    # robust value removes unobservable cross-product alignment.
    assert market.value(own) != market.robust_value(own)
