from agents.v2_market import end_of_day_sell_orders


def make_obs(shed, inventories, prices=None):
    return {
        "private": {"shed": dict(shed), "inventories": list(inventories)},
        "market": {"prices": dict(prices or {})},
    }


def test_final_step_sells_only_capacity_overflow():
    obs = make_obs(
        {"MELON": 85, "WHEAT": 10},
        [{"MELON": 8}, {"FERTILIZER": 2}],
        {"MELON": 100, "WHEAT": 10, "FERTILIZER": 5},
    )

    # 95 in shed + 10 carried = 105, so only five slots must be freed.
    assert end_of_day_sell_orders(obs, capacity=100) == [["SELL", "MELON", 5]]


def test_final_step_preserves_direct_inputs_until_other_goods_are_exhausted():
    obs = make_obs(
        {"WHEAT": 10, "FERTILIZER": 10, "CARROT": 3},
        [{"MELON": 84}],
        {"CARROT": 30, "WHEAT": 10, "FERTILIZER": 5},
    )

    # Total is 107. Sell three CARROT first, then four FERTILIZER; WHEAT stays.
    assert end_of_day_sell_orders(obs, capacity=100) == [
        ["SELL", "CARROT", 3],
        ["SELL", "FERTILIZER", 4],
    ]
