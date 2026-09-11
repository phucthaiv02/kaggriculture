"""Isolated tests for agents/selling.py."""

from agents.selling import sell_orders, update_selling_state


def make_obs(shed):
    return {"private": {"shed": shed}}


def test_sells_surplus_wheat_and_other_inventory():
    obs = make_obs({"WHEAT": 5, "WOOL": 2})
    orders = sell_orders(obs, reserved={})
    assert ["SELL", "WHEAT", 5] in orders
    assert ["SELL", "WOOL", 2] in orders


def test_wheat_reserves_feed_and_sells_only_surplus():
    obs = make_obs({"WHEAT": 5})
    orders = sell_orders(obs, reserved={"WHEAT": 2})
    assert orders == [["SELL", "WHEAT", 3]]


def test_reservation_covering_the_whole_shed_sells_nothing():
    obs = make_obs({"WHEAT": 2})
    orders = sell_orders(obs, reserved={"WHEAT": 5})
    assert orders == []


def test_zero_shed_amount_produces_no_order():
    obs = make_obs({"WHEAT": 0, "WOOL": 3})
    orders = sell_orders(obs, reserved={})
    assert orders == [["SELL", "WOOL", 3]]


def test_ignores_non_sellable_items_like_seeds_or_animals():
    obs = make_obs({"SHEEP": 3})
    orders = sell_orders(obs, reserved={})
    assert orders == []


def test_surplus_sales_keep_four_days_of_animal_feed():
    obs = make_obs({"WHEAT": 10})
    obs.update(
        day=4,
        player=0,
        farms=[
            {
                "tiles": [
                    [
                        {
                            "animal": "SHEEP",
                            "placed_day": 0,
                            "fed_today": False,
                        }
                    ]
                ]
            }
        ],
    )
    assert sell_orders(obs, {}) == [["SELL", "WHEAT", 6]]


def competitive_obs(tile=None, hour=0):
    return dict(day=4, hour=hour, player=0,
                private={"shed": {"CARROT": 5}},
                farms=[{"tiles": [[]]}, {"tiles": [[tile]]}])


def test_wheat_sells_surplus_independently_of_competition_time_and_investment():
    for tile in (None, dict(kind="PLANT", crop="WHEAT", planted_day=3, yield_units=0),
                 dict(kind="PLANT", crop="WHEAT", planted_day=0, yield_units=4)):
        obs = competitive_obs(tile)
        obs["private"]["shed"] = {"WHEAT": 5}
        state = {}
        update_selling_state(obs, state)
        for hour in (0, 1):
            obs["hour"] = hour
            update_selling_state(obs, state)
            for needs_investment in (False, True):
                assert sell_orders(
                    obs, {"WHEAT": 2}, selling_state=state,
                    needs_investment=needs_investment,
                ) == [["SELL", "WHEAT", 3]]


def test_holds_opening_crop_without_competition_or_with_immature_crop():
    for tile in (None, dict(kind="PLANT", crop="CARROT", planted_day=3, yield_units=2),
                 dict(kind="PLANT", crop="CARROT", planted_day=0, yield_units=0)):
        assert sell_orders(competitive_obs(tile), {}) == []


def test_sells_on_opponent_harvest_day_only_for_matching_crop():
    obs = competitive_obs(dict(kind="PLANT", crop="CARROT", planted_day=2, yield_units=1))
    obs["private"]["shed"]["MELON"] = 3
    assert sell_orders(obs, {}) == [["SELL", "CARROT", 5]]


def test_investment_releases_stock_but_respects_reservations():
    assert sell_orders(competitive_obs(), {"CARROT": 2}, needs_investment=True) == [["SELL", "CARROT", 3]]


def test_midday_arrivals_sell_without_releasing_opening_stock():
    obs, state = competitive_obs(), {}
    update_selling_state(obs, state)
    obs["hour"] = 1
    update_selling_state(obs, state)
    assert sell_orders(obs, {}, selling_state=state) == []
    obs["private"]["shed"]["CARROT"] = 8
    update_selling_state(obs, state)
    assert sell_orders(obs, {}, selling_state=state) == [["SELL", "CARROT", 3]]
    obs["private"]["shed"]["CARROT"] = 5
    obs["hour"] = 2
    update_selling_state(obs, state)
    assert sell_orders(obs, {}, selling_state=state) == []


def test_readiness_latched_until_next_day_for_deferred_sales():
    obs = competitive_obs(dict(kind="PLANT", crop="CARROT", planted_day=0, yield_units=4))
    state = {}
    update_selling_state(obs, state)
    obs["farms"][1]["tiles"] = [[None]]
    obs["hour"] = 1
    update_selling_state(obs, state)
    assert sell_orders(obs, {}, selling_state=state) == [["SELL", "CARROT", 5]]
    obs.update(day=5, hour=0)
    update_selling_state(obs, state)
    assert sell_orders(obs, {}, selling_state=state) == []


def test_final_day_liquidates_held_and_carried_crops():
    obs = competitive_obs()
    obs["_planning_end_day"] = 4
    obs["private"]["inventories"] = [{"CARROT": 2}]
    assert sell_orders(obs, {}) == [["SELL", "CARROT", 7]]


def test_investment_sales_uses_planned_cost_and_available_cash(monkeypatch):
    from agents import expansion_agent as agent

    obs = competitive_obs()
    obs["market"] = {"inventory": {"CARROT": 0}}
    obs["farms"][0].update(money=19, hands=[], unlocked_quadrants=["NW"])
    monkeypatch.setattr(agent, "_active_positions", lambda farm: [])
    monkeypatch.setattr(agent, "purchase_orders", lambda *args, **kwargs: [["BUY_SEED", "CARROT", 1]])
    state = {}
    update_selling_state(obs, state)
    assert agent._investment_sales(obs, {}, state, {}, 0) == [["SELL", "CARROT", 5]]
    obs["farms"][0]["money"] = 20
    assert agent._investment_sales(obs, {}, state, {}, 0) == []
    monkeypatch.setattr(agent, "purchase_orders", lambda *args, **kwargs: [])
    obs["farms"][0]["money"] = 0
    assert agent._investment_sales(obs, {}, state, {}, 0) == []
