"""Regression checks for newly freed tiles, cash and remaining worker time."""

from agents.farm_tasks import purchase_orders
from agents.intraday import queue_commitments, schedule_open_tiles
from agents.scheduler import SHED_ACCESS, WorkerPlan
from test_agents_farm_tasks import make_obs


def test_harvester_replants_before_leaving_and_is_not_scheduled_twice():
    obs = make_obs(day=8, seeds={"WHEAT": 1})
    plans = [WorkerPlan((4, 4), [["EAST"], ["WATER"]])]
    targets = {(4, 4): ("WHEAT", False)}
    eligible, hires = schedule_open_tiles(obs, targets, plans, SHED_ACCESS)
    assert plans[0].queue == [["PLANT", "WHEAT"], ["WATER"], ["EAST"], ["WATER"]]
    assert eligible == targets
    assert hires == 0
    schedule_open_tiles(obs, targets, plans, SHED_ACCESS)
    assert plans[0].queue.count(["PLANT", "WHEAT"]) == 1


def test_harvester_waits_for_affordable_seed_instead_of_walking_away():
    obs = make_obs(day=8, money=10)
    plans = [WorkerPlan((4, 4), [["EAST"], ["WATER"]])]
    targets = {(4, 4): ("WHEAT", False)}
    eligible, hires = schedule_open_tiles(obs, targets, plans, SHED_ACCESS)
    assert plans[0].queue[:2] == [["PLANT", "WHEAT"], ["WATER"]]
    assert purchase_orders(obs, eligible, list(eligible)) == [["BUY_SEED", "WHEAT", 1]]
    assert hires == 0


def test_new_land_uses_only_the_steps_left_after_existing_queues():
    obs = make_obs(day=8, seeds={"WHEAT": 10}, hands=[[5, 4]])
    obs["hour"] = 20
    plans = [WorkerPlan((4, 4), [["WATER"]] * 4), WorkerPlan((5, 4), [["EAST"]])]
    targets = {(6, 4): ("WHEAT", False), (9, 9): ("WHEAT", False)}
    eligible, hires = schedule_open_tiles(obs, targets, plans, SHED_ACCESS)
    assert (6, 4) in eligible and (9, 9) not in eligible
    assert plans[0].queue == [["WATER"]] * 4
    assert plans[1].queue == [["EAST"], ["PLANT", "WHEAT"], ["WATER"]]
    assert hires == 0


def test_hire_adds_capacity_only_when_seed_and_hire_are_both_affordable():
    targets = {(5, 4): ("WHEAT", False)}
    for money, expected_hires in ((11, 1), (10, 0), (0, 0)):
        obs = make_obs(day=8, money=money)
        obs["hour"] = 20
        plans = [WorkerPlan((4, 4), [["WATER"]] * 4)]
        eligible, hires = schedule_open_tiles(obs, targets, plans, SHED_ACCESS, [1, 2, 4])
        assert hires == expected_hires
        assert len(plans) == 1  # A pending hand has no route until it really spawns.
        assert len(plans[0].queue) == 4
        if hires:
            assert eligible == targets


def test_no_hire_or_purchase_when_delivery_leaves_too_few_steps():
    obs = make_obs(day=8, money=1000)
    obs["hour"] = 23
    plans = [WorkerPlan((4, 4), [])]
    eligible, hires = schedule_open_tiles(obs, {(5, 4): ("WHEAT", False)}, plans, SHED_ACCESS, [1, 2])
    assert eligible == {}
    assert hires == 0
    assert plans[0].queue == []


def test_new_work_does_not_take_seed_reserved_by_existing_queue():
    obs = make_obs(day=8, money=0, seeds={"WHEAT": 1})
    plans = [WorkerPlan((4, 4), [["NORTH"], ["PLANT", "WHEAT"], ["WATER"]])]
    schedule_open_tiles(obs, {(4, 4): ("WHEAT", False)}, plans, SHED_ACCESS)
    assert plans[0].queue == [["NORTH"], ["PLANT", "WHEAT"], ["WATER"]]
    endpoints, occupied, seeds, supplies = queue_commitments([(4, 4)], plans)
    assert endpoints == [(4, 3)]
    assert occupied == {(4, 3)}
    assert seeds == {"WHEAT": 1}
    assert not supplies


def test_land_bought_midday_gets_targets_and_production_before_next_dawn():
    from kaggle_environments import make
    from agents.expansion_agent import make_agent
    from experiments.crop_schedules import pass_agent
    from experiments.play_match import configuration

    config = configuration(1)
    config.update(episodeSteps=25, startingMoney=50000)
    env = make("kaggriculture", configuration=config, debug=False)
    production = make_agent(30, seed=1)
    observations_after_purchase = []

    def buy_midday(obs):
        action = production(obs)
        if obs.day == 0 and obs.hour == 12:
            action["market"] = [["BUY_LAND"], *action["market"]][:10]
        if obs.day == 0 and obs.hour > 12:
            cells = dict(zip(production.__code__.co_freevars, production.__closure__))
            targets = cells["targets"].cell_contents
            observations_after_purchase.append(sum(x >= 5 and y < 5 for x, y in targets))
        return action

    env.run([buy_midday, pass_agent])
    assert len(env.steps) == 25  # Do not accept an early termination after an agent exception.
    assert observations_after_purchase[0] == 25
    tiles = env.steps[-1][0].observation.farms[0].tiles
    assert any(
        isinstance(tiles[y][x], dict) and (tiles[y][x].get("crop") or tiles[y][x].get("animal"))
        for y in range(5) for x in range(5, 10)
    )


def test_pending_morning_hand_reserves_its_tile_and_inputs():
    obs = make_obs(day=8, seeds={"WHEAT": 1})
    obs["hour"] = 1
    plans = [
        WorkerPlan((4, 4), []),
        WorkerPlan((5, 4), [["PLANT", "WHEAT"], ["WATER"]]),
    ]
    eligible, hires = schedule_open_tiles(obs, {(5, 4): ("WHEAT", False)}, plans, SHED_ACCESS)
    assert plans[0].queue == []
    assert plans[1].queue == [["PLANT", "WHEAT"], ["WATER"]]
    assert (5, 4) in eligible
    assert hires == 0


def test_hire_cannot_spend_cash_needed_for_placement_feed():
    obs = make_obs(day=8, money=1, shed={"SHEEP": 1}, tiles={(5, 4): {"kind": "PASTURE"}})
    obs["hour"] = 18
    plans = [WorkerPlan((4, 4), [["WATER"]] * 6)]
    _, hires = schedule_open_tiles(obs, {(5, 4): ("SHEEP", False)}, plans, SHED_ACCESS, [1, 2])
    assert hires == 0
    assert len(plans[0].queue) == 6


def test_hire_does_not_reuse_money_for_an_already_promised_seed():
    obs = make_obs(day=8, money=11)
    obs["hour"] = 20
    plans = [WorkerPlan((4, 4), [["NORTH"], ["PLANT", "WHEAT"], ["WATER"], ["SOUTH"]])]
    targets = {(4, 3): ("WHEAT", False), (5, 4): ("WHEAT", False)}
    _, hires = schedule_open_tiles(obs, targets, plans, SHED_ACCESS, [1, 2])
    assert hires == 0
