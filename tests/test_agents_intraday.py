"""Regression checks for turn-by-turn rolling execution."""

from collections import Counter

from agents.farm_tasks import Task, build_tasks
from agents.intraday import queue_commitments
from agents.rolling_scheduler import build_rolling_queues, planning_observation
from agents.scheduler import SHED_ACCESS, WorkerPlan
from test_agents_farm_tasks import animal_tile, make_obs


def test_planning_observation_exposes_carried_inputs_without_mutating_real_shed():
    obs = make_obs(
        day=3,
        shed={"WHEAT": 1},
        inventories=[{"WHEAT": 2, "FERTILIZER": 1}],
    )
    planned = planning_observation(obs)
    assert planned["private"]["shed"]["WHEAT"] == 3
    assert planned["private"]["shed"]["FERTILIZER"] == 1
    assert obs["private"]["shed"] == {"WHEAT": 1}


def test_carried_feed_is_consumed_in_place_instead_of_returning_to_shed():
    task = Task((4, 4), [["FEED"], ["CARE"]], needs=Counter({"WHEAT": 1}), mandatory=True)
    plans, unassigned = build_rolling_queues(
        [task],
        starts=[(4, 4)],
        inventories=[{"WHEAT": 1}],
        budgets=[8],
        shed_access=SHED_ACCESS,
        shed={},
    )
    assert not unassigned
    assert plans[0].queue[:2] == [["FEED"], ["CARE"]]
    assert not any(op[0] == "PICKUP" for op in plans[0].queue)


def test_carried_wheat_keeps_feed_task_visible_after_previous_turn_pickup():
    tile = animal_tile("SHEEP", placed_day=0)
    obs = make_obs(
        day=3,
        tiles={(4, 4): tile},
        shed={},
        inventories=[{"WHEAT": 1}],
    )
    tasks = build_tasks(planning_observation(obs), {(4, 4): ("SHEEP", False)})
    assert len(tasks) == 1
    assert tasks[0].actions == [["FEED"], ["CARE"]]
    assert tasks[0].needs == {"WHEAT": 1}


def test_rolling_route_prefers_new_same_tile_work_after_state_change():
    local = Task((4, 4), [["PLANT", "WHEAT"], ["WATER"]], mandatory=True)
    remote = Task((0, 0), [["WATER"]], mandatory=True)
    plans, unassigned = build_rolling_queues(
        [remote, local],
        starts=[(4, 4)],
        inventories=[{}],
        budgets=[20],
        shed_access=SHED_ACCESS,
        shed={},
    )
    assert not unassigned
    assert plans[0].queue[0] == ["PLANT", "WHEAT"]
    assert plans[0].queue[1] == ["WATER"]


def test_queue_commitments_describes_only_current_ephemeral_routes():
    plans = [WorkerPlan((4, 4), [["PICKUP", "WHEAT", 1], ["EAST"], ["FEED"]])]
    endpoints, occupied, seeds, supplies = queue_commitments([(4, 4)], plans)
    assert endpoints == [(5, 4)]
    assert occupied == {(5, 4)}
    assert not seeds
    assert supplies == {"WHEAT": 1}
