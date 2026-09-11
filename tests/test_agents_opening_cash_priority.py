"""Opening days are calendar days 1-7, with cash deliveries before field work."""
from collections import Counter

import pytest

from agents.farm_tasks import Task, build_tasks
from agents.intraday import schedule_priority_drops
from agents.opening_book import PLANNER_HANDOFF_DAY
from agents.scheduler import WorkerPlan, build_queues
from agents.selling import sell_orders
from test_agents_farm_tasks import make_obs, animal_tile, plant


@pytest.mark.parametrize("day", range(7))
def test_opening_banks_fertilizer_and_animal_output_before_feed(day):
    obs = make_obs(day=day, shed={"WHEAT": 1}, tiles={
        (4, 4): animal_tile("SHEEP", day - 3, yield_units=2, fertilizer_available=True),
    })
    obs["_opening_cash_first"] = 0 <= day <= PLANNER_HANDOFF_DAY
    tasks = build_tasks(obs, {(4, 4): ("SHEEP", False)}, prioritize_fertilizer_drop=True)
    plans, rejected = build_queues(tasks, (4, 4), 0, worker_budgets=[23])
    assert not rejected
    queue = plans[0].queue
    assert queue[:3] == [["COLLECT_FERTILIZER"], ["HARVEST"], ["DROP"]]
    assert queue.index(["FEED"]) > queue.index(["DROP"])
    obs["private"]["inventories"] = [{"FERTILIZER": 1, "WOOL": 2, "WHEAT": 1}]
    assert sell_orders(obs, {}) == [["SELL", "FERTILIZER", 1], ["SELL", "WOOL", 2]]


def test_fertilizer_precedes_nearer_cash_harvest():
    tasks = [
        Task((4, 4), [["HARVEST"]], sells=Counter(MILK=2), urgent=True,
             immediate_drop=True, must_liquidate=True, cash_priority=1),
        Task((3, 4), [["COLLECT_FERTILIZER"]], sells=Counter(FERTILIZER=1), urgent=True,
             immediate_drop=True, must_liquidate=True, cash_priority=0),
    ]
    plans, rejected = build_queues(tasks, (4, 4), 0, worker_budgets=[23])
    assert not rejected
    queue = plans[0].queue
    assert queue.index(["COLLECT_FERTILIZER"]) < queue.index(["DROP"]) < queue.index(["HARVEST"])


def test_cash_crop_is_dropped_before_its_replacement_is_planted():
    obs = make_obs(day=6, tiles={(4, 4): plant("CARROT", 3, 6, 4)}, seeds={"CARROT": 1})
    obs["_opening_cash_first"] = True
    tasks = build_tasks(obs, {(4, 4): ("CARROT", False)})
    plans, rejected = build_queues(tasks, (4, 4), 0, worker_budgets=[23])
    assert not rejected
    assert plans[0].queue[:3] == [["WATER"], ["HARVEST"], ["DROP"]]
    assert ["PLANT", "CARROT"] not in plans[0].queue


def test_carried_cash_goes_home_before_continuing_and_restores_feed():
    obs = make_obs(day=6, inventories=[{"MILK": 2, "WHEAT": 1}])
    obs["_opening_cash_first"] = True
    obs["farms"][0]["farmer"] = [3, 4]
    plans = [WorkerPlan((3, 4), [["NORTH"], ["WATER"]])]
    schedule_priority_drops(obs, plans, ((4, 4),))
    expected = [["EAST"], ["DROP"], ["PICKUP", "WHEAT", 1], ["WEST"], ["NORTH"], ["WATER"]]
    assert plans[0].queue == expected
    schedule_priority_drops(obs, plans, ((4, 4),))
    assert plans[0].queue == expected


def test_wheat_alone_does_not_interrupt_maintenance():
    obs = make_obs(day=6, inventories=[{"WHEAT": 2}])
    obs["_opening_cash_first"] = True
    plans = [WorkerPlan((4, 4), [["NORTH"], ["FEED"]])]
    schedule_priority_drops(obs, plans, ((4, 4),))
    assert plans[0].queue == [["NORTH"], ["FEED"]]
