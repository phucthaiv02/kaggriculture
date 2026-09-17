from collections import Counter

from agents.farm_tasks import Task
from agents.v2_model import TileJob, jobs_from_tasks
from agents.v2_scheduler import _pickup_requirement, build_plans


def test_same_tile_animal_work_is_one_atomic_chain():
    tasks = [
        Task((2, 2), [["HARVEST"]], sells=Counter({"MILK": 4})),
        Task(
            (2, 2),
            [["FEED"], ["CARE"], ["COLLECT_FERTILIZER"]],
            needs=Counter({"WHEAT": 1}),
            sells=Counter({"FERTILIZER": 1}),
        ),
    ]
    jobs = jobs_from_tasks(tasks)
    assert len(jobs) == 1
    assert [action[0] for action in jobs[0].actions] == [
        "HARVEST", "COLLECT_FERTILIZER", "FEED", "CARE"
    ]
    assert jobs[0].split_after == 2


def test_future_harvest_reduces_initial_pickup_without_reordering_route():
    jobs = [
        TileJob((1, 1), (("HARVEST",),), produces=Counter({"WHEAT": 1})),
        TileJob((1, 2), (("FEED",),), needs=Counter({"WHEAT": 3})),
    ]
    assert _pickup_requirement(jobs) == Counter({"WHEAT": 2})


def test_harvest_after_feed_does_not_reduce_pickup():
    jobs = [
        TileJob((1, 2), (("FEED",),), needs=Counter({"WHEAT": 3})),
        TileJob((1, 1), (("HARVEST",),), produces=Counter({"WHEAT": 1})),
    ]
    assert _pickup_requirement(jobs) == Counter({"WHEAT": 3})


def test_scheduler_keeps_atomic_job_actions_together():
    job = TileJob(
        (0, 0),
        (("WATER",), ("HARVEST",), ("PLANT", "MELON"), ("WATER",)),
    )
    plans, unassigned = build_plans([job], [(4, 4)], [23])
    assert not unassigned
    action_names = [
        op[0]
        for op in plans[0].queue
        if op[0] in {"WATER", "HARVEST", "PLANT"}
    ]
    assert action_names == ["WATER", "HARVEST", "PLANT", "WATER"]
