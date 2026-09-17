from collections import Counter

from agents.v2_core import JobAction, TileJob
from agents.v2_scheduler import (
    WorkerSpec,
    build_worker_plan,
    optimize_routes,
    required_initial_inventory,
    spawn_positions,
)


def step(op, consume=None, produce=None):
    return JobAction.simple(op, consume=consume, produce=produce)


def test_harvest_before_feed_reduces_only_the_needed_pickup():
    harvest = TileJob(
        (4, 4),
        (step(["HARVEST"], produce={"WHEAT": 1}),),
        "harvest",
    )
    feed = TileJob(
        (4, 4),
        (step(["FEED"], consume={"WHEAT": 1}),),
        "feed",
    )

    assert required_initial_inventory([harvest, feed]) == Counter()
    assert required_initial_inventory([feed, harvest]) == Counter({"WHEAT": 1})


def test_worker_batches_same_item_into_one_initial_pickup():
    left = TileJob(
        (3, 4),
        (step(["FEED"], consume={"WHEAT": 1}),),
        "left",
    )
    right = TileJob(
        (6, 4),
        (step(["FEED"], consume={"WHEAT": 1}),),
        "right",
    )
    plan = build_worker_plan(WorkerSpec((4, 4), 23, 1), [left, right])

    pickups = [op for op in plan.queue if op[0] == "PICKUP"]
    assert pickups == [["PICKUP", "WHEAT", 2]]


def test_tail_split_obeys_release_step_without_global_dag():
    actions = (
        step(["WATER"]),
        step(["HARVEST"]),
        step(["PLANT", "MELON"]),
        step(["WATER"]),
    )
    job = TileJob((4, 4), actions, "melon-cycle", split_index=2)
    specs = [
        WorkerSpec((4, 4), 2, 1),
        WorkerSpec((4, 4), 2, 3),
    ]

    plans = optimize_routes([job], specs, iterations=1, allow_tail_split=True)
    assert plans is not None

    windows = {}
    for plan in plans:
        windows.update(plan.job_windows)
    assert windows["melon-cycle:suffix"][0] >= windows["melon-cycle:prefix"][1]


def test_hire_spawn_uses_post_move_occupancy():
    # Before movement the farmer occupies NW shed access, so a stale predictor
    # would choose NE. After moving east, exact engine occupancy chooses NW.
    assert spawn_positions([(4, 4)], 1) == [(5, 4)]
    assert spawn_positions([(5, 4)], 1) == [(4, 4)]
