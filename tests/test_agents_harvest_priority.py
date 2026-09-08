from collections import Counter

from agents.farm_tasks import Task
from agents.scheduler import build_queues, hands_needed


def ripe_wheat(position):
    return Task(
        position,
        [["WATER"], ["HARVEST"], ["PLANT", "WHEAT"], ["WATER"]],
        urgent=True,
        ends_cycle=True,
    )


def test_ripe_wheat_wall_packs_every_harvest_before_replant_tails():
    tasks = [ripe_wheat((x, 0)) for x in range(4)]

    hands, dropped = hands_needed(tasks, (4, 4), max_hands=0)
    assert hands == 0
    assert dropped == []

    plans, unassigned = build_queues(
        tasks,
        (4, 4),
        0,
        worker_budgets=[16],
    )
    assert unassigned == []
    assert plans[0].queue.count(["HARVEST"]) == 4
    assert ["PLANT", "WHEAT"] not in plans[0].queue


def test_retryable_replants_are_only_postlude_after_all_harvests():
    tasks = [ripe_wheat((4, 4)), ripe_wheat((4, 3))]
    plans, unassigned = build_queues(
        tasks,
        (4, 4),
        0,
        worker_budgets=[23],
    )
    assert unassigned == []
    queue = plans[0].queue
    harvest_indices = [index for index, op in enumerate(queue) if op == ["HARVEST"]]
    plant_indices = [index for index, op in enumerate(queue) if op == ["PLANT", "WHEAT"]]
    assert len(harvest_indices) == 2
    assert plant_indices
    assert max(harvest_indices) < min(plant_indices)


def test_safe_single_replant_tail_uses_spare_budget_after_harvest():
    plans, unassigned = build_queues(
        [ripe_wheat((4, 4))],
        (4, 4),
        0,
        worker_budgets=[4],
    )
    assert unassigned == []
    assert plans[0].queue == [
        ["WATER"],
        ["HARVEST"],
        ["PLANT", "WHEAT"],
        ["WATER"],
    ]


def test_post_harvest_replacement_inputs_do_not_block_harvest_prefix():
    task = Task(
        (4, 4),
        [
            ["WATER"],
            ["HARVEST"],
            ["BUILD_PASTURE"],
            ["PLACE", "COW"],
            ["FEED"],
            ["CARE"],
        ],
        needs=Counter({"COW": 1, "WHEAT": 1}),
        urgent=True,
        ends_cycle=True,
    )

    plans, unassigned = build_queues(
        [task],
        (4, 4),
        0,
        worker_budgets=[2],
    )
    assert unassigned == []
    assert plans[0].queue == [["WATER"], ["HARVEST"]]


def test_non_finished_work_keeps_full_task_actions():
    task = Task((4, 4), [["PLANT", "WHEAT"], ["WATER"]])
    plans, unassigned = build_queues(
        [task],
        (4, 4),
        0,
        worker_budgets=[2],
    )
    assert unassigned == []
    assert plans[0].queue == [["PLANT", "WHEAT"], ["WATER"]]
