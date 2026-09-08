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


def test_ripe_wheat_wall_packs_harvest_even_when_replant_tails_do_not_fit():
    # From (4, 4), visiting these four top-row tiles costs 8 movement steps.
    # WATER+HARVEST for all four is another 8 steps and fits comfortably in
    # one 23-turn worker day.  Treating PLANT+WATER as part of the same atomic
    # task adds 8 more steps and used to reject part of the harvest wall.
    tasks = [ripe_wheat((x, 0)) for x in range(4)]

    hands, dropped = hands_needed(tasks, (4, 4), max_hands=0)
    assert hands == 0
    assert dropped == []

    plans, unassigned = build_queues(
        tasks,
        (4, 4),
        0,
        worker_budgets=[23],
    )
    assert unassigned == []
    assert plans[0].queue.count(["HARVEST"]) == 4
    assert ["PLANT", "WHEAT"] not in plans[0].queue


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
