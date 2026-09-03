"""Isolated tests for agents/scheduler.py -- synthetic Task lists, no environment."""

from collections import Counter

from agents.farm_tasks import Task
from agents.scheduler import (
    FARMER_BUDGET,
    HAND_BUDGET,
    SHED,
    build_queues,
    hands_needed,
    predicted_hand_starts,
    route,
)


def test_route_manhattan_step_counts():
    ops = route((2, 3), (5, 1))
    assert ops.count(["EAST"]) == 3
    assert ops.count(["NORTH"]) == 2
    assert not any(op in (["WEST"], ["SOUTH"]) for op in ops)
    assert len(ops) == 5


def test_route_from_shed_to_itself_is_empty():
    assert route(SHED, SHED) == []


def test_predicted_hires_match_engine_nwse_least_occupied_spawn_rule():
    assert predicted_hand_starts((4, 4), (), 4) == [
        (5, 4), (4, 5), (5, 5), (4, 4)
    ]


def test_queue_uses_each_hands_real_spawn_instead_of_shifting_its_route():
    task = Task((4, 3), [["WATER"]], needs=Counter({"WHEAT": 1}))
    farmer_filler = Task((4, 4), [["WATER"]] * FARMER_BUDGET)
    plans, unassigned = build_queues(
        [farmer_filler, task], farmer_start=(4, 4), hand_count=1, hand_starts=((5, 4),)
    )
    assert unassigned == []
    hand_queue = plans[1].queue
    assert hand_queue[:3] == [["PICKUP", "WHEAT", 1], ["WEST"], ["NORTH"]]
    assert hand_queue[-1] == ["WATER"]


def test_hand_leaves_locked_spawn_before_accessing_shed():
    task = Task((4, 3), [["WATER"]], needs=Counter({"WHEAT": 1}))
    farmer_filler = Task((4, 4), [["WATER"]] * FARMER_BUDGET)
    plans, unassigned = build_queues(
        [farmer_filler, task],
        farmer_start=(4, 4),
        hand_count=1,
        hand_starts=((5, 4),),
        shed_access=((4, 4),),
    )
    assert unassigned == []
    assert plans[1].queue[:2] == [["WEST"], ["PICKUP", "WHEAT", 1]]


def test_tasks_are_actioned_near_to_far_even_if_far_task_is_given_first():
    far = Task((0, 0), [["HARVEST"]])
    near = Task((4, 3), [["WATER"]])

    plans, unassigned = build_queues([far, near], farmer_start=(4, 4), hand_count=0)

    assert unassigned == []
    queue = plans[0].queue
    assert queue.index(["WATER"]) < queue.index(["HARVEST"])
    assert queue[:2] == [["NORTH"], ["WATER"]]


def test_later_task_is_inserted_before_far_work_for_that_specific_worker():
    """Route order is optimized inside the assigned worker's own bucket."""
    far = Task((0, 0), [["HARVEST"]])
    middle = Task((2, 2), [["FERTILIZE"]])
    near = Task((4, 3), [["WATER"]])

    plans, unassigned = build_queues(
        [far, middle, near], farmer_start=(4, 4), hand_count=0
    )

    assert unassigned == []
    queue = plans[0].queue
    assert queue.index(["WATER"]) < queue.index(["FERTILIZE"]) < queue.index(["HARVEST"])


def test_packing_counts_pickup_steps_and_uses_spare_hand_capacity():
    """A distinct PICKUP is a real turn and must not overflow one worker."""
    first = Task(
        (4, 3),
        [["WATER"]] * 20,
        needs=Counter({"COW": 1, "WHEAT": 1}),
    )
    second = Task((4, 2), [["HARVEST"]])

    plans, unassigned = build_queues(
        [first, second], farmer_start=(4, 4), hand_count=1, hand_starts=((5, 4),)
    )

    assert unassigned == []
    assert all(
        len(plan.queue) <= budget
        for plan, budget in zip(plans, (FARMER_BUDGET, HAND_BUDGET))
    )
    assert sum(["HARVEST"] in plan.queue for plan in plans) == 1


def test_fertilizer_is_not_dropped_before_the_workers_next_task():
    fertilizer = Task(
        (4, 3),
        [["COLLECT_FERTILIZER"]],
        sells=Counter({"FERTILIZER": 1}),
    )
    later = Task((3, 3), [["WATER"]])
    plans, unassigned = build_queues(
        [fertilizer, later], farmer_start=(4, 4), hand_count=0, shed_access=((4, 4),)
    )
    assert unassigned == []
    queue = plans[0].queue
    assert queue.index(["COLLECT_FERTILIZER"]) < queue.index(["WATER"]) < queue.index(["DROP"])


def test_opening_fertilizer_is_dropped_before_the_workers_next_task():
    fertilizer = Task(
        (4, 3),
        [["COLLECT_FERTILIZER"]],
        sells=Counter({"FERTILIZER": 1}),
        immediate_drop=True,
    )
    later = Task((3, 3), [["WATER"]])
    plans, unassigned = build_queues(
        [fertilizer, later], farmer_start=(4, 4), hand_count=0, shed_access=((4, 4),)
    )
    assert unassigned == []
    queue = plans[0].queue
    assert queue.index(["COLLECT_FERTILIZER"]) < queue.index(["DROP"]) < queue.index(["WATER"])


def test_hand_hired_at_hour_one_is_limited_to_22_steps():
    """A pending hire cannot execute until hour 2, unlike an existing hand."""
    farmer_filler = Task((4, 4), [["WATER"]] * FARMER_BUDGET)
    pending_work = [
        Task((5, 4), [["WATER"]] * 12),
        Task((5, 4), [["HARVEST"]] * 11),
    ]

    count, unassigned = hands_needed(
        [farmer_filler, *pending_work],
        farmer_start=(4, 4),
        existing_hand_starts=(),
        pending_hand_budget=22,
    )

    assert count == 2
    assert unassigned == []


def test_hands_needed_zero_for_no_tasks():
    count, dropped = hands_needed([], farmer_start=SHED)
    assert count == 0
    assert dropped == []


def test_ready_animal_harvest_precedes_other_urgent_animal_work():
    feed = Task((4, 3), [["FEED"], ["CARE"]], urgent=True)
    harvest = Task(
        (0, 0), [["HARVEST"]], urgent=True, animal_harvest=True,
    )
    plans, unassigned = build_queues(
        [feed, harvest], farmer_start=SHED, hand_count=0,
    )
    assert unassigned == []
    assert plans[0].queue.index(["HARVEST"]) < plans[0].queue.index(["FEED"])


def test_hands_needed_zero_when_farmer_alone_fits():
    tasks = [Task((4, 5), [["WATER"]]) for _ in range(3)]
    count, dropped = hands_needed(tasks, farmer_start=SHED)
    assert count == 0
    assert dropped == []


def test_hands_needed_scales_up_for_heavy_workload():
    """More work than one farmer (24 turns) can possibly fit must hire hands,
    and every worker's assigned load must respect its own budget."""
    far_positions = [(x, 0) for x in range(10)] + [(x, 9) for x in range(10)]
    tasks = [Task(pos, [["WATER"], ["FERTILIZE"], ["HARVEST"]]) for pos in far_positions]
    count, dropped = hands_needed(tasks, farmer_start=SHED)
    assert count > 0
    assert dropped == [] or all(not t.urgent for t in dropped)

    plans, unassigned = build_queues(tasks, farmer_start=SHED, hand_count=count)
    assert len(plans) == count + 1
    for plan, budget in zip(plans, [FARMER_BUDGET] + [HAND_BUDGET] * count):
        assert len(plan.queue) <= budget


def test_urgent_tasks_are_not_dropped_when_they_fit_within_max_hands():
    urgent_tasks = [Task((x, 0), [["FEED"], ["CARE"]], urgent=True) for x in range(10)]
    count, dropped = hands_needed(urgent_tasks, farmer_start=SHED)
    assert dropped == []


def test_build_queues_picks_up_needs_before_first_task():
    tasks = [Task((6, 6), [["FEED"], ["CARE"]], needs=Counter({"WHEAT": 1}))]
    plans, _ = build_queues(tasks, farmer_start=SHED, hand_count=0)
    farmer_plan = plans[0]
    pickup_index = farmer_plan.queue.index(["PICKUP", "WHEAT", 1])
    feed_index = farmer_plan.queue.index(["FEED"])
    assert pickup_index < feed_index


def test_build_queues_skips_pickup_when_no_needs():
    tasks = [Task((4, 5), [["WATER"]])]
    plans, _ = build_queues(tasks, farmer_start=SHED, hand_count=0)
    assert not any(op[0] == "PICKUP" for op in plans[0].queue)


def test_build_queues_drops_sellable_only_if_it_fits_the_budget():
    """A task far enough away that farmer has no spare turns left must not
    get a forced return trip -- that would just eat a task that could have
    been done instead, and inventory drops to the shed automatically at day
    end regardless (see kaggriculture's _end_of_day)."""
    padding = [["WATER"]] * 18  # burn most of the farmer's 24-turn budget
    far_task = Task((9, 9), padding, sells=Counter({"WHEAT": 4}))
    plans, _ = build_queues([far_task], farmer_start=SHED, hand_count=0)
    queue = plans[0].queue
    assert len(queue) <= FARMER_BUDGET
    assert queue.count(["DROP"]) == 0


def test_build_queues_drops_sellable_when_it_fits():
    task = Task((5, 5), [["HARVEST"]], sells=Counter({"WHEAT": 4}))
    plans, _ = build_queues([task], farmer_start=SHED, hand_count=0)
    assert plans[0].queue.count(["DROP"]) == 1


def test_farmer_far_from_shed_is_charged_the_approach_distance():
    """The farmer's default spawn is not the shed; hands_needed must budget
    for that trip (see _pack's module docstring) or it will systematically
    under-hire on day 0."""
    far_start = (0, 0)  # distance 8 from SHED=(4,4)
    tasks = [Task((4, 4), [["WATER"]] * 15)]  # 15 + 8 baseline = 23, exactly the budget
    count, dropped = hands_needed(tasks, farmer_start=far_start)
    assert dropped == []
    assert count == 0
    tasks_over_budget = [Task((4, 4), [["WATER"]] * 16)]  # now 1 turn over budget
    count2, dropped2 = hands_needed(tasks_over_budget, farmer_start=far_start)
    assert count2 >= 1


if __name__ == "__main__":
    import sys
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"{'ALL PASSED' if not failures else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)
