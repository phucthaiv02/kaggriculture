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


def test_equal_scheduled_work_is_routed_by_geometry_not_action_kind():
    """Execution routing must not invent a HARVEST-before-FEED runtime priority."""
    feed = Task((4, 3), [["FEED"], ["CARE"]], urgent=True)
    harvest = Task(
        (0, 0), [["HARVEST"]], urgent=True, animal_harvest=True,
    )
    plans, unassigned = build_queues(
        [feed, harvest], farmer_start=SHED, hand_count=0,
    )
    assert unassigned == []
    assert plans[0].queue.index(["FEED"]) < plans[0].queue.index(["HARVEST"])


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
    padding = [["WATER"]] * 18
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
    far_start = (0, 0)
    tasks = [Task((4, 4), [["WATER"]] * 15)]
    count, dropped = hands_needed(tasks, farmer_start=far_start)
    assert dropped == []
    assert count == 0
    tasks_over_budget = [Task((4, 4), [["WATER"]] * 16)]
    count2, dropped2 = hands_needed(tasks_over_budget, farmer_start=far_start)
    assert count2 >= 1


def test_nearby_tasks_share_worker_to_reduce_total_travel():
    tasks = [
        Task((4, 3), [["WATER"]] * 3),
        Task((4, 2), [["HARVEST"]] * 3),
    ]
    plans, unassigned = build_queues(
        tasks, farmer_start=SHED, hand_count=1, hand_starts=((5, 4),),
        existing_hand_budget=8,
    )
    assert unassigned == []
    assert sum(len(plan.queue) for plan in plans) == 8
    assert plans[1].queue == []


def test_supplies_for_later_work_stay_in_shed_until_after_drop():
    fertilizer = Task(
        (4, 3), [["COLLECT_FERTILIZER"]],
        sells=Counter({"FERTILIZER": 1}), immediate_drop=True,
    )
    placement = Task(
        (4, 2), [["PLACE", "COW"]], needs=Counter({"COW": 1}),
    )
    plans, unassigned = build_queues(
        [fertilizer, placement], farmer_start=SHED, hand_count=0,
        existing_hand_budget=8,
    )
    assert unassigned == []
    queue = plans[0].queue
    assert len(queue) == 8
    assert queue.count(["PICKUP", "COW", 1]) == 1
    assert queue.index(["DROP"]) < queue.index(["PICKUP", "COW", 1])


def test_multiple_drops_pick_up_only_each_segments_feed():
    tasks = [
        Task((4, 3), [["FEED"]], needs=Counter({"WHEAT": 1}),
             urgent=True, immediate_drop=True),
        Task((4, 2), [["FEED"]], needs=Counter({"WHEAT": 2}),
             urgent=True, immediate_drop=True),
        Task((4, 1), [["FEED"]], needs=Counter({"WHEAT": 3}), urgent=True),
    ]
    plans, unassigned = build_queues(tasks, farmer_start=SHED, hand_count=0)
    assert unassigned == []
    queue = plans[0].queue
    pickups = [op for op in queue if op[0] == "PICKUP"]
    assert sorted(op[2] for op in pickups) == [1, 2, 3]
    assert pickups[-1] == ["PICKUP", "WHEAT", 3]
    assert queue.count(["DROP"]) == 2
    assert len(plans[0].queue) <= FARMER_BUDGET


def test_scheduled_tasks_may_share_worker_when_route_is_cheaper():
    """The compatibility `urgent` flag must not force runtime worker splitting."""
    tasks = [
        Task((4, 3), [["FEED"], ["CARE"]], urgent=True),
        Task((4, 2), [["FEED"], ["CARE"]], urgent=True),
    ]
    plans, unassigned = build_queues(
        tasks, farmer_start=SHED, hand_count=1, hand_starts=((5, 4),),
    )
    assert unassigned == []
    assert sum(plan.queue.count(["FEED"]) for plan in plans) == 2
    assert sum(plan.queue.count(["CARE"]) for plan in plans) == 2
    assert sum(bool(plan.queue) for plan in plans) == 1


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


def test_length_projection_matches_executable_queues_with_drops_and_refinancing():
    from random import Random
    from agents.scheduler import _task_length, _task_queue, nearest_shed, SHED_ACCESS

    rng = Random(713)
    for _ in range(100):
        access = SHED_ACCESS[:rng.randint(1, 4)]
        start = (rng.randrange(10), rng.randrange(10))
        tasks = [
            Task(
                (rng.randrange(10), rng.randrange(10)),
                [["HARVEST"]] * rng.randint(1, 4),
                needs=Counter({"WHEAT": rng.randrange(3), "COW": rng.randrange(2)}),
                immediate_drop=bool(rng.randrange(2)),
                refinance_feed=bool(rng.randrange(2)),
            )
            for _ in range(rng.randrange(8))
        ]
        assert _task_length(start, tasks, lambda p: nearest_shed(p, access)) == len(
            _task_queue(start, tasks, access)[0]
        )


def test_packing_finds_feasible_assignment_without_extra_hand():
    """Packing outcome matters; no test should depend on a legacy rescue pass firing."""
    from agents.scheduler import _pack
    tasks = [Task(position, [["WATER"]] * count, urgent=True)
             for position, count in [((1, 1), 3), ((2, 4), 4), ((3, 4), 2),
                                     ((4, 2), 3), ((0, 4), 2), ((0, 0), 4), ((2, 4), 4)]]
    starts = [(4, 4), (5, 4)]
    buckets, missing = _pack(tasks, starts, [23, 23])
    assert not missing
    assert Counter(id(task) for bucket in buckets for task in bucket) == Counter(map(id, tasks))
    count, missing = hands_needed(tasks, (4, 4), max_hands=1)
    assert (count, missing) == (1, [])
    plans, missing = build_queues(tasks, (4, 4), count)
    assert not missing
    assert all(len(plan.queue) <= 23 for plan in plans)
    assert sum(plan.queue.count(["WATER"]) for plan in plans) == 22


def test_packing_preserves_nw_opening_assignments():
    from agents.scheduler import _pack_greedy, _pack
    tasks = [Task((x, y), [["WATER"]] * 3, urgent=True)
             for x, y in [(0, 0), (4, 4), (1, 0), (2, 1), (3, 4), (4, 0)]]
    args = (tasks, [(4, 4), (5, 4)], [23, 23], ((4, 4),))
    assert _pack(*args) == _pack_greedy(*args)


def _feed_pair():
    return [Task(position, [["FEED"], ["CARE"], ["COLLECT_FERTILIZER"]],
                 needs=Counter(WHEAT=1), sells=Counter(FERTILIZER=1), urgent=True)
            for position in ((4, 3), (5, 3))]


def test_stocked_feed_routes_share_one_bulk_pickup():
    plans, missing = build_queues(_feed_pair(), (4, 4), 1, available_wheat=2)
    assert missing == []
    queues = [plan.queue for plan in plans if plan.queue]
    assert len(queues) == 1
    assert [op for op in queues[0] if op[0] == "PICKUP"] == [["PICKUP", "WHEAT", 2]]
    assert queues[0].count(["FEED"]) == 2
    assert queues[0].count(["CARE"]) == 2
    assert queues[0][-1] == ["DROP"]
    assert len(queues[0]) < 14


def test_feed_consolidation_requires_stock_and_preserves_opening():
    tasks = _feed_pair()
    normal, _ = build_queues(tasks, (4, 4), 1)
    scarce, _ = build_queues(tasks, (4, 4), 1, available_wheat=1)
    assert [p.queue for p in scarce] == [p.queue for p in normal]
    opening, _ = build_queues(tasks, (4, 4), 1, shed_access=((4, 4),))
    stocked, _ = build_queues(tasks, (4, 4), 1, shed_access=((4, 4),), available_wheat=2)
    assert [p.queue for p in stocked] == [p.queue for p in opening]


def test_feed_consolidation_respects_return_and_worker_budget():
    plans, missing = build_queues(_feed_pair(), (4, 4), 1,
                                 available_wheat=2, worker_budgets=[8, 8])
    assert missing == []
    assert all(plan.queue.count(["FEED"]) == 1 for plan in plans)
    assert all(len(plan.queue) <= 8 for plan in plans)


def test_feed_delivery_packing_preserves_all_work_and_shared_pickups():
    positions = [(4, 3), (5, 3), (6, 3), (4, 2), (5, 2)]
    tasks = [Task(p, [["FEED"], ["CARE"], ["COLLECT_FERTILIZER"]],
                  needs=Counter(WHEAT=1), sells=Counter(FERTILIZER=1), urgent=True)
             for p in positions]
    plans, missing = build_queues(tasks, (4, 4), 4, available_wheat=5)
    assert missing == []
    assert sum(plan.queue.count(["FEED"]) for plan in plans) == 5
    assert sum(plan.queue.count(["CARE"]) for plan in plans) == 5
    pickups = [op for plan in plans for op in plan.queue if op[0] == "PICKUP"]
    assert sum(op[2] for op in pickups) == 5
    assert len(pickups) <= sum(bool(plan.queue) for plan in plans)
    assert all(len(plan.queue) <= HAND_BUDGET for plan in plans)


def test_feed_rebalancing_keeps_investment_on_its_original_worker():
    from agents.scheduler import _rebalance_feed
    feed1, feed2 = _feed_pair()
    plant = Task((5, 4), [["PLANT", "CARROT"], ["WATER"]])
    buckets = [[feed1, plant], [feed2]]
    result = _rebalance_feed(buckets, [(4, 4), (5, 4)], [23, 23],
                             ((4, 4), (5, 4), (4, 5), (5, 5)), [feed1, feed2, plant])
    assert plant in result[0] and plant not in result[1]
    assert sorted(id(t) for b in result for t in b) == sorted(map(id, [feed1, feed2, plant]))
