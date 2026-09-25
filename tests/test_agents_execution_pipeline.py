from collections import Counter

from agents.expansion_agent import (
    _market_remainder,
    _needs_route_rebuild,
    _plans_have_work,
    _pop_market_batch,
    _should_keep_incumbent_route,
    _strip_partial_animal_builds,
)
from agents.farm_tasks import Task
from agents.rolling_scheduler import build_rolling_queues
from agents.scheduler import WorkerPlan


def test_market_batch_preserves_orders_beyond_engine_cap():
    queue = [["HIRE", index] for index in range(13)]

    first = _pop_market_batch(queue)
    second = _pop_market_batch(queue)

    assert len(first) == 10
    assert len(second) == 3
    assert first + second == [["HIRE", index] for index in range(13)]
    assert queue == []


def test_market_remainder_keeps_seed_hire_and_other_orders_beyond_engine_cap():
    orders = [["SELL", "MILK", 1] for _ in range(10)] + [
        ["BUY_SEED", "WHEAT", 3],
        ["HIRE"],
        ["BUY_ANIMAL", "COW", 1],
        ["BUY_PRODUCT", "WHEAT", 2],
    ]

    assert _market_remainder(orders) == orders[10:]


def test_post_opening_never_leaves_standalone_animal_build():
    task = Task((0, 0), [["HARVEST"], ["BUILD_PASTURE"]])

    cleaned = _strip_partial_animal_builds([task], opening_active=False)

    assert cleaned == [task]
    assert task.actions == [["HARVEST"]]


def test_post_opening_drops_pure_standalone_animal_build():
    task = Task((0, 0), [["BUILD_COOP"]])

    assert _strip_partial_animal_builds([task], opening_active=False) == []


def test_opening_keeps_bootstrap_build_behavior():
    task = Task((0, 0), [["BUILD_PASTURE"]])

    cleaned = _strip_partial_animal_builds([task], opening_active=True)

    assert cleaned == [task]
    assert task.actions == [["BUILD_PASTURE"]]


def test_ordinary_movement_and_maintenance_keep_current_route():
    assert not _needs_route_rebuild([
        ["WEST"], ["EAST"], ["WATER"], ["FEED"], ["CARE"], ["FERTILIZE"]
    ])


def test_dependency_or_tile_topology_change_requests_candidate_rebuild():
    for operation in (
        ["PICKUP", "WHEAT", 1], ["DROP"], ["HARVEST"], ["DIG"],
        ["PLANT", "WHEAT"], ["PLACE", "COW"], ["BUILD_PASTURE"],
        ["COLLECT_FERTILIZER"],
    ):
        assert _needs_route_rebuild([operation])


def test_empty_worker_plan_container_is_not_treated_as_live_schedule():
    assert not _plans_have_work([
        WorkerPlan((4, 4), []), WorkerPlan((5, 4), [])
    ])
    assert _plans_have_work([
        WorkerPlan((4, 4), []), WorkerPlan((5, 4), [["WEST"]])
    ])


def test_incomplete_rolling_candidate_keeps_valid_incumbent():
    incumbent = [WorkerPlan((1, 1), [["WEST"], ["WATER"]])]
    missing = Task((0, 1), [["WATER"]], mandatory=True)

    assert _should_keep_incumbent_route(
        incumbent, [missing], previous_invalidated=False, worker_count=1
    )
    assert not _should_keep_incumbent_route(
        incumbent, [missing], previous_invalidated=True, worker_count=1
    )
    assert not _should_keep_incumbent_route(
        [WorkerPlan((1, 1), [])], [missing],
        previous_invalidated=False, worker_count=1,
    )
    assert not _should_keep_incumbent_route(
        incumbent, [missing], previous_invalidated=False, worker_count=2
    )


def test_complete_market_candidate_without_new_work_keeps_incumbent():
    incumbent = [WorkerPlan((1, 1), [["WEST"], ["HARVEST"]])]

    assert _should_keep_incumbent_route(
        incumbent,
        [],
        previous_invalidated=False,
        worker_count=1,
        candidate_adds_work=False,
    )
    assert not _should_keep_incumbent_route(
        incumbent,
        [],
        previous_invalidated=False,
        worker_count=1,
        candidate_adds_work=True,
    )


def test_optional_rolling_loss_does_not_block_newly_materialized_candidate():
    incumbent = [WorkerPlan((1, 1), [["WEST"]])]
    optional = Task((0, 1), [["WATER"]], mandatory=False)

    assert not _should_keep_incumbent_route(
        incumbent,
        [optional],
        previous_invalidated=False,
        worker_count=1,
        candidate_adds_work=True,
    )


def test_rolling_keeps_zero_distance_admitted_successor_before_remote_feed():
    """A worker must finish the admitted chain underfoot before leaving.

    This models the observation immediately after HARVEST: the worker carries
    WHEAT, the tile now exposes its admitted PLANT->WATER successor, and a
    remote animal can also consume the carried WHEAT. Route optimization should
    take the zero-distance successor first instead of creating a needless
    leave-and-return trip.
    """
    local = Task(
        (1, 1),
        [["PLANT", "WHEAT"], ["WATER"]],
        mandatory=True,
    )
    remote_feed = Task(
        (4, 1),
        [["FEED"]],
        needs=Counter({"WHEAT": 1}),
        mandatory=True,
    )

    plans, unassigned = build_rolling_queues(
        [local, remote_feed],
        starts=[(1, 1)],
        inventories=[{"WHEAT": 1}],
        budgets=[20],
        shed_access=((4, 4),),
        shed={},
    )

    assert not unassigned
    assert plans[0].queue[:2] == [["PLANT", "WHEAT"], ["WATER"]]
    assert ["FEED"] in plans[0].queue[2:]
