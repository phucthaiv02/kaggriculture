from agents.expansion_agent import (
    _needs_route_rebuild,
    _plans_have_work,
    _pop_market_batch,
    _strip_partial_animal_builds,
)
from agents.farm_tasks import Task
from agents.scheduler import WorkerPlan


def test_market_batch_preserves_orders_beyond_engine_cap():
    queue = [["HIRE", index] for index in range(13)]

    first = _pop_market_batch(queue)
    second = _pop_market_batch(queue)

    assert len(first) == 10
    assert len(second) == 3
    assert first + second == [["HIRE", index] for index in range(13)]
    assert queue == []


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


def test_dependency_or_tile_topology_change_rebuilds_route():
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
