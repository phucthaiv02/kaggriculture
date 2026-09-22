from agents.expansion_agent import _pop_market_batch, _strip_partial_animal_builds
from agents.farm_tasks import Task


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
