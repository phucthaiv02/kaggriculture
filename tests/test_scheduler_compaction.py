"""Regression tests for route compaction before hiring another hand."""

from agents.farm_tasks import Task
from agents.scheduler import FARMER_BUDGET, HAND_BUDGET, SHED, build_queues, hands_needed


def _fragmentation_case():
    """Near-first fragments two workers; hard-first can keep the remote cluster whole."""
    return [
        Task((0, 9), [["WATER"]]),
        Task((6, 0), [["WATER"]] * 2),
        Task((7, 4), [["WATER"]] * 4),
        Task((1, 2), [["WATER"]] * 3),
        Task((0, 6), [["WATER"]] * 7),
    ]


def test_hard_first_compaction_avoids_one_extra_hand():
    count, unassigned = hands_needed(_fragmentation_case(), farmer_start=SHED)

    assert count == 1
    assert unassigned == []


def test_compacted_routes_still_fit_each_workers_real_budget():
    plans, unassigned = build_queues(
        _fragmentation_case(),
        farmer_start=SHED,
        hand_count=1,
        hand_starts=((5, 4),),
    )

    assert unassigned == []
    assert [len(plan.queue) for plan in plans] == [17, 23]
    assert len(plans[0].queue) <= FARMER_BUDGET
    assert len(plans[1].queue) <= HAND_BUDGET
