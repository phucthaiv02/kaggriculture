from collections import Counter

from agents.farm_tasks import Task
from agents.scheduler import build_queues, hands_needed


def test_terminal_sale_reserve_only_reduces_cashout_workers_budget():
    """The shared SELL turn must not idle workers that have no goods to DROP.

    On the terminal day every worker loses the terminal observation itself,
    but only the worker carrying sellable output must finish one additional
    turn early so its DROP is visible to the following market SELL.
    """
    non_selling = [
        Task((5, 4), [["WATER"]] * 12, cashout=True),
        Task((5, 4), [["WATER"]] * 10, cashout=True),
    ]
    cashout = Task(
        (4, 4),
        [["HARVEST"]] * 20,
        sells=Counter({"WHEAT": 1}),
        cashout=True,
    )

    count, missing = hands_needed(
        [*non_selling, cashout],
        farmer_start=(5, 4),
        max_hands=2,
    )

    assert (count, missing) == (1, [])
    plans, missing = build_queues(
        [*non_selling, cashout],
        farmer_start=(5, 4),
        hand_count=count,
    )
    assert not missing
    assert sorted(len(plan.queue) for plan in plans) == [21, 22]
    assert sum(plan.queue.count(["DROP"]) for plan in plans) == 1


def test_terminal_day_still_loses_a_turn_without_mandatory_returns():
    task = Task((4, 4), [['WATER']] * 23, terminal_day=True)
    plans, missing = build_queues([task], (4, 4), 0)
    assert missing == [task]
    assert plans[0].queue == []
