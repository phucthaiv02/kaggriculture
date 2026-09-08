from agents.expansion_agent import _capacity_safe_queues
from agents.farm_tasks import Task
from agents.scheduler import SHED


def test_capacity_guard_drops_expansion_before_protective_debt():
    water_a = Task(SHED, [["WATER"]], urgent=True)
    water_b = Task(SHED, [["WATER"]], urgent=True)
    expansion = Task(SHED, [["PLANT", "WHEAT"], ["WATER"]])
    plans, admitted, unassigned, debt = _capacity_safe_queues(
        [expansion, water_a, water_b], SHED, 0, (), (SHED,),
        pending_hand_budget=1, existing_hand_budget=1,
    )
    assert plans[0].queue == [["WATER"]]
    assert len(admitted) == 1 and admitted[0] in (water_a, water_b)
    assert len(debt) == 1 and debt[0] in (water_a, water_b)
    assert expansion not in admitted


def test_capacity_guard_keeps_full_plan_when_only_discretionary_work_drops():
    water = Task(SHED, [["WATER"]], urgent=True)
    expansion = Task(SHED, [["PLANT", "WHEAT"], ["WATER"]])
    plans, admitted, unassigned, debt = _capacity_safe_queues(
        [expansion, water], SHED, 0, (), (SHED,),
        pending_hand_budget=1, existing_hand_budget=1,
    )
    assert plans[0].queue == [["WATER"]]
    assert water in admitted
    assert expansion in unassigned
    assert debt == []
