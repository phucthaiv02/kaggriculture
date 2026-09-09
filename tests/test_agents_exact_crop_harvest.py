from agents.farm_tasks import Task, build_tasks
from agents.scheduler import SHED, build_queues
from tests.test_agents_farm_tasks import make_obs, plant

def test_target_change_does_not_harvest_early():
    tile = plant("WHEAT", planted_day=0, day=2, yield_units=2)
    task = build_tasks(make_obs(day=2, tiles={(0, 0): tile}), {(0, 0): ("CARROT", False)})[0]
    assert ["HARVEST"] not in task.actions

def test_exact_max_yield_harvest_exists():
    tile = plant("WHEAT", planted_day=0, day=4, yield_units=4)
    task = build_tasks(make_obs(day=4, tiles={(0, 0): tile}), {(0, 0): ("WHEAT", False)})[0]
    assert ["HARVEST"] in task.actions

def test_due_harvest_outranks_water_when_only_one_fits():
    harvest = Task(SHED, [["HARVEST"]], urgent=True, ends_cycle=True, deadline=120)
    water = Task(SHED, [["WATER"]], urgent=True)
    plans, unassigned = build_queues([water, harvest], SHED, 0, (), (SHED,), existing_hand_budget=1)
    assert plans[0].queue == [["HARVEST"]]
    assert unassigned == [water]
