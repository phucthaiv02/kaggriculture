import pytest

from agents.farm_tasks import build_tasks, purchase_orders
from agents.horizon import first_yield_age
from agents.intraday import schedule_open_tiles
from agents.planner import _candidates
from agents.scheduler import SHED_ACCESS, WorkerPlan
from test_agents_farm_tasks import make_obs, plant


@pytest.mark.parametrize('name', ['WHEAT', 'CARROT', 'MELON', 'TOMATO', 'STRAWBERRY', 'GOOSE', 'COW', 'SHEEP'])
@pytest.mark.parametrize('late', [False, True])
def test_candidate_task_and_purchase_share_first_yield_deadline(name, late):
    day = 29 - first_yield_age(name) + int(late)
    obs = make_obs(day, shed={name: 1, 'WHEAT': 10}, seeds={name: 1})
    target = {(4, 4): (name, False)}
    candidates = [choice[0] for choice, _, _ in _candidates(day, 29)]
    actions = [a for t in build_tasks(obs, target) for a in t.actions]
    assert (name in candidates) == (not late)
    assert any(a[0] in ('PLANT', 'PLACE') for a in actions) == (not late)
    obs['private']['seeds'] = {}
    obs['private']['shed'] = {'WHEAT': 10}
    buys = purchase_orders(obs, target, [(4, 4)])
    assert any(a[0] in ('BUY_SEED', 'BUY_ANIMAL') for a in buys) == (not late)


@pytest.mark.parametrize('crop,day,planted', [('TOMATO', 22, 11), ('STRAWBERRY', 24, 8)])
def test_finished_ongoing_target_cannot_replant_midday(crop, day, planted):
    tile = plant(crop, planted, day, yield_units=0)
    obs = make_obs(day, tiles={(4, 4): tile}, seeds={crop: 10})
    target = {(4, 4): (crop, False)}
    actions = [a for t in build_tasks(obs, target) for a in t.actions]
    assert ['DIG'] in actions
    assert not any(a[0] == 'PLANT' for a in actions)
    obs['farms'][0]['tiles'][4][4] = None
    plans = [WorkerPlan((4, 4), [])]
    eligible, hires = schedule_open_tiles(obs, target, plans, SHED_ACCESS)
    assert not eligible and not hires
    assert not plans[0].queue


def test_late_target_still_harvests_existing_ready_crop():
    obs = make_obs(29, tiles={(4, 4): plant('STRAWBERRY', 13, 29, yield_units=2)})
    tasks = build_tasks(obs, {(4, 4): ('STRAWBERRY', False)})
    assert ['HARVEST'] in tasks[0].actions


def test_real_game_config_overrides_erroneous_end_day_and_blocks_queued_plant(monkeypatch):
    from agents.expansion_agent import make_agent
    import agents.expansion_agent as module
    agent = make_agent(30)
    cells = dict(zip(agent.__code__.co_freevars, (c.cell_contents for c in agent.__closure__)))
    obs = make_obs(22, seeds={'TOMATO': 1})
    obs['hour'] = 22
    obs['town'] = {'unlocked_shops': []}
    cells['targets'].update({(x, y): None for y in range(10) for x in range(10)})
    cells['state'].update(day=22, plans=[WorkerPlan((4, 4), [['PLANT', 'TOMATO'], ['WATER'], ['EAST'], ['HARVEST']])])
    monkeypatch.setattr(module, 'schedule_open_tiles', lambda *args: ({}, 0))
    result = agent(obs, {'episodeSteps': 720, 'turnsPerDay': 24})
    assert result['farmer'] == ['PASS']
    assert cells['state']['plans'][0].queue == [['EAST'], ['HARVEST']]
