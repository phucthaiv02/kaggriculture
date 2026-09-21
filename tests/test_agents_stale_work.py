from agents.expansion_agent import _prune_stale_harvests, _valid_harvest_operations
from agents.farm_tasks import build_tasks, purchase_orders
from agents.scheduler import SHED_ACCESS, WorkerPlan
from test_agents_farm_tasks import make_obs, plant, animal_tile


def test_pending_finished_melon_harvests_without_replant_or_seed_purchase():
    position = (4, 4)
    obs = make_obs(10, tiles={position: plant('MELON', 0, 10, yield_units=4)}, seeds={'MELON': 9})
    obs['_pending_targets'] = {position}
    targets = {position: ('MELON', False)}
    actions = [op for task in build_tasks(obs, targets, assume_crop_seeds=True) for op in task.actions]
    assert ['HARVEST'] in actions
    assert not any(op[0] == 'PLANT' for op in actions)
    obs['private']['seeds'] = {}
    assert not purchase_orders(obs, targets, [position], replant_same_crop=True)
    obs['farms'][0]['tiles'][4][4] = None
    obs['private']['seeds'] = {'MELON': 1}


def test_pending_growing_crop_keeps_maintenance():
    obs = make_obs(6, tiles={(4, 4): plant('MELON', 0, 6)})
    obs['_pending_targets'] = {(4, 4)}
    tasks = build_tasks(obs, {(4, 4): ('MELON', False)})
    assert ['WATER'] in tasks[0].actions


def test_animal_cadence_and_terminal_flush():
    obs = make_obs(10, tiles={(4, 4): animal_tile('COW', 0, yield_units=2)}, shed={'WHEAT': 1})
    targets = {(4, 4): ('COW', False)}
    assert not any(['HARVEST'] in t.actions for t in build_tasks(obs, targets))
    obs['_planning_end_day'] = 10
    assert any(['HARVEST'] in t.actions for t in build_tasks(obs, targets))


def test_stale_harvest_removed_without_losing_feed_or_route():
    obs = make_obs(10, tiles={(5, 4): animal_tile('COW', 0)})
    farm = obs['farms'][0]
    plans = [WorkerPlan((4, 4), [['EAST'], ['HARVEST'], ['FEED'], ['CARE']])]
    _prune_stale_harvests(farm, plans)
    assert plans[0].queue == [['EAST'], ['FEED'], ['CARE']]


def test_only_first_worker_can_harvest_same_tile():
    obs = make_obs(10, tiles={(4, 4): animal_tile('COW', 0, yield_units=2)}, hands=[[4, 4]])
    farm = obs['farms'][0]
    assert _valid_harvest_operations(farm, [['HARVEST'], ['HARVEST']]) == [['HARVEST'], ['PASS']]
    farm['tiles'][4][4]['yield_units'] = 0
    assert _valid_harvest_operations(farm, [['HARVEST'], ['HARVEST']]) == [['PASS'], ['PASS']]


def test_agent_pending_reprice_blocks_morning_purchase_and_queued_replant(monkeypatch):
    import agents.expansion_agent as expansion

    monkeypatch.setattr(expansion, 'make_opening_controller', lambda: lambda *args: False)
    monkeypatch.setattr(expansion, 'plan_targets', lambda *args, **kwargs: kwargs['replan_positions'])
    monkeypatch.setattr(expansion, 'should_buy_land_on_schedule', lambda *args: False)
    agent = expansion.make_agent()
    cells = dict(zip(agent.__code__.co_freevars, agent.__closure__))
    cells['targets'].cell_contents[(4, 4)] = ('MELON', False)
    obs = make_obs(10, tiles={(4, 4): plant('MELON', 0, 10, yield_units=4)}, seeds={'MELON': 1})
    obs['hour'] = 0
    result = agent(obs)
    assert not any(op[0] == 'BUY_SEED' for op in result['market'])
    obs['hour'] = 1
    result = agent(obs)
    state = cells['state'].cell_contents
    assert (4, 4) in state['pending_targets']
    assert not any(op[0] == 'PLANT' for plan in state['plans'] for op in plan.queue)
    assert result['farmer'][0] != 'PLANT'


def test_invalid_front_action_triggers_global_remaining_day_replan():
    import agents.expansion_agent as expansion

    position = (5, 4)
    tile = animal_tile('COW', 0)
    tile['consecutive_unfed'] = 1
    obs = make_obs(8, tiles={position: tile}, shed={'WHEAT': 1})
    obs['hour'] = 10
    agent = expansion.make_agent()
    cells = dict(zip(agent.__code__.co_freevars, agent.__closure__))
    cells['targets'].cell_contents.update({(x, y): None for y in range(10) for x in range(10)})
    cells['targets'].cell_contents[position] = ('COW', False)
    state = cells['state'].cell_contents
    state.update(
        day=8,
        plan_frozen=True,
        frozen_positions={position},
        plans=[WorkerPlan((4, 4), [['HARVEST']])],
    )
    result = agent(obs)
    assert result['farmer'] != ['HARVEST']
    assert any(['FEED'] in plan.queue for plan in state['plans'])


def test_daily_target_backlog_does_not_restart_near_shed(monkeypatch):
    import agents.expansion_agent as expansion

    monkeypatch.setattr(expansion, 'make_opening_controller',
                        lambda: lambda *args: False)
    seen = []

    def batch(_obs, _targets, _positions, _end_day, **kwargs):
        current = set(kwargs['replan_positions'])
        seen.append(current)
        return set(sorted(current)[10:])

    monkeypatch.setattr(expansion, 'plan_targets', batch)
    agent = expansion.make_agent()
    first = make_obs(10)
    first['hour'] = 0
    agent(first)
    second = make_obs(11)
    second['hour'] = 0
    agent(second)
    assert len(seen[0]) == 100
    assert seen[1] == set(sorted(seen[0])[10:])
