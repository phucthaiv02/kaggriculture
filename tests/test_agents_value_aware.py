from collections import Counter
from copy import deepcopy

import pytest
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from agents.farm_tasks import Task, build_tasks
from agents.forecast import production
from agents.scheduler import build_queues, hands_needed
from agents.schedules import should_feed_animal, should_care_animal, should_harvest_animal
from test_agents_farm_tasks import animal_tile, make_obs, plant


def test_optional_work_does_not_buy_an_expensive_hand():
    required = Task((4, 4), [['WATER']] * 23, mandatory=True)
    optional = Task((4, 4), [['COLLECT_FERTILIZER']], mandatory=False, value=10)
    assert hands_needed([required, optional], (4, 4), max_hands=1,
                        marginal_hire_costs=[11]) == (0, [optional])
    assert hands_needed([required, optional], (4, 4), max_hands=1,
                        marginal_hire_costs=[9]) == (1, [])


def test_mandatory_work_can_justify_cost_above_sale_value():
    work = [Task((4, 4), [['WATER']] * 22, mandatory=True),
            Task((4, 4), [['FEED']] * 2, mandatory=True)]
    assert hands_needed(work, (4, 4), max_hands=1,
                        marginal_hire_costs=[1000]) == (1, [])


def test_optional_values_are_combined_and_existing_workers_are_sunk_cost():
    work = [Task((4, 4), [['WATER']] * 23, mandatory=True)]
    work += [Task((4, 4), [['HARVEST']], mandatory=False, value=6) for _ in range(2)]
    assert hands_needed(work, (4, 4), max_hands=1, marginal_hire_costs=[10]) == (1, [])
    assert hands_needed(work, (4, 4), [(4, 4)], max_hands=1,
                        marginal_hire_costs=[]) == (1, [])


@pytest.mark.parametrize('target', [('STRAWBERRY', False), None])
def test_off_schedule_dry_strawberry_is_rescued_before_other_urgent_work(target):
    tile = plant('STRAWBERRY', 0, 3, yield_units=0)
    tile['consecutive_unwatered'] = 1
    obs = make_obs(3, {(4, 4): tile})
    tasks = build_tasks(obs, {(4, 4): target})
    assert len(tasks) == 1
    assert tasks[0].rescue and tasks[0].mandatory
    harvest = Task((4, 4), [['HARVEST']] * 23, urgent=True, animal_harvest=True)
    plans, missing = build_queues([harvest, *tasks], (4, 4), 0)
    assert plans[0].queue[0] == ['WATER']
    assert missing == [harvest]
    farm = deepcopy(obs['farms'][0])
    game._apply_unit_action(farm, deepcopy(obs['private']), 0, ['WATER'], 10, 3, 24)
    game._daily_refresh_plants(farm, 3, 24)
    assert farm['tiles'][4][4]['kind'] == 'PLANT'


@pytest.mark.parametrize('crop', ['WHEAT', 'STRAWBERRY'])
def test_terminal_early_harvest_reserves_return_and_sell(crop):
    tile = plant(crop, 28, 29, yield_units=2)
    obs = make_obs(29, {(0, 0): tile})
    task, = build_tasks(obs, {(0, 0): (crop, False)})
    assert task.actions == [['HARVEST']]
    assert task.cashout and task.terminal_day and not task.mandatory
    assert task.value > 0
    plans, missing = build_queues([task], (4, 4), 0)
    assert not missing
    assert plans[0].queue[-1] == ['DROP']
    assert len(plans[0].queue) <= 21
    plans, missing = build_queues([task], (4, 4), 0, existing_hand_budget=16)
    assert missing == [task]


@pytest.mark.parametrize('animal', ['GOOSE', 'COW', 'SHEEP'])
@pytest.mark.parametrize('placed_day,end_day', [(0, 29), (7, 29), (20, 29), (4, 12)])
def test_animal_forecast_matches_runtime_actions_and_visits(animal, placed_day, end_day):
    farm = {'tiles': [[game._new_animal(animal, placed_day)]], 'farmer': [0, 0], 'hands': []}
    private = {'inventories': [{}], 'shed': {}, 'seeds': {}}
    initial = deepcopy(farm['tiles'][0][0])
    predicted = production(animal, False, placed_day, end_day, initial)
    for day in range(placed_day, end_day + 1):
        tile = farm['tiles'][0][0]
        age, last_age = day - placed_day, end_day - placed_day
        ops = []
        if should_harvest_animal(animal, age, tile.get('yield_units', 0), force=day == end_day):
            ops.append('HARVEST')
        if should_feed_animal(animal, age, last_age):
            ops.append('FEED')
            private['inventories'][0]['WHEAT'] = 1
        if should_care_animal(animal, age, last_age):
            ops.append('CARE')
        if tile.get('fertilizer_available'):
            ops.append('COLLECT_FERTILIZER')
        for op in ops:
            game._apply_unit_action(farm, private, 0, [op], 1, day, 24)
        assert predicted.sales[day] == Counter(private['inventories'][0])
        assert predicted.inputs[day]['WHEAT'] == int('FEED' in ops)
        assert sum(v[1] for v in predicted.visits[day]) == len(ops)
        private['inventories'][0].clear()
        if day < end_day:
            game._daily_refresh_animals(farm, day)


def test_unreachable_mandatory_task_does_not_force_sixteen_hires():
    task = Task((4, 4), [['WATER']] * 24, mandatory=True)
    assert hands_needed([task], (4, 4), max_hands=2,
                        marginal_hire_costs=[1, 1]) == (0, [task])


def test_intraday_expansion_cannot_bypass_expensive_hire_gate():
    from agents.intraday import schedule_open_tiles
    from agents.scheduler import SHED_ACCESS, WorkerPlan
    obs = make_obs(8, money=10000)
    obs['hour'] = 20
    plans = [WorkerPlan((4, 4), [['WATER']] * 4)]
    _, hires = schedule_open_tiles(obs, {(5, 4): ('WHEAT', False)}, plans,
                                   SHED_ACCESS, [1000, 2000])
    assert hires == 0


def test_terminal_water_keeps_immediate_yield_increment():
    tile = game._new_plant('WHEAT', 25, 24)
    tile['yield_units'] = 3
    obs = make_obs(29, {(4, 4): tile})
    task, = build_tasks(obs, {(4, 4): ('WHEAT', False)})
    assert task.actions == [['WATER'], ['HARVEST']]
    farm, private = deepcopy(obs['farms'][0]), deepcopy(obs['private'])
    for op in task.actions:
        game._apply_unit_action(farm, private, 0, op, 10, 29, 24)
    assert private['inventories'][0] == task.sells == {'WHEAT': 4}
    assert not task.mandatory and task.cashout


def test_terminal_harvest_does_not_require_a_standing_target():
    tile = plant('STRAWBERRY', 10, 29, yield_units=2)
    task, = build_tasks(make_obs(29, {(0, 0): tile}), {})
    assert task.position == (0, 0)
    assert task.actions == [['HARVEST']]
    assert task.sells == {'STRAWBERRY': 2}


def test_due_animal_harvest_is_mandatory_cashout_work():
    obs = make_obs(8, {(4, 4): animal_tile('COW', 0, yield_units=6)})
    task = next(task for task in build_tasks(obs, {(4, 4): ('COW', False)})
                if task.animal_harvest)
    assert task.mandatory


def test_routine_water_and_feed_are_valued_but_not_mandatory():
    crop = plant('WHEAT', 0, 2, yield_units=1)
    crop_task, = build_tasks(make_obs(2, {(4, 4): crop}),
                             {(4, 4): ('WHEAT', False)})
    assert not crop_task.mandatory and crop_task.value > 0

    animal = animal_tile('GOOSE', 0)
    animal_task, = build_tasks(make_obs(1, {(4, 4): animal}, shed={'WHEAT': 1}),
                               {(4, 4): ('GOOSE', False)})
    assert not animal_task.mandatory and animal_task.value > 0


def test_feed_becomes_mandatory_after_one_miss():
    animal = animal_tile('GOOSE', 0)
    animal['consecutive_unfed'] = 1
    tasks = build_tasks(make_obs(1, {(4, 4): animal}, shed={'WHEAT': 1}),
                        {(4, 4): ('GOOSE', False)})
    task = next(task for task in tasks if ['FEED'] in task.actions)
    assert task.mandatory and task.actions == [['FEED']]
    assert all(not other.mandatory for other in tasks if other is not task)


@pytest.mark.parametrize('targets', [{}, {(4, 4): None}, {(4, 4): ('WHEAT', False)}])
def test_live_animal_survives_target_removal(targets):
    from agents.farm_tasks import feed_wheat_order
    tile = animal_tile('COW', 0, yield_units=6)
    tile['consecutive_unfed'] = 1
    obs = make_obs(8, {(4, 4): tile})
    tasks = build_tasks(obs, targets)
    assert any(['FEED'] in task.actions and task.mandatory for task in tasks)
    assert any(task.animal_harvest for task in tasks)
    assert feed_wheat_order(obs, targets, [(4, 4)])


def test_terminal_packing_prefers_valuable_recovery():
    cheap = Task((4, 4), [['HARVEST']], mandatory=False, value=1, cashout=True, sells=Counter(WHEAT=1))
    valuable = Task((6, 4), [['HARVEST']], mandatory=False, value=500, cashout=True, sells=Counter(MELON=1))
    plans, missing = build_queues([cheap, valuable], (4, 4), 0, existing_hand_budget=7)
    assert missing == [cheap]
    assert plans[0].queue == [['EAST'], ['EAST'], ['HARVEST'], ['WEST'], ['DROP']]


def test_hire_search_crosses_optional_plateau(monkeypatch):
    import agents.scheduler as scheduler
    valuable = Task((4, 4), [['HARVEST']], mandatory=False, value=100)
    monkeypatch.setattr(scheduler, '_pack', lambda tasks, starts, budgets, shed:
                        ([[] for _ in starts], [valuable] if len(starts) < 3 else []))
    assert hands_needed([valuable], (4, 4), max_hands=2,
                        marginal_hire_costs=[10, 20]) == (2, [])


def test_rescue_preempts_busy_worker_and_reaches_crop_before_rollover():
    from agents.intraday import rescue_survival
    from agents.scheduler import WorkerPlan
    tile = plant('STRAWBERRY', 0, 3, yield_units=0)
    tile['consecutive_unwatered'] = 1
    obs = make_obs(3, {(5, 4): tile})
    obs['hour'] = 22
    plans = [WorkerPlan((4, 4), [['PASS'], ['EAST'], ['WATER']])]
    rescue_survival(obs, plans)
    assert plans[0].queue[:2] == [['EAST'], ['WATER']]
    farm, private = deepcopy(obs['farms'][0]), deepcopy(obs['private'])
    for op in plans[0].queue[:2]:
        game._apply_unit_action(farm, private, 0, op, 10, 3, 24)
    game._daily_refresh_plants(farm, 3, 24)
    assert farm['tiles'][4][5]['kind'] == 'PLANT'


def test_owned_animal_reconciles_missing_target():
    from agents.intraday import reconcile_animals
    obs = make_obs(8, {(5, 4): {'kind': 'PASTURE'}}, shed={'COW': 1})
    targets = {(5, 4): None}
    assert reconcile_animals(obs, targets) == {(5, 4)}
    assert targets[(5, 4)] == ('COW', False)


def test_animal_purchase_has_committed_delivery_and_placement_route():
    from agents.intraday import schedule_open_tiles, queue_commitments
    from agents.scheduler import SHED_ACCESS, WorkerPlan
    from agents.farm_tasks import purchase_orders
    obs = make_obs(8, {(5, 4): {'kind': 'PASTURE'}}, money=1000, shed={'WHEAT': 5})
    plans = [WorkerPlan((4, 4), [])]
    target = {(5, 4): ('COW', False)}
    eligible, hires = schedule_open_tiles(obs, target, plans, SHED_ACCESS)
    assert hires == 0
    assert ['PLACE', 'COW'] in plans[0].queue
    assert plans[0].queue[0] == ['PASS']
    assert queue_commitments([(4, 4)], plans)[3]['COW'] == 1
    assert ['BUY_ANIMAL', 'COW', 1] in purchase_orders(obs, eligible, list(eligible))
    # Until the purchase lands, the committed target must remain eligible.
    eligible, _ = schedule_open_tiles(obs, target, plans, SHED_ACCESS)
    assert eligible == target


def test_unreachable_animal_is_not_eligible_for_purchase():
    from agents.intraday import schedule_open_tiles
    from agents.scheduler import SHED_ACCESS, WorkerPlan
    obs = make_obs(8, money=10000)
    obs['hour'] = 22
    eligible, _ = schedule_open_tiles(obs, {(0, 0): ('SHEEP', False)},
                                     [WorkerPlan((4, 4), [])], SHED_ACCESS)
    assert eligible == {}


def test_rescue_survival_is_separate_from_fertilizer_and_harvest():
    tile = plant('STRAWBERRY', 0, 3, yield_units=2)
    tile['consecutive_unwatered'] = 1
    tasks = build_tasks(make_obs(3, {(4, 4): tile}), {(4, 4): ('STRAWBERRY', False)})
    rescue = next(task for task in tasks if task.rescue)
    assert rescue.actions == [['WATER']]
    assert all(not task.mandatory for task in tasks if task is not rescue)


def test_rescue_feed_fetches_real_wheat_before_rollover():
    from agents.intraday import rescue_survival
    from agents.scheduler import WorkerPlan
    tile = animal_tile('COW', 0)
    tile['consecutive_unfed'] = 1
    obs = make_obs(8, {(5, 4): tile}, shed={'WHEAT': 1})
    obs['hour'] = 21
    # The queued FEED has no wheat: metadata/queue presence is insufficient.
    plans = [WorkerPlan((4, 4), [['EAST'], ['FEED']])]
    rescue_survival(obs, plans)
    assert plans[0].queue[:3] == [['PICKUP', 'WHEAT', 1], ['EAST'], ['FEED']]
    farm, private = deepcopy(obs['farms'][0]), deepcopy(obs['private'])
    for op in plans[0].queue[:3]:
        game._apply_unit_action(farm, private, 0, op, 10, 8, 24)
    game._daily_refresh_animals(farm, 8)
    assert farm['tiles'][4][5]['animal'] == 'COW'


def test_harvest_is_mandatory_before_next_production_clips():
    tile = animal_tile('COW', 0, yield_units=5)
    tile['pending_care_bonus'] = 1
    obs = make_obs(9, {(4, 4): tile})
    # Force cadence-independent output detection when the next refresh clips.
    tasks = build_tasks(obs, {(4, 4): None})
    harvests = [task for task in tasks if task.animal_harvest]
    assert harvests and harvests[0].mandatory


def test_intraday_rescues_unassigned_capped_animal_output():
    from agents.intraday import rescue_survival
    from agents.scheduler import WorkerPlan
    obs = make_obs(8, {(5, 4): animal_tile('COW', 0, yield_units=6)})
    obs['hour'] = 22
    plans = [WorkerPlan((4, 4), [['PASS'], ['PASS']])]
    rescue_survival(obs, plans)
    assert plans[0].queue[:2] == [['EAST'], ['HARVEST']]
