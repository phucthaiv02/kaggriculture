from copy import deepcopy
from pathlib import Path

import pytest
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from experiments.sales_report import analyze, daily_cash, render


def replay_fixture():
    farms = [game._new_farm(10, 3000) for _ in range(2)]
    market = game._new_market()
    states = []
    for player in range(2):
        private = game._new_private()
        private['shed']['WHEAT'] = 2
        states.append({'observation': {'step': 0, 'farms': farms, 'market': market,
                                       'private': private}, 'action': {}})
    after = deepcopy(states)
    # Both players sell in lockstep: quote 25 for unit 1, then the
    # same new quote for unit 2. Oversized orders cannot sell missing stock.
    revenue = 25 + game.market_price('WHEAT', 10002)
    for i, state in enumerate(after):
        state['action'] = {'market': [['SELL', 'WHEAT', 99]]}
        state['observation']['step'] = 1
        state['observation']['farms'][i]['money'] += revenue
        state['observation']['private']['shed']['WHEAT'] = 0
    after[0]['observation']['market']['inventory']['WHEAT'] = 10004
    after[0]['observation']['market']['prices']['WHEAT'] = game.market_price('WHEAT', 10004)
    final = deepcopy(after)
    for state in final:
        state['action'] = {'market': [['SELL', 'WHEAT', 1]]}
        state['observation']['step'] = 2
    return {'name': 'kaggriculture', 'configuration': {'turnsPerDay': 24},
            'steps': [states, after, final]}, revenue


def test_executed_sales_and_step_alignment():
    replay, revenue = replay_fixture()
    report = analyze(replay)
    for row in report['rows']:
        if row['product'] == 'WHEAT':
            assert row['quantity'] == 2
            assert row['revenue'] == revenue
            assert row['average_price'] == revenue / 2
            assert row['timeline'][0]['quantity'] == 2
            assert row['timeline'][1]['quantity'] == 0
            assert row['timeline'][1]['average_price'] is None
            assert row['timeline'][1]['cumulative_quantity'] == 2
        else:
            assert row['average_price'] is None
    assert replay['steps'][0][0]['observation']['farms'][0]['money'] == 3000


def test_rejects_unreconciled_cash():
    replay, _ = replay_fixture()
    replay['steps'][1][0]['observation']['farms'][0]['money'] += 100
    with pytest.raises(ValueError, match='cash mismatch'):
        analyze(replay)


def test_offline_html_escapes_source():
    replay, _ = replay_fixture()
    result = render(analyze(replay), '<script>alert(1)</script>')
    assert '<svg' in result
    assert '&lt;script&gt;' in result
    assert '<script>' not in result
    assert 'theo step' in result
    assert 'Chi phí và dòng tiền' in result
    assert 'Cash theo ngày' in result
    assert 'Chênh lệch P0 − P1' in result
    assert 'Player 0' in result and 'Player 1' in result


def test_costs_include_successful_purchases_hiring_and_land():
    replay, _ = replay_fixture()
    initial = replay['steps'][0]
    final = deepcopy(initial)
    seed_cost = game.CROPS['WHEAT']['seed']
    hire_cost = game._hire_cost(0)
    land_cost = game.LAND_PRICES[0]
    total = 2 * seed_cost + hire_cost + land_cost
    final[0]['action'] = {'market': [['BUY_SEED', 'WHEAT', 2], ['HIRE'], ['BUY_LAND']]}
    final[0]['observation']['farms'][0]['money'] -= total
    for state in final:
        state['observation']['step'] = 1
    replay['steps'] = [initial, final]
    report = analyze(replay)
    p0, p1 = report['finances']
    assert p0['cost'] == total
    assert p0['net_cashflow'] == -total
    assert p0['breakdown'] == {'BUY_SEED:WHEAT': 2 * seed_cost,
                                'HIRE:HIRE': hire_cost, 'BUY_LAND:BUY_LAND': land_cost}
    assert p0['timeline'][0]['cumulative_cost'] == total
    assert p1['cost'] == 0


def test_unaffordable_purchases_only_count_executed_units():
    replay, _ = replay_fixture()
    initial = replay['steps'][0]
    seed_cost = game.CROPS['WHEAT']['seed']
    initial[0]['observation']['farms'][0]['money'] = seed_cost * 2
    final = deepcopy(initial)
    final[0]['action'] = {'market': [['BUY_SEED', 'WHEAT', 99], ['BUY_LAND'], ['HIRE']]}
    final[0]['observation']['farms'][0]['money'] = 0
    for state in final:
        state['observation']['step'] = 1
    replay['steps'] = [initial, final]
    report = analyze(replay)
    assert report['finances'][0]['cost'] == seed_cost * 2
    assert len(report['cost_events']) == 2


def test_daily_cash_uses_post_action_day_boundary_and_partial_final_day():
    replay = {"configuration": {"turnsPerDay": 2}, "steps": []}
    for step, cash in enumerate([3000, 2900, 3100, 2800]):
        replay["steps"].append([{"observation": {"step": step,
                                 "farms": [{"money": cash}, {"money": cash + 100}]}}])
    assert daily_cash(replay, 0) == [
        {"day": 0, "step": 1, "cash": 3100, "complete_day": True},
        {"day": 1, "step": 2, "cash": 2800, "complete_day": False},
    ]
    assert daily_cash(replay, 1)[-1]["cash"] == 2900


def test_cli_uses_project_environment_and_absolute_artifact_paths(monkeypatch, tmp_path):
    from experiments import sales_report
    root = tmp_path / 'project'
    python = root / '.venv' / 'bin' / 'python'
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(sales_report, '__file__', str(root / 'experiments' / 'sales_report.py'))
    monkeypatch.setattr(sales_report.sys, 'prefix', '/system-python')
    monkeypatch.chdir(tmp_path)
    assert sales_report._project_command(Path('match.json'), Path('report.html')) == [
        str(python), '-m', 'experiments.sales_report', str(tmp_path / 'match.json'),
        '--output', str(tmp_path / 'report.html'),
    ]


def test_cli_does_not_relaunch_inside_project_environment(monkeypatch, tmp_path):
    from experiments import sales_report
    python = tmp_path / '.venv' / 'bin' / 'python'
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(sales_report, '__file__', str(tmp_path / 'experiments' / 'sales_report.py'))
    monkeypatch.setattr(sales_report.sys, 'prefix', str(tmp_path / '.venv'))
    assert sales_report._project_command(Path('match.json'), None) is None


def test_cli_keeps_active_environment_when_project_venv_is_absent(monkeypatch, tmp_path):
    from experiments import sales_report
    monkeypatch.setattr(sales_report, '__file__', str(tmp_path / 'experiments' / 'sales_report.py'))
    assert sales_report._project_command(Path('match.json'), None) is None
