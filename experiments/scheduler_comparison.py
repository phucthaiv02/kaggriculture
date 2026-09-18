"""Compare routing against a replay, freezing its targets and start dates.

The baseline source directory must contain the agents package as it was before
optimization. Outputs include the frozen target trace and a calendar audit;
a run with missing or extra starts is not a valid fixed-calendar improvement.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import inspect
import hashlib
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from kaggle_environments import make
from kaggle_environments.utils import structify
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from agents import expansion_agent, scheduler
from experiments.play_match import pass_agent

STARTS = {'PLANT', 'PLACE', 'BUILD_COOP', 'BUILD_PASTURE', 'DIG'}


def start_calendar(replay):
    """Read successful structural actions, with their pre-action coordinates."""
    calendar = Counter()
    config = replay['configuration']
    for previous, current in zip(replay['steps'], replay['steps'][1:]):
        obs = deepcopy(previous[0]['observation'])
        farm, private = obs['farms'][0], obs['private']
        action = current[0].get('action') or {}
        ops = [action.get('farmer', ['PASS']), *action.get('hands', [])]
        seeds = Counter(op[1] for op in ops if op and op[0] == 'PLANT')
        blocked = {name for name, count in seeds.items() if count > private['seeds'].get(name, 0)}
        for unit in range(1 + len(farm['hands'])):
            op = ops[unit] if unit < len(ops) else ['PASS']
            if op[0] == 'PLANT' and op[1] in blocked:
                continue
            x, y = farm['farmer'] if unit == 0 else farm['hands'][unit-1]
            before = deepcopy(farm['tiles'][y][x])
            game._apply_unit_action(farm, private, unit, op, config['boardSize'],
                                    obs['day'], config['turnsPerDay'], config['shedCapacity'])
            if op[0] in STARTS and before != farm['tiles'][y][x]:
                calendar[(obs['day'], x, y, tuple(op))] += 1
    return calendar


def capture_targets(replay, baseline_source):
    """Run the saved agent on recorded observations; do not simulate a new farm."""
    # The saved modules use absolute agents imports. Load them under that name
    # temporarily, then restore every production module before comparison.
    import sys
    saved = {key: value for key, value in sys.modules.items()
             if key == 'agents' or key.startswith('agents.')}
    try:
        for key in saved:
            del sys.modules[key]
        spec = importlib.util.spec_from_file_location(
            'agents', baseline_source / '__init__.py',
            submodule_search_locations=[str(baseline_source)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules['agents'] = module
        spec.loader.exec_module(module)
        from agents.expansion_agent import make_agent
        agent = make_agent()
        targets = inspect.getclosurevars(agent).nonlocals['targets']
        trace = []
        for step in replay['steps'][:-1]:
            agent(structify(deepcopy(step[0]['observation'])), replay['configuration'])
            trace.append(dict(targets))
        return trace
    finally:
        for key in list(sys.modules):
            if key == 'agents' or key.startswith('agents.'):
                del sys.modules[key]
        sys.modules.update(saved)


def run(replay_path, baseline_source, output, baseline=False):
    replay = json.loads(replay_path.read_text())
    fingerprint = hashlib.sha256(replay_path.read_bytes())
    for source in sorted(baseline_source.glob('*.py')):
        fingerprint.update(source.name.encode())
        fingerprint.update(source.read_bytes())
    trace_path = output.parent / f'frozen_targets_{fingerprint.hexdigest()[:16]}.json'
    if trace_path.exists():
        trace = [{(x, y): tuple(target) if target else None for x, y, target in row}
                 for row in json.loads(trace_path.read_text())]
    else:
        trace = capture_targets(replay, baseline_source)
        output.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps([[[x, y, target] for (x, y), target in row.items()]
                                          for row in trace]))
    expected = start_calendar(replay)
    agent = expansion_agent.make_agent()
    target_state = inspect.getclosurevars(agent).nonlocals['targets']
    timings = []

    def fixed_agent(obs, config):
        step = obs['step']
        # Preserve discovery of newly unlocked positions in the coordinator.
        if len(obs['farms'][obs['player']]['unlocked_quadrants']) > 1:
            for position in list(target_state):
                if position in trace[step]:
                    target_state[position] = trace[step][position]
        start = perf_counter()
        result = agent(obs, config)
        farm = obs['farms'][obs['player']]
        if len(farm['unlocked_quadrants']) > 1:
            positions = [farm['farmer'], *farm['hands']]
            operations = [result['farmer'], *result['hands']]
            for index, ((x, y), op) in enumerate(zip(positions, operations)):
                if op[0] in STARTS and not expected[(obs['day'], x, y, tuple(op))]:
                    operations[index] = ['PASS']
            result['farmer'], result['hands'] = operations[0], operations[1:]
        timings.append(perf_counter() - start)
        return result

    def frozen_plan(obs, targets, *args, **kwargs):
        targets.clear()
        targets.update(trace[obs['step']])
        # Keep pending entries alive so the frozen trace is applied next turn.
        return kwargs.get('replan_positions', ())

    metadata = replay_path.with_name('result.json')
    seed = json.loads(metadata.read_text()).get('seed') if metadata.exists() else replay['configuration'].get('seed')
    if seed is None:
        raise ValueError('Replay configuration omits its seed; provide the original result.json beside it')
    config = dict(replay['configuration'], actTimeout=.75, seed=seed)
    env = make('kaggriculture', configuration=config, debug=False)
    with ExitStack() as stack:
        stack.enter_context(patch.object(expansion_agent, 'plan_targets', frozen_plan))
        if baseline:
            stack.enter_context(patch.object(scheduler, '_pack', scheduler._pack_greedy))
        env.run([fixed_agent, pass_agent])
    generated = env.toJSON()
    actual = start_calendar(generated)
    missing, extra = expected - actual, actual - expected
    output.mkdir(parents=True, exist_ok=True)
    (output / 'replay.json').write_text(json.dumps(generated))
    (output / 'replay.html').write_text(env.render(mode='html', width=1200, height=800))
    result = {
        'cash': env.steps[-1][0].reward,
        'seed': seed,
        'target_trace': str(trace_path),
        'statuses': [s.status for s in env.steps[-1]],
        'baseline_scheduler': baseline,
        'fixed_calendar_valid': not missing and not extra,
        'missing_starts': [[*key[:3], key[3], count] for key, count in missing.items()],
        'extra_starts': [[*key[:3], key[3], count] for key, count in extra.items()],
        'p95_seconds': sorted(timings)[int(.95*(len(timings)-1))],
        'max_seconds': max(timings),
        'over_075_seconds': sum(t > .75 for t in timings),
    }
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replay', type=Path)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    run(args.replay, args.baseline_source, args.output, args.baseline)
