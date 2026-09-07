"""Play the current agent against the executable embedded in a public notebook."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kaggle_environments import make
from agents.expansion_agent import make_agent


def public_agent(notebook):
    document = json.loads(notebook.read_text())
    sources = [
        ''.join(cell.get('source', [])) for cell in document['cells']
        if cell['cell_type'] == 'code'
        and ''.join(cell.get('source', [])).startswith('%%writefile main.py\n')
    ]
    if len(sources) != 1:
        raise ValueError('Expected exactly one main.py source cell')
    source = sources[0].split('\n', 1)[1]
    namespace = {'__name__': 'public_solution_agent'}
    exec(compile(source, str(notebook) + ':main.py', 'exec'), namespace)
    return namespace['agent'], hashlib.sha256(source.encode()).hexdigest()


def run(seed=1, seats=(0, 1)):
    notebook = ROOT / 'public_solution/v17-r1-rc2-high-score-10c-4s-market-storage.ipynb'
    results = []
    for seat in seats:
        opponent, source_hash = public_agent(notebook)
        current = make_agent(seed=seed)
        players = [current, opponent] if seat == 0 else [opponent, current]
        config = {'episodeSteps': 720, 'townCenterSellInterval': 24, 'farmHandCostMult': 1, 'seed': seed}
        env = make('kaggriculture', configuration=config, debug=False)
        env.run(players)
        final = env.steps[-1]
        stem = f'current_vs_public_seed{seed}_seat{seat}'
        html = ROOT / (stem + '.html')
        html.write_text(env.render(mode='html', width=1200, height=800), encoding='utf-8')
        replay = ROOT / 'replays' / (stem + '.json')
        replay.parent.mkdir(exist_ok=True)
        replay.write_text(json.dumps(env.toJSON()), encoding='utf-8')
        row = {
            'seed': seed, 'current_seat': seat,
            'current_cash': final[seat].reward, 'public_cash': final[1-seat].reward,
            'margin': final[seat].reward - final[1-seat].reward,
            'statuses': [s.status for s in final], 'frames': len(env.steps),
            'engine': version('kaggle-environments'), 'public_source_sha256': source_hash,
            'configuration': dict(env.configuration),
            'html': str(html), 'replay_json': str(replay),
        }
        results.append(row)
        (ROOT / 'replays/public_match_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
        if row['statuses'] != ['DONE', 'DONE']:
            raise RuntimeError(f'Game did not finish normally: {row["statuses"]}')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--seats', type=int, nargs='+', choices=(0, 1), default=[0, 1])
    args = parser.parse_args()
    run(args.seed, args.seats)
