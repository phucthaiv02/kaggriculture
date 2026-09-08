import json
import sys
from collections import Counter
from copy import copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents import planner
from agents.expansion_agent import make_agent
from agents.forecast import Production
from experiments.play_match import configuration, resolve_opponent
from kaggle_environments import make

records = []
original = planner._choose


def traced(market, baseline, candidates, counts, labor=None, position=(4, 4)):
    candidates = list(candidates)
    results = planner.evaluate_targets(market, baseline, candidates, labor, position)
    profitable = [r for r in results if r.profit > 0]
    best = max(profitable, key=lambda r: (r.profit, -counts[r.choice[0]], r.choice), default=None)
    no_rival = copy(market)
    no_rival.external = Production()
    detail = []
    for r in results:
        sales = sum(r.output.sales.values(), Counter())
        inputs = sum(r.output.inputs.values(), Counter())
        static = sum((sales[p] - inputs[p]) * market.price(p, market.inventory.get(p, 0))
                     for p in sales.keys() | inputs.keys()) - r.capital_cost - r.labor_cost
        detail.append(dict(name=r.choice[0], fertilize=r.choice[1], static_profit=static,
                           solo_profit=no_rival.value(r.output)-r.capital_cost-r.labor_cost,
                           no_rival_profit=no_rival.marginal_profit(baseline,r.output,r.capital_cost)-r.labor_cost,
                           actual_profit=r.profit,
                           baseline_sales=dict(sum(baseline.sales.values(),Counter())),
                           rival_sales=dict(sum(market.external.sales.values(),Counter()))))
    records.append(dict(day=market.day, hour=market.hour, end_day=market.end_day,
                        position=position, selected=best.choice[0] if best else None,
                        fertilize=best.choice[1] if best else None, breakdown=detail,
                        inventory=dict(market.inventory), shops=dict(market.shop_demand),
                        excluded=[dict(name=name, reason=(
                            'first_yield_after_horizon' if market.day + planner._first_yield_age(name) > market.end_day
                            else 'tile_candidate_restriction'))
                            for name in (*planner.CROPS, *planner.ANIMALS)
                            if name not in {r.choice[0] for r in results}],
                        scores=[dict(name=r.choice[0], fertilize=r.choice[1], cash=r.market_cash, capital=r.capital_cost,
                                     labor=r.labor_cost, profit=r.profit,
                                     sales=dict(sum(r.output.sales.values(), Counter())),
                                     inputs=dict(sum(r.output.inputs.values(), Counter())))
                                for r in results]))
    return (best.choice, best.output) if best else (None, None)


opponent, _, _ = resolve_opponent('public/A.ipynb')
env = make('kaggriculture', configuration=configuration(1), debug=False)
planner._choose = traced
try:
    env.run([make_agent(29), opponent])
finally:
    planner._choose = original
Path('replays/target_scores.json').write_text(json.dumps(records, indent=2))
Path('replays/target_score_breakdown.json').write_text(json.dumps([
    {k: r[k] for k in ('day', 'hour', 'position', 'selected', 'breakdown')} for r in records
], indent=2))
print('PICKS', Counter(r['selected'] for r in records))
for record in records[:8]:
    print(json.dumps(record))
print('FINAL', Counter(t['animal'] for row in env.steps[-1][0].observation.farms[0]['tiles']
                       for t in row if isinstance(t, dict) and t.get('animal')))

# Count successful planting/placement from board changes, not target votes.
plantings, daily, previous = [], [], {}
for step in env.steps:
    obs = step[0].observation
    current, counts = {}, Counter()
    for y, row in enumerate(obs.farms[0]['tiles']):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict):
                continue
            name = tile.get('animal') or tile.get('crop')
            if not name:
                continue
            counts[name] += 1
            identity = (name, tile.get('planted_day'), tile.get('placed_day'))
            current[x, y] = identity
            if previous.get((x, y)) != identity:
                plantings.append(dict(day=obs.day, hour=obs.hour, position=[x, y], name=name,
                                      planted_day=tile.get('planted_day'), placed_day=tile.get('placed_day')))
    previous = current
    if obs.hour == 0:
        daily.append(dict(day=obs.day, counts=dict(counts)))
summary = dict(seed=1, opponent='public/A.ipynb',
               picks=dict(Counter(r['selected'] or 'NONE' for r in records)),
               plantings=dict(Counter(p['name'] for p in plantings)),
               final=dict(counts), daily=daily, planting_events=plantings)
Path('replays/target_summary.json').write_text(json.dumps(summary, indent=2))
print('PLANTINGS', summary['plantings'])
print('FINAL_ALL', summary['final'])
