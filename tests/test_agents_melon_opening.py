"""Opening deadlines checked against actual engine market/worker execution."""
from collections import Counter

import pytest
from kaggle_environments import make

from agents.expansion_agent import make_agent


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
def test_melon_opening_places_animals_on_the_scheduled_days(seed):
    env = make("kaggriculture", configuration={"seed": seed, "weedSpawnChance": 0.0}, debug=False)
    agent = make_agent(opening_version="melon_v2")
    bought = {}
    placed = {}
    for _ in range(96):
        obs = env.state[0].observation
        day, hour = obs.day, obs.hour
        action = agent(obs)
        if day in (2, 3):
            if hour > 0:
                assert ["HIRE"] not in action["market"]
            animal = "COW" if day == 2 else "SHEEP"
            if ["BUY_ANIMAL", animal, 1] in action["market"]:
                bought[animal] = hour
                assert any(op[:2] == ["SELL", "FERTILIZER"] for op in action["market"])
            if ["PLACE", animal] in [action["farmer"], *action["hands"]]:
                placed[animal] = hour
        state = env.step([action, {"farmer": ["PASS"], "hands": [], "market": []}])
        if hour == 23:
            counts = Counter(
                tile.get("animal", tile.get("crop"))
                for row in state[0].observation.farms[0].tiles
                for tile in row if isinstance(tile, dict)
            )
            assert counts == Counter({
                "COW": 3 if day >= 2 else 2,
                "SHEEP": 3 if day >= 3 else 2,
                "WHEAT": 7 if day >= 2 else 9,
                "MELON": 12,
            })
    assert bought.keys() == placed.keys() == {"COW", "SHEEP"}
    assert all(bought[name] < placed[name] < 24 for name in bought)


@pytest.mark.parametrize('seed,opponent', [(0, 'pass'), (1, 'pass'), (7, 'pass'), (42, 'pass'), (1, 'A')])
def test_opening_hands_off_on_day_five_and_preserves_live_melons(seed, opponent, monkeypatch):
    from pathlib import Path
    from experiments.play_match import notebook_agent
    import agents.expansion_agent as expansion

    real_plan_targets = expansion.plan_targets

    def guarded_planner(obs, *args, **kwargs):
        if obs['day'] < 4:
            pytest.fail('planner must not run before day 4')
        return real_plan_targets(obs, *args, **kwargs)

    monkeypatch.setattr(expansion, 'plan_targets', guarded_planner)
    env = make('kaggriculture', configuration={
        'seed': seed, 'weedSpawnChance': 0.0, 'episodeSteps': 169,
    }, debug=False)
    agent = make_agent(opening_version='melon_v2')
    other = lambda obs: {'farmer': ['PASS'], 'market': []}
    if opponent == 'A':
        notebook = Path(__file__).resolve().parents[1] / 'public' / 'A.ipynb'
        if not notebook.is_file():
            pytest.skip('local opponent notebook is not available')
        other, _ = notebook_agent(notebook)
    # Preserve the full-season planning horizon while bounding this engine
    # regression to the seven opening days. env.run copies observations like
    # real matches; direct env.step with live observations misses route bugs.
    env.run([lambda obs: agent(obs), other])
    assert len(env.steps) == 169
    opening_tiles = env.steps[24][0].observation.farms[0].tiles
    melons = {(x, y) for y, row in enumerate(opening_tiles) for x, tile in enumerate(row)
              if isinstance(tile, dict) and tile.get('crop') == 'MELON'}
    assert len(melons) == 12
    for frame in range(24, 169):
        tiles = env.steps[frame][0].observation.farms[0].tiles
        for x, y in melons:
            tile = tiles[y][x]
            assert isinstance(tile, dict) and tile.get('crop') == 'MELON', (frame, (x, y), tile)
            assert tile['planted_day'] == 0, (frame, (x, y), tile)
        if frame >= 96:
            animals = Counter(tile.get('animal') for row in tiles for tile in row
                              if isinstance(tile, dict) and tile.get('animal'))
            assert animals['COW'] >= 3
            assert animals['SHEEP'] >= 3
