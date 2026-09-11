"""Multi-day integration tests for agents/expansion_agent.py against the real
Kaggriculture interpreter.

Unlike the per-function tests (test_agents_farm_tasks.py, _scheduler.py,
_selling.py, _planner.py), which each exercise one function against a single
hand-built obs, the bugs these tests are meant to catch only show up once
state accumulates across many real days: a plan that is locally correct on
day 5 can still starve the operation of cash by day 15 if, say, replanting
after a finished cycle keeps losing a race against affordability. Per-
function tests cannot see that; only playing the real game across many days
can.
"""

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official_game

from agents.expansion_agent import make_agent
from agents.schedules import should_feed_animal
from experiments.crop_schedules import pass_agent

END_DAY = 30
STARTING_MONEY = 3000.0


def configuration(seed):
    return {
        "episodeSteps": (END_DAY + 1) * 24, "boardSize": 10, "startingMoney": STARTING_MONEY,
        "maxMarketOrdersPerTurn": 10, "turnsPerDay": 24, "shedCapacity": 100,
        "weedSpawnChance": 0.0, "townShopUnlockInterval": 3,
        "townShopSellInterval": 4, "townCenterSellInterval": 24, "seed": seed,
    }


def run_solo(seed=1, end_day=END_DAY):
    """Play make_agent alone against a passive opponent, snapshotting money
    and portfolio composition at the start of each day."""
    env = make("kaggriculture", configuration=configuration(seed), debug=False)
    agent = make_agent(end_day, seed=seed)
    free_vars = agent.__code__.co_freevars
    cells = {name: cell.cell_contents for name, cell in zip(free_vars, agent.__closure__)}
    targets = cells["targets"]

    agents = (agent, pass_agent)
    state = env.state
    daily = []
    for step in range(int(env.configuration.episodeSteps) - 1):
        for player in range(2):
            state[player].action = agents[player](state[player].observation)
        state = official_game.interpreter(state, env)
        state[0].observation.step = step + 1
        obs = state[0].observation
        if obs.hour == 0:
            tiles = obs.farms[0]["tiles"]
            occupied = sum(
                1 for (x, y) in targets
                if isinstance(tiles[y][x], dict) and (tiles[y][x].get("kind") == "PLANT" or "animal" in tiles[y][x])
            )
            daily.append({
                "day": obs.day, "money": obs.farms[0]["money"], "occupied": occupied,
                "targets": dict(targets),
            })
    final_cash = float(state[0].reward)
    return final_cash, daily


def test_final_cash_beats_starting_money():
    """The whole point of the agent: end the season with more than it
    started with, playing alone against a passive opponent (no market
    competition to blame a loss on)."""
    final_cash, _ = run_solo(seed=1)
    assert final_cash > STARTING_MONEY, f"lost money overall: final={final_cash}"


def test_day_zero_fills_all_25_opening_tiles():
    _, daily = run_solo(seed=1)
    day_one = next(row for row in daily if row["day"] == 1)
    assert day_one["occupied"] == 25


def test_thirty_day_game_never_starts_production_past_first_yield_deadline():
    config = configuration(1)
    config['episodeSteps'] = 30 * 24
    env = make('kaggriculture', configuration=config, debug=False)
    # Deliberately pass the old erroneous endpoint: actual config must win.
    env.run([make_agent(30, seed=1), pass_agent])
    seen = set()
    for step in env.steps:
        for y, row in enumerate(step[0].observation.farms[0]['tiles']):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict):
                    continue
                name = tile.get('animal') or tile.get('crop')
                if not name:
                    continue
                start = tile.get('placed_day', tile.get('planted_day'))
                key = (x, y, name, start)
                if key in seen:
                    continue
                seen.add(key)
                rules = official_game.ANIMALS if tile.get('animal') else official_game.CROPS
                assert start + rules[name]['first_yield_day'] <= 29, key
    assert seen


def test_money_does_not_stay_near_zero_past_day_15():
    """Regression test for the cash-flow deadlock repeatedly found by hand:
    money oscillating in the $0-$100 range for most of the game while a
    handful of tiles cycle between planted and empty, instead of growing as
    more tiles come online. By day 15 (half the season), the farm should
    have visibly more cash than it started with, not be flirting with zero."""
    _, daily = run_solo(seed=1)
    after_day_15 = [row["money"] for row in daily if row["day"] >= 15]
    assert after_day_15, "no snapshots past day 15"
    assert max(after_day_15) > 500, f"money never climbed past day 15: {after_day_15[:10]}"


def test_occupied_tiles_grow_and_stay_up_not_oscillate_to_near_zero():
    """Regression test: occupied tiles repeatedly collapsing back down to a
    handful (crops dying / not being replanted for lack of cash) even after
    the farm had earlier reached a healthy occupancy is the signature of the
    cash-flow deadlock this test suite was written to catch."""
    _, daily = run_solo(seed=1)
    late = [row["occupied"] for row in daily if row["day"] >= 20]
    assert late, "no snapshots past day 20"
    assert min(late) >= 10, f"occupancy collapsed late-game: {late}"


def test_portfolio_uses_both_crops_and_animals():
    """Regression test: an earlier scoring formula (raw ROI, then ROI/day
    without a category term) picked crops for the entire board and never
    touched an animal -- losing the WOOL/MILK/EGG market entirely and the
    daily FERTILIZER income animals produce as a side effect of being fed."""
    _, daily = run_solo(seed=1)
    final_targets = daily[-1]["targets"]
    names = {value[0] for value in final_targets.values() if value is not None}
    crops = {"WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY"}
    animals = {"GOOSE", "COW", "SHEEP"}
    assert names & crops, f"no crop in final portfolio: {names}"
    assert names & animals, f"no animal in final portfolio: {names}"


def test_animal_buys_are_followed_by_same_day_place_actions():
    """An animal bought from same-day fertilizer proceeds is placed that day."""
    env = make("kaggriculture", configuration=configuration(1), debug=False)
    agent = make_agent(END_DAY, seed=1)
    state = env.state
    bought_days, placed_days = set(), set()
    for step in range(int(env.configuration.episodeSteps) - 1):
        obs = state[0].observation
        action = agent(obs)
        if any(order and order[0] == "BUY_ANIMAL" for order in action.get("market", [])):
            bought_days.add(obs.day)
        worker_ops = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
        if any(op and op[0] == "PLACE" for op in worker_ops):
            placed_days.add(obs.day)
        state[0].action = action
        state[1].action = pass_agent(state[1].observation)
        state = official_game.interpreter(state, env)
        state[0].observation.step = step + 1
    assert 2 in bought_days
    assert 2 in placed_days


def test_day_three_places_converted_cow_on_the_nearest_conversion_tile():
    """Checking PLACE alone is insufficient: it must land on (4,3), the
    nearest WHEAT-to-COW conversion tile, and mutate that exact pasture."""
    env = make("kaggriculture", configuration=configuration(1), debug=False)
    env.run([make_agent(END_DAY, seed=1), pass_agent])

    matches = []
    for step in env.steps:
        state = step[0]
        obs = state.observation
        if obs.day != 2:
            continue
        action = state.action or {}
        operations = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
        positions = [tuple(obs.farms[0].farmer), *map(tuple, obs.farms[0].hands)]
        for operation, position in zip(operations, positions):
            if operation == ["PLACE", "COW"]:
                matches.append((obs.hour, position, obs.farms[0].tiles[position[1]][position[0]]))

    assert len(matches) == 1
    hour, position, tile = matches[0]
    assert hour < 24
    assert position == (4, 3)
    assert tile["kind"] == "PASTURE"
    assert tile["animal"] == "COW"
    assert tile["placed_day"] == 2


def test_conversion_batch_harvests_two_wheat_and_buys_cow_without_market_feed():
    env = make("kaggriculture", configuration=configuration(1), debug=False)
    env.run([make_agent(END_DAY, seed=1), pass_agent])

    conversion_harvests = []
    cow_purchase_passes = []
    for step in env.steps:
        state = step[0]
        obs = state.observation
        action = state.action or {}
        operations = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
        positions = [tuple(obs.farms[0].farmer), *map(tuple, obs.farms[0].hands)]
        if obs.day == 2:
            conversion_harvests += [
                position
                for operation, position in zip(operations, positions)
                if operation == ["HARVEST"] and position in {(4, 3), (3, 3)}
            ]
        market = action.get("market", [])
        if obs.day == 2 and ["BUY_ANIMAL", "COW", 1] in market:
            cow_purchase_passes.append(market)

    assert set(conversion_harvests) == {(4, 3), (3, 3)}
    assert cow_purchase_passes
    assert all(
        not any(order[0] == "BUY_PRODUCT" and order[1] == "WHEAT" for order in market)
        for market in cow_purchase_passes
    )


def test_day_five_replacement_seeds_follow_planner_targets():
    """UI Day 5 is engine index 4: replacement purchases must reflect the
    planner's repriced targets rather than the seven-WHEAT opening book."""
    env = make("kaggriculture", configuration=configuration(1), debug=False)
    env.run([make_agent(END_DAY, seed=1, opening_version="melon_v2"), pass_agent])

    actions = [
        step[0].action or {} for step in env.steps
        if step[0].observation.day == 4
    ]
    market = [order for action in actions for order in action.get("market", [])]
    wheat_seeds = sum(
        int(order[2])
        for order in market
        if order[:2] == ["BUY_SEED", "WHEAT"]
    )
    day_five_end = next(
        step[0].observation
        for step in env.steps
        if step[0].observation.day == 5 and step[0].observation.hour == 0
    )
    replanted_wheat = [
        tile
        for row in day_five_end.farms[0]["tiles"]
        for tile in row
        if isinstance(tile, dict)
        and tile.get("crop") == "WHEAT"
        and tile.get("planted_day") == 4
    ]
    assert len(replanted_wheat) == wheat_seeds
    assert wheat_seeds < 7
    weeds = [
        tile for step in env.steps if step[0].observation.day in (4, 5)
        for row in step[0].observation.farms[0]["tiles"]
        for tile in row
        if isinstance(tile, dict) and tile.get("kind") == "WEED"
    ]
    assert not weeds, "a tile decayed to WEED instead of just replanting a day late"


def test_every_opening_animal_is_fed_on_each_verified_feed_age():
    env = make("kaggriculture", configuration=configuration(1), debug=False)
    env.run([make_agent(END_DAY, seed=1), pass_agent])

    for step in env.steps:
        obs = step[0].observation
        if not (1 <= obs.day <= 10 and obs.hour == 0):
            continue
        animals = [
            tile
            for row in obs.farms[0]["tiles"]
            for tile in row
            if isinstance(tile, dict) and tile.get("animal")
        ]
        required_yesterday = [
            tile for tile in animals
            if should_feed_animal(tile["animal"], obs.day - 1 - tile["placed_day"])
        ]
        assert all(tile.get("consecutive_unfed", 0) == 0 for tile in required_yesterday), (
            obs.day,
            [(tile["animal"], tile.get("consecutive_unfed", 0)) for tile in required_yesterday],
        )


def test_day_three_places_cow_on_the_nearest_conversion_pasture():
    env = make("kaggriculture", configuration=configuration(1), debug=False)
    env.run([make_agent(END_DAY, seed=1), pass_agent])

    placements = []
    for step in env.steps:
        state = step[0]
        obs = state.observation
        if obs.day != 2:
            continue
        action = state.action or {}
        operations = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
        positions = [tuple(obs.farms[0]["farmer"]), *map(tuple, obs.farms[0]["hands"])]
        placements += [
            (position, operation)
            for position, operation in zip(positions, operations)
            if operation and operation[0] == "PLACE"
        ]

    assert ((4, 3), ["PLACE", "COW"]) in placements
    next_day = next(
        step[0].observation
        for step in env.steps
        if step[0].observation.day == 3 and step[0].observation.hour == 0
    )
    tile = next_day.farms[0]["tiles"][3][4]
    assert tile["kind"] == "PASTURE"
    assert tile["animal"] == "COW"
    assert tile["placed_day"] == 2


def test_sixth_opening_animal_is_placed_on_day_index_three():
    env = make("kaggriculture", configuration=configuration(1), debug=False)
    env.run([make_agent(END_DAY, seed=1), pass_agent])

    day_four = next(
        step[0].observation
        for step in env.steps
        if step[0].observation.day == 4 and step[0].observation.hour == 0
    )
    animals = [
        tile
        for row in day_four.farms[0]["tiles"]
        for tile in row
        if isinstance(tile, dict) and tile.get("animal")
    ]
    assert len(animals) == 6
    sixth = day_four.farms[0]["tiles"][3][3]
    assert sixth["animal"] == "SHEEP"
    assert sixth["placed_day"] == 3


def test_day_three_late_placement_does_not_drop_crop_work():
    env = make("kaggriculture", configuration=configuration(1), debug=False)
    env.run([make_agent(END_DAY, seed=1), pass_agent])

    for day in (4, 5, 6):
        observation = next(
            step[0].observation
            for step in env.steps
            if step[0].observation.day == day and step[0].observation.hour == 0
        )
        weeds = [
            (x, y)
            for y, row in enumerate(observation.farms[0]["tiles"])
            for x, tile in enumerate(row)
            if isinstance(tile, dict) and tile.get("kind") == "WEED"
        ]
        assert weeds == [], (day, weeds)


def test_ready_animal_output_is_harvested_the_same_day():
    """Ready animal output doesn't need to be collected the instant it's
    ready -- only before the day it became ready ends, so it never carries
    over into the next day's yield_units cap or sits there indefinitely."""
    env = make("kaggriculture", configuration=configuration(1), debug=False)
    env.run([make_agent(END_DAY, seed=1), pass_agent])

    for step in env.steps:
        observation = step[0].observation
        if observation.hour < 23:
            continue
        ready = [
            (x, y, tile["animal"], tile.get("yield_units", 0))
            for y, row in enumerate(observation.farms[0]["tiles"])
            for x, tile in enumerate(row)
            if isinstance(tile, dict)
            and tile.get("animal")
            and tile.get("yield_units", 0) > 0
        ]
        assert ready == [], (observation.day, observation.hour, ready)


if __name__ == "__main__":
    import sys
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"{'ALL PASSED' if not failures else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)
