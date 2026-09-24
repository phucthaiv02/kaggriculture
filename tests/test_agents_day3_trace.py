"""Opening regression for an admitted animal dependency unlocked by BUY.

A successful intraday BUY_ANIMAL must materialize PICKUP -> PLACE for the
already-admitted target.  The target stays inside the frozen daily schedule
while its market input is in transit; rolling execution may reroute it but may
not silently forget it.
"""

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official_game

from agents.expansion_agent import make_agent
from agents.farm_tasks import ANIMALS, build_tasks
from agents.rolling_scheduler import planning_observation
from experiments.crop_schedules import pass_agent
from tests.test_agents_integration import END_DAY, configuration


def _animal_targets(targets):
    return {
        position: target for position, target in targets.items()
        if target and target[0] in ANIMALS
    }


def _empty_animal_targets(obs, targets):
    return {
        position: target
        for position, target in _animal_targets(targets).items()
        if not (
            isinstance(obs.farms[0]["tiles"][position[1]][position[0]], dict)
            and obs.farms[0]["tiles"][position[1]][position[0]].get("animal")
            == target[0]
        )
    }


def test_intraday_animal_buy_materializes_admitted_place_dependency():
    env = make("kaggriculture", configuration=configuration(1), debug=False)
    agent = make_agent(END_DAY, seed=1)
    cells = {
        name: cell.cell_contents
        for name, cell in zip(agent.__code__.co_freevars, agent.__closure__)
    }
    planner_state = cells["state"]
    targets = cells["targets"]
    state = env.state
    buy_hour = None
    missing_position = None
    materialized = False

    for step in range(int(env.configuration.episodeSteps) - 1):
        obs = state[0].observation
        action = agent(obs)

        if obs.day == 3:
            animal_orders = [
                order for order in action.get("market", [])
                if order[:1] == ["BUY_ANIMAL"]
            ]
            if animal_orders and buy_hour is None:
                buy_hour = obs.hour
            elif buy_hour is not None and obs.hour == buy_hour + 1:
                empty_targets = _empty_animal_targets(obs, targets)
                assert len(empty_targets) == 1, empty_targets
                missing_position, missing_target = next(iter(empty_targets.items()))

                frozen = set(planner_state.get("frozen_positions", ()))
                assert missing_position in frozen, (
                    missing_position, sorted(frozen), planner_state.get("purchase_positions")
                )

                frozen_targets = {
                    position: targets.get(position)
                    for position in frozen
                }
                replanned = build_tasks(
                    planning_observation(obs),
                    frozen_targets,
                    prioritize_fertilizer_drop=planner_state.get("opening_active", False),
                    include_physical=False,
                )
                target_tasks = [
                    task for task in replanned if task.position == missing_position
                ]
                assert target_tasks, (
                    missing_position,
                    missing_target,
                    dict(obs["private"]["shed"]),
                    frozen_targets,
                )
                assert any(
                    ["PLACE", missing_target[0]] in task.actions
                    and task.needs.get(missing_target[0], 0) >= 1
                    for task in target_tasks
                ), [(task.actions, dict(task.needs)) for task in target_tasks]
                materialized = True

        if obs.day == 4 and obs.hour == 0:
            assert buy_hour is not None
            assert materialized
            assert missing_position is not None
            x, y = missing_position
            tile = obs.farms[0]["tiles"][y][x]
            assert isinstance(tile, dict) and tile.get("animal") == targets[missing_position][0], (
                missing_position, tile, planner_state.get("plans")
            )
            animal_count = sum(
                1
                for row in obs.farms[0]["tiles"]
                for current in row
                if isinstance(current, dict) and current.get("animal")
            )
            assert animal_count == 6, animal_count
            return

        state[0].action = action
        state[1].action = pass_agent(state[1].observation)
        state = official_game.interpreter(state, env)
        state[0].observation.step = step + 1

    raise AssertionError("simulation ended before the opening animal regression completed")
