"""Trace the admitted animal dependency immediately after an intraday BUY.

This is a focused diagnostic for the opening conversion path.  A successful
BUY_ANIMAL should materialize PICKUP -> PLACE (and FEED when age-0 feeding is
required) for the already-admitted target without reopening target selection.
"""

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official_game

from agents.expansion_agent import make_agent
from agents.farm_tasks import ANIMALS, build_tasks
from agents.rolling_scheduler import planning_observation
from experiments.crop_schedules import pass_agent
from tests.test_agents_integration import END_DAY, configuration


def _task_summary(task):
    return {
        "position": task.position,
        "actions": task.actions,
        "needs": dict(task.needs),
        "mandatory": task.mandatory,
    }


def _tile_summary(tile):
    if not isinstance(tile, dict):
        return tile
    return {
        key: tile.get(key)
        for key in (
            "kind", "crop", "animal", "planted_day", "placed_day",
            "yield_units", "fed_today", "consecutive_unfed",
        )
        if key in tile
    }


def test_trace_intraday_animal_unlock_task_materialization():
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
    buy_orders = None

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
                buy_orders = animal_orders
            elif buy_hour is not None and obs.hour == buy_hour + 1:
                frozen = set(planner_state.get("frozen_positions", ()))
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
                animal_targets = {
                    position: target for position, target in targets.items()
                    if target and target[0] in ANIMALS
                }
                empty_animal_targets = {
                    position: target
                    for position, target in animal_targets.items()
                    if not (
                        isinstance(obs.farms[0]["tiles"][position[1]][position[0]], dict)
                        and obs.farms[0]["tiles"][position[1]][position[0]].get("animal")
                        == target[0]
                    )
                }
                raise AssertionError({
                    "buy_hour": buy_hour,
                    "buy_orders": buy_orders,
                    "hour": obs.hour,
                    "money": obs.farms[0]["money"],
                    "shed": {
                        key: obs["private"]["shed"].get(key, 0)
                        for key in ("WHEAT", "COW", "SHEEP")
                    },
                    "animal_targets": animal_targets,
                    "empty_animal_targets": empty_animal_targets,
                    "empty_target_tiles": {
                        position: _tile_summary(
                            obs.farms[0]["tiles"][position[1]][position[0]]
                        )
                        for position in empty_animal_targets
                    },
                    "frozen_animal_targets": {
                        position: target
                        for position, target in frozen_targets.items()
                        if target and target[0] in ANIMALS
                    },
                    "daily_animal_targets": {
                        position: target
                        for position, target in planner_state.get("daily_targets", {}).items()
                        if target and target[0] in ANIMALS
                    },
                    "generated_animal_tasks": [
                        _task_summary(task) for task in replanned
                        if frozen_targets.get(task.position)
                        and frozen_targets[task.position][0] in ANIMALS
                    ],
                    "all_generated_tasks": [_task_summary(task) for task in replanned],
                    "plans": [list(plan.queue[:16]) for plan in planner_state.get("plans", ())],
                    "unassigned": [
                        _task_summary(task)
                        for task in planner_state.get("unassigned", ())
                    ],
                    "frozen_count": len(frozen),
                    "route_invalidated": planner_state.get("route_invalidated"),
                    "replan_needed": planner_state.get("replan_needed"),
                })

        state[0].action = action
        state[1].action = pass_agent(state[1].observation)
        state = official_game.interpreter(state, env)
        state[0].observation.step = step + 1

        if state[0].observation.day > 3 and buy_hour is None:
            raise AssertionError("no day-three animal BUY found")

    raise AssertionError("simulation ended before the day-three unlock trace")
