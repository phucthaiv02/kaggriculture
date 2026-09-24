"""Find the first crop -> WEED transition in the opening simulation.

The useful question is not which later rolling rebuild notices the damage, but
whether the dying tile still had a scheduled survival task immediately before
the engine ended the day.  Fail at that first transition with the planner,
route and generated-task state for only the affected positions.
"""

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official_game

from agents.expansion_agent import make_agent
from agents.farm_tasks import build_tasks
from agents.rolling_scheduler import planning_observation
from experiments.crop_schedules import pass_agent
from tests.test_agents_integration import END_DAY, configuration


def _tile_summary(tile):
    if not isinstance(tile, dict):
        return tile
    return {
        key: tile.get(key)
        for key in (
            "kind", "crop", "animal", "planted_day", "placed_day",
            "yield_units", "watered_today", "consecutive_unwatered",
            "fed_today", "consecutive_unfed", "fertilized_until_day",
        )
        if key in tile
    }


def _task_summary(task):
    return {
        "position": task.position,
        "actions": task.actions,
        "needs": dict(task.needs),
        "mandatory": task.mandatory,
        "urgent": task.urgent,
        "sells": dict(task.sells),
    }


def _plan_summary(plans, limit=14):
    return [
        {
            "start": tuple(plan.start),
            "queue": list(plan.queue[:limit]),
            "queue_len": len(plan.queue),
        }
        for plan in plans
    ]


def test_first_opening_weed_transition_keeps_its_survival_task():
    env = make("kaggriculture", configuration=configuration(1), debug=False)
    agent = make_agent(END_DAY, seed=1)
    cells = {
        name: cell.cell_contents
        for name, cell in zip(agent.__code__.co_freevars, agent.__closure__)
    }
    planner_state = cells["state"]
    state = env.state

    for step in range(int(env.configuration.episodeSteps) - 1):
        obs = state[0].observation
        before_tiles = obs.farms[0]["tiles"]
        action = agent(obs)

        # Rebuild the same physical/daily task view from the pre-transition
        # observation.  This is diagnostic only: if WATER is absent here the
        # bug is task generation/admission; if present but the tile still dies,
        # the bug is route/execution coverage.
        generated = build_tasks(
            planning_observation(obs),
            planner_state.get("daily_targets", {}),
            prioritize_fertilizer_drop=planner_state.get("opening_active", False),
        )
        generated_by_position = {}
        for task in generated:
            generated_by_position.setdefault(task.position, []).append(task)

        positions = [
            tuple(obs.farms[0]["farmer"]),
            *map(tuple, obs.farms[0]["hands"]),
        ]
        operations = [
            action.get("farmer", ["PASS"]),
            *action.get("hands", []),
        ]
        worker_snapshot = [
            {
                "worker": index,
                "position": position,
                "operation": operation,
                "inventory": dict(obs["private"]["inventories"][index]),
            }
            for index, (position, operation)
            in enumerate(zip(positions, operations))
        ]
        route_snapshot = _plan_summary(planner_state.get("plans", ()))
        unassigned_snapshot = [
            _task_summary(task) for task in planner_state.get("unassigned", ())
        ]
        frozen_snapshot = set(planner_state.get("frozen_positions", ()))
        daily_targets = dict(planner_state.get("daily_targets", {}))

        state[0].action = action
        state[1].action = pass_agent(state[1].observation)
        state = official_game.interpreter(state, env)
        state[0].observation.step = step + 1
        next_obs = state[0].observation

        new_weeds = []
        for y, row in enumerate(next_obs.farms[0]["tiles"]):
            for x, tile in enumerate(row):
                before = before_tiles[y][x]
                if (isinstance(tile, dict) and tile.get("kind") == "WEED"
                        and isinstance(before, dict)
                        and before.get("kind") == "PLANT"):
                    new_weeds.append((x, y))

        if new_weeds:
            affected = {}
            for position in new_weeds:
                x, y = position
                affected[position] = {
                    "before": _tile_summary(before_tiles[y][x]),
                    "after": _tile_summary(next_obs.farms[0]["tiles"][y][x]),
                    "target": daily_targets.get(position),
                    "frozen": position in frozen_snapshot,
                    "generated": [
                        _task_summary(task)
                        for task in generated_by_position.get(position, ())
                    ],
                    "unassigned": [
                        task for task in unassigned_snapshot
                        if tuple(task["position"]) == position
                    ],
                }
            raise AssertionError({
                "transition_from": (obs.day, obs.hour),
                "transition_to": (next_obs.day, next_obs.hour),
                "new_weeds": new_weeds,
                "affected": affected,
                "workers": worker_snapshot,
                "plans_after_dispatch": route_snapshot,
                "hands": len(obs.farms[0]["hands"]),
                "hand_target": planner_state.get("hand_target"),
                "mandatory_hand_target": planner_state.get("mandatory_hand_target"),
                "route_invalidated": planner_state.get("route_invalidated"),
                "replan_needed": planner_state.get("replan_needed"),
            })

        if next_obs.day > 4:
            raise AssertionError("no crop -> WEED transition found through day 4")

    raise AssertionError("simulation ended before the diagnostic horizon")
