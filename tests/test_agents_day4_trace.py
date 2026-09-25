"""Focused regressions/diagnostics for opening crop turnover and survival.

The Day-4 WHEAT schedule is a useful stress case for rolling execution:
successor PLANT+WATER chains admitted that morning must survive every
intraday route rebuild after HARVEST changes tile state.
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
        )
        if key in tile
    }


def _compact_task(task):
    return (task.position, tuple(tuple(op) for op in task.actions), task.mandatory)


def test_all_opening_wheat_successors_finish_on_day_four():
    env = make("kaggriculture", configuration=configuration(1), debug=False)
    agent = make_agent(END_DAY, seed=1)
    cells = {
        name: cell.cell_contents
        for name, cell in zip(agent.__code__.co_freevars, agent.__closure__)
    }
    planner_state = cells["state"]
    state = env.state

    live_wheat_at_start = set()
    scheduled_wheat = set()
    frozen_wheat = set()
    purchase_positions = set()
    day_four_start_hands = None
    hand_target = None
    mandatory_hand_target = None
    trace = []
    unassigned_trace = []

    for step in range(int(env.configuration.episodeSteps) - 1):
        obs = state[0].observation

        if obs.day == 4 and obs.hour == 0:
            day_four_start_hands = len(obs.farms[0]["hands"])
            live_wheat_at_start = {
                (x, y)
                for y, row in enumerate(obs.farms[0]["tiles"])
                for x, tile in enumerate(row)
                if isinstance(tile, dict) and tile.get("crop") == "WHEAT"
            }

        action = agent(obs)

        if obs.day == 4 and obs.hour == 0:
            scheduled_wheat = {
                position
                for position, target in planner_state["daily_targets"].items()
                if target and target[0] == "WHEAT"
            }
            purchase_positions = set(planner_state["purchase_positions"])
            hand_target = planner_state["hand_target"]
            mandatory_hand_target = planner_state["mandatory_hand_target"]

        if obs.day == 4 and obs.hour == 1:
            frozen_wheat = {
                position
                for position in planner_state["frozen_positions"]
                if planner_state["daily_targets"].get(position)
                and planner_state["daily_targets"][position][0] == "WHEAT"
            }

        if obs.day == 4:
            if planner_state["unassigned"]:
                unassigned_trace.append((
                    obs.hour,
                    tuple(_compact_task(task) for task in planner_state["unassigned"]),
                ))
            positions = [
                tuple(obs.farms[0]["farmer"]),
                *map(tuple, obs.farms[0]["hands"]),
            ]
            operations = [
                action.get("farmer", ["PASS"]),
                *action.get("hands", []),
            ]
            for worker, (position, operation) in enumerate(zip(positions, operations)):
                if position in scheduled_wheat or (
                    operation and operation[0] in ("HARVEST", "PLANT", "WATER", "DROP", "PICKUP")
                ):
                    x, y = position
                    trace.append((
                        obs.hour,
                        worker,
                        position,
                        tuple(operation),
                        tuple(sorted(obs["private"]["inventories"][worker].items())),
                        _tile_summary(obs.farms[0]["tiles"][y][x]),
                    ))

        state[0].action = action
        state[1].action = pass_agent(state[1].observation)
        state = official_game.interpreter(state, env)
        state[0].observation.step = step + 1

        next_obs = state[0].observation
        if next_obs.day == 5 and next_obs.hour == 0:
            replanted = {
                (x, y)
                for y, row in enumerate(next_obs.farms[0]["tiles"])
                for x, tile in enumerate(row)
                if isinstance(tile, dict)
                and tile.get("crop") == "WHEAT"
                and tile.get("planted_day") == 4
            }
            missing = scheduled_wheat - replanted
            missing_trace = [event for event in trace if event[2] in missing]
            missing_unassigned = [
                (hour, tuple(task for task in tasks if task[0] in missing))
                for hour, tasks in unassigned_trace
                if any(task[0] in missing for task in tasks)
            ]
            assert not missing, (
                f"missing={sorted(missing)} replanted={sorted(replanted)} "
                f"live_start={sorted(live_wheat_at_start)} "
                f"frozen={sorted(frozen_wheat)} "
                f"purchase={sorted(scheduled_wheat & purchase_positions)} "
                f"hands={day_four_start_hands}/{hand_target}/{mandatory_hand_target} "
                f"missing_unassigned={missing_unassigned} "
                f"missing_trace={missing_trace}"
            )
            return

    raise AssertionError("simulation never reached day 5 hour 0")


def test_no_opening_crop_turns_to_weed_through_day_five():
    """Protect every live crop, not just the Day-4 WHEAT turnover cohort."""
    env = make("kaggriculture", configuration=configuration(1), debug=False)
    agent = make_agent(END_DAY, seed=1)
    cells = {
        name: cell.cell_contents
        for name, cell in zip(agent.__code__.co_freevars, agent.__closure__)
    }
    planner_state = cells["state"]
    targets = cells["targets"]
    state = env.state
    history = {}

    for step in range(int(env.configuration.episodeSteps) - 1):
        obs = state[0].observation
        before_day, before_hour = obs.day, obs.hour
        before_snapshot = {
            (x, y): _tile_summary(tile)
            for y, row in enumerate(obs.farms[0]["tiles"])
            for x, tile in enumerate(row)
        }
        action = agent(obs)

        if 3 <= before_day <= 5:
            worker_positions = [
                tuple(obs.farms[0]["farmer"]),
                *map(tuple, obs.farms[0]["hands"]),
            ]
            worker_ops = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
            ops_by_position = {
                position: tuple(operation)
                for position, operation in zip(worker_positions, worker_ops)
                if operation != ["PASS"]
            }
            frozen = set(planner_state.get("frozen_positions", ()))
            frozen_targets = {position: targets.get(position) for position in frozen}
            generated = build_tasks(
                planning_observation(obs),
                frozen_targets,
                prioritize_fertilizer_drop=planner_state.get("opening_active", False),
                include_physical=False,
            )
            generated_by_position = {
                task.position: tuple(op[0] for op in task.actions) for task in generated
            }
            unassigned_positions = {
                task.position for task in planner_state.get("unassigned", ())
            }
            for position, tile in before_snapshot.items():
                if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
                    continue
                history.setdefault(position, []).append((
                    before_day,
                    before_hour,
                    tile.get("crop"),
                    tile.get("yield_units", 0),
                    bool(tile.get("watered_today")),
                    tile.get("consecutive_unwatered", 0),
                    position in frozen,
                    position in unassigned_positions,
                    generated_by_position.get(position),
                    ops_by_position.get(position),
                ))

        state[0].action = action
        state[1].action = pass_agent(state[1].observation)
        state = official_game.interpreter(state, env)
        state[0].observation.step = step + 1
        next_obs = state[0].observation

        if 3 <= before_day <= 5:
            transitions = []
            for position, before in before_snapshot.items():
                x, y = position
                after = next_obs.farms[0]["tiles"][y][x]
                if (
                    isinstance(before, dict)
                    and before.get("kind") == "PLANT"
                    and isinstance(after, dict)
                    and after.get("kind") == "WEED"
                ):
                    transitions.append(position)
            if transitions:
                compact = {
                    position: [
                        event for event in history.get(position, [])
                        if event[0] == before_day
                    ]
                    for position in transitions
                }
                raise AssertionError(
                    f"transition={before_day}:{before_hour}->{next_obs.day}:{next_obs.hour}; "
                    f"positions={transitions}; hands={len(obs.farms[0]['hands'])}; "
                    f"hand_target={planner_state.get('hand_target')}; "
                    f"mandatory_hand_target={planner_state.get('mandatory_hand_target')}; "
                    f"trace={compact}"
                )

        if next_obs.day > 5:
            return

    raise AssertionError("simulation ended before day five crop survival trace completed")
