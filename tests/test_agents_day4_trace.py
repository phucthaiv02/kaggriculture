"""Focused regression/diagnostic for the opening Day-4 WHEAT turnover.

The Day-4 WHEAT schedule is a useful stress case for rolling execution:
successor PLANT+WATER chains admitted that morning must survive every
intraday route rebuild after HARVEST changes tile state.
"""

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official_game

from agents.expansion_agent import make_agent
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
                    trace.append({
                        "hour": obs.hour,
                        "worker": worker,
                        "position": position,
                        "operation": operation,
                        "tile": _tile_summary(obs.farms[0]["tiles"][y][x]),
                        "hands": len(obs.farms[0]["hands"]),
                        "inventory": dict(obs["private"]["inventories"][worker]),
                        "wheat_seeds": obs["private"]["seeds"].get("WHEAT", 0),
                    })

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
            missing_trace = [
                event for event in trace if event["position"] in missing
            ]
            final_tiles = {
                position: _tile_summary(
                    next_obs.farms[0]["tiles"][position[1]][position[0]]
                )
                for position in sorted(scheduled_wheat)
            }
            assert len(replanted) >= len(scheduled_wheat) and not missing, {
                "live_wheat_at_start": sorted(live_wheat_at_start),
                "scheduled_wheat": sorted(scheduled_wheat),
                "frozen_wheat": sorted(frozen_wheat),
                "purchase_wheat": sorted(scheduled_wheat & purchase_positions),
                "missing": sorted(missing),
                "replanted": sorted(replanted),
                "day_four_start_hands": day_four_start_hands,
                "hand_target": hand_target,
                "mandatory_hand_target": mandatory_hand_target,
                "day_five_hands": len(next_obs.farms[0]["hands"]),
                "missing_trace": missing_trace,
                "all_trace": trace,
                "final_tiles": final_tiles,
            }
            return

    raise AssertionError("simulation never reached day 5 hour 0")
