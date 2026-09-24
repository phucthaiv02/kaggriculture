"""Focused diagnostic for the opening day-3 animal unlock.

The sixth opening animal is financed intraday. When BUY_ANIMAL unlocks PLACE,
rolling routing must incorporate that successor without dropping already-admitted
crop survival work. This test keeps the failure local to the first bad day and
prints the rolling candidate state needed to diagnose either side of that trade.
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
            "fed_today", "consecutive_unfed",
        )
        if key in tile
    }


def _task_summary(task):
    return {
        "position": task.position,
        "actions": task.actions,
        "needs": dict(task.needs),
        "mandatory": task.mandatory,
    }


def test_intraday_animal_unlock_keeps_all_day_three_survival_work():
    env = make("kaggriculture", configuration=configuration(1), debug=False)
    agent = make_agent(END_DAY, seed=1)
    cells = {
        name: cell.cell_contents
        for name, cell in zip(agent.__code__.co_freevars, agent.__closure__)
    }
    planner_state = cells["state"]
    state = env.state
    trace = []

    for step in range(int(env.configuration.episodeSteps) - 1):
        obs = state[0].observation
        action = agent(obs)

        if obs.day == 3:
            positions = [
                tuple(obs.farms[0]["farmer"]),
                *map(tuple, obs.farms[0]["hands"]),
            ]
            operations = [
                action.get("farmer", ["PASS"]),
                *action.get("hands", []),
            ]
            market = action.get("market", [])
            unassigned = list(planner_state.get("unassigned", ()))
            overdue = []
            weeds = []
            for y, row in enumerate(obs.farms[0]["tiles"]):
                for x, tile in enumerate(row):
                    if not isinstance(tile, dict):
                        continue
                    if tile.get("kind") == "WEED":
                        weeds.append((x, y))
                    if (tile.get("crop") and not tile.get("watered_today")
                            and tile.get("consecutive_unwatered", 0) >= 1):
                        overdue.append((x, y, tile.get("crop")))

            # Keep every state transition around the market unlock and every
            # hour where admitted work is reported missing. Ordinary movement
            # hours are summarized only when a survival signal is present.
            interesting = bool(
                market
                or unassigned
                or overdue
                or weeds
                or any(op and op[0] in ("WATER", "HARVEST", "PLANT", "PLACE")
                       for op in operations)
            )
            if interesting:
                trace.append({
                    "hour": obs.hour,
                    "money": obs.farms[0]["money"],
                    "hands": len(obs.farms[0]["hands"]),
                    "market": market,
                    "workers": [
                        {
                            "worker": index,
                            "position": position,
                            "operation": operation,
                            "inventory": dict(obs["private"]["inventories"][index]),
                            "tile": _tile_summary(
                                obs.farms[0]["tiles"][position[1]][position[0]]
                            ),
                        }
                        for index, (position, operation)
                        in enumerate(zip(positions, operations))
                    ],
                    "unassigned": [_task_summary(task) for task in unassigned],
                    "overdue": overdue,
                    "weeds": weeds,
                    "route_invalidated": planner_state.get("route_invalidated"),
                    "replan_needed": planner_state.get("replan_needed"),
                    "frozen": sorted(planner_state.get("frozen_positions", ())),
                })

        state[0].action = action
        state[1].action = pass_agent(state[1].observation)
        state = official_game.interpreter(state, env)
        state[0].observation.step = step + 1

        next_obs = state[0].observation
        if next_obs.day == 4 and next_obs.hour == 0:
            animals = []
            weeds = []
            overdue = []
            for y, row in enumerate(next_obs.farms[0]["tiles"]):
                for x, tile in enumerate(row):
                    if not isinstance(tile, dict):
                        continue
                    if tile.get("animal"):
                        animals.append((x, y, tile.get("animal")))
                    if tile.get("kind") == "WEED":
                        weeds.append((x, y))
                    if (tile.get("crop") and not tile.get("watered_today")
                            and tile.get("consecutive_unwatered", 0) >= 1):
                        overdue.append((x, y, tile.get("crop"),
                                        tile.get("consecutive_unwatered")))

            assert len(animals) == 6 and not weeds, {
                "animals": animals,
                "weeds": weeds,
                "overdue_at_day4": overdue,
                "trace": trace,
            }
            return

    raise AssertionError("simulation never reached day 4 hour 0")
