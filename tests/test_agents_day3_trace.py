"""Focused diagnostic for the opening day-3 animal unlock.

The sixth opening animal is financed intraday. When BUY_ANIMAL unlocks PLACE,
rolling routing must incorporate that successor without dropping already-admitted
crop survival work. Fail at the first incomplete rolling candidate so CI prints
only the causal state instead of a whole-day trace.
"""

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official_game

from agents.expansion_agent import make_agent
from experiments.crop_schedules import pass_agent
from tests.test_agents_integration import END_DAY, configuration


def _task_summary(task):
    return {
        "position": task.position,
        "actions": task.actions,
        "needs": dict(task.needs),
        "mandatory": task.mandatory,
    }


def _board_signals(obs):
    overdue, weeds = [], []
    for y, row in enumerate(obs.farms[0]["tiles"]):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "WEED":
                weeds.append((x, y))
            if (tile.get("crop") and not tile.get("watered_today")
                    and tile.get("consecutive_unwatered", 0) >= 1):
                overdue.append((x, y, tile.get("crop"),
                                tile.get("consecutive_unwatered")))
    return overdue, weeds


def test_intraday_animal_unlock_keeps_all_day_three_survival_work():
    env = make("kaggriculture", configuration=configuration(1), debug=False)
    agent = make_agent(END_DAY, seed=1)
    cells = {
        name: cell.cell_contents
        for name, cell in zip(agent.__code__.co_freevars, agent.__closure__)
    }
    planner_state = cells["state"]
    state = env.state
    saw_intraday_animal_buy = False
    buy_hour = None

    for step in range(int(env.configuration.episodeSteps) - 1):
        obs = state[0].observation
        action = agent(obs)

        if obs.day == 3:
            market = action.get("market", [])
            if any(order[:1] == ["BUY_ANIMAL"] for order in market):
                saw_intraday_animal_buy = True
                buy_hour = obs.hour

            if saw_intraday_animal_buy and obs.hour > buy_hour:
                mandatory_unassigned = [
                    task for task in planner_state.get("unassigned", ())
                    if task.mandatory is not False
                ]
                overdue, weeds = _board_signals(obs)
                if mandatory_unassigned:
                    positions = [
                        tuple(obs.farms[0]["farmer"]),
                        *map(tuple, obs.farms[0]["hands"]),
                    ]
                    operations = [
                        action.get("farmer", ["PASS"]),
                        *action.get("hands", []),
                    ]
                    raise AssertionError({
                        "buy_hour": buy_hour,
                        "hour": obs.hour,
                        "money": obs.farms[0]["money"],
                        "hands": len(obs.farms[0]["hands"]),
                        "unassigned": [
                            _task_summary(task) for task in mandatory_unassigned
                        ],
                        "overdue": overdue,
                        "weeds": weeds,
                        "workers": [
                            (index, position, operation,
                             dict(obs["private"]["inventories"][index]))
                            for index, (position, operation)
                            in enumerate(zip(positions, operations))
                        ],
                        "shed": dict(obs["private"]["shed"]),
                        "route_invalidated": planner_state.get("route_invalidated"),
                        "replan_needed": planner_state.get("replan_needed"),
                    })

        state[0].action = action
        state[1].action = pass_agent(state[1].observation)
        state = official_game.interpreter(state, env)
        state[0].observation.step = step + 1

        next_obs = state[0].observation
        if next_obs.day == 4 and next_obs.hour == 0:
            animals = []
            for y, row in enumerate(next_obs.farms[0]["tiles"]):
                for x, tile in enumerate(row):
                    if isinstance(tile, dict) and tile.get("animal"):
                        animals.append((x, y, tile.get("animal")))
            overdue, weeds = _board_signals(next_obs)
            assert saw_intraday_animal_buy
            assert len(animals) == 6 and not weeds, {
                "buy_hour": buy_hour,
                "animals": animals,
                "weeds": weeds,
                "overdue_at_day4": overdue,
            }
            return

    raise AssertionError("simulation never reached day 4 hour 0")
