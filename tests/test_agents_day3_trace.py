"""Focused diagnostic for the opening day-3 animal unlock."""

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official_game

from agents.expansion_agent import make_agent
from experiments.crop_schedules import pass_agent
from tests.test_agents_integration import END_DAY, configuration


ANIMALS = ("CHICKEN", "COW", "SHEEP")


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


def _plan_heads(plans, limit=8):
    return [list(plan.queue[:limit]) for plan in plans]


def test_intraday_animal_unlock_keeps_all_day_three_survival_work():
    env = make("kaggriculture", configuration=configuration(1), debug=False)
    agent = make_agent(END_DAY, seed=1)
    cells = {
        name: cell.cell_contents
        for name, cell in zip(agent.__code__.co_freevars, agent.__closure__)
    }
    planner_state = cells["state"]
    state = env.state
    buy_hour = None
    buy_orders = None

    for step in range(int(env.configuration.episodeSteps) - 1):
        obs = state[0].observation
        action = agent(obs)

        if obs.day == 3:
            market = action.get("market", [])
            animal_orders = [order for order in market if order[:1] == ["BUY_ANIMAL"]]
            if animal_orders and buy_hour is None:
                buy_hour = obs.hour
                buy_orders = animal_orders
            elif buy_hour is not None and obs.hour == buy_hour + 1:
                overdue, weeds = _board_signals(obs)
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
                    "buy_orders": buy_orders,
                    "hour": obs.hour,
                    "money": obs.farms[0]["money"],
                    "hands": len(obs.farms[0]["hands"]),
                    "shed_animals": {
                        name: obs["private"]["shed"].get(name, 0)
                        for name in ANIMALS
                    },
                    "carried_animals": [
                        {name: inventory.get(name, 0) for name in ANIMALS}
                        for inventory in obs["private"]["inventories"]
                    ],
                    "workers": [
                        (index, position, operation)
                        for index, (position, operation)
                        in enumerate(zip(positions, operations))
                    ],
                    "plan_heads": _plan_heads(planner_state.get("plans", ())),
                    "unassigned": [
                        (task.position, task.actions, dict(task.needs), task.mandatory)
                        for task in planner_state.get("unassigned", ())
                    ],
                    "overdue": overdue,
                    "weeds": weeds,
                    "route_invalidated": planner_state.get("route_invalidated"),
                    "replan_needed": planner_state.get("replan_needed"),
                })

        state[0].action = action
        state[1].action = pass_agent(state[1].observation)
        state = official_game.interpreter(state, env)
        state[0].observation.step = step + 1

    raise AssertionError("simulation never reached the first day-3 BUY_ANIMAL landing")
