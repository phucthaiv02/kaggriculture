"""Focused opening regressions for the rolling execution architecture."""

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official_game

from agents.expansion_agent import make_agent
from agents.farm_tasks import build_tasks
from agents.rolling_scheduler import planning_observation
from experiments.crop_schedules import pass_agent


END_DAY = 30
INTERESTING = {"PLACE", "FEED", "CARE", "COLLECT_FERTILIZER", "DROP", "PICKUP"}


def _configuration(seed=1):
    return {
        "episodeSteps": (END_DAY + 1) * 24,
        "boardSize": 10,
        "startingMoney": 3000.0,
        "maxMarketOrdersPerTurn": 10,
        "turnsPerDay": 24,
        "shedCapacity": 100,
        "weedSpawnChance": 0.0,
        "townShopUnlockInterval": 3,
        "townShopSellInterval": 4,
        "townCenterSellInterval": 24,
        "seed": seed,
    }


def _animals(obs):
    result = []
    for y, row in enumerate(obs.farms[0]["tiles"]):
        for x, tile in enumerate(row):
            if not isinstance(tile, dict) or not tile.get("animal"):
                continue
            result.append((
                (x, y), tile.get("animal"), tile.get("placed_day"),
                tile.get("fed_today"), tile.get("cared_today"),
                tile.get("fertilizer_available"), tile.get("consecutive_unfed"),
            ))
    return result


def _task_summary(task):
    return {
        "position": task.position,
        "actions": task.actions,
        "needs": dict(task.needs),
        "mandatory": task.mandatory,
        "immediate_drop": task.immediate_drop,
        "refinance_feed": task.refinance_feed,
    }


def test_day_two_opening_conversion_is_admitted_and_buys_cow():
    env = make("kaggriculture", configuration=_configuration(), debug=False)
    agent = make_agent(END_DAY, seed=1)
    cells = {
        name: cell.cell_contents
        for name, cell in zip(agent.__code__.co_freevars, agent.__closure__)
    }
    targets = cells["targets"]
    planner_state = cells["state"]
    state = env.state
    market_ledger = []
    execution_ledger = []
    animal_snapshots = []
    late_trace = []

    for step in range(int(env.configuration.episodeSteps) - 1):
        obs = state[0].observation

        before_tasks = []
        if obs.day == 1 and obs.hour >= 13 and planner_state.get("daily_targets"):
            task_obs = planning_observation(dict(obs, _planning_end_day=END_DAY))
            generated = build_tasks(
                task_obs,
                planner_state["daily_targets"],
                prioritize_fertilizer_drop=planner_state["opening_active"],
            )
            before_tasks = [
                _task_summary(task) for task in generated
                if task.position == (1, 4)
            ]

        action = agent(obs)
        market = action.get("market", [])
        if market:
            market_ledger.append((
                obs.day, obs.hour, obs.farms[0]["money"],
                len(obs.farms[0]["hands"]), obs.farms[0].get("hires_today", 0),
                obs.private["shed"].get("FERTILIZER", 0), market,
            ))

        if obs.day == 1 and obs.hour >= 13:
            tile = obs.farms[0]["tiles"][4][1]
            late_trace.append({
                "hour": obs.hour,
                "money": obs.farms[0]["money"],
                "shed_wheat": obs.private["shed"].get("WHEAT", 0),
                "tile": dict(tile) if isinstance(tile, dict) else tile,
                "generated_task": before_tasks,
                "frozen": (1, 4) in planner_state["frozen_positions"],
                "schedule_admitted": planner_state["schedule_admitted"],
                "unassigned": [_task_summary(task) for task in planner_state["unassigned"]],
                "worker_positions": [tuple(obs.farms[0]["farmer"]), *map(tuple, obs.farms[0]["hands"])],
                "inventories": [dict(inventory) for inventory in obs.private["inventories"]],
                "action": action,
                "remaining_plans": [list(plan.queue) for plan in planner_state["plans"]],
            })

        if obs.day <= 1:
            positions = [tuple(obs.farms[0]["farmer"]), *map(tuple, obs.farms[0]["hands"])]
            operations = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
            for worker, (position, operation) in enumerate(zip(positions, operations)):
                if operation and operation[0] in INTERESTING:
                    execution_ledger.append((
                        obs.day, obs.hour, worker, position, operation,
                        dict(obs.private["inventories"][worker]),
                    ))
            if obs.hour == 0 or any(op and op[0] in ("PLACE", "CARE", "COLLECT_FERTILIZER") for op in operations):
                animal_snapshots.append((obs.day, obs.hour, _animals(obs)))

        if obs.day == 2 and obs.hour == 0:
            cow_targets = sorted(
                position for position, target in targets.items()
                if target and target[0] == "COW"
            )
            diagnostic_lines = [
                f"day2 money={obs.farms[0]['money']} shed={dict(obs.private['shed'])}",
                f"day2 animals={_animals(obs)}",
                f"market={market}",
                f"cow_targets={cow_targets}",
                f"purchase_positions={sorted(planner_state['purchase_positions'])}",
                f"daily(4,3)={planner_state['daily_targets'].get((4, 3))}",
                f"daily(3,3)={planner_state['daily_targets'].get((3, 3))}",
                f"hands={planner_state['hand_target']} mandatory={planner_state['mandatory_hand_target']}",
                "late trace:",
                *map(str, late_trace),
                "animal snapshots:",
                *map(str, animal_snapshots),
                "execution(day,hour,worker,pos,op,inventory):",
                *map(str, execution_ledger),
                "market(day,hour,money,hands,hires_today,fertilizer,orders):",
                *map(str, market_ledger),
            ]
            diagnostic = "\n".join(diagnostic_lines)
            assert ["BUY_ANIMAL", "COW", 1] in market, diagnostic
            assert (4, 3) in planner_state["purchase_positions"], diagnostic
            return

        state[0].action = action
        state[1].action = pass_agent(state[1].observation)
        state = official_game.interpreter(state, env)
        state[0].observation.step = step + 1

    raise AssertionError("never reached opening day 2 hour 0")
