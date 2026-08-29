"""Schedule farmers/hands for one of every crop and animal, then render HTML."""

import random
from pathlib import Path

from kaggle_environments import make


CROP_SCHEDULES = {
    "WHEAT": {"water": {0, 2, 3, 4}, "fertilize": {2}, "harvest": {4}, "ongoing": False},
    "CARROT": {"water": {0, 2, 3}, "fertilize": {2}, "harvest": {3}, "ongoing": False},
    "MELON": {"water": {0, 2, 4, 6, 8, 9}, "fertilize": {8}, "harvest": {10}, "ongoing": False},
    "TOMATO": {"water": {0, 2, 4, 5, 7, 8, 9, 10}, "fertilize": {7, 10}, "harvest": {9, 11}, "ongoing": True},
    "STRAWBERRY": {"water": {0, 2, 4, 6, 7, 9, 11, 13, 15}, "fertilize": {9, 13}, "harvest": {12, 16}, "ongoing": True},
}

ANIMALS = {
    "GOOSE": {"build": "BUILD_COOP"},
    "COW": {"build": "BUILD_PASTURE"},
    "SHEEP": {"build": "BUILD_PASTURE"},
}

SELLABLE = ("WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY", "EGG", "MILK", "WOOL")


def pass_agent(_obs):
    return {"farmer": ["PASS"], "hands": [], "market": []}


def route(start, target):
    x, y = start
    tx, ty = target
    moves = []
    moves += [["EAST"]] * max(0, tx - x)
    moves += [["WEST"]] * max(0, x - tx)
    moves += [["SOUTH"]] * max(0, ty - y)
    moves += [["NORTH"]] * max(0, y - ty)
    return moves


def make_scheduler(seed=2026):
    rng = random.Random(seed)
    objects = list(CROP_SCHEDULES) + list(ANIMALS)
    # Use only the initially unlocked NW quadrant; BUY_LAND is never issued.
    positions = rng.sample([(x, y) for y in range(5) for x in range(5)], len(objects))
    targets = dict(zip(objects, positions))
    queues = []
    queue_day = -1
    harvest_log = []

    def create_tasks(obs):
        day = obs["day"]
        farm = obs["farms"][obs["player"]]
        tasks = []

        for plot_index, initial_crop in enumerate(CROP_SCHEDULES):
            x, y = targets[initial_crop]
            tile = farm["tiles"][y][x]
            prefix, actions = [], []
            if day == 0:
                crop = initial_crop
                actions = [["PLANT", crop], ["WATER"]]
            elif not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                crop = rng.choice(list(CROP_SCHEDULES))
                if isinstance(tile, dict):
                    actions.append(["DIG"])
                actions += [["PLANT", crop], ["WATER"]]
            else:
                crop = tile["crop"]
                schedule = CROP_SCHEDULES[crop]
                age = day - tile["planted_day"]
                if age in schedule["fertilize"]:
                    prefix.append(["PICKUP", "FERTILIZER", 1])
                if schedule["ongoing"] and age in schedule["harvest"]:
                    actions.append(["HARVEST"])
                if age in schedule["fertilize"]:
                    actions.append(["FERTILIZE"])
                if age in schedule["water"]:
                    actions.append(["WATER"])
                if not schedule["ongoing"] and age in schedule["harvest"]:
                    actions.append(["HARVEST"])
                final_harvest = age == max(schedule["harvest"])
                if final_harvest:
                    replacement = rng.choice(list(CROP_SCHEDULES))
                    if schedule["ongoing"]:
                        actions.append(["DIG"])
                    actions += [["PLANT", replacement], ["WATER"]]
            if actions:
                tasks.append((f"PLOT_{plot_index}:{crop}", (x, y), prefix, actions, tile))

        for animal, config in ANIMALS.items():
            x, y = targets[animal]
            tile = farm["tiles"][y][x]
            if day == 0:
                prefix = [["PICKUP", animal, 1], ["PICKUP", "WHEAT", 1]]
                actions = [[config["build"]], ["PLACE", animal], ["FEED"], ["CARE"]]
            else:
                prefix = [["PICKUP", "WHEAT", 1]]
                actions = []
                if isinstance(tile, dict) and tile.get("yield_units", 0) > 0:
                    actions.append(["HARVEST"])
                actions += [["FEED"], ["CARE"]]
                if isinstance(tile, dict) and tile.get("fertilizer_available"):
                    actions.append(["COLLECT_FERTILIZER"])
            tasks.append((animal, (x, y), prefix, actions, tile))
        return tasks

    def assign(obs):
        farm = obs["farms"][obs["player"]]
        starts = [tuple(farm["farmer"])] + [tuple(p) for p in farm["hands"]]
        assigned = [[] for _ in starts]
        available = set(range(len(starts)))
        for name, target, prefix, actions, _tile in create_tasks(obs):
            worker = min(
                available,
                key=lambda i: abs(starts[i][0] - target[0]) + abs(starts[i][1] - target[1]),
            )
            available.remove(worker)
            if prefix:
                # Some hands spawn on locked quadrants. Move through them to the
                # unlocked NW shed tile before attempting PICKUP.
                assigned[worker] = (
                    route(starts[worker], (4, 4))
                    + prefix
                    + route((4, 4), target)
                    + actions
                )
            else:
                assigned[worker] = route(starts[worker], target) + actions
        return assigned

    def agent(obs):
        nonlocal queues, queue_day
        day, hour = obs["day"], obs["hour"]
        farm = obs["farms"][obs["player"]]
        market = []

        if hour == 0:
            market.extend([["HIRE"] for _ in range(8)])
            shed = obs["private"]["shed"]
            market.extend([["SELL", item, shed.get(item, 0)] for item in SELLABLE if shed.get(item, 0)])
            market.append(["BUY_PRODUCT", "WHEAT", 3])
            if day == 0:
                market.extend([["BUY_SEED", crop, 50] for crop in CROP_SCHEDULES])
                market.extend([["BUY_ANIMAL", animal, 1] for animal in ANIMALS])
            return {"farmer": ["PASS"], "hands": [["PASS"] for _ in farm["hands"]], "market": market}

        if queue_day != day:
            queues = assign(obs)
            queue_day = day

        ops = [(queue.pop(0) if queue else ["PASS"]) for queue in queues]
        unit_positions = [tuple(farm["farmer"])] + [tuple(p) for p in farm["hands"]]
        names_by_target = {position: name for name, position in targets.items()}
        for position, op in zip(unit_positions, ops):
            if op[0] != "HARVEST":
                continue
            x, y = position
            tile = farm["tiles"][y][x]
            name = names_by_target.get(position, "UNKNOWN")
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                name = tile.get("crop", name)
            amount = tile.get("yield_units", 0) if isinstance(tile, dict) else 0
            harvest_log.append((day, name, amount))
        farmer = ops[0] if ops else ["PASS"]
        hands = ops[1 : 1 + len(farm["hands"])]
        hands += [["PASS"] for _ in range(len(farm["hands"]) - len(hands))]
        return {"farmer": farmer, "hands": hands, "market": market}

    return agent, targets, harvest_log


def run(output="farm_schedule_replay.html", seed=2026):
    agent, targets, harvest_log = make_scheduler(seed)
    env = make(
        "kaggriculture",
        configuration={
            "episodeSteps": 30 * 24,
            "turnsPerDay": 24,
            "startingMoney": 100_000,
            "maxMarketOrdersPerTurn": 100,
            "weedSpawnChance": 0,
            "seed": seed,
        },
        debug=True,
    )
    env.run([agent, pass_agent])
    output_path = Path(output).resolve()
    output_path.write_text(env.render(mode="html", width=1200, height=800), encoding="utf-8")
    print("Random targets:", targets)
    print("Harvest log:", harvest_log)
    print("Final rewards:", [state.reward for state in env.steps[-1]])
    print("HTML:", output_path)
    return env, targets, harvest_log, output_path


if __name__ == "__main__":
    run()
