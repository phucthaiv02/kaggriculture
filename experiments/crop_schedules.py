"""Experiment with maximum-yield, minimum-tile-action crop schedules."""

from collections import Counter

from kaggle_environments import make


CASES = {
    ("WHEAT", False): {
        "ongoing": False,
        "water": {0, 2, 3, 4},
        "fertilize": set(),
        "harvest": {4},
        "expected_yields": [4],
        "expected_tile_actions": 6,
    },
    ("WHEAT", True): {
        "ongoing": False,
        "water": {0, 2, 3, 4},
        "fertilize": {2},
        "harvest": {4},
        "expected_yields": [6],
        "expected_tile_actions": 7,
    },
    ("CARROT", False): {
        "ongoing": False,
        "water": {0, 2, 3},
        "fertilize": set(),
        "harvest": {3},
        "expected_yields": [3],
        "expected_tile_actions": 5,
    },
    ("CARROT", True): {
        "ongoing": False,
        "water": {0, 2, 3},
        "fertilize": {2},
        "harvest": {3},
        "expected_yields": [4],
        "expected_tile_actions": 6,
    },
    ("MELON", False): {
        "ongoing": False,
        "water": {0, 2, 4, 6, 7, 8, 9, 10},
        "fertilize": set(),
        "harvest": {10},
        "expected_yields": [6],
        "expected_tile_actions": 10,
    },
    ("MELON", True): {
        "ongoing": False,
        "water": {0, 2, 4, 6, 8, 9},
        "fertilize": {8},
        "harvest": {10},
        "expected_yields": [6],
        "expected_tile_actions": 9,
    },
    ("TOMATO", False): {
        "ongoing": True,
        "water": {0, 2, 4, 6, 8, 9},
        "fertilize": set(),
        "harvest": {11},
        "expected_yields": [4],
        "expected_tile_actions": 8,
    },
    ("TOMATO", True): {
        "ongoing": True,
        "water": {0, 2, 4, 5, 7, 8, 9, 10},
        "fertilize": {7, 10},
        "harvest": {9, 11},
        "expected_yields": [4, 4],
        "expected_tile_actions": 13,
    },
    ("STRAWBERRY", False): {
        "ongoing": True,
        "water": {0, 2, 4, 6, 8, 10, 12, 14},
        "fertilize": set(),
        "harvest": {16},
        "expected_yields": [4],
        "expected_tile_actions": 10,
    },
    ("STRAWBERRY", True): {
        "ongoing": True,
        "water": {0, 2, 4, 6, 7, 9, 11, 13, 15},
        "fertilize": {9, 13},
        "harvest": {12, 16},
        "expected_yields": [4, 4],
        "expected_tile_actions": 14,
    },
}

def pass_agent(_obs):
    return {"farmer": ["PASS"], "hands": [], "market": []}


def run_case(crop, use_fertilizer, seed):
    schedule = CASES[(crop, use_fertilizer)]
    trace = []
    action_counts = Counter()

    def agent(obs):
        day, hour = obs["day"], obs["hour"]
        farm = obs["farms"][obs["player"]]
        x, y = farm["farmer"]
        tile = farm["tiles"][y][x]
        market = []

        # Market orders run before field actions. Buy supplies on the first turn,
        # then plant on the next turn after the seed appears in private state.
        if day == 0 and hour == 0:
            market.append(["BUY_SEED", crop, 1])
            if use_fertilizer:
                market.append(
                    ["BUY_PRODUCT", "FERTILIZER", len(schedule["fertilize"])]
                )

        op = ["PASS"]

        # Build the exact within-day order. One-time crops must WATER before
        # HARVEST to receive the last day's bonus. Ongoing crops HARVEST first
        # when both operations share a day, freeing their held-yield capacity
        # before the production tick at end of day.
        day_ops = []
        if day == 0:
            day_ops.append(["PLANT", crop])
            if day in schedule["water"]:
                day_ops.append(["WATER"])
        else:
            if schedule["ongoing"] and day in schedule["harvest"]:
                day_ops.append(["HARVEST"])
            if day in schedule["fertilize"]:
                # Inventories return to the shed nightly; PICKUP is a real game
                # action but is not an action on the crop tile, so it is excluded
                # from expected_tile_actions.
                day_ops.extend([["PICKUP", "FERTILIZER", 1], ["FERTILIZE"]])
            if day in schedule["water"]:
                day_ops.append(["WATER"])
            if not schedule["ongoing"] and day in schedule["harvest"]:
                day_ops.append(["HARVEST"])

        # Hour 0 is reserved for initial market purchases on day 0.
        op_index = hour - 1 if day == 0 else hour
        if 0 <= op_index < len(day_ops):
            op = day_ops[op_index]
            name = op[0]
            if name == "HARVEST":
                amount = tile.get("yield_units", 0) if isinstance(tile, dict) else 0
                trace.append((day, "HARVEST", amount))
            if name in {"PLANT", "WATER", "FERTILIZE", "HARVEST"}:
                action_counts[name] += 1

        return {"farmer": op, "hands": [], "market": market}

    last_harvest = max(schedule["harvest"])
    env = make(
        "kaggriculture",
        configuration={
            "episodeSteps": (last_harvest + 2) * 24,
            "turnsPerDay": 24,
            "weedSpawnChance": 0,
            "seed": seed,
        },
        debug=True,
    )
    env.run([agent, pass_agent])

    yields = [amount for _, event, amount in trace if event == "HARVEST"]
    tile_actions = sum(action_counts.values())
    assert yields == schedule["expected_yields"], (crop, use_fertilizer, trace)
    assert tile_actions == schedule["expected_tile_actions"], action_counts
    return trace, action_counts


def main():
    for seed in (1, 7, 42):
        for crop, use_fertilizer in CASES:
            trace, counts = run_case(crop, use_fertilizer, seed)
            print(
                f"seed={seed:>2} crop={crop:<10} fertilizer={str(use_fertilizer):<5} "
                f"harvests={trace} actions={dict(counts)} total={sum(counts.values())}"
            )
    print(f"All {len((1, 7, 42)) * len(CASES)} crop games passed.")


def test_crop_schedules():
    for seed in (1, 7, 42):
        for crop, use_fertilizer in CASES:
            run_case(crop, use_fertilizer, seed)


if __name__ == "__main__":
    main()
