"""Experimentally measure animal yield over the playable 30-day season."""

from collections import Counter

from kaggle_environments import make


ANIMALS = {
    "GOOSE": {"structure": "BUILD_COOP", "expected": {False: 26, True: 54}},
    "COW": {"structure": "BUILD_PASTURE", "expected": {False: 11, True: 36}},
    "SHEEP": {"structure": "BUILD_PASTURE", "expected": {False: 8, True: 34}},
}

# Product-maximizing 30-day schedules, indexed by age since PLACE. Among all
# schedules with the same product yield, these minimize FEED + CARE + HARVEST.
# Fertilizer is independent: collect on ages 1..29 to receive all 29 units.
OPTIMAL_SCHEDULES = {
    "GOOSE": {
        "feed": set(range(29)),
        "care": set(range(28)),
        "harvest": {4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29},
        "collect_fertilizer": set(range(1, 30)),
        "max_yield": 54,
    },
    "COW": {
        "feed": {1, *range(3, 28)},
        "care": {1, *range(3, 27)},
        "harvest": {8, 12, 16, 21, 25, 29},
        "collect_fertilizer": set(range(1, 30)),
        "max_yield": 36,
    },
    "SHEEP": {
        "feed": set(range(28)),
        "care": set(range(26)),
        "harvest": {6, 9, 12, 17, 18, 21, 24, 27},
        "collect_fertilizer": set(range(1, 30)),
        "max_yield": 34,
    },
}


def pass_agent(_obs):
    return {"farmer": ["PASS"], "hands": [], "market": []}


def measure_animal(animal, use_care, seed, days=30):
    """Feed daily and harvest immediately so max_held cannot discard yield."""
    cfg = ANIMALS[animal]
    trace = []
    action_counts = Counter()
    flags = {}

    def agent(obs):
        day, hour = obs["day"], obs["hour"]
        farm = obs["farms"][obs["player"]]
        x, y = farm["farmer"]
        tile = farm["tiles"][y][x]
        if day == 0 and hour == 0:
            market = [
                ["BUY_ANIMAL", animal, 1],
                ["BUY_PRODUCT", "WHEAT", days],
            ]
            return {"farmer": ["PASS"], "hands": [], "market": market}

        today = flags.setdefault(day, {"animal_picked": False, "wheat_picked": False})
        op = ["PASS"]
        if tile is None:
            op = [cfg["structure"]]
            action_counts[op[0]] += 1
        elif isinstance(tile, dict) and "animal" not in tile:
            if not today["animal_picked"]:
                today["animal_picked"] = True
                op = ["PICKUP", animal, 1]
            else:
                op = ["PLACE", animal]
                action_counts["PLACE"] += 1
        elif isinstance(tile, dict) and tile.get("animal") == animal:
            if tile.get("yield_units", 0) > 0:
                amount = tile["yield_units"]
                trace.append((day, hour, amount))
                op = ["HARVEST"]
                action_counts["HARVEST"] += 1
            elif not today["wheat_picked"]:
                today["wheat_picked"] = True
                op = ["PICKUP", "WHEAT", 1]
            elif not tile["fed_today"]:
                op = ["FEED"]
                action_counts["FEED"] += 1
            elif use_care and not tile["cared_today"]:
                op = ["CARE"]
                action_counts["CARE"] += 1
        return {"farmer": op, "hands": [], "market": []}

    env = make(
        "kaggriculture",
        configuration={
            "episodeSteps": days * 24,
            "turnsPerDay": 24,
            "weedSpawnChance": 0,
            "seed": seed,
        },
        debug=True,
    )
    env.run([agent, pass_agent])
    total = sum(amount for _, _, amount in trace)
    if days == 30:
        assert total == cfg["expected"][use_care], (animal, use_care, trace)
    return trace, action_counts


def run_optimal_schedule(animal, seed, collect_fertilizer=True, days=30):
    """Run the max-product, minimum-maintenance-action 30-day schedule."""
    cfg = ANIMALS[animal]
    schedule = OPTIMAL_SCHEDULES[animal]
    trace = []
    action_counts = Counter()
    queues = {}

    def agent(obs):
        day, hour = obs["day"], obs["hour"]
        farm = obs["farms"][obs["player"]]
        x, y = farm["farmer"]
        tile = farm["tiles"][y][x]

        if day == 0 and hour == 0:
            market = [["BUY_ANIMAL", animal, 1], ["BUY_PRODUCT", "WHEAT", days]]
            return {"farmer": ["PASS"], "hands": [], "market": market}

        if day not in queues:
            ops = []
            if tile is None:
                ops += [[cfg["structure"]], ["PICKUP", animal, 1], ["PLACE", animal]]
            if day in schedule["harvest"]:
                amount = tile.get("yield_units", 0) if isinstance(tile, dict) else 0
                trace.append((day, amount))
                ops.append(["HARVEST"])
            if day in schedule["feed"]:
                ops += [["PICKUP", "WHEAT", 1], ["FEED"]]
            if day in schedule["care"]:
                ops.append(["CARE"])
            if collect_fertilizer and day in schedule["collect_fertilizer"]:
                ops.append(["COLLECT_FERTILIZER"])
            queues[day] = ops

        op = queues[day].pop(0) if queues[day] else ["PASS"]
        if op[0] in {"BUILD_COOP", "BUILD_PASTURE", "PLACE", "FEED", "CARE", "HARVEST", "COLLECT_FERTILIZER"}:
            action_counts[op[0]] += 1
        return {"farmer": op, "hands": [], "market": []}

    env = make(
        "kaggriculture",
        configuration={
            "episodeSteps": days * 24,
            "turnsPerDay": 24,
            "weedSpawnChance": 0,
            "seed": seed,
        },
        debug=True,
    )
    env.run([agent, pass_agent])
    total = sum(amount for _, amount in trace)
    if days == 30:
        assert total == schedule["max_yield"], (animal, trace)
        expected_fertilizer = 29 if collect_fertilizer else 0
        assert action_counts["COLLECT_FERTILIZER"] == expected_fertilizer
    return trace, action_counts


def test_animal_yields():
    for seed in (1, 7, 42):
        for animal in ANIMALS:
            for use_care in (False, True):
                measure_animal(animal, use_care, seed)


def test_sheep_yield_through_day_16():
    trace, _ = measure_animal("SHEEP", True, seed=1, days=17)
    assert [amount for _, _, amount in trace] == [6, 4, 4, 4]
    assert sum(amount for _, _, amount in trace) == 18


def test_optimal_animal_schedules():
    for seed in (1, 7, 42):
        for animal in ANIMALS:
            run_optimal_schedule(animal, seed)


def main():
    for seed in (1, 7, 42):
        for animal in ANIMALS:
            for use_care in (False, True):
                trace, counts = measure_animal(animal, use_care, seed)
                print(
                    f"seed={seed:>2} animal={animal:<5} care={str(use_care):<5} "
                    f"harvests={trace} total_yield={sum(x[2] for x in trace)} "
                    f"tile_actions={sum(counts.values())}"
                )
    print("All 18 animal games passed.")
    for animal in ANIMALS:
        trace, counts = run_optimal_schedule(animal, seed=1)
        print(
            f"optimal animal={animal:<5} harvests={trace} "
            f"total_yield={sum(x[1] for x in trace)} tile_actions={sum(counts.values())}"
        )


if __name__ == "__main__":
    main()
