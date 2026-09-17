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

PRODUCTION = {
    "GOOSE": {"first": 4, "interval": 1, "max_held": 4},
    "COW": {"first": 8, "interval": 2, "max_held": 6},
    "SHEEP": {"first": 6, "interval": 3, "max_held": 6},
}


def find_equivalent_schedules(animal, days=30, limit=20):
    """Find several low-action schedules matching daily FEED+CARE output.

    This is a small dynamic program over the public animal state.  It models
    survival, the CARE bank, scheduled production, ``max_held``, and optional
    harvests.  The winning schedule is still replayed through the real
    interpreter by :func:`run_optimal_schedule`; this model is only the search
    mechanism.
    """
    target = ANIMALS[animal]["expected"][True]
    production = PRODUCTION[animal]
    # (held, care bank, consecutive misses, harvested) -> (actions, schedule)
    states = {(0, 0, 0, 0): [(0, (frozenset(), frozenset(), frozenset()))]}
    for age in range(days):
        next_states = {}
        produces = (
            age + 1 >= production["first"]
            and (age + 1 - production["first"]) % production["interval"] == 0
        )
        for (held, bank, missed, harvested), paths in states.items():
            for cost, schedule in paths:
                harvest_days, feed_days, care_days = schedule
                for harvest in (False, True):
                    if harvest and not held:
                        continue
                    held_after_harvest = 0 if harvest else held
                    next_harvested = harvested + (held if harvest else 0)
                    if next_harvested > target:
                        continue
                    for feed, care in ((False, False), (True, False), (True, True)):
                        next_held = held_after_harvest
                        next_missed = 0 if feed else missed + 1
                        # Escaping after the final playable day cannot reduce yield.
                        if next_missed >= 2 and age < days - 1:
                            continue
                        next_bank = bank
                        if produces:
                            produced = 1 + (next_bank if feed else 0)
                            next_bank = 0
                            next_held = min(
                                production["max_held"], next_held + produced
                            )
                        # The current day's CARE is banked after its production
                        # tick, so it applies to the following scheduled yield.
                        next_bank += int(feed and care)
                        key = (next_held, next_bank, next_missed, next_harvested)
                        candidate = (
                            cost + int(harvest) + int(feed) + int(care),
                            (
                                harvest_days | ({age} if harvest else set()),
                                feed_days | ({age} if feed else set()),
                                care_days | ({age} if care else set()),
                            ),
                        )
                        bucket = next_states.setdefault(key, [])
                        bucket.append(candidate)
                        # Keeping several paths per engine state preserves schedules
                        # with different busy days without making the DP explode.
                        bucket.sort(key=lambda value: value[0])
                        unique = []
                        seen = set()
                        for value in bucket:
                            if value[1] not in seen:
                                seen.add(value[1])
                                unique.append(value)
                            if len(unique) == limit:
                                break
                        next_states[key] = unique
        states = next_states

    matches = [
        value
        for state, values in states.items()
        if state[3] == target
        for value in values
    ]
    if not matches:
        raise RuntimeError(f"no {animal} schedule reaches baseline yield {target}")
    matches.sort(key=lambda value: value[0])
    results = []
    seen = set()
    for _, (harvest, feed, care) in matches:
        signature = (harvest, feed, care)
        if signature in seen:
            continue
        seen.add(signature)
        results.append(
            {
                "feed": set(feed),
                "care": set(care),
                "harvest": set(harvest),
                "collect_fertilizer": set(),
                "max_yield": target,
            }
        )
        if len(results) == limit:
            break
    return results


def find_equivalent_schedule(animal, days=30):
    """Backward-compatible shortcut returning the first searched schedule."""
    return find_equivalent_schedules(animal, days=days, limit=1)[0]


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


def run_optimal_schedule(
    animal, seed, collect_fertilizer=True, days=30, schedule=None
):
    """Run the max-product, minimum-maintenance-action 30-day schedule."""
    cfg = ANIMALS[animal]
    schedule = schedule or OPTIMAL_SCHEDULES[animal]
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


def test_searched_animal_schedules_match_daily_care_yield():
    for animal in ANIMALS:
        schedules = find_equivalent_schedules(animal, limit=5)
        assert len(schedules) == 5
        for schedule in (schedules[0], schedules[-1]):
            run_optimal_schedule(
                animal, seed=1, collect_fertilizer=False, schedule=schedule
            )


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
    print("\nSchedules found by dynamic programming (same yield as daily FEED+CARE):")
    for animal in ANIMALS:
        schedules = find_equivalent_schedules(animal, limit=5)
        for index, schedule in enumerate(schedules, 1):
            trace, counts = run_optimal_schedule(
                animal, seed=1, collect_fertilizer=False, schedule=schedule
            )
            maintenance = len(schedule["feed"]) + len(schedule["care"])
            baseline = 2 * 30
            print(
                f"searched animal={animal:<5} option={index} "
                f"yield={sum(x[1] for x in trace):>2} "
                f"feed={sorted(schedule['feed'])} care={sorted(schedule['care'])} "
                f"harvest={sorted(schedule['harvest'])} maintenance={maintenance} "
                f"saved_vs_daily={baseline - maintenance}"
            )


if __name__ == "__main__":
    main()
