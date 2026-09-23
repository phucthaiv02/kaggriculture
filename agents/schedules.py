"""Verified minimal-action maintenance schedules for crops and animals.

The crop and animal tables are validated by ``experiments/crop_schedules.py``
and ``experiments/animal_yields.py`` against the Kaggriculture interpreter.
Keep schedule changes grounded in those executable checks rather than deriving
them from engine source alone.

Crop ages are relative to the tile's ``planted_day``
(``age = day - planted_day``). A planted tile is not eligible for random weed
spawning, so these relative-age schedules stay valid regardless of the
calendar day on which planting happened. The remaining operational risk is a
missed scheduled action because of worker capacity; routing and capacity live
in ``agents/scheduler.py``.
"""

from __future__ import annotations

from functools import lru_cache

from agents.fertilizer import plan_ages


# (crop, fertilized) -> water ages relative to planted_day. Fertilizing an
# ongoing crop (TOMATO/STRAWBERRY) needs more frequent watering because each
# watered production tick banks a larger bonus. One-time crops use the same
# watering cadence with or without fertilizer.
CROP_WATER_DAYS = {
    ("WHEAT", False): {0, 2, 3, 4},
    ("WHEAT", True): {0, 2, 3, 4},
    ("CARROT", False): {0, 2, 3},
    ("CARROT", True): {0, 2, 3},
    ("MELON", False): {0, 2, 4, 6, 7, 8, 9, 10},
    ("MELON", True): {0, 2, 4, 6, 8, 9},
    ("TOMATO", False): {0, 2, 4, 6, 8, 9},
    ("TOMATO", True): {0, 2, 4, 5, 7, 8, 9, 10},
    ("STRAWBERRY", False): {0, 2, 4, 6, 8, 10, 12, 14},
    ("STRAWBERRY", True): {0, 2, 4, 6, 7, 9, 11, 13, 15},
}


# Interpreter-verified alternatives from experiments/crop_schedules.py.
# Profile 0 is the historical minimum-action calendar. Profile 1 shifts
# protective irrigation between neighboring spatial zones so same-day cohorts
# do not all create the same labor peak. Some one-time crops need one extra
# protective WATER to move earlier visits while preserving the same yield.
CROP_WATER_PROFILES = {
    ("MELON", False): (
        frozenset(CROP_WATER_DAYS[("MELON", False)]),
        frozenset({0, 1, 3, 5, 6, 7, 8, 9, 10}),
    ),
    ("MELON", True): (
        frozenset(CROP_WATER_DAYS[("MELON", True)]),
        frozenset({0, 1, 3, 5, 6, 8, 9}),
    ),
    ("TOMATO", False): (
        frozenset(CROP_WATER_DAYS[("TOMATO", False)]),
        frozenset({0, 1, 3, 5, 7, 9}),
    ),
    ("TOMATO", True): (
        frozenset(CROP_WATER_DAYS[("TOMATO", True)]),
        frozenset({0, 1, 3, 5, 7, 8, 9, 10}),
    ),
    ("STRAWBERRY", False): (
        frozenset(CROP_WATER_DAYS[("STRAWBERRY", False)]),
        frozenset({0, 1, 3, 5, 6, 8, 10, 12, 14}),
    ),
    ("STRAWBERRY", True): (
        frozenset(CROP_WATER_DAYS[("STRAWBERRY", True)]),
        frozenset({0, 1, 3, 5, 7, 9, 11, 13, 15}),
    ),
}

# Last age at which a one-time crop can still be harvested or an ongoing crop
# still produces a tick.
CROP_LAST_AGE = {
    "WHEAT": 4,
    "CARROT": 3,
    "MELON": 10,
    "TOMATO": 11,
    "STRAWBERRY": 16,
}
ONGOING_CROPS = {"TOMATO", "STRAWBERRY"}

# Fertilize-eligible ages verified against the interpreter. Dynamic planner
# targets may select any subset of these ages for the current cycle. Opening
# targets still pass a boolean and therefore retain the old all-or-none rule.
CROP_FERTILIZE_DAYS = {
    "WHEAT": {2},
    "CARROT": {2},
    "MELON": {8},
    "TOMATO": {7, 10},
    "STRAWBERRY": {9, 13},
}

# Product-maximizing, minimum-maintenance schedules measured by
# experiments/animal_yields.py. Ages are relative to placed_day. Deliberate
# gaps (notably COW ages 0 and 2) avoid feed/care actions that do not increase
# season yield.
ANIMAL_FEED_DAYS = {
    "GOOSE": set(range(29)),
    "COW": {1, *range(3, 28)},
    "SHEEP": set(range(28)),
}
ANIMAL_CARE_DAYS = {
    "GOOSE": set(range(28)),
    "COW": {1, *range(3, 27)},
    "SHEEP": set(range(26)),
}

# Max-yield harvest cadence found by the same interpreter-backed experiment.
# Harvesting every time yield_units becomes non-zero is safe but wastes a
# worker turn and a route visit. These ages let output accumulate up to the
# animal's held-product cap without clipping a later production tick.
ANIMAL_HARVEST_DAYS = {
    "GOOSE": {4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29},
    "COW": {8, 12, 16, 21, 25, 29},
    "SHEEP": {6, 9, 12, 17, 18, 21, 24, 27},
}
ANIMAL_MAX_HELD = {
    "GOOSE": 4,
    "COW": 6,
    "SHEEP": 6,
}


def _cycle_uses_fertilizer(fertilized):
    ages = plan_ages(fertilized)
    return bool(fertilized) if ages is None else bool(ages)


def water_profiles(crop, fertilized):
    """Equivalent verified WATER calendars available to one crop cycle."""
    key = (crop, _cycle_uses_fertilizer(fertilized))
    return CROP_WATER_PROFILES.get(key, (frozenset(CROP_WATER_DAYS[key]),))


def water_days(crop, fertilized, position=None, planted_day=0):
    """Stable spatially staggered WATER calendar for one physical crop.

    Adjacent 2x2 zones share a profile for compact routes; neighboring zones
    alternate profiles. Replanting on a different day rotates the phase too.
    position=None keeps the historical baseline and is used by opening.
    """
    profiles = water_profiles(crop, fertilized)
    if position is None or len(profiles) == 1:
        return profiles[0]
    x, y = position
    zone = x // 2 + y // 2 + int(planted_day)
    return profiles[zone % len(profiles)]


def is_maintenance_day(crop, age, fertilized, position=None, planted_day=0):
    return age in water_days(crop, fertilized, position, planted_day)


def should_fertilize_today(crop, age, fertilized):
    ages = plan_ages(fertilized)
    if ages is None:
        return age in CROP_FERTILIZE_DAYS[crop]
    return age in ages and age in CROP_FERTILIZE_DAYS[crop]


ANIMAL_PRODUCTION = {"GOOSE": (4, 1), "COW": (8, 2), "SHEEP": (6, 3)}


@lru_cache(maxsize=128)
def animal_last_yield_age(animal, last_age):
    first, interval = ANIMAL_PRODUCTION[animal]
    return first + ((last_age - first) // interval) * interval if last_age >= first else -1


@lru_cache(maxsize=128)
def animal_feed_end_age(animal, last_age=29):
    # Feed on the eve of the last useful production tick. Keep enough later
    # feeds to prevent two consecutive misses before the last playable day.
    survival_feed = min(
        (age for age in ANIMAL_FEED_DAYS[animal] if age >= last_age - 2),
        default=last_age - 1,
    )
    return max(animal_last_yield_age(animal, last_age) - 1, survival_feed)


def should_feed_animal(animal, age, last_age=29):
    return (age < last_age and age in ANIMAL_FEED_DAYS[animal]
            and age <= animal_feed_end_age(animal, last_age))


def should_care_animal(animal, age, last_age=29):
    # CARE is banked after today's production refresh, so care on the eve
    # of the final useful tick would only affect a tick beyond the horizon.
    # Trim in the terminal window only: earlier route capacity is still needed
    # for crop turnover and late target decisions.
    return age in ANIMAL_CARE_DAYS[animal] and (
        age < last_age - 2 or age <= animal_last_yield_age(animal, last_age) - 2
    )


def should_harvest_animal(animal, age, held=0, force=False):
    """Whether ready animal output should consume a worker turn today.

    ``force`` is used for terminal liquidation. The held-cap fallback keeps
    the policy safe if a missed action causes live state to diverge from the
    verified nominal cadence.
    """
    return bool(held) and (
        force
        or held >= ANIMAL_MAX_HELD[animal]
        or age in ANIMAL_HARVEST_DAYS[animal]
    )


def cycle_finished(crop, age, tile):
    """Return whether a crop is already empty and physically reusable now."""
    if crop not in ONGOING_CROPS:
        return age >= CROP_LAST_AGE[crop]
    return age >= CROP_LAST_AGE[crop] and not tile.get("yield_units", 0)


def cycle_turns_over_today(crop, age, tile):
    """Whether today's full plan can harvest/clear the crop and reuse its tile."""
    if crop in ONGOING_CROPS:
        return age >= CROP_LAST_AGE[crop]
    return cycle_finished(crop, age, tile)
