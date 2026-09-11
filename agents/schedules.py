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

# Fertilize-eligible ages verified against the interpreter. These are only
# meaningful for a planting that committed to fertilizer in the planner.
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
ANIMAL_HARVEST_DAYS = {
    "GOOSE": {4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29},
    "COW": {8, 12, 16, 21, 25, 29},
    "SHEEP": {6, 9, 12, 17, 18, 21, 24, 27},
}


def water_days(crop, fertilized):
    return CROP_WATER_DAYS[(crop, fertilized)]


def is_maintenance_day(crop, age, fertilized):
    return age in CROP_WATER_DAYS[(crop, fertilized)]


def should_fertilize_today(crop, age, fertilized):
    return fertilized and age in CROP_FERTILIZE_DAYS[crop]


def should_feed_animal(animal, age):
    return age in ANIMAL_FEED_DAYS[animal]


def should_care_animal(animal, age):
    return age in ANIMAL_CARE_DAYS[animal]


def animal_maintenance_can_still_pay(animal, age, day, end_day):
    """Whether FEED/CARE today can still contribute to a harvest we can sell.

    Animal production becomes harvestable on a later calendar day, so a
    maintenance action has end-game value only when a validated harvest age
    strictly after ``age`` still lands on or before the last actionable day.
    """
    return any(
        harvest_age > age and day + (harvest_age - age) <= end_day
        for harvest_age in ANIMAL_HARVEST_DAYS[animal]
    )


def cycle_finished(crop, age, tile):
    """Whether today's visit must retire this crop before the next dawn.

    For ongoing crops, waiting for ``yield_units`` to become zero before
    scheduling DIG is one turn too late at the final production age: the
    worker HARVESTs during that day, then the engine can convert the exhausted
    PLANT to WEED on the next hour-0 transition before our hour-1 execution
    queues exist.  Treat the final production age itself as cycle-finished;
    build_tasks already orders WATER/HARVEST before DIG, so the last yield is
    collected and the tile is cleared in the same visit.
    """
    del tile  # cycle end is age-defined; current yield is harvested before DIG
    return age >= CROP_LAST_AGE[crop]
