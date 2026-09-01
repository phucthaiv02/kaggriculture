"""Verified minimal-action maintenance schedules for crops and animals.

These tables come directly from experiments/crop_schedules.py and
experiments/animal_yields.py, both of which assert their numbers against the
real Kaggriculture interpreter (weedSpawnChance=0, multiple seeds) before
being trusted here -- see the project's [[feedback_verify_with_real_game]]
rule: never hand-derive a yield/action schedule from reading engine source
alone.

Crop ages are relative to the tile's planted_day (age = day - planted_day).
A crop tile is immune to weed spawning once planted (kaggriculture only
weeds tiles that are still `None`), so a schedule keyed by relative age stays
valid regardless of what calendar day the tile was actually planted on --
the only real risk is a *missed* scheduled WATER (worker capacity problem,
not a schedule problem; see agents/scheduler.py).
"""

from __future__ import annotations

# (crop, fertilized) -> water ages (relative to planted_day). Fertilizing an
# ongoing crop (TOMATO/STRAWBERRY) needs *more* frequent watering than not
# fertilizing it, not less -- each watered production tick banks a bigger
# bonus, so the schedule harvests twice before the tile's 4-unit yield cap
# would otherwise discard the extra. Fertilizing a one-time crop (WHEAT/
# CARROT/MELON) needs the same watering, since it just adds bonus units to a
# harvest that already had to happen at max_day. Never assume "unfertilized
# schedule is always a safe superset" -- it verifiably is not for TOMATO.
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

# Last age at which a one-time crop can still be harvested / an ongoing crop
# still produces a tick. Matches day16_allocation.MAX_DAY.
CROP_LAST_AGE = {"WHEAT": 4, "CARROT": 3, "MELON": 10, "TOMATO": 11, "STRAWBERRY": 16}
ONGOING_CROPS = {"TOMATO", "STRAWBERRY"}

# Fertilize-eligible ages, matching day16_allocation.FERT_DAYS and the
# engine's own crop_data window. Only meaningful when a planting has
# committed to fertilizing (see agents/planner.py).
CROP_FERTILIZE_DAYS = {
    "WHEAT": {2}, "CARROT": {2}, "MELON": {8}, "TOMATO": {7, 10}, "STRAWBERRY": {9, 13},
}

# Product-maximizing, minimum-maintenance schedules measured against the real
# interpreter in experiments/animal_yields.py. Ages are relative to
# placed_day. These deliberately contain gaps (notably COW ages 0 and 2):
# feeding/caring on those days costs actions and wheat without increasing the
# animal's season yield.
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


def cycle_finished(crop, age, tile):
    """True once a tile's crop will never yield again and can be reused."""
    if crop not in ONGOING_CROPS:
        return age >= CROP_LAST_AGE[crop]
    return age >= CROP_LAST_AGE[crop] and not tile.get("yield_units", 0)
