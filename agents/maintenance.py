"""Yield-preserving maintenance variants used to stagger daily labor.

The baseline schedules in agents.schedules stay the canonical forecast model.
This module only exposes experimentally equivalent variants for execution, so
same-age producers can be placed on different phases without changing action
count or harvest yield.
"""
from __future__ import annotations

from agents.schedules import ANIMAL_CARE_DAYS, ANIMAL_FEED_DAYS, CROP_WATER_DAYS


def _frozen(*days):
    return frozenset(days)


# All variants keep age-0 WATER and the same number of WATER actions as the
# validated baseline. They were enumerated against the Kaggriculture engine
# and retain the same harvest trace as experiments/crop_schedules.py.
CROP_WATER_VARIANTS = {
    ("MELON", True): (
        _frozen(0, 2, 4, 6, 8, 9),
        _frozen(0, 2, 4, 6, 8, 10),
    ),
    ("TOMATO", False): (
        _frozen(0, 2, 4, 6, 8, 9),
        _frozen(0, 1, 3, 5, 7, 9),
        _frozen(0, 2, 3, 5, 7, 9),
        _frozen(0, 2, 4, 5, 7, 9),
        _frozen(0, 2, 4, 6, 7, 9),
        _frozen(0, 2, 4, 6, 8, 10),
    ),
    ("TOMATO", True): (
        _frozen(0, 2, 4, 5, 7, 8, 9, 10),
        _frozen(0, 1, 3, 5, 7, 8, 9, 10),
        _frozen(0, 2, 3, 5, 7, 8, 9, 10),
        _frozen(0, 2, 4, 6, 7, 8, 9, 10),
    ),
    ("STRAWBERRY", True): (
        _frozen(0, 2, 4, 6, 7, 9, 11, 13, 15),
        _frozen(0, 1, 3, 5, 7, 9, 11, 13, 15),
        _frozen(0, 2, 3, 5, 7, 9, 11, 13, 15),
        _frozen(0, 2, 4, 5, 7, 9, 11, 13, 15),
        _frozen(0, 2, 4, 6, 8, 9, 11, 13, 15),
    ),
}


# COW has one useful early-life phase shift that preserves every production
# prefix as well as the full 36-MILK trace: move the age-3 FEED+CARE pair to
# age 2. It deliberately leaves age 0 unchanged, so PLACE remains independent
# of same-turn wheat availability.
_COW_BASE_FEED = frozenset(ANIMAL_FEED_DAYS["COW"])
_COW_BASE_CARE = frozenset(ANIMAL_CARE_DAYS["COW"])
COW_MAINTENANCE_VARIANTS = (
    (_COW_BASE_FEED, _COW_BASE_CARE),
    (
        frozenset({1, 2, *range(4, 28)}),
        frozenset({1, 2, *range(4, 27)}),
    ),
)


def variant_index(position, count):
    """Stable spatial phase; adjacent plots naturally land on different phases."""
    if count <= 1 or position is None:
        return 0
    x, y = position
    return (int(x) + 3 * int(y)) % count


def water_days(crop, fertilized, position=None):
    variants = CROP_WATER_VARIANTS.get((crop, bool(fertilized)))
    if not variants:
        return frozenset(CROP_WATER_DAYS[(crop, bool(fertilized))])
    return variants[variant_index(position, len(variants))]


def should_water(crop, age, fertilized, position=None):
    return age in water_days(crop, fertilized, position)


def animal_days(animal, position=None):
    if animal == "COW":
        return COW_MAINTENANCE_VARIANTS[
            variant_index(position, len(COW_MAINTENANCE_VARIANTS))
        ]
    return (
        frozenset(ANIMAL_FEED_DAYS[animal]),
        frozenset(ANIMAL_CARE_DAYS[animal]),
    )


def should_feed(animal, age, position=None):
    feed, _care = animal_days(animal, position)
    return age in feed


def should_care(animal, age, position=None):
    _feed, care = animal_days(animal, position)
    return age in care
