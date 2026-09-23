from pathlib import Path


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:120]
    return text.replace(old, new, 1)


path = Path("agents/schedules.py")
text = path.read_text()
marker = "\n# Last age at which a one-time crop can still be harvested or an ongoing crop\n"
assert marker in text
profiles = '''
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
'''
text = text.replace(marker, "\n" + profiles + marker, 1)

text = replace_once(
    text,
    '''def water_days(crop, fertilized):
    return CROP_WATER_DAYS[(crop, _cycle_uses_fertilizer(fertilized))]


def is_maintenance_day(crop, age, fertilized):
    return age in water_days(crop, fertilized)
''',
    '''def water_profiles(crop, fertilized):
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
''',
)

text = replace_once(
    text,
    '''def cycle_finished(crop, age, tile):
    """Return whether a crop will never yield again and its tile can be reused."""
    if crop not in ONGOING_CROPS:
        return age >= CROP_LAST_AGE[crop]
    return age >= CROP_LAST_AGE[crop] and not tile.get("yield_units", 0)
''',
    '''def cycle_finished(crop, age, tile):
    """Return whether a crop is already empty and physically reusable now."""
    if crop not in ONGOING_CROPS:
        return age >= CROP_LAST_AGE[crop]
    return age >= CROP_LAST_AGE[crop] and not tile.get("yield_units", 0)


def cycle_turns_over_today(crop, age, tile):
    """Whether today's full plan can harvest/clear the crop and reuse its tile."""
    if crop in ONGOING_CROPS:
        return age >= CROP_LAST_AGE[crop]
    return cycle_finished(crop, age, tile)
''',
)

path.write_text(text)
