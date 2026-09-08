"""Short fixed opening followed by the ROI planner.

The opening book only creates the initial cash-flow/animal base.  It governs
through day 2 so the two WHEAT -> COW/SHEEP conversions can happen, then hands
off to ``agents/planner.py`` from day 3 onward.  Land expansion is independent
of that handoff: buy attempts are derived from the number of already unlocked
quadrants, targeting the first two extra quadrants on days 6 and 9.
"""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture import kaggriculture as official_game

LAND_ORDER = official_game.LAND_ORDER

OPENING_COUNTS = {"MELON": 12, "WHEAT": 9, "COW": 2, "SHEEP": 2}
OPENING_SIZE = sum(OPENING_COUNTS.values())
OPENING_HANDOFF_DAY = 3

LAND_FIRST_DAY = 6
LAND_INTERVAL_DAYS = 3
LAND_MAX_EXTRA = 2
LAND_BUY_DAYS = tuple(
    LAND_FIRST_DAY + LAND_INTERVAL_DAYS * index for index in range(LAND_MAX_EXTRA)
)

CONVERSION_START_DAY = 2
CONVERSIONS = ("COW", "SHEEP")


def should_buy_land_on_schedule(obs, farm):
    n_extra = len(farm["unlocked_quadrants"]) - 1
    if n_extra < 0 or n_extra >= min(LAND_MAX_EXTRA, len(LAND_ORDER)):
        return False
    next_day = LAND_FIRST_DAY + LAND_INTERVAL_DAYS * n_extra
    return obs["day"] >= next_day and obs.get("hour", 0) == 0


def build_opening_targets(positions):
    if len(positions) != OPENING_SIZE:
        raise ValueError(
            f"opening book requires exactly {OPENING_SIZE} active positions, got {len(positions)}"
        )
    order = []
    for name, count in OPENING_COUNTS.items():
        order += [name] * count
    return {position: (name, False) for position, name in zip(positions, order)}


def make_opening_controller():
    book = {
        "applied": False,
        "handed_off": False,
        "wheat_positions": (),
        "next_conversion": 0,
    }

    def governs(obs, targets, active_positions):
        day = obs["day"]
        farm = obs["farms"][obs["player"]]
        tiles = farm["tiles"]

        if book["handed_off"]:
            return False
        if len(farm["unlocked_quadrants"]) > 1:
            book["handed_off"] = True
            return False
        if not book["applied"]:
            opening = build_opening_targets(active_positions)
            targets.update(opening)
            shed = (len(tiles[0]) // 2 - 1, len(tiles) // 2 - 1)
            book["wheat_positions"] = tuple(
                sorted(
                    (position for position, target in opening.items() if target[0] == "WHEAT"),
                    key=lambda position: (
                        abs(position[0] - shed[0]) + abs(position[1] - shed[1]),
                        -position[1],
                        -position[0],
                    ),
                )
            )
            book["applied"] = True
            return True
        if day >= CONVERSION_START_DAY:
            while book["next_conversion"] < len(CONVERSIONS):
                position = next(
                    (
                        (x, y)
                        for x, y in book["wheat_positions"]
                        if targets.get((x, y), (None, False))[0] == "WHEAT"
                        and isinstance(tiles[y][x], dict)
                        and tiles[y][x].get("kind") == "PLANT"
                        and tiles[y][x].get("crop") == "WHEAT"
                        and tiles[y][x].get("yield_units", 0) > 0
                    ),
                    None,
                )
                if position is None:
                    break
                index = book["next_conversion"]
                targets[position] = (CONVERSIONS[index], False)
                book["next_conversion"] += 1
        if day >= OPENING_HANDOFF_DAY:
            book["handed_off"] = True
            return False
        return True

    return governs
