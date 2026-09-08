"""Fixed opening allocation for the initial NW quadrant.

The opening book is used instead of the ROI planner until the first land
purchase succeeds. Isolated 25-tile tests consistently favored this fixed
allocation and its early cash-flow sequence over planning from scratch. Once
another quadrant unlocks, all new land and later replant decisions hand off to
the price-reactive planner in ``agents/planner.py``.
"""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture import kaggriculture as official_game

LAND_ORDER = official_game.LAND_ORDER

# 12 MELON + 9 WHEAT + 2 COW + 2 SHEEP = 25 (NW's whole board). Wheat is
# funded ahead of each animal's verified age-relative FEED schedule. In
# particular, placement is age 0: SHEEP feeds/cares then, while COW starts at
# age 1 (experiments/animal_yields.py).
OPENING_COUNTS = {"MELON": 12, "WHEAT": 9, "COW": 2, "SHEEP": 2}
OPENING_SIZE = sum(OPENING_COUNTS.values())

# Keep the fixed opening in control until the first land purchase really
# succeeds. Land attempts themselves follow successful land state: first extra
# quadrant is due on day 6, second on day 9, and an unfunded attempt retries on
# later mornings until the quadrant count changes.
LAND_FIRST_DAY = 6
LAND_INTERVAL_DAYS = 3
LAND_MAX_EXTRA = 2
LAND_BUY_DAYS = tuple(
    LAND_FIRST_DAY + LAND_INTERVAL_DAYS * index
    for index in range(LAND_MAX_EXTRA)
)

# Convert two of the initial WHEAT targets to one COW and one SHEEP together
# on day 2 (the UI's Day 3). Their two harvested WHEAT units can then feed
# the newly placed animals without buying feed from the market.
# farm_tasks.build_tasks deliberately supports harvesting a one-time crop
# early when its target changes, so each conversion can start as soon as the
# opening has generated some WHEAT instead of waiting for the full cycle.
CONVERSION_START_DAY = 2
CONVERSIONS = ("COW", "SHEEP")


def should_buy_land_on_schedule(obs, farm):
    """Submit the next due land order at hour 0, based on successful buys.

    Affordability itself is left to the engine. If BUY_LAND is a no-op because
    cash is insufficient, the same stage stays eligible on later mornings
    instead of being lost to a one-day schedule.
    """
    n_extra = len(farm["unlocked_quadrants"]) - 1
    if n_extra < 0 or n_extra >= min(LAND_MAX_EXTRA, len(LAND_ORDER)):
        return False
    next_day = LAND_FIRST_DAY + LAND_INTERVAL_DAYS * n_extra
    return obs["day"] >= next_day and obs.get("hour", 0) == 0


def build_opening_targets(positions):
    """Return the fixed day-0 allocation for the initial 25 active tiles.

    The opening deliberately does not commit fertilizer; early fertilizer is
    sold to support the initial cash-flow plan.
    """
    if len(positions) != OPENING_SIZE:
        raise ValueError(
            f"opening book requires exactly {OPENING_SIZE} active positions, got {len(positions)}"
        )

    order = []
    for name, count in OPENING_COUNTS.items():
        order += [name] * count
    return {position: (name, False) for position, name in zip(positions, order)}


def make_opening_controller():
    """Create the stateful opening-book controller.

    The returned callable mutates ``targets`` while the opening book governs
    the farm and returns ``True`` so the caller skips the dynamic planner.
    Once the first additional quadrant unlocks, it returns ``False``
    permanently.
    """
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
            # Converted animals produce every few days (and fertilizer every
            # day), so keep their recurring HARVEST/COLLECT/DROP route short.
            shed = (len(tiles[0]) // 2 - 1, len(tiles) // 2 - 1)
            book["wheat_positions"] = tuple(
                sorted(
                    (
                        position
                        for position, target in opening.items()
                        if target[0] == "WHEAT"
                    ),
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

        return True

    return governs
