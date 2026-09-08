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

# 12 MELON + 9 WHEAT + 2 COW + 2 SHEEP = 25 (NW's whole board). Wheat is
# funded ahead of each animal's verified age-relative FEED schedule.
OPENING_COUNTS = {"MELON": 12, "WHEAT": 9, "COW": 2, "SHEEP": 2}
OPENING_SIZE = sum(OPENING_COUNTS.values())

# The fixed book gets the farm started, then the planner can react to live
# prices for three full days before the first expansion attempt.
OPENING_HANDOFF_DAY = 3

# State-derived land cadence.  LAND_BUY_DAYS is kept as a readable summary;
# should_buy_land_on_schedule derives the next attempt from unlocked land so a
# failed/unfunded order retries instead of silently missing the schedule.
LAND_FIRST_DAY = 6
LAND_INTERVAL_DAYS = 3
LAND_MAX_EXTRA = 2
LAND_BUY_DAYS = tuple(
    LAND_FIRST_DAY + LAND_INTERVAL_DAYS * index for index in range(LAND_MAX_EXTRA)
)

# Convert two of the initial WHEAT targets to one COW and one SHEEP together
# on day 2 (the UI's Day 3). Their harvested WHEAT can feed the new animals
# without buying that feed from the market.
CONVERSION_START_DAY = 2
CONVERSIONS = ("COW", "SHEEP")


def should_buy_land_on_schedule(obs, farm):
    """Attempt the next of two land purchases at hour 0 on a 3-day cadence.

    The next target day comes from actual unlocked-quadrant state.  If an
    order is unaffordable, the same purchase remains due on subsequent days
    until the quadrant count changes.  Affordability itself is left to the
    engine.
    """
    n_extra = len(farm["unlocked_quadrants"]) - 1
    if n_extra < 0 or n_extra >= min(LAND_MAX_EXTRA, len(LAND_ORDER)):
        return False
    next_day = LAND_FIRST_DAY + LAND_INTERVAL_DAYS * n_extra
    return obs["day"] >= next_day and obs.get("hour", 0) == 0


def build_opening_targets(positions):
    """Return the fixed day-0 allocation for the initial 25 active tiles."""
    if len(positions) != OPENING_SIZE:
        raise ValueError(
            f"opening book requires exactly {OPENING_SIZE} active positions, got {len(positions)}"
        )

    order = []
    for name, count in OPENING_COUNTS.items():
        order += [name] * count
    return {position: (name, False) for position, name in zip(positions, order)}


def make_opening_controller():
    """Create the short fixed-opening controller.

    Returns ``True`` while the fixed book owns target selection.  From day 3
    onward it returns ``False`` permanently so the normal planner immediately
    reevaluates the live farm.  A successful early land unlock also forces an
    immediate handoff.
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

        if day >= OPENING_HANDOFF_DAY:
            book["handed_off"] = True
            return False

        return True

    return governs
