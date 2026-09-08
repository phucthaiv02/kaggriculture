"""No-opening experiment with state-derived land expansion cadence.

The ROI planner owns the initial quadrant from day 0.  Land expansion is
separate from target selection: buy at most two extra quadrants, with the
next eligible day derived from how many purchases have actually succeeded.
A failed/unfunded order is retried on later mornings instead of being lost.
"""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture import kaggriculture as official_game

LAND_ORDER = official_game.LAND_ORDER

LAND_FIRST_DAY = 6
LAND_INTERVAL_DAYS = 3
LAND_MAX_EXTRA = 2
# Exposed only for diagnostics/tests. should_buy_land_on_schedule derives the
# next day from farm state rather than indexing this tuple.
LAND_BUY_DAYS = tuple(
    LAND_FIRST_DAY + LAND_INTERVAL_DAYS * index
    for index in range(LAND_MAX_EXTRA)
)

# Legacy opening helpers remain import-compatible for isolated historical
# tests/tools, but make_opening_controller deliberately never applies them on
# this branch.
OPENING_COUNTS = {"MELON": 12, "WHEAT": 9, "COW": 2, "SHEEP": 2}
OPENING_SIZE = sum(OPENING_COUNTS.values())


def should_buy_land_on_schedule(obs, farm):
    """Buy the next extra quadrant at a three-day cadence from day 6.

    The schedule is based on *successful* land count.  With only NW unlocked,
    the first eligible morning is day 6.  After that purchase succeeds the
    next eligible morning becomes day 9.  We stop after two extra quadrants.
    If the engine rejects an order for affordability, the same stage remains
    eligible on subsequent hour-0 observations until it succeeds.
    """
    n_extra = max(0, len(farm["unlocked_quadrants"]) - 1)
    max_extra = min(LAND_MAX_EXTRA, len(LAND_ORDER))
    if n_extra >= max_extra or obs.get("hour", 0) != 0:
        return False
    next_day = LAND_FIRST_DAY + LAND_INTERVAL_DAYS * n_extra
    return obs["day"] >= next_day


def build_opening_targets(positions):
    """Legacy fixed allocation helper; not used by the no-opening controller."""
    if len(positions) != OPENING_SIZE:
        raise ValueError(
            f"opening book requires exactly {OPENING_SIZE} active positions, got {len(positions)}"
        )
    order = []
    for name, count in OPENING_COUNTS.items():
        order += [name] * count
    return {position: (name, False) for position, name in zip(positions, order)}


def make_opening_controller():
    """Return a controller that immediately hands every tile to the planner."""
    def governs(obs, targets, active_positions):
        del obs, targets, active_positions
        return False

    return governs
