"""Fixed two-day opening allocation for the initial NW quadrant.

The opening book owns only the initial deployment and the first fertilizer
cash-flow turn. It then hands the standing targets to the price-reactive
planner in ``agents/planner.py`` so the 19-WHEAT opening is not held longer
than necessary.
"""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture import kaggriculture as official_game

LAND_ORDER = official_game.LAND_ORDER

# 19 WHEAT + 2 COW + 2 SHEEP + 2 GOOSE = 25 (NW's whole board).
# purchase_orders buys feed alongside missing animals on day 0; that currently
# reserves one WHEAT per purchased animal, which is conservative because COW
# does not need its first FEED until age 1. The result is therefore guaranteed
# to cover every day-0 feed action for the two SHEEP and two GOOSE.
OPENING_COUNTS = {"WHEAT": 19, "COW": 2, "SHEEP": 2, "GOOSE": 2}
OPENING_SIZE = sum(OPENING_COUNTS.values())

# Engine days are zero-indexed. Keep the fixed opening through engine day 1
# (the UI's Day 2) so fertilizer can be collected/dropped/sold as refinancing,
# then let the planner take over from engine day 2 onward.
OPENING_REFINANCE_DAY = 1
PLANNER_HANDOFF_DAY = 2

# Expansion is deliberately fixed rather than utilization-driven: submit
# exactly two scheduled BUY_LAND attempts, one on day 7 and one on day 10.
# No later day may buy the third remaining quadrant.
LAND_BUY_DAYS = (7, 10)
MAX_SCHEDULED_LAND_PURCHASES = 2


def should_buy_land_on_schedule(obs, farm):
    """Submit BUY_LAND only at hour 0 on day 7 or day 10.

    Affordability itself is left to the engine (BUY_LAND is simply a no-op if
    the farm cannot cover the cost at that instant). The strategy never owns
    more than two extra quadrants through this schedule, so no third BUY_LAND
    is emitted even if another scheduled day is added accidentally later.
    """
    n_extra = len(farm["unlocked_quadrants"]) - 1
    return (
        obs["day"] in LAND_BUY_DAYS
        and obs.get("hour", 0) == 0
        and n_extra < min(MAX_SCHEDULED_LAND_PURCHASES, len(LAND_ORDER))
    )


def build_opening_targets(positions):
    """Return the fixed day-0 allocation for the initial 25 active tiles.

    The opening deliberately does not commit crop fertilizer. Animal-produced
    fertilizer is instead liquidated on the second day to replenish working
    capital before planner handoff.
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
    """Create the stateful two-day opening-book controller.

    The returned callable mutates ``targets`` while the opening governs the
    farm and returns ``True`` so the caller skips the dynamic planner. It hands
    off permanently either when extra land appears or at PLANNER_HANDOFF_DAY.
    Existing live animals remain protected by planner.py and are not replanned
    out from under the opening.
    """
    book = {"applied": False, "handed_off": False}

    def governs(obs, targets, active_positions):
        day = obs["day"]
        farm = obs["farms"][obs["player"]]

        if book["handed_off"]:
            return False

        if len(farm["unlocked_quadrants"]) > 1 or day >= PLANNER_HANDOFF_DAY:
            book["handed_off"] = True
            return False

        if not book["applied"]:
            targets.update(build_opening_targets(active_positions))
            book["applied"] = True

        return True

    return governs
