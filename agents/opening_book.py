"""Versioned opening allocations and fertilizer refinancing before planner handoff."""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture import kaggriculture as official_game

LAND_ORDER = official_game.LAND_ORDER

# 19 WHEAT + 2 COW + 2 SHEEP + 2 GOOSE = 25 (NW's whole board).
# Buy only feed needed on placement day; harvested WHEAT is retained.
OPENING_COUNTS = {"WHEAT": 19, "COW": 2, "SHEEP": 2, "GOOSE": 2}
MELON_OPENING_COUNTS = {"WHEAT": 9, "COW": 2, "SHEEP": 2, "MELON": 12}
OPENING_VERSIONS = {"classic": OPENING_COUNTS, "melon_v2": MELON_OPENING_COUNTS}

OPENING_SIZE = sum(OPENING_COUNTS.values())

# Engine days are zero-indexed. Both books govern indices 0 through 5;
# finished WHEAT plots are repriced separately before replacement purchases.
OPENING_REFINANCE_DAY = 1
PLANNER_HANDOFF_DAY = 6
MELON_PLANNER_HANDOFF_DAY = PLANNER_HANDOFF_DAY

# Expansion is deliberately fixed rather than utilization-driven: submit
# at most two purchases: calendar day 7 (index 6), then the existing index 10.
# No later day may buy the third remaining quadrant.
LAND_BUY_DAYS = (6, 10)
MAX_SCHEDULED_LAND_PURCHASES = 2


def should_buy_land_on_schedule(obs, farm):
    """Retry the scheduled expansion throughout the day, once per quadrant."""
    n_extra = len(farm["unlocked_quadrants"]) - 1
    return (
        obs["day"] in LAND_BUY_DAYS
        and n_extra < LAND_BUY_DAYS.index(obs["day"]) + 1
        and n_extra < min(MAX_SCHEDULED_LAND_PURCHASES, len(LAND_ORDER))
    )


def build_opening_targets(positions, version="classic"):
    """Return the fixed day-0 allocation for the initial 25 active tiles.

    The opening deliberately does not commit crop fertilizer. Animal-produced
    fertilizer is instead liquidated on the second day to replenish working
    capital before planner handoff.
    """
    if len(positions) != OPENING_SIZE:
        raise ValueError(
            f"opening book requires exactly {OPENING_SIZE} active positions, got {len(positions)}"
        )

    if version == "melon_v2":
        # Keep refinancing animals and conversion plots close to the shed.
        positions = sorted(positions, key=lambda p: (abs(p[0] - 4) + abs(p[1] - 4), p))
    order = []
    for name, count in OPENING_VERSIONS[version].items():
        order += [name] * count
    if version == "melon_v2":
        order = (
            ["WHEAT"] * 2 + ["COW"] * 2 + ["SHEEP"] * 2
            + ["WHEAT"] * 7 + ["MELON"] * 12
        )
    return {position: (name, False) for position, name in zip(positions, order)}


def make_opening_controller(version="classic"):
    """Create a stateful opening controller, active until calendar day 7 (engine index 6).

    The returned callable mutates ``targets`` while the opening governs the
    farm and returns ``True`` so the caller skips the dynamic planner. It hands
    off permanently when the engine reaches the handoff day.
    Existing live animals remain protected by planner.py and are not replanned
    out from under the opening.
    """
    if version not in OPENING_VERSIONS:
        raise ValueError(f"unknown opening version: {version}")
    handoff_day = (
        MELON_PLANNER_HANDOFF_DAY if version == "melon_v2"
        else PLANNER_HANDOFF_DAY
    )
    conversion_positions = []
    book = {"applied": False, "handed_off": False}

    def governs(obs, targets, active_positions):
        day = obs["day"]
        farm = obs["farms"][obs["player"]]

        if book["handed_off"]:
            return False

        ready = day >= handoff_day
        if ready:
            book["handed_off"] = True
            return False

        if not book["applied"]:
            targets.update(build_opening_targets(active_positions, version))
            conversion_positions.extend(
                position for position in targets
                if targets[position][0] == "WHEAT"
            )
            book["applied"] = True

        if version == "melon_v2":
            obs["_opening_refinance_first"] = True

        if version == "melon_v2" and day >= 2:
            first, second = conversion_positions[:2]
            targets[first] = ("COW", False)
            targets[second] = ("SHEEP", False) if day >= 3 else None
            obs["_opening_early_harvest_positions"] = {first, second}

        return True

    return governs
