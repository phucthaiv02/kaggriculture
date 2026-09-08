"""Fixed first-day portfolio, fertilizer refinancing on UI day 2, then planner."""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture import kaggriculture as official_game

LAND_ORDER = official_game.LAND_ORDER

# Fill NW on UI day 1; purchase_orders budgets feed before animals/seeds.
OPENING_COUNTS = {"WHEAT": 19, "COW": 2, "SHEEP": 2, "GOOSE": 2}
OPENING_SIZE = sum(OPENING_COUNTS.values())

# Keep UI day 2's immediate fertilizer drops, then release targets on day 3.
OPENING_HANDOFF_DAY = 2

# State-derived land cadence.  LAND_BUY_DAYS is kept as a readable summary;
# should_buy_land_on_schedule derives the next attempt from unlocked land so a
# failed/unfunded order retries instead of silently missing the schedule.
LAND_FIRST_DAY = 6
LAND_INTERVAL_DAYS = 3
LAND_MAX_EXTRA = 2
LAND_BUY_DAYS = tuple(
    LAND_FIRST_DAY + LAND_INTERVAL_DAYS * index for index in range(LAND_MAX_EXTRA)
)

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

    # NW's shed-access corner is its maximum x/y. Reserve the six nearest
    # tiles for the animals that repeatedly fetch feed and return products.
    shed = (max(x for x, _ in positions), max(y for _, y in positions))
    nearest = sorted(positions, key=lambda p: (
        abs(p[0] - shed[0]) + abs(p[1] - shed[1]), -p[1], -p[0],
    ))
    animals = [name for name, count in OPENING_COUNTS.items()
               if name != "WHEAT" for _ in range(count)]
    allocation = dict(zip(nearest, animals + ["WHEAT"] * OPENING_COUNTS["WHEAT"]))
    return {position: (allocation[position], False) for position in positions}


def make_opening_controller():
    """Own the first two days, then permanently hand target selection off."""
    book = {"applied": False, "handed_off": False}

    def governs(obs, targets, active_positions):
        farm = obs["farms"][obs["player"]]
        if book["handed_off"]:
            return False
        if obs["day"] >= OPENING_HANDOFF_DAY or len(farm["unlocked_quadrants"]) > 1:
            book["handed_off"] = True
            return False
        if not book["applied"]:
            targets.update(build_opening_targets(active_positions))
            book["applied"] = True
        return True

    return governs
