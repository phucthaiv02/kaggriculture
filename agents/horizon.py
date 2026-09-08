"""Shared elapsed-day bounds for new production commitments."""
from kaggle_environments.envs.kaggriculture import kaggriculture as game

SEASON_END_DAY = 29  # Inclusive day index in a 30-day game.

# Legacy/global forecast cap used for standing production and execution guards.
# The planner itself now uses the target-specific windows below.
TARGET_HORIZON_DAYS = 16

# Fresh-target comparison windows. Crops stop at their max-yield age. Animals
# stop at their first harvest, so profit/day compares the time needed to get
# the initial return rather than bundling later repeat harvests into the score:
#   GOOSE: first harvest at age 4
#   SHEEP: first harvest at age 6
#   COW:   first harvest at age 8
PLANNER_HORIZON_DAYS = {
    "WHEAT": 4,
    "CARROT": 3,
    "MELON": 10,
    "TOMATO": 11,
    "STRAWBERRY": 16,
    "GOOSE": 4,
    "COW": 8,
    "SHEEP": 6,
}


def cycle_end(day, end_day):
    return min(end_day, day + TARGET_HORIZON_DAYS)


def planner_cycle_end(name, day, end_day):
    """Inclusive endpoint for comparing a fresh planner target."""
    return min(end_day, day + PLANNER_HORIZON_DAYS[name])


def first_yield_age(name):
    rules = game.ANIMALS if name in game.ANIMALS else game.CROPS
    return rules[name]['first_yield_day']


def can_start(name, day, end_day):
    return day + first_yield_age(name) <= cycle_end(day, end_day)


def can_start_today(name, obs):
    # make_agent attaches its inclusive endpoint to its local observation copy.
    return can_start(name, obs['day'], obs.get('_planning_end_day', SEASON_END_DAY))
