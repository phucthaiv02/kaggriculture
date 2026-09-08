"""Shared elapsed-day bounds for new production commitments."""
from kaggle_environments.envs.kaggriculture import kaggriculture as game

SEASON_END_DAY = 29  # Inclusive day index in a 30-day game.

# Legacy/global forecast cap used for standing production and execution guards.
# The planner itself now uses the target-specific windows below.
TARGET_HORIZON_DAYS = 16

# Fresh-target comparison windows. Crops stop at their max-yield age. Animals
# stop after the requested number of post-first-yield harvests under the
# planner's harvest-as-soon-as-ready forecast:
#   GOOSE: age 4 + six more harvests -> age 15
#   COW:   age 8 + three more harvests -> age 14
#   SHEEP: age 6 + two more harvests -> age 12
PLANNER_HORIZON_DAYS = {
    "WHEAT": 4,
    "CARROT": 3,
    "MELON": 10,
    "TOMATO": 11,
    "STRAWBERRY": 16,
    "GOOSE": 15,
    "COW": 14,
    "SHEEP": 12,
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
