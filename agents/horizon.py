"""Shared elapsed-day bounds for new production commitments."""
from kaggle_environments.envs.kaggriculture import kaggriculture as game

SEASON_END_DAY = 29  # Inclusive day index in a 30-day game.
TARGET_HORIZON_DAYS = 16


def cycle_end(day, end_day):
    return min(end_day, day + TARGET_HORIZON_DAYS)


def first_yield_age(name):
    rules = game.ANIMALS if name in game.ANIMALS else game.CROPS
    return rules[name]['first_yield_day']


def can_start(name, day, end_day):
    return day + first_yield_age(name) <= cycle_end(day, end_day)


def can_start_today(name, obs):
    # make_agent attaches its inclusive endpoint to its local observation copy.
    return can_start(name, obs['day'], obs.get('_planning_end_day', SEASON_END_DAY))
