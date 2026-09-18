"""Horizon-specific maintenance must preserve engine yield and survival."""
import pytest
from experiments.animal_horizon import simulate
from agents.schedules import should_feed_animal, should_care_animal


@pytest.mark.parametrize('animal', ['GOOSE', 'COW', 'SHEEP'])
@pytest.mark.parametrize('days', range(1, 31))
def test_terminal_maintenance_preserves_harvest_fertilizer_and_survival(animal, days):
    expected, old_actions = simulate(animal, days, False)
    actual, actions = simulate(animal, days, True)
    assert actual == expected
    assert actions['FEED'] <= old_actions['FEED']
    assert actions['CARE'] <= old_actions['CARE']
    assert not should_feed_animal(animal, days - 1, days - 1)


def test_short_horizon_cow_still_gets_survival_feed():
    # Age zero is deliberately unfed for cows; another miss would cause escape.
    assert should_feed_animal('COW', 1, last_age=2)
    assert should_feed_animal('COW', 3, last_age=4)


def test_care_banked_after_last_production_is_not_scheduled():
    # A sheep placed on day 7 produces for the last time on day 28 (age 21).
    assert not should_care_animal('SHEEP', 20, last_age=22)
    assert not should_feed_animal('SHEEP', 21, last_age=22)
    assert not should_feed_animal('SHEEP', 22, last_age=22)
