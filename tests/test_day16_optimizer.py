"""Fast invariants for the cash-aware day-16 optimizer."""

import pytest

from experiments.day16_allocation import ANIMALS, CROPS, ROTATIONS, Policy, candidates


def test_policy_rejects_more_than_the_initial_quadrant():
    with pytest.raises(ValueError):
        Policy.from_dict({"WHEAT": 26})


def test_search_space_contains_every_crop_and_animal():
    names = {name for policy in candidates(limit=24) for name, _ in policy.allocation}
    assert set(CROPS + ANIMALS) <= names


def test_search_policies_never_require_land_purchase():
    for policy in candidates(limit=24):
        assert sum(policy.counts().values()) <= 25
        assert policy.rotation in ROTATIONS


def test_rotation_can_change_crop_or_convert_land_to_animal():
    crop = Policy.from_dict({"WHEAT": 1}, rotation="TOMATO")
    animal = Policy.from_dict({"WHEAT": 1}, rotation="GOOSE")
    assert crop.rotation == "TOMATO"
    assert animal.rotation == "GOOSE"
