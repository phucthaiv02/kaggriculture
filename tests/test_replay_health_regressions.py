"""Regressions extracted from replay failures where producers disappeared."""

from agents.intraday import queue_commitments
from agents.scheduler import WorkerPlan
from agents.schedules import cycle_finished


def test_harvest_on_tile_does_not_mask_missing_survival_work():
    plans = [WorkerPlan((4, 4), [["HARVEST"]])]

    # Normal occupancy still reserves the tile against duplicate planting.
    assert queue_commitments([(4, 4)], plans)[1] == {(4, 4)}
    # Survival rescue must not interpret HARVEST as FEED/WATER coverage.
    assert queue_commitments([(4, 4)], plans, max_steps=5)[1] == set()


def test_actual_feed_and_water_do_cover_survival_rescue():
    feed = [WorkerPlan((4, 4), [["FEED"]])]
    water = [WorkerPlan((4, 4), [["WATER"]])]

    assert queue_commitments([(4, 4)], feed, max_steps=5)[1] == {(4, 4)}
    assert queue_commitments([(4, 4)], water, max_steps=5)[1] == {(4, 4)}


def test_ongoing_crop_is_retired_on_its_last_production_age():
    # Replay failures showed TOMATO age 11 and STRAWBERRY age 16 harvested
    # during the day but left planted, then becoming WEED at the next dawn.
    assert cycle_finished("TOMATO", 11, {"yield_units": 1})
    assert cycle_finished("STRAWBERRY", 16, {"yield_units": 1})


def test_ongoing_crop_is_not_retired_early():
    assert not cycle_finished("TOMATO", 10, {"yield_units": 0})
    assert not cycle_finished("STRAWBERRY", 15, {"yield_units": 0})
