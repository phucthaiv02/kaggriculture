from collections import Counter

from agents.agent_v2 import _extra_hires_to_fit
from agents.farm_tasks import Task
from agents.schedules import should_care_animal, should_feed_animal
from agents.v2_lifecycle import schedule_variants
from agents.v2_model import TileJob, jobs_from_tasks
from agents.v2_scheduler import _pickup_requirement, build_plans


def test_same_tile_animal_work_is_one_atomic_chain():
    tasks = [
        Task((2, 2), [["HARVEST"]], sells=Counter({"MILK": 4})),
        Task(
            (2, 2),
            [["FEED"], ["CARE"], ["COLLECT_FERTILIZER"]],
            needs=Counter({"WHEAT": 1}),
            sells=Counter({"FERTILIZER": 1}),
        ),
    ]
    jobs = jobs_from_tasks(tasks)
    assert len(jobs) == 1
    assert [action[0] for action in jobs[0].actions] == [
        "HARVEST", "COLLECT_FERTILIZER", "FEED", "CARE"
    ]
    assert jobs[0].split_after == 2


def test_future_harvest_reduces_initial_pickup_without_reordering_route():
    jobs = [
        TileJob((1, 1), (("HARVEST",),), produces=Counter({"WHEAT": 1})),
        TileJob((1, 2), (("FEED",),), needs=Counter({"WHEAT": 3})),
    ]
    assert _pickup_requirement(jobs) == Counter({"WHEAT": 2})


def test_harvest_after_feed_does_not_reduce_pickup():
    jobs = [
        TileJob((1, 2), (("FEED",),), needs=Counter({"WHEAT": 3})),
        TileJob((1, 1), (("HARVEST",),), produces=Counter({"WHEAT": 1})),
    ]
    assert _pickup_requirement(jobs) == Counter({"WHEAT": 3})


def test_scheduler_keeps_atomic_job_actions_together():
    job = TileJob(
        (0, 0),
        (("WATER",), ("HARVEST",), ("PLANT", "MELON"), ("WATER",)),
    )
    plans, unassigned = build_plans([job], [(4, 4)], [23])
    assert not unassigned
    action_names = [
        op[0]
        for op in plans[0].queue
        if op[0] in {"WATER", "HARVEST", "PLANT"}
    ]
    assert action_names == ["WATER", "HARVEST", "PLANT", "WATER"]


def test_explicit_dig_is_preserved_for_weed_recovery():
    jobs = jobs_from_tasks(
        [Task((0, 0), [["DIG"], ["PLANT", "WHEAT"], ["WATER"]])]
    )
    assert len(jobs) == 1
    assert [action[0] for action in jobs[0].actions] == [
        "DIG", "PLANT", "WATER"
    ]


def test_animal_variants_match_verified_placement_day_actions():
    for animal in ("GOOSE", "COW", "SHEEP"):
        variants = schedule_variants(animal, False)
        assert variants
        assert all(
            (0 in variant.feed) == should_feed_animal(animal, 0)
            for variant in variants
        )
        assert all(
            (0 in variant.care) == should_care_animal(animal, 0)
            for variant in variants
        )


def test_scheduler_can_request_one_more_hand_before_execution():
    jobs = [
        TileJob((4, 4), tuple(("WATER",) for _ in range(15))),
        TileJob((0, 0), tuple(("WATER",) for _ in range(15))),
    ]
    starts = [(4, 4)]
    assert _extra_hires_to_fit(
        jobs,
        starts,
        hour=1,
        shed_access=((4, 4), (5, 4), (4, 5), (5, 5)),
    ) == 1
