from agents.expansion_agent import _pop_market_batch, _strip_partial_animal_builds
from agents.farm_tasks import Task


def test_market_batch_preserves_orders_beyond_engine_cap():
    queue = [["HIRE", index] for index in range(13)]

    first = _pop_market_batch(queue)
    second = _pop_market_batch(queue)

    assert len(first) == 10
    assert len(second) == 3
    assert first + second == [["HIRE", index] for index in range(13)]
    assert queue == []


def test_post_opening_never_leaves_standalone_animal_build():
    task = Task((0, 0), [["HARVEST"], ["BUILD_PASTURE"]])

    cleaned = _strip_partial_animal_builds([task], opening_active=False)

    assert cleaned == [task]
    assert task.actions == [["HARVEST"]]


def test_post_opening_drops_pure_standalone_animal_build():
    task = Task((0, 0), [["BUILD_COOP"]])

    assert _strip_partial_animal_builds([task], opening_active=False) == []


def test_opening_keeps_bootstrap_build_behavior():
    task = Task((0, 0), [["BUILD_PASTURE"]])

    cleaned = _strip_partial_animal_builds([task], opening_active=True)

    assert cleaned == [task]
    assert task.actions == [["BUILD_PASTURE"]]


def test_inventory_aware_queue_uses_carried_wheat_without_shed_roundtrip():
    from collections import Counter
    from agents.scheduler import build_queues

    task = Task((0, 0), [["FEED"]], needs=Counter(WHEAT=1), mandatory=True)
    plans, missing = build_queues(
        [task], (0, 1), 0, worker_inventories=[{"WHEAT": 1}],
    )

    assert missing == []
    assert plans[0].queue == [["NORTH"], ["FEED"]]
    assert ["DROP"] not in plans[0].queue
    assert not any(op[0] == "PICKUP" for op in plans[0].queue)


def test_turnover_waits_when_opening_successor_inputs_are_missing():
    from agents.farm_tasks import build_tasks
    from test_agents_farm_tasks import make_obs, plant

    obs = make_obs(
        2, tiles={(0, 0): plant("WHEAT", 0, 2, yield_units=1)}, shed={},
    )
    obs["_committed_targets"] = {(0, 0)}
    tasks = build_tasks(obs, {(0, 0): ("COW", False)},
                        prioritize_fertilizer_drop=True)

    assert tasks and tasks[0].actions == [["WATER"]]
    assert not any(["HARVEST"] in task.actions for task in tasks)


def test_temporarily_empty_pickup_does_not_block_independent_worker():
    from agents.expansion_agent import make_agent
    from agents.scheduler import WorkerPlan
    from test_agents_farm_tasks import make_obs, plant

    obs = make_obs(
        8, tiles={(5, 4): plant("WHEAT", 7, 8)}, hands=[[5, 4]], shed={},
    )
    obs["hour"] = 10
    obs["private"]["inventories"] = [{}, {}]
    agent = make_agent()
    cells = dict(zip(agent.__code__.co_freevars, agent.__closure__))
    cells["targets"].cell_contents.update(
        {(x, y): None for y in range(10) for x in range(10)}
    )
    state = cells["state"].cell_contents
    state.update(
        day=8, plan_frozen=True, frozen_positions=set(),
        plans=[
            WorkerPlan((4, 4), [["PICKUP", "WHEAT", 1]]),
            WorkerPlan((5, 4), [["WATER"]]),
        ],
    )

    result = agent(obs)

    assert result["farmer"] == ["PASS"]
    assert result["hands"] == [["WATER"]]


def test_max_yield_melon_harvests_even_when_atomic_successor_is_unavailable():
    from agents.farm_tasks import build_tasks
    from test_agents_farm_tasks import make_obs, plant

    obs = make_obs(12, tiles={(0, 0): plant("MELON", 0, 12, yield_units=6)})
    obs["_committed_targets"] = {(0, 0)}

    task = build_tasks(obs, {(0, 0): ("MELON", False)})[0]

    assert ["HARVEST"] in task.actions
    assert not any(op[0] == "PLANT" for op in task.actions)


def test_retargeted_melon_is_not_harvested_before_engine_first_yield_day():
    from agents.farm_tasks import build_tasks
    from test_agents_farm_tasks import make_obs, plant

    obs = make_obs(5, tiles={(0, 0): plant("MELON", 0, 5, yield_units=1)})

    tasks = build_tasks(obs, {(0, 0): ("WHEAT", False)})

    assert not any(["HARVEST"] in task.actions for task in tasks)


def test_final_ongoing_crop_cashout_keeps_harvest_and_dig_without_successor_seed():
    from agents.farm_tasks import build_tasks
    from test_agents_farm_tasks import make_obs, plant

    tile = plant("STRAWBERRY", 0, 16, yield_units=4)
    obs = make_obs(16, tiles={(0, 0): tile}, seeds={})
    obs["_committed_targets"] = {(0, 0)}

    task = build_tasks(
        obs, {(0, 0): ("WHEAT", False)}, preserve_turnovers={(0, 0)},
    )[0]

    assert task.actions == [["HARVEST"], ["DIG"]]
    assert task.ends_cycle
    assert task.sells == {"STRAWBERRY": 4}


def test_resource_free_cashout_runs_before_later_pickup():
    from collections import Counter
    from agents.scheduler import _task_queue

    cashout = Task(
        (4, 3), [["HARVEST"], ["DIG"]], mandatory=True, ends_cycle=True,
    )
    supplied = Task(
        (4, 2), [["FEED"]], needs=Counter(WHEAT=1), mandatory=True,
    )

    queue, _ = _task_queue((4, 4), [cashout, supplied], ((4, 4),))

    assert queue[:3] == [["NORTH"], ["HARVEST"], ["DIG"]]
    assert queue.index(["HARVEST"]) < queue.index(["PICKUP", "WHEAT", 1])
