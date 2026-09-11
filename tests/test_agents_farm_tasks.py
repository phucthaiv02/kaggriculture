"""Isolated tests for agents/farm_tasks.py -- no environment, hand-built obs.

Each tile shape here is copied verbatim from the engine's own constructors
(_new_plant, _new_animal in kaggriculture.py) so a test failure means a real
mismatch with build_tasks' expectations, not a fixture bug.
"""

from agents.farm_tasks import build_tasks, feed_wheat_order, purchase_orders, reserved_items

PRODUCTS = ("WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY", "EGG", "MILK", "WOOL", "FERTILIZER")


def make_obs(day, tiles=(), shed=None, seeds=None, money=3000.0, hands=None,
             inventories=None, prices=None, unlocked_quadrants=("NW",)):
    board = [[None] * 10 for _ in range(10)]
    for (x, y), tile in dict(tiles).items():
        board[y][x] = tile
    farm = {
        "money": money, "tiles": board, "farmer": [4, 4],
        "hands": hands or [], "unlocked_quadrants": list(unlocked_quadrants), "hires_today": 0,
    }
    return {
        "day": day, "hour": 2, "player": 0,
        "farms": [farm, farm],
        "private": {"shed": shed or {}, "seeds": seeds or {}, "inventories": inventories or [{}]},
        "market": {"prices": prices or {p: 25 for p in PRODUCTS}, "inventory": {p: 10000 for p in PRODUCTS}},
    }


def plant(crop, planted_day, day, yield_units=None, watered_today=False, fertilized_until_day=-1):
    ongoing = crop in ("TOMATO", "STRAWBERRY")
    return {
        "kind": "PLANT", "crop": crop, "planted_day": planted_day,
        "watered_today": watered_today,
        "yield_units": (0 if ongoing else 1) if yield_units is None else yield_units,
        "fertilized_until_day": fertilized_until_day,
    }


def animal_tile(animal, placed_day, yield_units=0, fed_today=False, fertilizer_available=False):
    structure = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}[animal]
    return {
        "kind": structure, "animal": animal, "placed_day": placed_day, "yield_units": yield_units,
        "fed_today": fed_today, "cared_today": False, "fertilizer_available": fertilizer_available,
    }


def one_task(tasks):
    assert len(tasks) == 1, tasks
    return tasks[0]


def test_empty_tile_plants_crop_when_seed_available():
    obs = make_obs(day=0, seeds={"WHEAT": 1})
    tasks = build_tasks(obs, {(0, 0): ("WHEAT", False)})
    task = one_task(tasks)
    assert task.actions == [["PLANT", "WHEAT"], ["WATER"]]
    assert not task.needs


def test_empty_tile_skipped_without_seed():
    obs = make_obs(day=0, seeds={})
    tasks = build_tasks(obs, {(0, 0): ("WHEAT", False)})
    assert tasks == []


def test_empty_tile_places_animal_when_shed_has_animal_and_wheat():
    obs = make_obs(day=0, shed={"SHEEP": 1, "WHEAT": 1})
    task = one_task(build_tasks(obs, {(0, 0): ("SHEEP", False)}))
    assert task.actions == [["BUILD_PASTURE"], ["PLACE", "SHEEP"], ["FEED"], ["CARE"]]
    assert task.needs == {"SHEEP": 1, "WHEAT": 1}
    assert not task.urgent


def test_empty_animal_target_waits_for_animal_and_placement_feed():
    for shed in ({}, {"SHEEP": 1}, {"WHEAT": 1}):
        obs = make_obs(day=5, money=200, shed=shed)
        assert build_tasks(obs, {(0, 0): ("SHEEP", False)}) == []
    obs = make_obs(day=5, money=200)
    assert build_tasks(obs, {(0, 0): ("COW", False)}) == []


def test_animal_onto_existing_empty_structure_skips_redundant_build():
    """A PASTURE an earlier animal escaped from is already the right kind and
    empty -- BUILD_PASTURE there would just no-op (engine requires tile is
    None), wasting a turn. Regression test for that fix."""
    obs = make_obs(day=5, tiles={(0, 0): {"kind": "PASTURE"}}, shed={"SHEEP": 1, "WHEAT": 1})
    task = one_task(build_tasks(obs, {(0, 0): ("SHEEP", False)}))
    assert task.actions == [["PLACE", "SHEEP"], ["FEED"], ["CARE"]]


def test_animal_onto_wrong_structure_is_not_scheduled():
    obs = make_obs(day=5, tiles={(0, 0): {"kind": "COOP"}}, shed={"SHEEP": 1, "WHEAT": 1})
    assert build_tasks(obs, {(0, 0): ("SHEEP", False)}) == []


def test_live_animal_fed_when_wheat_available():
    obs = make_obs(day=3, tiles={(0, 0): animal_tile("SHEEP", placed_day=0)}, shed={"WHEAT": 5})
    task = one_task(build_tasks(obs, {(0, 0): ("SHEEP", False)}))
    assert task.actions == [["FEED"], ["CARE"]]
    assert task.needs == {"WHEAT": 1}
    assert task.urgent


def test_live_animal_already_fed_is_not_fed_twice_when_tasks_are_rebuilt():
    tile = animal_tile("SHEEP", placed_day=0, fed_today=True)
    obs = make_obs(day=3, tiles={(0, 0): tile}, shed={"WHEAT": 5})
    task = one_task(build_tasks(obs, {(0, 0): ("SHEEP", False)}))
    assert task.actions == [["CARE"]]
    assert not task.needs


def test_live_animal_harvests_before_feeding():
    obs = make_obs(day=3, tiles={(0, 0): animal_tile("SHEEP", placed_day=0, yield_units=4)}, shed={"WHEAT": 5})
    tasks = build_tasks(obs, {(0, 0): ("SHEEP", False)})
    assert [task.actions for task in tasks] == [[["HARVEST"]], [["FEED"], ["CARE"]]]
    assert tasks[0].sells == {"WOOL": 4}
    assert tasks[0].urgent
    assert tasks[0].animal_harvest


def test_live_animal_harvests_urgently_even_outside_maintenance_schedule():
    obs = make_obs(
        day=28,
        tiles={(0, 0): animal_tile("SHEEP", placed_day=0, yield_units=4)},
        shed={"WHEAT": 5},
    )
    task = one_task(build_tasks(obs, {(0, 0): ("SHEEP", False)}))
    assert task.actions == [["HARVEST"]]
    assert task.sells == {"WOOL": 4}
    assert task.urgent
    assert task.animal_harvest


def test_live_animal_collects_fertilizer_without_prioritized_drop_when_no_cash_or_wheat():
    obs = make_obs(day=3, tiles={(0, 0): animal_tile("SHEEP", placed_day=0, fertilizer_available=True)}, shed={}, money=0)
    task = one_task(build_tasks(obs, {(0, 0): ("SHEEP", False)}))
    assert task.actions == [["COLLECT_FERTILIZER"]]
    assert task.sells == {"FERTILIZER": 1}
    assert not task.immediate_drop
    assert not task.refinance_feed


def test_opening_live_animal_prioritizes_fertilizer_refinance():
    obs = make_obs(day=3, tiles={(0, 0): animal_tile("SHEEP", placed_day=0, fertilizer_available=True)}, shed={})
    task = one_task(
        build_tasks(
            obs,
            {(0, 0): ("SHEEP", False)},
            prioritize_fertilizer_drop=True,
        )
    )
    assert task.immediate_drop
    assert task.refinance_feed


def test_day_five_fertilizer_is_cash_first_liquidation_work():
    obs = make_obs(
        day=4,
        tiles={(0, 0): animal_tile("SHEEP", placed_day=0, fertilizer_available=True)},
        shed={"WHEAT": 1},
    )
    obs["_liquidate_fertilizer_first"] = True
    task = one_task(build_tasks(
        obs,
        {(0, 0): ("SHEEP", False)},
        prioritize_fertilizer_drop=True,
    ))
    assert ["COLLECT_FERTILIZER"] in task.actions
    assert task.immediate_drop
    assert task.must_liquidate


def test_live_animal_collects_fertilizer_in_addition_to_normal_feed():
    """Regression test: fed normally *and* has a fertilizer bonus sitting on
    the tile -- both are free money and must not be mutually exclusive.
    (A prior version checked the post-decrement wheat counter and silently
    dropped this fertilizer collection whenever feeding succeeded.)"""
    obs = make_obs(
        day=3, tiles={(0, 0): animal_tile("SHEEP", placed_day=0, fertilizer_available=True)}, shed={"WHEAT": 5},
    )
    task = one_task(build_tasks(obs, {(0, 0): ("SHEEP", False)}))
    assert task.actions == [["FEED"], ["CARE"], ["COLLECT_FERTILIZER"]]
    assert task.sells == {"FERTILIZER": 1}
    assert task.needs == {"WHEAT": 1}
    assert not task.immediate_drop
    assert not task.refinance_feed


def test_one_time_crop_harvests_and_replants_on_last_age():
    """Age 4 is WHEAT's last maintenance day *and* its harvest day: the final
    WATER must still happen (it banks the last bonus -- see
    experiments/crop_schedules.py), strictly before HARVEST collects it."""
    obs = make_obs(day=4, tiles={(0, 0): plant("WHEAT", planted_day=0, day=4, yield_units=6)}, seeds={"WHEAT": 1})
    task = one_task(build_tasks(obs, {(0, 0): ("WHEAT", False)}))
    assert task.actions.index(["WATER"]) < task.actions.index(["HARVEST"])
    assert task.sells == {"WHEAT": 6}
    assert task.ends_cycle
    assert task.actions.index(["PLANT", "WHEAT"]) > task.actions.index(["HARVEST"])
    assert task.actions == [
        ["WATER"], ["HARVEST"], ["PLANT", "WHEAT"], ["WATER"]
    ]


def test_wheat_conversion_waits_for_exact_max_yield_day():
    obs = make_obs(
        day=1,
        tiles={(0, 0): plant("WHEAT", planted_day=0, day=1, yield_units=1)},
        shed={},
    )
    assert build_tasks(obs, {(0, 0): ("COW", False)}) == []


def test_wheat_conversion_keeps_maintenance_before_max_yield():
    obs = make_obs(
        day=2,
        tiles={(0, 0): plant("WHEAT", planted_day=0, day=2, yield_units=1)},
        shed={"COW": 1},
    )
    task = one_task(build_tasks(obs, {(0, 0): ("COW", False)}))
    assert task.actions == [["WATER"]]
    assert not task.ends_cycle
    assert not task.needs


def test_wheat_conversion_happens_on_exact_max_yield_day():
    obs = make_obs(
        day=4,
        tiles={(0, 0): plant("WHEAT", planted_day=0, day=4, yield_units=4)},
        shed={"COW": 1},
    )
    task = one_task(build_tasks(obs, {(0, 0): ("COW", False)}))
    assert task.actions == [
        ["WATER"], ["HARVEST"], ["BUILD_PASTURE"], ["PLACE", "COW"]
    ]
    assert task.ends_cycle
    assert task.needs == {"COW": 1}


def test_water_is_urgent_once_a_day_was_already_missed():
    """One more missed WATER kills the crop outright (kaggriculture turns it
    into a WEED at consecutive_unwatered >= 2), not just a delayed harvest --
    this must outrank an ordinary (non-urgent) task under hand-capacity
    pressure, the same way animal care already does."""
    tile = plant("WHEAT", planted_day=0, day=3, watered_today=False)
    tile["consecutive_unwatered"] = 1
    task = one_task(build_tasks(make_obs(day=3, tiles={(0, 0): tile}), {(0, 0): ("WHEAT", False)}))
    assert task.urgent


def test_water_is_urgent_before_the_first_miss():
    tile = plant("WHEAT", planted_day=0, day=3, watered_today=False)
    tile["consecutive_unwatered"] = 0
    task = one_task(build_tasks(make_obs(day=3, tiles={(0, 0): tile}), {(0, 0): ("WHEAT", False)}))
    assert task.urgent


def test_one_time_crop_not_yet_finished_only_waters():
    obs = make_obs(day=2, tiles={(0, 0): plant("WHEAT", planted_day=0, day=2)}, seeds={"WHEAT": 1})
    task = one_task(build_tasks(obs, {(0, 0): ("WHEAT", False)}))
    assert task.actions == [["WATER"]]
    assert not task.ends_cycle


def test_ongoing_crop_mid_cycle_harvests_without_ending():
    """TOMATO at age 9 has produced yield but is nowhere near its last age
    (11) -- HARVEST it, but the tile must not be treated as finished."""
    obs = make_obs(day=9, tiles={(0, 0): plant("TOMATO", planted_day=0, day=9, yield_units=4)})
    task = one_task(build_tasks(obs, {(0, 0): ("TOMATO", False)}))
    assert ["HARVEST"] in task.actions
    assert not task.ends_cycle
    assert task.sells == {"TOMATO": 4}


def test_ongoing_crop_finished_digs_and_replants():
    """Age >= last age AND yield_units == 0 (already harvested) -- only then
    is the tile free to reuse; harvesting alone must never DIG."""
    obs = make_obs(day=11, tiles={(0, 0): plant("TOMATO", planted_day=0, day=11, yield_units=0)}, seeds={"CARROT": 1})
    task = one_task(build_tasks(obs, {(0, 0): ("CARROT", False)}))
    assert task.actions[0] == ["DIG"]
    assert task.ends_cycle
    assert ["PLANT", "CARROT"] in task.actions


def test_reserved_items_sums_needs_across_tasks():
    obs = make_obs(
        day=3,
        tiles={(0, 0): animal_tile("SHEEP", placed_day=0), (1, 0): animal_tile("COW", placed_day=0)},
        shed={"WHEAT": 5},
    )
    tasks = build_tasks(obs, {(0, 0): ("SHEEP", False), (1, 0): ("COW", False)})
    assert reserved_items(tasks) == {"WHEAT": 2}


def test_purchase_orders_funds_feed_before_animals_and_caps_by_affordability():
    """Feed is bought first, but animal demand is pre-capped so it cannot
    consume an unbounded amount of WHEAT before placement."""
    obs = make_obs(day=0, money=1200.0)
    targets = {(0, 0): ("SHEEP", False), (1, 0): ("SHEEP", False), (2, 0): ("SHEEP", False)}
    orders = purchase_orders(obs, targets, list(targets))
    assert orders[0][0] == "BUY_PRODUCT"
    wheat_index = next(i for i, order in enumerate(orders) if order[0] == "BUY_PRODUCT")
    animal_index = next(i for i, order in enumerate(orders) if order[0] == "BUY_ANIMAL")
    assert wheat_index < animal_index
    # SHEEP costs 500; with a 1200 budget and a +15/head wheat buffer, at
    # most 2 fit (2*515=1030 <= 1200 < 3*515=1545).
    assert next(order for order in orders if order[0] == "BUY_ANIMAL")[2] == 2


def test_purchase_orders_funds_cheapest_seed_type_first_when_cash_is_short():
    """Regression test: with both WHEAT (cost 10) and MELON (cost 80)
    demanded and too little cash for both, funding whichever the position
    scan hit first (MELON, if it's targeted at lower positions) could starve
    WHEAT entirely for the day even though far more WHEAT tiles could have
    been planted with the same money."""
    obs = make_obs(day=0, money=100.0)
    targets = {(0, 0): ("MELON", False), (1, 0): ("WHEAT", False), (2, 0): ("WHEAT", False)}
    orders = purchase_orders(obs, targets, list(targets))
    seed_orders = {order[1]: order[2] for order in orders if order[0] == "BUY_SEED"}
    assert seed_orders.get("WHEAT") == 2
    # 100 - 2*10 = 80, exactly one MELON seed.
    assert seed_orders.get("MELON") == 1


def test_feed_wheat_order_covers_the_shortfall_for_live_animals():
    obs = make_obs(day=3, tiles={(0, 0): animal_tile("SHEEP", placed_day=0)}, shed={})
    orders = feed_wheat_order(obs, {(0, 0): ("SHEEP", False)}, [(0, 0)])
    assert orders == [["BUY_PRODUCT", "WHEAT", 1]]


def test_intraday_feed_credits_ripe_wheat_only_against_future_reserve():
    targets = {(0, 0): ("SHEEP", False), (1, 0): ("WHEAT", False)}
    obs = make_obs(day=4, tiles={
        (0, 0): animal_tile("SHEEP", placed_day=0),
        (1, 0): plant("WHEAT", planted_day=0, day=4, yield_units=10),
    })
    assert feed_wheat_order(obs, targets, list(targets)) == [["BUY_PRODUCT", "WHEAT", 1]]
    obs["private"]["shed"]["WHEAT"] = 1
    assert feed_wheat_order(obs, targets, list(targets)) == []


def test_feed_wheat_order_stops_when_today_is_covered():
    """Regression test: this must be safe to call every turn of the day
    (not just hour 0/1) so a mid-day FERTILIZER sale can fund the same
    day's feed -- calling it again after the shortfall is already covered
    must not re-request wheat that's already on hand."""
    obs = make_obs(day=3, tiles={(0, 0): animal_tile("SHEEP", placed_day=0)}, shed={"WHEAT": 1})
    orders = feed_wheat_order(obs, {(0, 0): ("SHEEP", False)}, [(0, 0)])
    assert orders == []


def test_feed_wheat_order_counts_carried_wheat_without_a_buffer():
    obs = make_obs(
        day=3,
        tiles={(0, 0): animal_tile("SHEEP", placed_day=0)},
        shed={},
        inventories=[{}, {"WHEAT": 1}],
    )
    orders = feed_wheat_order(obs, {(0, 0): ("SHEEP", False)}, [(0, 0)])
    assert orders == []


def test_purchase_orders_requests_seeds_for_empty_crop_tiles():
    obs = make_obs(day=0, money=100.0)
    orders = purchase_orders(obs, {(0, 0): ("WHEAT", False)}, [(0, 0)])
    assert ["BUY_SEED", "WHEAT", 1] in orders


def test_purchase_orders_requests_replant_seed_for_finished_same_crop():
    obs = make_obs(
        day=4,
        tiles={(0, 0): plant("WHEAT", planted_day=0, day=4, yield_units=4)},
        money=100.0,
    )
    orders = purchase_orders(
        obs, {(0, 0): ("WHEAT", False)}, [(0, 0)], replant_same_crop=True
    )
    assert ["BUY_SEED", "WHEAT", 1] in orders


def test_labor_forecast_includes_replant_actions_before_seed_arrives():
    obs = make_obs(
        day=4,
        tiles={(0, 0): plant("WHEAT", planted_day=0, day=4, yield_units=4)},
    )
    task = one_task(
        build_tasks(obs, {(0, 0): ("WHEAT", False)}, assume_crop_seeds=True)
    )
    assert task.actions == [
        ["WATER"], ["HARVEST"], ["PLANT", "WHEAT"], ["WATER"]
    ]


def test_labor_forecast_includes_unpurchased_animal_placement():
    obs = make_obs(day=3, tiles={(0, 0): {"kind": "PASTURE"}}, shed={})
    task = one_task(
        build_tasks(obs, {(0, 0): ("SHEEP", False)}, assume_animal_inputs=True)
    )
    assert task.actions == [["PLACE", "SHEEP"], ["FEED"], ["CARE"]]


def test_purchase_orders_can_budget_same_turn_sale_proceeds():
    obs = make_obs(day=2, money=100.0)
    targets = {(0, 0): ("COW", False)}
    assert not any(
        order[0] == "BUY_ANIMAL" for order in purchase_orders(obs, targets, list(targets))
    )
    assert ["BUY_ANIMAL", "COW", 1] in purchase_orders(
        obs, targets, list(targets), available_money=500.0
    )


def test_purchase_orders_buys_affordable_cow_before_costlier_sheep():
    obs = make_obs(day=3, money=500.0)
    # SHEEP deliberately appears first to reproduce row-major target order.
    targets = {(3, 3): ("SHEEP", False), (4, 3): ("COW", False)}
    animal_orders = [
        order for order in purchase_orders(obs, targets, list(targets))
        if order[0] == "BUY_ANIMAL"
    ]
    assert animal_orders == [["BUY_ANIMAL", "COW", 1]]


def test_purchase_orders_buys_seeds_while_animal_target_is_unfunded():
    obs = make_obs(day=3, money=100.0)
    targets = {(0, 0): ("SHEEP", False), (1, 0): ("WHEAT", False)}
    orders = purchase_orders(obs, targets, list(targets))
    assert ["BUY_SEED", "WHEAT", 1] in orders
    assert not any(order[0] == "BUY_ANIMAL" for order in orders)


def test_seed_orders_preserve_next_days_animal_feed_money():
    obs = make_obs(day=1, money=450.0)
    targets = {(0, 0): ("COW", False), (1, 0): ("WHEAT", False)}
    orders = purchase_orders(obs, targets, list(targets))
    assert ["BUY_ANIMAL", "COW", 1] in orders
    assert ["BUY_SEED", "WHEAT", 1] in orders
    assert not any(order[0] == "BUY_PRODUCT" for order in orders)


def test_cow_feed_and_care_follow_verified_age_schedule():
    target = {(0, 0): ("COW", False)}
    for age, expected in (
        (0, []),
        (1, [["FEED"], ["CARE"]]),
        (2, []),
        (3, [["FEED"], ["CARE"]]),
        (26, [["FEED"], ["CARE"]]),
        (27, [["FEED"]]),
        (28, []),
    ):
        obs = make_obs(
            day=age,
            tiles={(0, 0): animal_tile("COW", placed_day=0)},
            shed={"WHEAT": 1},
        )
        tasks = build_tasks(obs, target)
        assert ([action for task in tasks for action in task.actions]) == expected


def test_sheep_feed_and_care_follow_verified_age_schedule():
    target = {(0, 0): ("SHEEP", False)}
    for age, expected in (
        (0, [["FEED"], ["CARE"]]),
        (25, [["FEED"], ["CARE"]]),
        (26, [["FEED"]]),
        (27, [["FEED"]]),
        (28, []),
    ):
        obs = make_obs(
            day=age,
            tiles={(0, 0): animal_tile("SHEEP", placed_day=0)},
            shed={"WHEAT": 1},
        )
        tasks = build_tasks(obs, target)
        assert ([action for task in tasks for action in task.actions]) == expected


def test_existing_surplus_wheat_feeds_new_animal_without_market_purchase():
    obs = make_obs(day=3, money=500.0, shed={"WHEAT": 5})
    targets = {
        (0, 0): ("COW", False),
        (1, 0): ("COW", False),
    }
    obs["farms"][0]["tiles"][0][0] = animal_tile("COW", placed_day=0)
    orders = purchase_orders(
        obs, targets, list(targets), available_money=500.0, available_wheat=2
    )
    assert ["BUY_ANIMAL", "COW", 1] in orders
    assert not any(order[0] == "BUY_PRODUCT" for order in orders)


if __name__ == "__main__":
    import sys
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"{'ALL PASSED' if not failures else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)


def test_affordable_feed_is_scheduled_before_waiting_for_fertilizer_sale():
    obs = make_obs(day=3, tiles={(0, 0): animal_tile("SHEEP", 0, fertilizer_available=True)}, shed={}, money=100)
    task = one_task(build_tasks(obs, {(0, 0): ("SHEEP", False)}))
    assert task.actions == [["FEED"], ["CARE"], ["COLLECT_FERTILIZER"]]
    assert task.needs["WHEAT"] == 1


def test_stale_target_never_digs_animal_structures():
    for name in ("WHEAT", "COW"):
        obs = make_obs(day=7, tiles={(0, 0): {"kind": "COOP"}},
                       seeds={"WHEAT": 1}, shed={"COW": 1, "WHEAT": 1})
        assert build_tasks(obs, {(0, 0): (name, False)}) == []


def test_melon_opening_cuts_only_the_two_designated_wheat_tiles():
    positions = [(0, 0), (1, 0), (2, 0)]
    obs = make_obs(day=2, tiles={pos: plant("WHEAT", 0, 2, yield_units=2) for pos in positions})
    obs["_opening_early_harvest_positions"] = set(positions[:2])
    tasks = build_tasks(obs, {positions[0]: ("COW", False), positions[1]: None,
                              positions[2]: ("WHEAT", False)})
    harvested = {task.position for task in tasks if ["HARVEST"] in task.actions}
    assert harvested == set(positions[:2])


def test_harvest_replant_water_remains_one_scheduler_visit():
    from agents.scheduler import _mandatory_actions
    obs = make_obs(day=4, tiles={(0, 0): plant('WHEAT', 0, 4, 4)}, seeds={'WHEAT': 1})
    task = one_task(build_tasks(obs, {(0, 0): ('WHEAT', False)}))
    assert _mandatory_actions(task)[-3:] == [['HARVEST'], ['PLANT', 'WHEAT'], ['WATER']]


def test_early_wheat_returns_to_shed_before_next_task():
    from agents.scheduler import _task_queue
    obs = make_obs(day=2, tiles={(0, 0): plant('WHEAT', 0, 2, 2)})
    obs['_opening_early_harvest_positions'] = {(0, 0)}
    tasks = build_tasks(obs, {(0, 0): None})
    queue, end = _task_queue((0, 0), tasks, ((4, 4),))
    assert queue[-1] == ['DROP']
    assert end == (4, 4)


def test_no_future_feed_top_up_after_opening_placements():
    obs = make_obs(day=4, tiles={(0, 0): animal_tile('SHEEP', 0)}, shed={'WHEAT': 1})
    assert feed_wheat_order(obs, {(0, 0): ('SHEEP', False)}, [(0, 0)]) == []


def test_missed_water_is_rescued_on_scheduled_rest_day():
    tile = plant('MELON', 0, 1, 0)
    tile['consecutive_unwatered'] = 1
    tasks = build_tasks(make_obs(1, {(0, 0): tile}), {(0, 0): ('MELON', False)})
    assert any(['WATER'] in task.actions and task.urgent for task in tasks)
