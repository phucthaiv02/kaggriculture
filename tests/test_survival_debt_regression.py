"""Regression coverage for the seed-1 Day-10 animal-loss replay."""

from agents.farm_tasks import Task, build_tasks, feed_wheat_order
from agents.scheduler import SHED, build_queues


PRODUCTS = (
    "WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)


def _obs(day, tiles, shed=None, hour=2):
    board = [[None] * 10 for _ in range(10)]
    for (x, y), tile in tiles.items():
        board[y][x] = tile
    farm = {
        "money": 500.0,
        "tiles": board,
        "farmer": [4, 4],
        "hands": [],
        "unlocked_quadrants": ["NW", "NE", "SW", "SE"],
        "hires_today": 0,
    }
    return {
        "day": day,
        "hour": hour,
        "player": 0,
        "farms": [farm, farm],
        "private": {
            "shed": shed or {},
            "seeds": {},
            "inventories": [{}],
        },
        "market": {
            "prices": {name: 25 for name in PRODUCTS},
            "inventory": {name: 10000 for name in PRODUCTS},
        },
    }


def _cow(placed_day, consecutive_unfed=0):
    return {
        "kind": "PASTURE",
        "animal": "COW",
        "placed_day": placed_day,
        "yield_units": 0,
        "consecutive_unfed": consecutive_unfed,
        "fed_today": False,
        "cared_today": False,
        "fertilizer_available": False,
        "pending_care_bonus": 0,
    }


def _wheat(planted_day, consecutive_unwatered=0):
    return {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": planted_day,
        "watered_today": False,
        "consecutive_unwatered": consecutive_unwatered,
        "yield_units": 1,
        "max_lifespan_step": 600,
        "fertilized_until_day": -1,
    }


def test_starvation_debt_feed_precedes_ready_animal_harvest():
    position = (5, 2)
    obs = _obs(
        day=8,
        tiles={position: _cow(placed_day=7, consecutive_unfed=1)},
        shed={"WHEAT": 1},
    )
    feed_task = next(
        task for task in build_tasks(obs, {position: ("COW", False)})
        if ["FEED"] in task.actions
    )
    assert feed_task.urgent
    assert getattr(feed_task, "survival_debt", False)

    ordinary_harvest = Task(
        (0, 0), [["HARVEST"]], urgent=True, animal_harvest=True
    )
    plans, unassigned = build_queues(
        [ordinary_harvest, feed_task], farmer_start=SHED, hand_count=0
    )
    assert unassigned == []
    queue = plans[0].queue
    assert queue.index(["FEED"]) < queue.index(["HARVEST"])


def test_second_miss_water_is_marked_as_survival_debt():
    position = (3, 3)
    obs = _obs(
        day=8,
        tiles={position: _wheat(planted_day=5, consecutive_unwatered=1)},
    )
    task = next(
        task for task in build_tasks(obs, {position: ("WHEAT", False)})
        if ["WATER"] in task.actions
    )
    assert task.urgent
    assert getattr(task, "survival_debt", False)


def test_hour_23_does_not_buy_tomorrows_cow_feed_with_hire_cash():
    positions = ((5, 2), (5, 3), (5, 4), (6, 4))
    tiles = {position: _cow(placed_day=7) for position in positions}
    targets = {position: ("COW", False) for position in positions}

    # These COWs are age 0: the validated maintenance schedule intentionally
    # does not FEED them today. Four wheat already covers the core's
    # conservative same-day accounting, so any extra order is future reserve.
    last_hour = _obs(day=7, hour=23, tiles=tiles, shed={"WHEAT": 4})
    assert feed_wheat_order(last_hour, targets, list(positions)) == []

    # Future feed is never stockpiled; buy only feed due today.
    earlier = _obs(day=7, hour=22, tiles=tiles, shed={"WHEAT": 4})
    assert feed_wheat_order(earlier, targets, list(positions)) == []


def test_hour_23_still_buys_feed_needed_today():
    position = (4, 3)
    sheep = {
        "kind": "PASTURE",
        "animal": "SHEEP",
        "placed_day": 0,
        "yield_units": 0,
        "consecutive_unfed": 0,
        "fed_today": False,
        "cared_today": False,
        "fertilizer_available": False,
        "pending_care_bonus": 0,
    }
    obs = _obs(day=3, hour=23, tiles={position: sheep}, shed={})
    assert feed_wheat_order(obs, {position: ("SHEEP", False)}, [position]) == [
        ["BUY_PRODUCT", "WHEAT", 1]
    ]