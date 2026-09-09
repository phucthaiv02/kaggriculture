"""Focused regression tests for end-game monetization rules."""

from agents.farm_tasks import build_tasks
from agents.scheduler import build_queues
from agents.schedules import animal_maintenance_can_still_pay
from agents.selling import sell_orders


def _empty_board(size=10):
    return [[None for _ in range(size)] for _ in range(size)]


def _obs(day, tiles, inventories=None, hands=None, shed=None, end_day=29):
    return {
        "day": day,
        "hour": 1,
        "player": 0,
        "_planning_end_day": end_day,
        "farms": [{
            "tiles": tiles,
            "farmer": [4, 4],
            "hands": hands or [],
            "money": 10000,
        }],
        "private": {
            "shed": shed or {},
            "seeds": {},
            "inventories": inventories or [{} for _ in range(1 + len(hands or []))],
        },
        "market": {
            "prices": {"WHEAT": 50},
            "inventory": {"WHEAT": 10000},
        },
    }


def test_animal_maintenance_runs_only_if_a_future_harvest_still_fits():
    assert animal_maintenance_can_still_pay("GOOSE", 28, 28, 29)
    assert not animal_maintenance_can_still_pay("GOOSE", 29, 29, 29)
    assert not animal_maintenance_can_still_pay("SHEEP", 27, 28, 29)


def test_final_day_animal_only_monetizes_ready_output_and_fertilizer():
    tiles = _empty_board()
    tiles[0][0] = {
        "animal": "GOOSE",
        "placed_day": 0,
        "yield_units": 2,
        "fertilizer_available": True,
        "fed_today": False,
        "cared_today": False,
    }
    obs = _obs(29, tiles)
    tasks = build_tasks(obs, {(0, 0): ("GOOSE", False)})
    tile_tasks = [task for task in tasks if task.position == (0, 0)]
    assert len(tile_tasks) == 1
    assert tile_tasks[0].actions == [["HARVEST"], ["COLLECT_FERTILIZER"]]
    assert tile_tasks[0].must_liquidate
    assert all(action[0] not in {"FEED", "CARE", "PLACE"} for action in tile_tasks[0].actions)


def test_final_day_harvests_partial_crop_without_extra_water_or_replant():
    tiles = _empty_board()
    tiles[0][0] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 28,
        "yield_units": 1,
        "watered_today": False,
    }
    obs = _obs(29, tiles)
    tasks = build_tasks(obs, {(0, 0): ("WHEAT", False)})
    tile_tasks = [task for task in tasks if task.position == (0, 0)]
    assert len(tile_tasks) == 1
    assert tile_tasks[0].actions == [["HARVEST"]]
    assert tile_tasks[0].must_liquidate


def test_final_day_carried_inventory_is_pinned_to_its_worker_and_dropped():
    tiles = _empty_board()
    obs = _obs(
        29,
        tiles,
        inventories=[{}, {"CARROT": 3}],
        hands=[[9, 9]],
    )
    tasks = build_tasks(obs, {})
    assert len(tasks) == 1
    assert tasks[0].pinned_worker == 1
    assert tasks[0].must_liquidate

    plans, unassigned = build_queues(
        tasks,
        farmer_start=(4, 4),
        hand_count=1,
        hand_starts=((9, 9),),
    )
    assert not unassigned
    assert plans[0].queue == []
    assert plans[1].queue[-1] == ["DROP"]


def test_final_day_sell_order_includes_carried_stock_for_same_turn_drop():
    obs = {
        "day": 29,
        "player": 0,
        "_planning_end_day": 29,
        "farms": [{"tiles": [], "money": 0}],
        "private": {
            "shed": {"CARROT": 1},
            "inventories": [{"CARROT": 2}],
        },
    }
    assert sell_orders(obs, {}) == [["SELL", "CARROT", 3]]
