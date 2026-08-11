from __future__ import annotations

from collections import Counter

import pytest

from kaggriculture_agent.heuristic import (
    ANIMAL_LAYOUT,
    ANIMAL_STRUCTURE,
    CROP_HARVEST_DAY,
    CROP_LAYOUT,
    HeuristicAgent,
    NE_CROP_POSITIONS,
    ONGOING_CROPS,
    _animal_task,
    _crop_task,
    _route_pickup_action,
    _route_steps,
    estimate_asset_steps,
    generate_unit_clusters,
    market_actions,
    choose_replacement_crop,
    project_crop_profit,
    required_unit_count,
    unit_actions,
)


PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
ANIMALS = ("GOOSE", "COW", "SHEEP")


def observation() -> dict:
    tiles = [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)]
    farm = {
        "money": 3000,
        "tiles": tiles,
        "farmer": [4, 4],
        "hands": [],
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }
    return {
        "player": 0,
        "day": 0,
        "hour": 0,
        "farms": [farm, {**farm, "tiles": [row[:] for row in tiles]}],
        "private": {
            "shed": {item: 0 for item in (*PRODUCTS, *ANIMALS)},
            "seeds": {crop: 0 for crop in CROP_LAYOUT.values()},
            "inventories": [{}],
        },
        "market": {"prices": {item: 1 for item in PRODUCTS}, "inventory": {}},
        "town": {"unlocked_shops": []},
    }


def test_initial_market_plan_buys_exact_strategy_quantities_and_queues_next_land() -> None:
    orders = market_actions(observation())
    assert ["BUY_SEED", "WHEAT", 9] in orders
    assert ["BUY_SEED", "MELON", 12] in orders
    assert ["BUY_ANIMAL", "COW", 2] in orders
    assert ["BUY_ANIMAL", "SHEEP", 2] in orders
    assert sum(order[0] == "HIRE" for order in orders) == 4
    assert orders[-1] == ["BUY_LAND"]
    assert len(orders) <= 10


def test_products_are_sold_before_buying_next_land() -> None:
    obs = observation()
    obs["hour"] = 20
    obs["private"]["shed"].update({"WHEAT": 8, "MELON": 10, "COW": 2, "SHEEP": 2})
    obs["private"]["seeds"].update({"WHEAT": 9, "MELON": 12})
    orders = market_actions(obs, clusters=[[]])
    assert orders == [["SELL", "MELON", 10], ["BUY_LAND"]]


def test_strategy_fills_nw_with_requested_asset_counts() -> None:
    positions = {*CROP_LAYOUT, *ANIMAL_LAYOUT}
    assert len(positions) == 25
    assert all(0 <= x < 5 and 0 <= y < 5 for x, y in positions)
    assert Counter(CROP_LAYOUT.values()) == {"MELON": 12, "WHEAT": 9}
    assert Counter(ANIMAL_LAYOUT.values()) == {"COW": 2, "SHEEP": 2}
    assert CROP_LAYOUT[(4, 4)] == "WHEAT"


def test_unlocked_ne_becomes_one_complete_profit_ranked_crop_cohort() -> None:
    obs = observation()
    obs.update({"day": 5, "hour": 1, "step": 121})
    obs["farms"][0]["money"] = 100_000
    obs["farms"][0]["unlocked_quadrants"] = ["NW", "NE"]
    for y in range(5):
        for x in range(5, 10):
            obs["farms"][0]["tiles"][y][x] = None
    obs["market"]["prices"].update(
        {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250}
    )

    agent = HeuristicAgent()
    action = agent(obs)
    selected = agent.expansion_choices[5][0]
    assert len(NE_CROP_POSITIONS) == 25
    assert {agent._crop_layout[position] for position in NE_CROP_POSITIONS} == {selected}
    assert any(order[:2] == ["BUY_SEED", selected] and order[2] >= 25 for order in action["market"])


def test_clusters_are_generated_by_capacity_and_geography() -> None:
    clusters = generate_unit_clusters(observation())
    assert len(clusters) == 5
    assert sum(map(len, clusters)) == 25
    assert all(_route_steps(observation(), cluster) <= 24 for cluster in clusters)
    assert all(
        _route_steps(observation(), [*left, *right]) > 24
        for left_index, left in enumerate(clusters)
        for right in clusters[left_index + 1 :]
    )


def test_fully_serviced_day_uses_only_main_farmer() -> None:
    obs = observation()
    for position, crop in CROP_LAYOUT.items():
        x, y = position
        obs["farms"][0]["tiles"][y][x] = {
            "kind": "PLANT",
            "crop": crop,
            "planted_day": 0,
            "watered_today": True,
            "yield_units": 0,
        }
    for position, animal in ANIMAL_LAYOUT.items():
        x, y = position
        obs["farms"][0]["tiles"][y][x] = {
            "kind": ANIMAL_STRUCTURE[animal],
            "animal": animal,
            "fed_today": True,
            "cared_today": True,
            "fertilizer_available": False,
            "yield_units": 0,
        }
    clusters = generate_unit_clusters(obs)
    assert len(clusters) == 1
    assert required_unit_count(obs, clusters) == 1
    assert ["HIRE"] not in market_actions(obs, clusters)


def test_crop_workload_changes_with_lifecycle_phase() -> None:
    obs = observation()
    wheat = ("crop", (0, 4), "WHEAT")
    assert estimate_asset_steps(obs, wheat) == 2  # PLANT + WATER.

    obs["day"] = 4
    obs["farms"][0]["tiles"][4][0] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": False,
        "yield_units": 4,
    }
    # WATER + HARVEST + replacement PLANT + replacement WATER.
    assert estimate_asset_steps(obs, wheat) == 4


def test_crop_harvest_is_urgent_exactly_at_max_yield_day() -> None:
    melon = {
        "kind": "PLANT",
        "crop": "MELON",
        "planted_day": 2,
        "watered_today": True,
        "yield_units": 6,
    }
    assert _crop_task((0, 0), "MELON", melon, 11, {}) is None
    task = _crop_task((0, 0), "MELON", melon, 12, {})
    assert task is not None
    assert (task.operation, task.priority, task.deadline) == ("HARVEST", 0, 23)

    tomato = {**melon, "crop": "TOMATO", "watered_today": False, "yield_units": 1}
    ongoing_task = _crop_task((0, 0), "TOMATO", tomato, 10, {})
    assert ongoing_task is not None
    assert (ongoing_task.operation, ongoing_task.priority) == ("WATER", 0)
    assert _crop_task((0, 0), "TOMATO", {**tomato, "watered_today": True}, 10, {}).operation == "HARVEST"


def test_crop_profit_projection_respects_remaining_season_window() -> None:
    assert project_crop_profit("WHEAT", 27, 30, 25, slots=9).total_profit == 0
    obs = observation()
    obs["market"]["prices"].update({"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250})
    selected, projections = choose_replacement_crop(obs, plant_day=4, slots=9)
    assert selected == "MELON"
    assert projections["MELON"].profit_per_day > projections["CARROT"].profit_per_day
    assert choose_replacement_crop(obs, plant_day=29, slots=9)[0] is None


def test_harvest_day_selects_replacement_and_buys_seed_at_hour_zero() -> None:
    obs = observation()
    obs.update({"day": 4, "hour": 0, "step": 96})
    obs["market"]["prices"].update({"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120, "MELON": 250})
    for position, crop in CROP_LAYOUT.items():
        x, y = position
        obs["farms"][0]["tiles"][y][x] = {
            "kind": "PLANT",
            "crop": crop,
            "planted_day": 0,
            "watered_today": False,
            "yield_units": 6 if crop == "MELON" else 4,
        }
    agent = HeuristicAgent()
    action = agent(obs)
    assert agent.crop_choices[4][0] == "MELON"
    assert ["BUY_SEED", "MELON", 9] in action["market"]

    obs["farms"][0]["money"] = 417
    obs["private"]["shed"]["WHEAT"] = 8
    constrained_agent = HeuristicAgent()
    constrained_action = constrained_agent(obs)
    assert constrained_agent.crop_choices[4][0] == "CARROT"
    assert ["BUY_SEED", "CARROT", 9] in constrained_action["market"]


def test_no_replant_when_even_fastest_crop_cannot_be_harvested() -> None:
    obs = observation()
    obs.update({"day": 27, "hour": 0, "step": 648})
    for position, crop in CROP_LAYOUT.items():
        x, y = position
        obs["farms"][0]["tiles"][y][x] = {
            "kind": "PLANT",
            "crop": crop,
            "planted_day": 26,
            "watered_today": True,
            "yield_units": 0,
        }
    target = (0, 4)
    obs["farms"][0]["tiles"][target[1]][target[0]] = {
        "kind": "PLANT",
        "crop": "CARROT",
        "planted_day": 24,
        "watered_today": False,
        "yield_units": 3,
    }

    agent = HeuristicAgent()
    action = agent(obs)
    assert agent.crop_choices[27][0] is None
    assert agent._crop_layout[target] is None
    assert all(order[0] != "BUY_SEED" for order in action["market"])


def test_animal_workload_includes_place_and_harvest() -> None:
    obs = observation()
    cow = ("animal", (3, 3), "COW")
    # PLACE is a required game action in addition to the requested PICKUP+BUILD.
    assert estimate_asset_steps(obs, cow) == 4
    obs["farms"][0]["tiles"][3][3] = {
        "kind": "PASTURE",
        "animal": "COW",
        "fed_today": False,
        "yield_units": 1,
    }
    assert estimate_asset_steps(obs, cow) == 3  # FEED + HARVEST + CARE.


def test_fertilizer_is_a_daily_deadline_before_optional_care() -> None:
    cow = {
        "kind": "PASTURE",
        "animal": "COW",
        "placed_day": 3,
        "fed_today": True,
        "cared_today": False,
        "fertilizer_available": True,
        "yield_units": 0,
    }
    task = _animal_task((3, 3), "COW", cow)
    assert task is not None
    assert (task.operation, task.priority, task.deadline) == ("COLLECT_FERTILIZER", 0, 23)


def test_animal_route_loads_duplicate_animals_in_one_pickup() -> None:
    obs = observation()
    cluster = [("animal", (3, 4), "COW"), ("animal", (3, 3), "COW")]
    shed = {"COW": 2, "WHEAT": 2}
    assert _route_pickup_action(obs, cluster, (4, 4), {}, shed) == ["PICKUP", "COW", 2]
    assert _route_pickup_action(obs, cluster, (4, 4), {"COW": 2}, shed) == [
        "PICKUP",
        "WHEAT",
        2,
    ]


def test_crop_owner_immediately_plants_and_waters_an_empty_target() -> None:
    obs = observation()
    obs["farms"][0]["farmer"] = [0, 4]
    obs["private"]["seeds"]["WHEAT"] = 1
    wheat_route = [[("crop", (0, 4), "WHEAT")]]
    farmer, _ = unit_actions(obs, wheat_route)
    assert farmer == ["PLANT", "WHEAT"]

    obs["private"]["seeds"]["WHEAT"] = 0
    obs["farms"][0]["tiles"][4][0] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": False,
        "consecutive_unwatered": 1,
        "yield_units": 1,
    }
    farmer, _ = unit_actions(obs, wheat_route)
    assert farmer == ["WATER"]


def test_harvest_inventory_returns_and_drops_before_optional_route_work() -> None:
    obs = observation()
    route = [[("crop", (0, 4), "WHEAT")]]
    obs["farms"][0]["tiles"][4][0] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 1,
        "watered_today": True,
        "yield_units": 0,
    }
    obs["farms"][0]["farmer"] = [0, 4]
    obs["private"]["inventories"] = [{"MELON": 6}]
    farmer, _ = unit_actions(obs, route, prioritize_drop=True)
    assert farmer == ["EAST"]

    obs["farms"][0]["farmer"] = [4, 4]
    farmer, _ = unit_actions(obs, route, prioritize_drop=True)
    assert farmer == ["DROP"]

    # A survival/deadline WATER still precedes returning the harvest.
    obs["farms"][0]["farmer"] = [0, 4]
    obs["farms"][0]["tiles"][4][0]["watered_today"] = False
    farmer, _ = unit_actions(obs, route, prioritize_drop=True)
    assert farmer == ["WATER"]


def test_fast_drop_preempts_optional_care_only_when_enabled() -> None:
    obs = observation()
    obs["farms"][0]["farmer"] = [3, 3]
    obs["farms"][0]["tiles"][3][3] = {
        "kind": "PASTURE",
        "animal": "COW",
        "fed_today": True,
        "cared_today": False,
        "fertilizer_available": False,
        "yield_units": 0,
    }
    obs["private"]["inventories"] = [{"MILK": 1}]
    route = [[("animal", (3, 3), "COW")]]

    farmer, _ = unit_actions(obs, route, prioritize_drop=False)
    assert farmer == ["CARE"]
    farmer, _ = unit_actions(obs, route, prioritize_drop=True)
    assert farmer == ["EAST"]


def test_initial_action_has_complete_schema() -> None:
    action = HeuristicAgent()(observation())
    assert set(action) == {"farmer", "hands", "market"}
    assert action["farmer"] == ["PASS"]
    assert action["hands"] == []


def test_two_units_plant_and_water_the_same_tile_in_one_turn() -> None:
    obs = observation()
    obs["private"]["seeds"]["WHEAT"] = 1
    obs["farms"][0]["farmer"] = [0, 4]
    obs["farms"][0]["hands"] = [[0, 4]]
    obs["private"]["inventories"] = [{}, {}]
    action = HeuristicAgent()(obs)
    assert [action["farmer"], *action["hands"]] == [
        ["PLANT", "WHEAT"],
        ["WATER"],
    ]


def test_four_units_replace_a_harvested_crop_in_one_turn() -> None:
    obs = observation()
    obs["day"] = 4
    obs["market"]["prices"]["WHEAT"] = 25
    obs["private"]["seeds"]["WHEAT"] = 1
    obs["farms"][0]["farmer"] = [0, 4]
    obs["farms"][0]["hands"] = [[0, 4], [0, 4], [0, 4]]
    obs["private"]["inventories"] = [{}, {}, {}, {}]
    obs["farms"][0]["tiles"][4][0] = {
        "kind": "PLANT",
        "crop": "WHEAT",
        "planted_day": 0,
        "watered_today": False,
        "yield_units": 4,
    }
    action = HeuristicAgent()(obs)
    assert [action["farmer"], *action["hands"]] == [
        ["WATER"],
        ["HARVEST"],
        ["PLANT", "WHEAT"],
        ["WATER"],
    ]


def test_three_hands_feed_care_and_collect_on_the_same_turn() -> None:
    obs = observation()
    obs["farms"][0]["hands"] = [[3, 3], [3, 3], [3, 3]]
    obs["private"]["inventories"] = [{}, {"WHEAT": 1}, {}, {}]
    obs["farms"][0]["tiles"][3][3] = {
        "kind": "PASTURE",
        "animal": "COW",
        "fed_today": False,
        "cared_today": False,
        "fertilizer_available": True,
        "yield_units": 0,
    }
    action = HeuristicAgent()(obs)
    assert action["hands"] == [["FEED"], ["COLLECT_FERTILIZER"], ["CARE"]]


def test_full_season_creates_every_crop_and_animal_and_expands_ne() -> None:
    kaggle = pytest.importorskip("kaggle_environments")
    env = kaggle.make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "weedSpawnChance": 0, "seed": 42},
        debug=True,
    )
    agent = HeuristicAgent()
    env.run([agent, "pass"])
    seen_crops: set[str] = set()
    seen_animals: set[str] = set()
    harvested_assets: set[str] = set()
    fertilizer_animals: set[str] = set()
    required_fertilizer_days: set[tuple[int, tuple[int, int], str]] = set()
    collected_fertilizer_days: set[tuple[int, tuple[int, int], str]] = set()
    peak_assets: Counter[str] = Counter()
    peak_crop_tiles = 0
    crop_tiles_at_day_start: dict[int, int] = {}
    for step_index, step in enumerate(env.steps):
        obs = step[0].observation
        if not getattr(obs, "farms", None):
            continue
        current_assets: Counter[str] = Counter()
        for row in obs.farms[0]["tiles"]:
            for tile in row:
                if isinstance(tile, dict):
                    if tile.get("crop"):
                        seen_crops.add(tile["crop"])
                        current_assets[tile["crop"]] += 1
                    if tile.get("animal"):
                        seen_animals.add(tile["animal"])
                        current_assets[tile["animal"]] += 1
        peak_assets |= current_assets
        peak_crop_tiles = max(peak_crop_tiles, sum(current_assets[crop] for crop in CROP_HARVEST_DAY))
        if obs.hour == 0:
            crop_tiles_at_day_start[obs.day] = sum(
                current_assets[crop] for crop in CROP_HARVEST_DAY
            )
        if obs.hour == 0:
            for position, animal in ANIMAL_LAYOUT.items():
                tile = obs.farms[0]["tiles"][position[1]][position[0]]
                if (
                    isinstance(tile, dict)
                    and tile.get("animal") == animal
                    and tile.get("fertilizer_available", False)
                ):
                    required_fertilizer_days.add((obs.day, position, animal))

        if step_index == 0:
            continue
        source = env.steps[step_index - 1][0].observation
        action = step[0].action
        if isinstance(action, dict):
            commands = [action.get("farmer", ["PASS"]), *(action.get("hands", []) or [])]
            positions = [source.farms[0]["farmer"], *(source.farms[0].get("hands", []) or [])]
            for unit_index, command in enumerate(commands):
                if not command or unit_index >= len(positions):
                    continue
                x, y = positions[unit_index]
                tile = source.farms[0]["tiles"][y][x]
                if not isinstance(tile, dict):
                    asset = CROP_LAYOUT.get((x, y)) or ANIMAL_LAYOUT.get((x, y))
                else:
                    asset = tile.get("crop") or tile.get("animal")
                if command[0] == "HARVEST" and asset:
                    harvested_assets.add(asset)
                    if isinstance(tile, dict) and tile.get("crop"):
                        age = source.day - int(tile["planted_day"])
                        if tile["crop"] in ONGOING_CROPS:
                            assert tile.get("watered_today", False)
                            assert int(tile.get("yield_units", 0)) > 0
                        else:
                            assert age == CROP_HARVEST_DAY[tile["crop"]]
                if command[0] == "COLLECT_FERTILIZER" and isinstance(tile, dict) and tile.get("animal"):
                    fertilizer_animals.add(tile["animal"])
                    collected_fertilizer_days.add((source.day, (x, y), tile["animal"]))
    final_farm = env.steps[-1][0].observation.farms[0]
    assert seen_crops >= set(CROP_LAYOUT.values())
    assert seen_animals == set(ANIMAL_LAYOUT.values())
    assert harvested_assets >= {*CROP_LAYOUT.values(), *ANIMAL_LAYOUT.values()}
    assert fertilizer_animals == set(ANIMAL_LAYOUT.values())
    assert required_fertilizer_days <= collected_fertilizer_days
    assert peak_crop_tiles == len(CROP_LAYOUT) + len(NE_CROP_POSITIONS)
    expansion_day = min(agent.expansion_choices)
    assert all(crop_tiles_at_day_start[day] == len(CROP_LAYOUT) for day in range(1, expansion_day + 1))
    assert all(
        crop_tiles_at_day_start[day] == len(CROP_LAYOUT) + len(NE_CROP_POSITIONS)
        for day in range(expansion_day + 1, 28)
    )
    assert peak_assets["COW"] == peak_assets["SHEEP"] == 2
    assert max(len(step[0].observation.farms[0].get("hands", [])) for step in env.steps[1:]) <= 16
    assert final_farm["unlocked_quadrants"] == ["NW", "NE"]
