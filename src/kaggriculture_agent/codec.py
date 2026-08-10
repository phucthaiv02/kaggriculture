"""Lossless-enough state features, action targets, and legality masks."""

from __future__ import annotations

from typing import Any

import numpy as np

from .constants import (
    ANIMALS,
    BOARD_CHANNELS,
    BOARD_CHANNELS_PER_PLAYER,
    BOARD_SIZE,
    CROPS,
    CROP_RULES,
    GLOBAL_FEATURES,
    ITEM_TO_ID,
    MARKET_OPS,
    MARKET_OP_TO_ID,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    MOVE_DELTA,
    PRODUCTS,
    QUANTITY_BUCKETS,
    SHED_ACCESS,
    SHED_ITEMS,
    SHOPS,
    UNIT_FEATURES,
    UNIT_OPS,
    UNIT_OP_TO_ID,
)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _byte(value: float) -> np.uint8:
    return np.uint8(round(np.clip(value, 0.0, 1.0) * 255.0))


def tile_category(tile: Any) -> int:
    if tile is None:
        return 0
    if tile == "LOCKED":
        return 1
    if not isinstance(tile, dict):
        return 0
    if tile.get("kind") == "WEED":
        return 2
    if tile.get("kind") == "PLANT" and tile.get("crop") in CROPS:
        return 3 + CROPS.index(tile["crop"])
    if tile.get("animal") in ANIMALS:
        return 10 + ANIMALS.index(tile["animal"])
    if tile.get("kind") == "COOP":
        return 8
    if tile.get("kind") == "PASTURE":
        return 9
    return 0


def _encode_farm(board: np.ndarray, offset: int, farm: dict[str, Any], day: int) -> None:
    farmer = tuple(farm.get("farmer", (-1, -1)))
    hand_counts: dict[tuple[int, int], int] = {}
    for position in farm.get("hands", []):
        key = tuple(position)
        hand_counts[key] = hand_counts.get(key, 0) + 1
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            category = tile_category(tile)
            board[offset + category, y, x] = 255
            if isinstance(tile, dict):
                board[offset + 13, y, x] = 255 if tile.get("watered_today") or tile.get("fed_today") else 0
                board[offset + 14, y, x] = 255 if tile.get("cared_today") else 0
                board[offset + 15, y, x] = 255 if tile.get("fertilizer_available") else 0
                board[offset + 16, y, x] = 255 if int(tile.get("fertilized_until_day", -1)) >= day else 0
                board[offset + 17, y, x] = _byte(float(tile.get("yield_units", 0)) / 10.0)
                placed = int(tile.get("planted_day", tile.get("placed_day", day)))
                board[offset + 18, y, x] = _byte((day - placed) / 30.0)
                unmet = int(tile.get("consecutive_unwatered", tile.get("consecutive_unfed", 0)))
                board[offset + 19, y, x] = _byte(unmet / 2.0)
            if (x, y) == farmer:
                board[offset + 20, y, x] = 255
            if (x, y) in hand_counts:
                board[offset + 21, y, x] = _byte(hand_counts[(x, y)] / 8.0)


def encode_observation(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    player = int(_get(obs, "player", 0))
    opponent = 1 - player
    farms = _get(obs, "farms", [])
    mine, other = farms[player], farms[opponent]
    day, hour = int(_get(obs, "day", 0)), int(_get(obs, "hour", 0))
    private = _get(obs, "private", {}) or {}

    board = np.zeros((BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
    _encode_farm(board, 0, mine, day)
    _encode_farm(board, BOARD_CHANNELS_PER_PLAYER, other, day)

    features: list[float] = [
        day / 30.0,
        hour / 24.0,
        float(mine.get("money", 0)) / 250_000.0,
        float(other.get("money", 0)) / 250_000.0,
        float(mine.get("hires_today", 0)) / 32.0,
        float(other.get("hires_today", 0)) / 32.0,
    ]
    for farm in (mine, other):
        unlocked = set(farm.get("unlocked_quadrants", []))
        features.extend(float(name in unlocked) for name in ("NW", "NE", "SW", "SE"))
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    inventories = private.get("inventories", []) or []
    carried = {item: 0 for item in SHED_ITEMS}
    for inventory in inventories:
        for item in SHED_ITEMS:
            carried[item] += int((inventory or {}).get(item, 0))
    features.extend(float(seeds.get(item, 0)) / 100.0 for item in CROPS)
    features.extend(float(shed.get(item, 0)) / 100.0 for item in SHED_ITEMS)
    features.extend(float(carried[item]) / 100.0 for item in SHED_ITEMS)
    market = _get(obs, "market", {}) or {}
    prices = market.get("prices", {}) or {}
    market_inventory = market.get("inventory", {}) or {}
    for item in PRODUCTS:
        features.append(float(prices.get(item, 0)) / 500.0)
        features.append((float(market_inventory.get(item, 10_000)) - 10_000.0) / 5_000.0)
    unlocked_shops = _get(_get(obs, "town", {}) or {}, "unlocked_shops", []) or []
    features.extend(unlocked_shops.count(shop) / 8.0 for shop in SHOPS)
    features.append((1 + len(mine.get("hands", []))) / MAX_UNITS)
    global_features = np.asarray(features, dtype=np.float32)
    if global_features.shape != (GLOBAL_FEATURES,):
        raise ValueError(f"global feature mismatch: {global_features.shape} != {(GLOBAL_FEATURES,)}")

    positions = [mine["farmer"], *mine.get("hands", [])]
    units = np.zeros((MAX_UNITS, UNIT_FEATURES), dtype=np.float32)
    unit_mask = np.zeros(MAX_UNITS, dtype=np.bool_)
    for index, position in enumerate(positions[:MAX_UNITS]):
        x, y = map(int, position)
        tile = mine["tiles"][y][x]
        inventory = inventories[index] if index < len(inventories) else {}
        category = tile_category(tile)
        row: list[float] = [
            float(index == 0),
            index / MAX_UNITS,
            x / (BOARD_SIZE - 1),
            y / (BOARD_SIZE - 1),
            float((x, y) in SHED_ACCESS),
        ]
        row.extend(float(category == value) for value in range(13))
        if isinstance(tile, dict):
            placed = int(tile.get("planted_day", tile.get("placed_day", day)))
            unmet = int(tile.get("consecutive_unwatered", tile.get("consecutive_unfed", 0)))
            row.extend(
                (
                    float(bool(tile.get("watered_today") or tile.get("fed_today"))),
                    float(bool(tile.get("cared_today"))),
                    float(bool(tile.get("fertilizer_available"))),
                    min(1.0, float(tile.get("yield_units", 0)) / 10.0),
                    min(1.0, max(0.0, (day - placed) / 30.0)),
                    min(1.0, unmet / 2.0),
                )
            )
        else:
            row.extend((0.0,) * 6)
        row.extend(float((inventory or {}).get(item, 0)) / 100.0 for item in SHED_ITEMS)
        units[index] = np.asarray(row, dtype=np.float32)
        unit_mask[index] = True
    return {"board": board, "global": global_features, "units": units, "unit_mask": unit_mask}


def quantity_id(value: Any) -> int:
    try:
        number = max(0, int(value))
    except (TypeError, ValueError):
        return 0
    return min(range(len(QUANTITY_BUCKETS)), key=lambda index: abs(QUANTITY_BUCKETS[index] - number))


def encode_action(action: dict[str, Any], unit_count: int) -> dict[str, np.ndarray]:
    unit_op = np.zeros(MAX_UNITS, dtype=np.int64)
    unit_item = np.zeros(MAX_UNITS, dtype=np.int64)
    unit_quantity = np.zeros(MAX_UNITS, dtype=np.int64)
    unit_item_mask = np.zeros(MAX_UNITS, dtype=np.bool_)
    unit_quantity_mask = np.zeros(MAX_UNITS, dtype=np.bool_)
    commands = [action.get("farmer") or ["PASS"], *(action.get("hands") or [])]
    for index, command in enumerate(commands[: min(unit_count, MAX_UNITS)]):
        command = command or ["PASS"]
        op = str(command[0])
        unit_op[index] = UNIT_OP_TO_ID.get(op, 0)
        if op in ("PICKUP", "PLACE", "PLANT") and len(command) > 1:
            unit_item[index] = ITEM_TO_ID.get(str(command[1]), 0)
            unit_item_mask[index] = True
        if op in ("PICKUP", "PLACE") and len(command) > 2:
            unit_quantity[index] = quantity_id(command[2])
            unit_quantity_mask[index] = True

    market_op = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)
    market_item = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)
    market_quantity = np.zeros(MAX_MARKET_ORDERS, dtype=np.int64)
    market_item_mask = np.zeros(MAX_MARKET_ORDERS, dtype=np.bool_)
    market_quantity_mask = np.zeros(MAX_MARKET_ORDERS, dtype=np.bool_)
    for index, order in enumerate((action.get("market") or [])[:MAX_MARKET_ORDERS]):
        if not order:
            continue
        op = str(order[0])
        market_op[index] = MARKET_OP_TO_ID.get(op, 0)
        if op in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL") and len(order) > 1:
            market_item[index] = ITEM_TO_ID.get(str(order[1]), 0)
            market_item_mask[index] = True
        if op in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL") and len(order) > 2:
            market_quantity[index] = quantity_id(order[2])
            market_quantity_mask[index] = True
    return {
        "unit_op": unit_op,
        "unit_item": unit_item,
        "unit_quantity": unit_quantity,
        "unit_item_mask": unit_item_mask,
        "unit_quantity_mask": unit_quantity_mask,
        "market_op": market_op,
        "market_item": market_item,
        "market_quantity": market_quantity,
        "market_item_mask": market_item_mask,
        "market_quantity_mask": market_quantity_mask,
    }


def legal_unit_ops(obs: dict[str, Any], unit_index: int) -> np.ndarray:
    mask = np.zeros(len(UNIT_OPS), dtype=np.bool_)
    mask[UNIT_OP_TO_ID["PASS"]] = True
    player = int(_get(obs, "player", 0))
    farm = _get(obs, "farms", [])[player]
    private = _get(obs, "private", {}) or {}
    positions = [farm["farmer"], *farm.get("hands", [])]
    if unit_index >= len(positions):
        return mask
    x, y = map(int, positions[unit_index])
    tile = farm["tiles"][y][x]
    inventories = private.get("inventories", []) or []
    inventory = inventories[unit_index] if unit_index < len(inventories) else {}
    for op, (dx, dy) in MOVE_DELTA.items():
        if 0 <= x + dx < BOARD_SIZE and 0 <= y + dy < BOARD_SIZE:
            mask[UNIT_OP_TO_ID[op]] = True
    if (x, y) in SHED_ACCESS:
        if any(int(value) > 0 for value in private.get("shed", {}).values()):
            mask[UNIT_OP_TO_ID["PICKUP"]] = True
        if inventory:
            mask[UNIT_OP_TO_ID["DROP"]] = True
            mask[UNIT_OP_TO_ID["PLACE"]] = True
    if tile == "LOCKED":
        return mask
    if tile is None:
        if any(int(private.get("seeds", {}).get(crop, 0)) > 0 for crop in CROPS):
            mask[UNIT_OP_TO_ID["PLANT"]] = True
        mask[UNIT_OP_TO_ID["BUILD_COOP"]] = True
        mask[UNIT_OP_TO_ID["BUILD_PASTURE"]] = True
        return mask
    if not isinstance(tile, dict):
        return mask
    if tile.get("kind") == "WEED":
        mask[UNIT_OP_TO_ID["DIG"]] = True
    elif tile.get("kind") == "PLANT":
        if not tile.get("watered_today"):
            mask[UNIT_OP_TO_ID["WATER"]] = True
        crop = tile.get("crop")
        age = int(_get(obs, "day", 0)) - int(tile.get("planted_day", 0))
        if int(tile.get("yield_units", 0)) > 0 and age >= CROP_RULES.get(crop, {}).get("first", 999):
            mask[UNIT_OP_TO_ID["HARVEST"]] = True
        if int(inventory.get("FERTILIZER", 0)) > 0:
            mask[UNIT_OP_TO_ID["FERTILIZE"]] = True
        mask[UNIT_OP_TO_ID["DIG"]] = True
    elif tile.get("animal") in ANIMALS:
        if not tile.get("fed_today") and int(inventory.get("WHEAT", 0)) > 0:
            mask[UNIT_OP_TO_ID["FEED"]] = True
        if not tile.get("cared_today"):
            mask[UNIT_OP_TO_ID["CARE"]] = True
        if tile.get("fertilizer_available"):
            mask[UNIT_OP_TO_ID["COLLECT_FERTILIZER"]] = True
        if int(tile.get("yield_units", 0)) > 0:
            mask[UNIT_OP_TO_ID["HARVEST"]] = True
    elif tile.get("kind") in ("COOP", "PASTURE"):
        structure = tile.get("kind")
        if any(int(inventory.get(animal, 0)) > 0 for animal in ANIMALS if (animal == "GOOSE") == (structure == "COOP")):
            mask[UNIT_OP_TO_ID["PLACE"]] = True
        mask[UNIT_OP_TO_ID["DIG"]] = True
    return mask


def legal_unit_op_matrix(obs: dict[str, Any]) -> np.ndarray:
    result = np.zeros((MAX_UNITS, len(UNIT_OPS)), dtype=np.bool_)
    for index in range(MAX_UNITS):
        result[index] = legal_unit_ops(obs, index)
    return result


def legal_market_ops(obs: dict[str, Any]) -> np.ndarray:
    mask = np.zeros(len(MARKET_OPS), dtype=np.bool_)
    mask[MARKET_OP_TO_ID["NONE"]] = True
    player = int(_get(obs, "player", 0))
    farm = _get(obs, "farms", [])[player]
    private = _get(obs, "private", {}) or {}
    money = float(farm.get("money", 0))
    shed = private.get("shed", {}) or {}
    if money >= 10:
        mask[MARKET_OP_TO_ID["BUY_SEED"]] = True
    if money >= 1 and sum(int(value) for value in shed.values()) < 100:
        mask[MARKET_OP_TO_ID["BUY_PRODUCT"]] = True
    if money >= 300 and sum(int(value) for value in shed.values()) < 100:
        mask[MARKET_OP_TO_ID["BUY_ANIMAL"]] = True
    if any(int(shed.get(item, 0)) > 0 for item in PRODUCTS):
        mask[MARKET_OP_TO_ID["SELL"]] = True
    if money >= 1:
        mask[MARKET_OP_TO_ID["HIRE"]] = True
    if len(farm.get("unlocked_quadrants", [])) < 4 and money >= (1000, 2000, 4000)[len(farm.get("unlocked_quadrants", [])) - 1]:
        mask[MARKET_OP_TO_ID["BUY_LAND"]] = True
    return mask
