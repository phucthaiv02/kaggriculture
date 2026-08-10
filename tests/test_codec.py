from __future__ import annotations

import numpy as np

from kaggriculture_agent.codec import (
    encode_action,
    encode_observation,
    legal_market_ops,
    legal_unit_op_matrix,
    tile_category,
)
from kaggriculture_agent.constants import (
    BOARD_CHANNELS,
    GLOBAL_FEATURES,
    MARKET_OP_TO_ID,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    UNIT_FEATURES,
    UNIT_OP_TO_ID,
)


def observation() -> dict:
    empty = [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)]
    farms = [
        {
            "money": 3000,
            "tiles": [[cell for cell in row] for row in empty],
            "farmer": [4, 4],
            "hands": [],
            "unlocked_quadrants": ["NW"],
            "hires_today": 0,
        }
        for _ in range(2)
    ]
    return {
        "player": 0,
        "day": 0,
        "hour": 0,
        "farms": farms,
        "private": {
            "shed": {item: 0 for item in ("WHEAT", "COW", "FERTILIZER")},
            "seeds": {"WHEAT": 2, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 0, "MELON": 0},
            "inventories": [{}],
        },
        "market": {
            "prices": {item: 25 for item in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")},
            "inventory": {item: 10000 for item in ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")},
        },
        "town": {"unlocked_shops": []},
    }


def test_observation_shapes() -> None:
    encoded = encode_observation(observation())
    assert encoded["board"].shape == (BOARD_CHANNELS, 10, 10)
    assert encoded["global"].shape == (GLOBAL_FEATURES,)
    assert encoded["units"].shape == (MAX_UNITS, UNIT_FEATURES)
    assert encoded["unit_mask"].sum() == 1


def test_legality_and_action_targets() -> None:
    obs = observation()
    unit_legal = legal_unit_op_matrix(obs)
    assert unit_legal[0, UNIT_OP_TO_ID["PLANT"]]
    assert unit_legal[0, UNIT_OP_TO_ID["NORTH"]]
    market_legal = legal_market_ops(obs)
    assert market_legal[MARKET_OP_TO_ID["BUY_SEED"]]
    targets = encode_action(
        {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": [["BUY_SEED", "WHEAT", 5]]},
        1,
    )
    assert targets["unit_op"][0] == UNIT_OP_TO_ID["PLANT"]
    assert targets["market_op"].shape == (MAX_MARKET_ORDERS,)
    assert np.count_nonzero(targets["unit_item_mask"]) == 1


def test_animal_tiles_are_not_encoded_as_empty_structures() -> None:
    assert tile_category({"kind": "PASTURE", "animal": "COW"}) != tile_category({"kind": "PASTURE"})
