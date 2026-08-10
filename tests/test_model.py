from __future__ import annotations

import torch

from kaggriculture_agent.codec import encode_action, encode_observation, legal_market_ops, legal_unit_op_matrix
from kaggriculture_agent.constants import ITEMS, MARKET_OPS, MAX_MARKET_ORDERS, MAX_UNITS, QUANTITY_BUCKETS, UNIT_OPS
from kaggriculture_agent.losses import behavior_cloning_loss
from kaggriculture_agent.model import DynamicPolicy, ModelConfig
from test_codec import observation


def test_forward_and_loss_are_finite() -> None:
    encoded = encode_observation(observation())
    target = encode_action({"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}, 1)
    batch = {
        "board": torch.from_numpy(encoded["board"]).unsqueeze(0),
        "global": torch.from_numpy(encoded["global"]).unsqueeze(0),
        "units": torch.from_numpy(encoded["units"]).unsqueeze(0),
        "unit_mask": torch.from_numpy(encoded["unit_mask"]).unsqueeze(0),
        "unit_legal": torch.from_numpy(legal_unit_op_matrix(observation())).unsqueeze(0),
        "market_legal": torch.from_numpy(legal_market_ops(observation())).view(1, 1, -1).repeat(1, MAX_MARKET_ORDERS, 1),
        "sample_weight": torch.ones(1),
        "value_target": torch.zeros(1),
    }
    for key, value in target.items():
        batch[key] = torch.from_numpy(value).unsqueeze(0)
    model = DynamicPolicy(
        ModelConfig(d_model=64, board_blocks=1, transformer_layers=2, attention_heads=4, dropout=0.0)
    )
    targets = {key: batch[key].long() for key in ("unit_op", "unit_item", "unit_quantity", "market_op", "market_item", "market_quantity")}
    output = model(batch["board"], batch["global"], batch["units"], batch["unit_mask"], targets)
    assert output["unit_op"].shape == (1, MAX_UNITS, len(UNIT_OPS))
    assert output["unit_item"].shape == (1, MAX_UNITS, len(ITEMS))
    assert output["market_op"].shape == (1, MAX_MARKET_ORDERS, len(MARKET_OPS))
    assert output["market_quantity"].shape[-1] == len(QUANTITY_BUCKETS)
    loss, metrics = behavior_cloning_loss(output, batch)
    assert torch.isfinite(loss)
    for name in (
        "unit_op_accuracy",
        "unit_op_top3_accuracy",
        "unit_item_accuracy",
        "unit_quantity_accuracy",
        "unit_exact_accuracy",
        "market_op_accuracy",
        "market_op_top3_accuracy",
        "market_item_accuracy",
        "market_quantity_accuracy",
        "market_exact_accuracy",
        "value_mae",
    ):
        assert torch.isfinite(metrics[name])
    loss.backward()
