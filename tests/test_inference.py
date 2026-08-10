from __future__ import annotations

import torch

from kaggriculture_agent.inference import DynamicAgent
from kaggriculture_agent.model import DynamicPolicy, ModelConfig
from test_codec import observation


def test_random_policy_decodes_a_well_formed_action(tmp_path) -> None:
    model = DynamicPolicy(
        ModelConfig(d_model=64, board_blocks=1, transformer_layers=1, attention_heads=4, dropout=0.0)
    )
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "architecture": "dynamic_autoregressive_v1",
            "model_config": model.config.to_dict(),
            "model": model.state_dict(),
        },
        checkpoint,
    )
    action = DynamicAgent(checkpoint)(observation())
    assert isinstance(action["farmer"], list) and action["farmer"]
    assert action["hands"] == []
    assert isinstance(action["market"], list)
    assert len(action["market"]) <= 10


def test_rollout_trace_matches_the_decoded_action_distribution(tmp_path) -> None:
    model = DynamicPolicy(
        ModelConfig(d_model=64, board_blocks=1, transformer_layers=1, attention_heads=4, dropout=0.0)
    )
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "architecture": "dynamic_autoregressive_v1",
            "model_config": model.config.to_dict(),
            "model": model.state_dict(),
        },
        checkpoint,
    )
    agent = DynamicAgent(checkpoint, temperature=1.0)
    torch.manual_seed(1234)
    for _ in range(20):
        action, trace = agent.act_with_trace(observation())
        assert action["farmer"]
        assert trace["unit_component_active"][0, 0]
        assert trace["market_component_active"][0, 0]
        assert torch.isfinite(torch.tensor(trace["old_log_probability"]))
        assert trace["policy_temperature"] == 1.0
