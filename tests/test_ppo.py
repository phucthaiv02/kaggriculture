from __future__ import annotations

import torch

from kaggriculture_agent.ppo import generalized_advantage_estimate, ppo_loss


def test_gae_propagates_terminal_reward_backwards() -> None:
    reward = torch.tensor([0.0, 0.0, 1.0])
    value = torch.zeros(3)
    done = torch.tensor([False, False, True])
    advantage, returns = generalized_advantage_estimate(reward, value, done, 1.0, 1.0)
    assert torch.equal(advantage, torch.ones(3))
    assert torch.equal(returns, torch.ones(3))


def _ppo_batch(batch_size: int = 2) -> tuple[dict, dict, dict]:
    slots = {"unit": 2, "market": 2}
    vocabs = {"op": 3, "item": 4, "quantity": 5}
    outputs = {"value": torch.zeros(batch_size, requires_grad=True)}
    reference = {"value": torch.zeros(batch_size)}
    batch = {
        "old_log_probability": torch.zeros(batch_size),
        "advantage": torch.tensor([1.0, -1.0]),
        "return": torch.zeros(batch_size),
        "policy_temperature": torch.ones(batch_size),
    }
    for prefix, slot_count in slots.items():
        choices = torch.zeros(batch_size, slot_count, 3, dtype=torch.long)
        active = torch.zeros(batch_size, slot_count, 3, dtype=torch.bool)
        active[:, 0, 0] = True
        batch[f"{prefix}_choices"] = choices
        batch[f"{prefix}_component_active"] = active
        for component, vocab in vocabs.items():
            logits = torch.zeros(batch_size, slot_count, vocab, requires_grad=True)
            outputs[f"{prefix}_{component}"] = logits
            reference[f"{prefix}_{component}"] = torch.zeros_like(logits)
            legal = torch.zeros(batch_size, slot_count, vocab, dtype=torch.bool)
            legal[:, 0] = True
            batch[f"{prefix}_{component}_legal"] = legal
    # Two active uniform ternary operation choices: log probability = -2*log(3).
    batch["old_log_probability"].fill_(-2.0 * torch.log(torch.tensor(3.0)))
    return outputs, reference, batch


def test_ppo_loss_is_finite_and_differentiable() -> None:
    outputs, reference, batch = _ppo_batch()
    loss, metrics = ppo_loss(outputs, reference, batch, 0.2, 0.5, 0.01, 0.02)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    loss.backward()
