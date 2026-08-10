"""PPO utilities for autoregressive, legality-masked Kaggriculture actions."""

from __future__ import annotations

import json
import multiprocessing
import random
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info


def generalized_advantage_estimate(
    reward: torch.Tensor,
    value: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantage = torch.zeros_like(reward, dtype=torch.float32)
    accumulator = torch.zeros((), dtype=torch.float32)
    next_value = torch.zeros((), dtype=torch.float32)
    for index in range(len(reward) - 1, -1, -1):
        continuing = 1.0 - done[index].float()
        delta = reward[index].float() + gamma * next_value * continuing - value[index].float()
        accumulator = delta + gamma * gae_lambda * continuing * accumulator
        advantage[index] = accumulator
        next_value = value[index].float()
    return advantage, advantage + value.float()


class PPORolloutDataset(IterableDataset):
    def __init__(
        self, directory: str | Path, gamma: float, gae_lambda: float, seed: int = 0
    ) -> None:
        super().__init__()
        self.directory = Path(directory)
        manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.files = [self.directory / row["file"] for row in manifest["trajectories"]]
        self.sample_count = sum(int(row["steps"]) for row in manifest["trajectories"])
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.seed = seed
        self._epoch = multiprocessing.Value("i", 0)

    def __len__(self) -> int:
        return self.sample_count

    def set_epoch(self, epoch: int) -> None:
        self._epoch.value = epoch

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        rng = random.Random(self.seed + self._epoch.value * 1_000_003 + worker_id)
        files = list(self.files)
        rng.shuffle(files)
        for path in files[worker_id::worker_count]:
            trajectory = torch.load(path, map_location="cpu", weights_only=True)
            advantage, returns = generalized_advantage_estimate(
                trajectory["reward"],
                trajectory["old_value"],
                trajectory["done"],
                self.gamma,
                self.gae_lambda,
            )
            indices = list(range(len(trajectory["reward"])))
            rng.shuffle(indices)
            for index in indices:
                row = {key: value[index] for key, value in trajectory.items()}
                row["advantage"] = advantage[index]
                row["return"] = returns[index]
                yield row


def model_targets(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "unit_op": batch["unit_choices"][..., 0].long(),
        "unit_item": batch["unit_choices"][..., 1].long(),
        "unit_quantity": batch["unit_choices"][..., 2].long(),
        "market_op": batch["market_choices"][..., 0].long(),
        "market_item": batch["market_choices"][..., 1].long(),
        "market_quantity": batch["market_choices"][..., 2].long(),
    }


def _component_statistics(
    logits: torch.Tensor,
    choices: torch.Tensor,
    legal: torch.Tensor,
    active: torch.Tensor,
    temperature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    legal = legal.bool().clone()
    legal[..., 0] |= ~active.bool()
    scaled = logits.float() / temperature.float().view(-1, *([1] * (logits.ndim - 1))).clamp_min(1e-6)
    masked = scaled.masked_fill(~legal, -torch.inf)
    log_distribution = F.log_softmax(masked, dim=-1)
    distribution = log_distribution.exp()
    selected = log_distribution.gather(-1, choices.long().unsqueeze(-1)).squeeze(-1)
    entropy = -(distribution * log_distribution.masked_fill(~legal, 0.0)).sum(-1)
    return selected * active.float(), entropy * active.float()


def joint_log_probability_and_entropy(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    temperature = batch["policy_temperature"].float()
    log_probabilities = []
    entropies = []
    active_counts = torch.zeros_like(temperature)
    for prefix in ("unit", "market"):
        choices = batch[f"{prefix}_choices"]
        active = batch[f"{prefix}_component_active"].bool()
        for component_index, component in enumerate(("op", "item", "quantity")):
            component_active = active[..., component_index]
            selected, entropy = _component_statistics(
                outputs[f"{prefix}_{component}"],
                choices[..., component_index],
                batch[f"{prefix}_{component}_legal"],
                component_active,
                temperature,
            )
            log_probabilities.append(selected.sum(-1))
            entropies.append(entropy.sum(-1))
            active_counts += component_active.float().sum(-1)
    joint = torch.stack(log_probabilities).sum(0)
    mean_entropy = torch.stack(entropies).sum(0) / active_counts.clamp_min(1.0)
    return joint, mean_entropy


def reference_kl(
    outputs: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    total = torch.zeros(outputs["value"].shape[0], device=outputs["value"].device)
    count = torch.zeros_like(total)
    for prefix in ("unit", "market"):
        active = batch[f"{prefix}_component_active"].bool()
        for component_index, component in enumerate(("op", "item", "quantity")):
            component_active = active[..., component_index]
            legal = batch[f"{prefix}_{component}_legal"].bool().clone()
            legal[..., 0] |= ~component_active
            current_logits = outputs[f"{prefix}_{component}"].float().masked_fill(~legal, -torch.inf)
            reference_logits = reference[f"{prefix}_{component}"].float().masked_fill(~legal, -torch.inf)
            current_log = F.log_softmax(current_logits, dim=-1)
            reference_log = F.log_softmax(reference_logits, dim=-1)
            current_probability = current_log.exp()
            kl = (
                current_probability
                * (current_log - reference_log).masked_fill(~legal, 0.0)
            ).sum(-1)
            total += (kl * component_active.float()).sum(-1)
            count += component_active.float().sum(-1)
    return total / count.clamp_min(1.0)


def ppo_loss(
    outputs: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    clip_ratio: float,
    value_coefficient: float,
    entropy_coefficient: float,
    kl_coefficient: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    new_log_probability, entropy = joint_log_probability_and_entropy(outputs, batch)
    old_log_probability = batch["old_log_probability"].float()
    advantage = batch["advantage"].float()
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
    log_ratio = (new_log_probability - old_log_probability).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
    policy = -torch.minimum(unclipped, clipped).mean()
    value = F.smooth_l1_loss(outputs["value"].float(), batch["return"].float())
    kl = reference_kl(outputs, reference, batch).mean()
    entropy_mean = entropy.mean()
    total = (
        policy
        + value_coefficient * value
        - entropy_coefficient * entropy_mean
        + kl_coefficient * kl
    )
    metrics = {
        "loss": total.detach(),
        "policy": policy.detach(),
        "value": value.detach(),
        "entropy": entropy_mean.detach(),
        "reference_kl": kl.detach(),
        "approx_kl": (old_log_probability - new_log_probability).mean().detach(),
        "clip_fraction": ((ratio - 1.0).abs() > clip_ratio).float().mean().detach(),
        "ratio": ratio.mean().detach(),
        "return": batch["return"].float().mean().detach(),
    }
    return total, metrics
