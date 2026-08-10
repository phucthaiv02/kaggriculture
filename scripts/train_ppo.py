#!/usr/bin/env python3
"""Legality-aware PPO learner with a frozen behavior-cloning KL anchor."""

from __future__ import annotations

import argparse
import math
import sys
import tomllib
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from kaggriculture_agent.model import DynamicPolicy, ModelConfig  # noqa: E402
from kaggriculture_agent.ppo import (  # noqa: E402
    PPORolloutDataset,
    model_targets,
    ppo_loss,
)


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[DynamicPolicy, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = DynamicPolicy(config).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def save_checkpoint(
    path: Path,
    model: DynamicPolicy,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "dynamic_autoregressive_v1",
            "model_config": model.config.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "algorithm": "ppo",
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo_h100.toml")
    parser.add_argument("--checkpoint", required=True, help="Current on-policy checkpoint.")
    parser.add_argument("--reference", required=True, help="Frozen BC checkpoint used as a KL anchor.")
    parser.add_argument("--rollouts", help="Override train.rollout_dir from config.")
    parser.add_argument("--output", help="Override train.output from config.")
    parser.add_argument("--batch-size", type=int, help="Override train.batch_size.")
    parser.add_argument("--workers", type=int, help="Override train.workers.")
    parser.add_argument("--epochs", type=int, help="Override train.epochs.")
    args = parser.parse_args()

    with Path(args.config).open("rb") as config_file:
        config = tomllib.load(config_file)["train"]
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.workers is not None:
        config["workers"] = args.workers
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if int(config["batch_size"]) < 1 or int(config["workers"]) < 0 or int(config["epochs"]) < 1:
        raise SystemExit("batch-size/epochs must be positive and workers cannot be negative")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = bool(config.get("bf16", True) and device.type == "cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, source_checkpoint = load_model(args.checkpoint, device)
    reference, _ = load_model(args.reference, device)
    reference.eval()
    reference.requires_grad_(False)
    model.train()
    rollout_dir = args.rollouts or config["rollout_dir"]
    dataset = PPORolloutDataset(
        rollout_dir,
        gamma=float(config["gamma"]),
        gae_lambda=float(config["gae_lambda"]),
        seed=int(config["seed"]),
    )
    workers = int(config["workers"])
    loader_kwargs = {
        "batch_size": int(config["batch_size"]),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = int(config["prefetch_factor"])
    loader = DataLoader(dataset, **loader_kwargs)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        fused=device.type == "cuda",
    )
    output = Path(args.output or config["output"])
    writer = SummaryWriter(config["run_dir"])
    max_steps = int(config.get("max_steps", 0)) or math.ceil(
        len(dataset) / int(config["batch_size"])
    ) * int(config["epochs"])
    step = 0
    for epoch in range(int(config["epochs"])):
        dataset.set_epoch(epoch)
        for batch in loader:
            batch = move_batch(batch, device)
            targets = model_targets(batch)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                outputs = model(
                    batch["board"], batch["global"].float(), batch["units"].float(),
                    batch["unit_mask"], targets,
                )
                with torch.no_grad():
                    reference_outputs = reference(
                        batch["board"], batch["global"].float(), batch["units"].float(),
                        batch["unit_mask"], targets,
                    )
                loss, metrics = ppo_loss(
                    outputs,
                    reference_outputs,
                    batch,
                    clip_ratio=float(config["clip_ratio"]),
                    value_coefficient=float(config["value_coefficient"]),
                    entropy_coefficient=float(config["entropy_coefficient"]),
                    kl_coefficient=float(config["kl_coefficient"]),
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
            optimizer.step()
            step += 1
            if step % int(config["log_interval"]) == 0:
                summary = " ".join(f"{key}={float(value):.4f}" for key, value in metrics.items())
                print(f"epoch={epoch + 1} step={step}/{max_steps} {summary}")
                for key, value in metrics.items():
                    writer.add_scalar(f"ppo/{key}", float(value), step)
            if step % int(config["checkpoint_interval"]) == 0:
                save_checkpoint(output.with_name(f"ppo-step-{step:08d}.pt"), model, optimizer, step, epoch)
            if step >= max_steps:
                break
        save_checkpoint(output, model, optimizer, step, epoch + 1)
        if step >= max_steps:
            break
    if step == 0:
        raise RuntimeError("PPO DataLoader produced no batches; check rollout files and worker errors")
    writer.close()
    print(
        f"Saved PPO checkpoint to {output.resolve()} "
        f"(initialized from step={source_checkpoint.get('step', 0)})"
    )


if __name__ == "__main__":
    main()
