#!/usr/bin/env python3
"""Distributed BF16 behavior cloning trainer tuned for NVIDIA H100."""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from kaggriculture_agent.dataset import ShardedReplayDataset
from kaggriculture_agent.losses import behavior_cloning_loss
from kaggriculture_agent.model import DynamicPolicy, ModelConfig


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
    return rank, local_rank, world_size


def unwrap(model: nn.Module) -> nn.Module:
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        model = getattr(model, "module", getattr(model, "_orig_mod", model))
    return model


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    if device.type == "cuda":
        moved["board"] = moved["board"].contiguous(memory_format=torch.channels_last)
    return moved


def model_targets(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].long()
        for key in (
            "unit_op",
            "unit_item",
            "unit_quantity",
            "market_op",
            "market_item",
            "market_quantity",
        )
    }


def make_loader(dataset: ShardedReplayDataset, config: dict, shuffle: bool) -> DataLoader:
    workers = int(config["workers"])
    kwargs = {
        "batch_size": int(config["batch_size"]),
        "num_workers": workers,
        "pin_memory": True,
        "drop_last": shuffle,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(config["prefetch_factor"])
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    bf16: bool,
    max_batches: int = 100,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16)
    with autocast:
        for batch in loader:
            batch = move_batch(batch, device)
            output = model(
                batch["board"], batch["global"].float(), batch["units"].float(), batch["unit_mask"],
                model_targets(batch),
            )
            _, metrics = behavior_cloning_loss(output, batch)
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            batches += 1
            if batches >= max_batches:
                break
    model.train()
    return {key: value / max(1, batches) for key, value in totals.items()}


def save_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, step: int, epoch: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = unwrap(model)
    torch.save(
        {
            "architecture": "dynamic_autoregressive_v1",
            "model_config": raw.config.to_dict(),
            "model": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
        },
        path,
    )


def train(config_path: Path, resume: str | None, init: str | None) -> None:
    with config_path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)
    train_config = raw_config["train"]
    model_config = ModelConfig(**raw_config["model"])
    model_config.gradient_checkpointing = bool(train_config.get("gradient_checkpointing", False))
    rank, local_rank, world_size = setup_distributed()
    primary = rank == 0
    seed = int(train_config["seed"]) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    bf16 = bool(train_config["bf16"] and device.type == "cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    train_dataset = ShardedReplayDataset(train_config["train_dir"], shuffle=True, seed=seed)
    validation_dataset = ShardedReplayDataset(train_config["validation_dir"], shuffle=False, seed=seed)
    train_loader = make_loader(train_dataset, train_config, shuffle=True)
    validation_loader = make_loader(validation_dataset, train_config, shuffle=False)
    model: nn.Module = DynamicPolicy(model_config).to(device)
    if init:
        initial = torch.load(init, map_location=device, weights_only=False)
        model.load_state_dict(initial["model"])
    if device.type == "cuda":
        model.board_encoder.to(memory_format=torch.channels_last)
    if bool(train_config.get("compile", False)):
        model = torch.compile(model, mode="max-autotune", fullgraph=False)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        fused=device.type == "cuda",
    )
    per_epoch = max(1, math.ceil(len(train_dataset) / (int(train_config["batch_size"]) * world_size)))
    max_steps = int(train_config.get("max_steps", 0)) or per_epoch * int(train_config["epochs"])
    warmup = int(train_config["warmup_steps"])
    min_ratio = float(train_config["min_learning_rate"]) / float(train_config["learning_rate"])

    def lr_factor(step: int) -> float:
        if step < warmup:
            return max(1e-3, step / max(1, warmup))
        progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    step, start_epoch = 0, 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        unwrap(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step, start_epoch = int(checkpoint["step"]), int(checkpoint["epoch"])
    writer = SummaryWriter(train_config["run_dir"]) if primary else None
    accumulation = int(train_config["gradient_accumulation"])
    output_path = Path(train_config["output"])
    optimizer.zero_grad(set_to_none=True)
    model.train()
    started = time.monotonic()

    for epoch in range(start_epoch, int(train_config["epochs"])):
        train_dataset.set_epoch(epoch)
        for micro_step, batch in enumerate(train_loader):
            batch = move_batch(batch, device)
            synchronize = (micro_step + 1) % accumulation == 0
            sync_context = contextlib.nullcontext()
            if isinstance(model, DistributedDataParallel) and not synchronize:
                sync_context = model.no_sync()
            with sync_context, torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
                outputs = model(
                    batch["board"],
                    batch["global"].float(),
                    batch["units"].float(),
                    batch["unit_mask"],
                    model_targets(batch),
                )
                loss_batch = batch
                if str(train_config.get("algorithm", "bc")).lower() == "awr":
                    advantage = batch["value_target"].float() - outputs["value"].detach().float()
                    temperature = float(train_config.get("awr_temperature", 0.10))
                    max_weight = float(train_config.get("awr_max_weight", 20.0))
                    awr_weight = torch.exp(advantage / temperature).clamp(max=max_weight)
                    loss_batch = dict(batch)
                    loss_batch["sample_weight"] = batch["sample_weight"].float() * awr_weight
                loss, metrics = behavior_cloning_loss(outputs, loss_batch)
                loss = loss / accumulation
            loss.backward()
            if not synchronize:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_config["max_grad_norm"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            if primary and step % int(train_config["log_interval"]) == 0:
                samples = step * int(train_config["batch_size"]) * accumulation * world_size
                elapsed = max(1e-6, time.monotonic() - started)
                print(
                    f"epoch={epoch + 1} step={step}/{max_steps} loss={float(metrics['loss']):.4f} "
                    f"samples/s={samples / elapsed:.0f} lr={scheduler.get_last_lr()[0]:.2e}"
                )
                assert writer is not None
                for key, value in metrics.items():
                    writer.add_scalar(f"train/{key}", float(value), step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
            if step % int(train_config["validation_interval"]) == 0:
                validation = validate(model, validation_loader, device, bf16)
                if primary and writer is not None:
                    print(f"validation step={step} loss={validation.get('loss', float('nan')):.4f}")
                    for key, value in validation.items():
                        writer.add_scalar(f"validation/{key}", value, step)
            if primary and step % int(train_config["checkpoint_interval"]) == 0:
                save_checkpoint(output_path.with_name(f"step-{step:08d}.pt"), model, optimizer, scheduler, step, epoch)
                save_checkpoint(output_path, model, optimizer, scheduler, step, epoch)
            if step >= max_steps:
                break
        if primary:
            save_checkpoint(output_path, model, optimizer, scheduler, step, epoch + 1)
        if step >= max_steps:
            break
    if writer is not None:
        writer.close()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bc_h100.toml")
    parser.add_argument("--resume")
    parser.add_argument("--init", help="Load model weights only; start a fresh optimizer/schedule.")
    args = parser.parse_args()
    train(Path(args.config), args.resume, args.init)


if __name__ == "__main__":
    main()
