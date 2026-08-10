#!/usr/bin/env python3
"""Measure model-only H100 throughput before launching a long run."""

from __future__ import annotations

import argparse
import time
import tomllib
from pathlib import Path

import torch

from kaggriculture_agent.constants import (
    BOARD_CHANNELS,
    GLOBAL_FEATURES,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    UNIT_FEATURES,
)
from kaggriculture_agent.model import DynamicPolicy, ModelConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/bc_h100.toml")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    with Path(args.config).open("rb") as config_file:
        config = tomllib.load(config_file)
    model_config = ModelConfig(**config["model"])
    model_config.gradient_checkpointing = bool(config["train"].get("gradient_checkpointing", False))
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = DynamicPolicy(model_config).to(device).train()
    if args.compile:
        model = torch.compile(model, mode="max-autotune", fullgraph=False)
    batch_size = args.batch_size
    board = torch.randint(
        0, 256, (batch_size, BOARD_CHANNELS, 10, 10), dtype=torch.uint8, device=device
    ).contiguous(memory_format=torch.channels_last)
    global_features = torch.randn(batch_size, GLOBAL_FEATURES, device=device)
    units = torch.randn(batch_size, MAX_UNITS, UNIT_FEATURES, device=device)
    unit_mask = torch.ones(batch_size, MAX_UNITS, dtype=torch.bool, device=device)
    targets = {
        "unit_op": torch.zeros(batch_size, MAX_UNITS, dtype=torch.long, device=device),
        "unit_item": torch.zeros(batch_size, MAX_UNITS, dtype=torch.long, device=device),
        "unit_quantity": torch.zeros(batch_size, MAX_UNITS, dtype=torch.long, device=device),
        "market_op": torch.zeros(batch_size, MAX_MARKET_ORDERS, dtype=torch.long, device=device),
        "market_item": torch.zeros(batch_size, MAX_MARKET_ORDERS, dtype=torch.long, device=device),
        "market_quantity": torch.zeros(batch_size, MAX_MARKET_ORDERS, dtype=torch.long, device=device),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(board, global_features, units, unit_mask, targets)
            loss = sum(value.float().square().mean() for value in output.values())
        loss.backward()
        optimizer.step()

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"batch_size={batch_size} steps={args.steps}")
    print(f"throughput={batch_size * args.steps / elapsed:.1f} samples/s")
    print(f"step_time={elapsed / args.steps * 1000:.1f} ms")
    print(f"peak_allocated={peak:.2f} GiB")


if __name__ == "__main__":
    main()
