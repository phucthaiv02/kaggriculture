#!/usr/bin/env python3
"""Detect local hardware, derive a safe BC config, and launch training."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]


def available_ram_gib() -> float:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, value = line.split(":", 1)
            values[name] = int(value.strip().split()[0])
        return values.get("MemAvailable", values.get("MemFree", 0)) / 1024**2
    except (OSError, ValueError):
        return 0.0


def hardware_profile() -> dict[str, Any]:
    cpu_count = os.cpu_count() or 1
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gib = free_bytes / 1024**3
        total_gib = total_bytes / 1024**3
        profiles = (
            (70, 256, 2),
            (40, 128, 4),
            (24, 64, 8),
            (16, 32, 16),
            (10, 16, 32),
            (7, 8, 64),
            (4, 4, 128),
            (0, 2, 256),
        )
        _, batch_size, accumulation = next(row for row in profiles if free_gib >= row[0])
        return {
            "device": "cuda",
            "description": f"{torch.cuda.get_device_name(0)} ({free_gib:.1f}/{total_gib:.1f} GiB free)",
            "batch_size": batch_size,
            "gradient_accumulation": accumulation,
            "workers": min(8, max(1, cpu_count // 2)),
            "bf16": bool(torch.cuda.is_bf16_supported()),
            "compile": free_gib >= 10,
            "gradient_checkpointing": free_gib < 24,
        }

    ram_gib = available_ram_gib()
    if ram_gib >= 12:
        batch_size, accumulation = 4, 16
    elif ram_gib >= 6:
        batch_size, accumulation = 2, 32
    else:
        batch_size, accumulation = 1, 32
    return {
        "device": "cpu",
        "description": f"CPU-only, {cpu_count} logical CPUs, {ram_gib:.1f} GiB RAM available",
        "batch_size": batch_size,
        "gradient_accumulation": accumulation,
        "workers": 0,
        "bf16": False,
        "compile": False,
        "gradient_checkpointing": ram_gib < 8,
    }


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return str(value)


def write_config(path: Path, model: dict[str, Any], train: dict[str, Any]) -> None:
    lines = ["[model]"]
    lines.extend(f"{key} = {toml_value(value)}" for key, value in model.items())
    lines.extend(("", "[train]"))
    lines.extend(f"{key} = {toml_value(value)}" for key, value in train.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/bc_h100.toml")
    parser.add_argument("--resume", help="Resume model, optimizer and scheduler state.")
    parser.add_argument("--init", help="Load model weights but start a new optimizer.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", default="checkpoints/agent_local.pt")
    parser.add_argument("--small", action="store_true", help="Use a small debug model, not production.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.resume and args.init:
        raise SystemExit("Use only one of --resume or --init")

    base_path = (ROOT / args.base_config).resolve()
    with base_path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    model = dict(raw["model"])
    train = dict(raw["train"])
    profile = hardware_profile()

    batch_size = args.batch_size or int(profile["batch_size"])
    accumulation = args.gradient_accumulation or int(profile["gradient_accumulation"])
    workers = args.workers if args.workers is not None else int(profile["workers"])
    if batch_size < 1 or accumulation < 1 or workers < 0:
        raise SystemExit("Batch size/accumulation must be positive; workers cannot be negative")

    if args.small:
        model.update(
            d_model=128,
            board_blocks=2,
            transformer_layers=2,
            attention_heads=4,
            ff_multiplier=2,
            dropout=0.05,
        )
        if args.batch_size is None:
            batch_size = min(4, batch_size * 4)
        if args.gradient_accumulation is None:
            accumulation = max(1, math.ceil(64 / batch_size))

    effective_batch = batch_size * accumulation
    base_lr = float(train["learning_rate"])
    reference_batch = int(raw["train"]["batch_size"]) * int(
        raw["train"]["gradient_accumulation"]
    )
    learning_rate = base_lr * math.sqrt(effective_batch / reference_batch)
    train.update(
        batch_size=batch_size,
        gradient_accumulation=accumulation,
        workers=workers,
        learning_rate=learning_rate,
        min_learning_rate=min(float(train["min_learning_rate"]), learning_rate / 10),
        output=args.output,
        run_dir="runs/agent_bc_auto",
        bf16=bool(profile["bf16"]),
        compile=bool(profile["compile"]),
        gradient_checkpointing=bool(profile["gradient_checkpointing"]),
    )
    if args.epochs is not None:
        train["epochs"] = args.epochs
    if args.max_steps is not None:
        train["max_steps"] = args.max_steps
    if workers == 0:
        train["prefetch_factor"] = 2  # Ignored by the trainer when workers=0.

    generated = ROOT / "runs/auto_bc_config.toml"
    write_config(generated, model, train)
    parameter_estimate = "small debug" if args.small else "production (~20.8M parameters)"
    print(f"Hardware: {profile['description']}")
    print(f"Model: {parameter_estimate}")
    print(
        f"batch_size={batch_size}, gradient_accumulation={accumulation}, "
        f"effective_batch={effective_batch}, workers={workers}"
    )
    print(
        f"lr={learning_rate:.3g}, bf16={train['bf16']}, compile={train['compile']}, "
        f"gradient_checkpointing={train['gradient_checkpointing']}"
    )
    print(f"Generated config: {generated}")

    train_manifest = ROOT / str(train["train_dir"]) / "manifest.json"
    validation_manifest = ROOT / str(train["validation_dir"]) / "manifest.json"
    if not train_manifest.exists() or not validation_manifest.exists():
        message = (
            "Dataset shards are missing. Run collect_replays.py and build_dataset.py first; "
            "see README.md."
        )
        if args.dry_run:
            print(f"WARNING: {message}")
        else:
            raise SystemExit(message)

    command = [sys.executable, "scripts/train_bc.py", "--config", str(generated)]
    if args.resume:
        command.extend(("--resume", args.resume))
    if args.init:
        command.extend(("--init", args.init))
    print("Command:", " ".join(command))
    if args.dry_run:
        return
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "src"), str(ROOT), environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
