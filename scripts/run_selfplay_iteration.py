#!/usr/bin/env python3
"""Run one complete self-play collection and AWR fine-tuning iteration."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/agent_h100.pt")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument(
        "--base-manifest",
        default="data/raw/manifest.json",
        help="Expert manifest for iteration 1, or the previous merged manifest to accumulate replay data.",
    )
    parser.add_argument("--opponent", action="append", default=[])
    parser.add_argument("--seeds", type=int, default=500)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--dataset-config", default="configs/dataset.toml")
    parser.add_argument("--train-config", default="configs/awr_h100.toml")
    parser.add_argument("--shards", default="data/shards")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    args = parser.parse_args()

    if args.iteration < 1:
        raise SystemExit("--iteration must be at least 1")
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    base_manifest = Path(args.base_manifest)
    if not base_manifest.exists():
        raise SystemExit(f"Base manifest not found: {base_manifest}")

    iteration_root = Path("data/raw/selfplay") / f"iteration-{args.iteration:02d}"
    selfplay_manifest = iteration_root / "manifest.json"
    merged_manifest = iteration_root / "merged-manifest.json"
    seed_start = (
        args.seed_start
        if args.seed_start is not None
        else 10_000 + args.iteration * 10_000
    )
    opponents = args.opponent or ["self", "starter"]

    collect = [
        sys.executable,
        "scripts/collect_selfplay.py",
        "--checkpoint",
        str(checkpoint),
        "--seeds",
        str(args.seeds),
        "--seed-start",
        str(seed_start),
        "--temperature",
        str(args.temperature),
        "--output",
        str(iteration_root),
    ]
    for opponent in opponents:
        collect.extend(("--opponent", opponent))
    run(collect)
    run(
        [
            sys.executable,
            "scripts/merge_manifests.py",
            str(base_manifest),
            str(selfplay_manifest),
            "--output",
            str(merged_manifest),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/build_dataset.py",
            "--config",
            args.dataset_config,
            "--manifest",
            str(merged_manifest),
            "--output-dir",
            args.shards,
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            "scripts/train_bc.py",
            "--config",
            args.train_config,
            "--init",
            str(checkpoint),
        ]
    )
    print(f"Completed self-play iteration {args.iteration}: {merged_manifest}")


if __name__ == "__main__":
    main()
