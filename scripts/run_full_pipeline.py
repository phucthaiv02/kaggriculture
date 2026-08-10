#!/usr/bin/env python3
"""Run expert BC -> AWR recovery -> league PPO -> packaging as one resumable pipeline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def snapshot(source: str | Path, destination: str | Path) -> None:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def ensure_league(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "strategy": "weighted_snapshot_pool",
                "opponents": [
                    {"agent": "self", "weight": 2.0, "active": True},
                    {"agent": "starter", "weight": 1.0, "active": True},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.toml")
    parser.add_argument("--skip-expert-collection", action="store_true")
    parser.add_argument("--skip-bc", action="store_true")
    parser.add_argument("--skip-awr", action="store_true")
    parser.add_argument("--skip-ppo", action="store_true")
    parser.add_argument("--skip-package", action="store_true")
    args = parser.parse_args()
    with Path(args.config).open("rb") as config_file:
        config = tomllib.load(config_file)
    pipeline = config["pipeline"]
    expert = config["expert"]
    awr = config["awr"]
    ppo = config["ppo"]
    manifest = Path(pipeline["expert_manifest"])
    checkpoint = Path(pipeline["bc_checkpoint"])
    anchor = Path(pipeline["bc_anchor"])
    best = Path(pipeline["best_checkpoint"])
    league = Path(pipeline["league"])
    python = sys.executable

    needs_manifest = not args.skip_bc or not args.skip_awr
    if needs_manifest and not manifest.exists() and not args.skip_expert_collection:
        run(
            [
                python,
                "scripts/collect_replays.py",
                "--top-teams",
                str(expert["top_teams"]),
                "--episodes-per-team",
                str(expert["episodes_per_team"]),
                "--max-discovery-queries",
                str(expert["max_discovery_queries"]),
                "--output",
                str(manifest.parent),
            ]
        )
    if needs_manifest and not manifest.exists():
        raise SystemExit(f"Expert manifest not found: {manifest}")

    if not args.skip_bc:
        run(
            [
                python,
                "scripts/build_dataset.py",
                "--config",
                "configs/dataset.toml",
                "--manifest",
                str(manifest),
                "--output-dir",
                str(pipeline["dataset_dir"]),
            ]
        )
        run(
            [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={pipeline['nproc_per_node']}",
                "scripts/train_bc.py",
                "--config",
                "configs/bc_h100.toml",
            ]
        )
        snapshot(checkpoint, anchor)
    if (not args.skip_awr or not args.skip_ppo) and (not checkpoint.exists() or not anchor.exists()):
        raise SystemExit("BC checkpoint/anchor missing; run BC or provide both configured files")

    if not args.skip_awr:
        base_manifest = manifest
        for iteration in range(1, int(pipeline["awr_iterations"]) + 1):
            command = [
                python,
                "scripts/run_selfplay_iteration.py",
                "--checkpoint",
                str(checkpoint),
                "--iteration",
                str(iteration),
                "--base-manifest",
                str(base_manifest),
                "--seeds",
                str(awr["seeds"]),
                "--temperature",
                str(awr["temperature"]),
                "--nproc-per-node",
                str(pipeline["nproc_per_node"]),
            ]
            for opponent in awr["opponents"]:
                command.extend(("--opponent", str(opponent)))
            run(command)
            base_manifest = Path("data/raw/selfplay") / f"iteration-{iteration:02d}" / "merged-manifest.json"

    if (not best.exists() or not args.skip_bc or not args.skip_awr) and checkpoint.exists():
        snapshot(checkpoint, best)
    if not best.exists():
        raise SystemExit(f"Best checkpoint not found: {best}")
    ensure_league(league)
    if not args.skip_ppo:
        for iteration in range(1, int(pipeline["ppo_iterations"]) + 1):
            rollout_dir = Path("data/ppo") / f"iteration-{iteration:03d}"
            candidate = Path("checkpoints/candidates") / f"ppo-iteration-{iteration:03d}.pt"
            incumbent = Path("checkpoints/incumbents") / f"before-iteration-{iteration:03d}.pt"
            snapshot(best, incumbent)
            run(
                [
                    python,
                    "scripts/collect_ppo_rollouts.py",
                    "--checkpoint",
                    str(best),
                    "--league",
                    str(league),
                    "--seeds",
                    str(ppo["rollout_seeds"]),
                    "--seed-start",
                    str(50_000 + iteration * 1_000),
                    "--temperature",
                    str(ppo["rollout_temperature"]),
                    "--workers",
                    str(ppo["rollout_workers"]),
                    "--margin-scale",
                    str(ppo["margin_scale"]),
                    "--win-bonus",
                    str(ppo["win_bonus"]),
                    "--output",
                    str(rollout_dir),
                ]
            )
            run(
                [
                    python,
                    "scripts/train_ppo.py",
                    "--config",
                    "configs/ppo_h100.toml",
                    "--checkpoint",
                    str(best),
                    "--reference",
                    str(anchor),
                    "--rollouts",
                    str(rollout_dir),
                    "--output",
                    str(candidate),
                ]
            )
            run(
                [
                    python,
                    "scripts/promote_candidate.py",
                    "--candidate",
                    str(candidate),
                    "--incumbent",
                    str(incumbent),
                    "--best",
                    str(best),
                    "--league",
                    str(league),
                    "--iteration",
                    str(iteration),
                    "--seed-start",
                    str(90_000 + iteration * 1_000),
                    "--seed-count",
                    str(ppo["promotion_seeds"]),
                    "--minimum-win-rate",
                    str(ppo["minimum_win_rate"]),
                    "--minimum-mean-margin",
                    str(ppo["minimum_mean_margin"]),
                    "--minimum-p10-margin",
                    str(ppo["minimum_p10_margin"]),
                    "--output",
                    f"runs/promotion-iteration-{iteration:03d}.json",
                ]
            )
    if not args.skip_package:
        run(
            [
                python,
                "scripts/package_submission.py",
                "--checkpoint",
                str(best),
                "--output",
                str(pipeline["package_output"]),
            ]
        )
        run([python, "scripts/audit_submission.py", str(pipeline["package_output"])])
    print(f"Full pipeline complete. Best checkpoint: {best.resolve()}")


if __name__ == "__main__":
    main()
