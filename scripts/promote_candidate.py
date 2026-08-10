#!/usr/bin/env python3
"""Evaluate a PPO candidate head-to-head and promote it into the opponent league."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from kaggle_environments import make

from kaggriculture_agent.inference import DynamicAgent


def evaluate(
    candidate_path: str,
    incumbent_path: str,
    seed_start: int,
    seed_count: int,
    episode_steps: int,
) -> list[dict[str, Any]]:
    candidate = DynamicAgent(candidate_path, temperature=0.0)
    incumbent = DynamicAgent(incumbent_path, temperature=0.0)
    rows = []
    for seed in range(seed_start, seed_start + seed_count):
        for position in (0, 1):
            agents = [incumbent, incumbent]
            agents[position] = candidate
            env = make(
                "kaggriculture",
                configuration={"episodeSteps": episode_steps, "seed": seed},
                debug=False,
            )
            env.run(agents)
            mine = float(env.steps[-1][position].reward or 0.0)
            theirs = float(env.steps[-1][1 - position].reward or 0.0)
            margin = mine - theirs
            rows.append(
                {
                    "seed": seed,
                    "position": position,
                    "candidate_reward": mine,
                    "incumbent_reward": theirs,
                    "margin": margin,
                    "result": "win" if margin > 0 else "loss" if margin < 0 else "draw",
                }
            )
        print(f"promotion evaluation seed={seed}")
    return rows


def update_league(path: Path, snapshot: Path) -> None:
    if path.exists():
        league = json.loads(path.read_text(encoding="utf-8"))
    else:
        league = {
            "strategy": "weighted_snapshot_pool",
            "opponents": [
                {"agent": "self", "weight": 2.0, "active": True},
                {"agent": "starter", "weight": 1.0, "active": True},
            ],
        }
    resolved = str(snapshot.resolve())
    if not any(entry.get("agent") == resolved for entry in league["opponents"]):
        league["opponents"].append({"agent": resolved, "weight": 1.0, "active": True})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(league, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--incumbent", required=True)
    parser.add_argument("--best", default="checkpoints/agent_best.pt")
    parser.add_argument("--league", default="configs/opponent_league.json")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--seed-start", type=int, default=90_000)
    parser.add_argument("--seed-count", type=int, default=50)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--minimum-win-rate", type=float, default=0.52)
    parser.add_argument("--minimum-mean-margin", type=float, default=0.0)
    parser.add_argument("--minimum-p10-margin", type=float, default=-25_000.0)
    parser.add_argument("--output", default="runs/promotion.json")
    args = parser.parse_args()
    rows = evaluate(
        args.candidate, args.incumbent, args.seed_start, args.seed_count, args.episode_steps
    )
    margins = np.asarray([row["margin"] for row in rows], dtype=np.float64)
    wins = sum(row["result"] == "win" for row in rows)
    win_rate = wins / len(rows)
    mean_margin = float(margins.mean())
    p10_margin = float(np.percentile(margins, 10))
    promoted = (
        win_rate >= args.minimum_win_rate
        and mean_margin >= args.minimum_mean_margin
        and p10_margin >= args.minimum_p10_margin
    )
    summary = {
        "candidate": str(Path(args.candidate).resolve()),
        "incumbent": str(Path(args.incumbent).resolve()),
        "games": len(rows),
        "win_rate": win_rate,
        "mean_margin": mean_margin,
        "p10_margin": p10_margin,
        "promoted": promoted,
        "thresholds": {
            "minimum_win_rate": args.minimum_win_rate,
            "minimum_mean_margin": args.minimum_mean_margin,
            "minimum_p10_margin": args.minimum_p10_margin,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "results": rows}, indent=2), encoding="utf-8")
    if promoted:
        best = Path(args.best)
        best.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.candidate, best)
        snapshot = best.parent / "league" / f"iteration-{args.iteration:03d}.pt"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.candidate, snapshot)
        update_league(Path(args.league), snapshot)
        print(f"PROMOTED candidate to {best.resolve()}")
    else:
        print("REJECTED candidate; incumbent remains unchanged")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
