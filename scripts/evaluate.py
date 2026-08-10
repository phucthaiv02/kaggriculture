#!/usr/bin/env python3
"""Risk-aware complete-game evaluation across seeds and both positions."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from kaggle_environments import make

from kaggriculture_agent.inference import DynamicAgent


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty value sequence")
    tail = max(1, int(np.ceil(len(array) * 0.10)))
    standard_error = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    mean = float(array.mean())
    return {
        "mean": mean,
        "mean_ci95": [mean - 1.959963984540054 * standard_error, mean + 1.959963984540054 * standard_error],
        "median": float(np.median(array)),
        "std": float(array.std()),
        "minimum": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
        "cvar10": float(np.sort(array)[:tail].mean()),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def result_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    games = len(rows)
    wins = sum(row["result"] == "win" for row in rows)
    draws = sum(row["result"] == "draw" for row in rows)
    losses = games - wins - draws
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / max(1, games),
        "draw_rate": draws / max(1, games),
        "loss_rate": losses / max(1, games),
        "win_rate_wilson95": wilson_interval(wins, games),
    }


class TimedPolicy:
    def __init__(self, policy: DynamicAgent) -> None:
        self.policy = policy
        self.latencies: list[float] = []

    def __call__(self, observation: dict[str, Any], configuration: Any = None):
        started = time.perf_counter()
        try:
            return self.policy(observation, configuration)
        finally:
            self.latencies.append(time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponent", default="starter")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--output", default="runs/evaluation.json")
    parser.add_argument("--save-worst-replay", action="store_true")
    args = parser.parse_args()
    if args.seed_count < 1:
        raise SystemExit("--seed-count must be at least 1")
    policy = TimedPolicy(DynamicAgent(args.checkpoint, temperature=0.0))
    rows = []
    worst = None
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        for position in (0, 1):
            agents = [args.opponent, args.opponent]
            agents[position] = policy
            env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
            env.run(agents)
            mine = float(env.steps[-1][position].reward)
            theirs = float(env.steps[-1][1 - position].reward)
            margin = mine - theirs
            if margin > 0:
                result = "win"
            elif margin < 0:
                result = "loss"
            else:
                result = "draw"
            row = {
                "seed": seed,
                "position": position,
                "reward": mine,
                "opponent_reward": theirs,
                "margin": margin,
                "result": result,
                "win": result == "win",
                "status": str(env.steps[-1][position].status),
            }
            rows.append(row)
            if worst is None or mine < worst[0]:
                worst = (mine, seed, position, env.toJSON())
        print(f"completed seed={seed}")
    reward_stats = distribution([row["reward"] for row in rows])
    opponent_reward_stats = distribution([row["opponent_reward"] for row in rows])
    margin_stats = distribution([row["margin"] for row in rows])
    results = result_metrics(rows)
    statuses = Counter(row["status"] for row in rows)
    by_position = {}
    for position in (0, 1):
        position_rows = [row for row in rows if row["position"] == position]
        by_position[str(position)] = {
            **result_metrics(position_rows),
            "reward": distribution([row["reward"] for row in position_rows]),
            "margin": distribution([row["margin"] for row in position_rows]),
        }

    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    complete_pairs = [pair for pair in by_seed.values() if len(pair) == 2]
    paired_rewards = [statistics.mean(row["reward"] for row in pair) for pair in complete_pairs]
    paired_margins = [statistics.mean(row["margin"] for row in pair) for pair in complete_pairs]
    paired = {
        "seeds": len(complete_pairs),
        "both_positions_win_rate": sum(
            all(row["result"] == "win" for row in pair) for pair in complete_pairs
        ) / max(1, len(complete_pairs)),
        "at_least_one_win_rate": sum(
            any(row["result"] == "win" for row in pair) for pair in complete_pairs
        ) / max(1, len(complete_pairs)),
        "reward": distribution(paired_rewards),
        "margin": distribution(paired_margins),
        "mean_absolute_position_reward_gap": statistics.mean(
            abs(pair[0]["reward"] - pair[1]["reward"]) for pair in complete_pairs
        ),
    }
    latency_ms = [latency * 1000.0 for latency in policy.latencies]
    latency = {
        "calls": len(latency_ms),
        "total_seconds": sum(policy.latencies),
        **distribution(latency_ms),
    }
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "opponent": args.opponent,
        **results,
        # Keep the original top-level reward fields for existing result consumers.
        **reward_stats,
        "reward": reward_stats,
        "opponent_reward": opponent_reward_stats,
        "margin": margin_stats,
        "status_counts": dict(statuses),
        "completion_rate": sum(
            count for status, count in statuses.items() if status.upper().endswith("DONE")
        ) / max(1, len(rows)),
        "by_position": by_position,
        "paired_seed": paired,
        "inference_latency_ms": latency,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "results": rows}, indent=2), encoding="utf-8")
    if args.save_worst_replay and worst is not None:
        replay_path = output.with_name(f"worst-seed-{worst[1]}-position-{worst[2]}.json")
        replay_path.write_text(json.dumps(worst[3], separators=(",", ":")), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
