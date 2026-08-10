#!/usr/bin/env python3
"""Collect on-policy, legality-aware trajectories for PPO updates."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from kaggle_environments import make

from kaggriculture_agent.inference import DynamicAgent


class TracedPolicy:
    def __init__(self, policy: DynamicAgent) -> None:
        self.policy = policy
        self.steps: list[dict[str, Any]] = []

    def __call__(self, observation: dict[str, Any], configuration: Any = None):
        action, trace = self.policy.act_with_trace(observation, configuration)
        self.steps.append(trace)
        return action


def load_league(path: str | None, explicit: list[str], checkpoint: str) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    if path:
        content = json.loads(Path(path).read_text(encoding="utf-8"))
        for entry in content.get("opponents", []):
            if entry.get("active", True):
                rows.append((str(entry["agent"]), float(entry.get("weight", 1.0))))
    rows.extend((name, 1.0) for name in explicit)
    if not rows:
        rows = [("self", 1.0), ("starter", 1.0)]
    return [(checkpoint if name == "self" else name, weight) for name, weight in rows]


def choose_opponent(pool: list[tuple[str, float]], rng: random.Random) -> str:
    return rng.choices([name for name, _ in pool], weights=[weight for _, weight in pool], k=1)[0]


def opponent_agent(name: str) -> Any:
    path = Path(name)
    if path.suffix == ".pt" and path.exists():
        return DynamicAgent(path, temperature=0.0)
    return name


_ROLLOUT_POLICY: DynamicAgent | None = None
_OPPONENT_CACHE: dict[str, Any] = {}


def initialize_worker(checkpoint: str, temperature: float) -> None:
    global _ROLLOUT_POLICY, _OPPONENT_CACHE
    torch.set_num_threads(1)
    _ROLLOUT_POLICY = DynamicAgent(checkpoint, temperature=temperature)
    _OPPONENT_CACHE = {}


def collect_game(task: dict[str, Any]) -> dict[str, Any]:
    if _ROLLOUT_POLICY is None:
        raise RuntimeError("Rollout worker was not initialized")
    seed = int(task["seed"])
    position = int(task["position"])
    selected = str(task["opponent"])
    torch.manual_seed(int(task["sampling_seed"]))
    traced = TracedPolicy(_ROLLOUT_POLICY)
    if selected not in _OPPONENT_CACHE:
        _OPPONENT_CACHE[selected] = opponent_agent(selected)
    opponent = _OPPONENT_CACHE[selected]
    agents = [opponent, opponent]
    agents[position] = traced
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": int(task["episode_steps"]), "seed": seed},
        debug=False,
    )
    env.run(agents)
    mine = float(env.steps[-1][position].reward or 0.0)
    theirs = float(env.steps[-1][1 - position].reward or 0.0)
    margin = mine - theirs
    result = 1.0 if margin > 0 else -1.0 if margin < 0 else 0.0
    terminal_reward = float(
        np.clip(
            margin / float(task["margin_scale"]) + float(task["win_bonus"]) * result,
            -float(task["reward_clip"]),
            float(task["reward_clip"]),
        )
    )
    output = Path(task["output"])
    trajectory_path = output / f"seed-{seed}-position-{position}.pt"
    torch.save(tensorize(traced.steps, terminal_reward), trajectory_path)
    return {
        "file": trajectory_path.name,
        "seed": seed,
        "position": position,
        "opponent": selected,
        "steps": len(traced.steps),
        "reward": mine,
        "opponent_reward": theirs,
        "margin": margin,
        "terminal_reward": terminal_reward,
    }


def tensorize(steps: list[dict[str, Any]], terminal_reward: float) -> dict[str, torch.Tensor]:
    if not steps:
        raise ValueError("Cannot save an empty trajectory")
    tensors: dict[str, torch.Tensor] = {}
    for key in steps[0]:
        tensors[key] = torch.from_numpy(np.stack([np.asarray(step[key]) for step in steps]))
    reward = torch.zeros(len(steps), dtype=torch.float32)
    reward[-1] = float(terminal_reward)
    tensors["reward"] = reward
    tensors["done"] = torch.zeros(len(steps), dtype=torch.bool)
    tensors["done"][-1] = True
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--league")
    parser.add_argument("--opponent", action="append", default=[])
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=50_000)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--margin-scale", type=float, default=50_000.0)
    parser.add_argument("--win-bonus", type=float, default=0.25)
    parser.add_argument("--reward-clip", type=float, default=5.0)
    parser.add_argument("--output", default="data/ppo/rollouts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.temperature <= 0:
        raise SystemExit("PPO rollout temperature must be positive")
    if args.seeds < 1 or args.episode_steps < 2:
        raise SystemExit("--seeds must be positive and --episode-steps must be at least 2")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    pool = load_league(args.league, args.opponent, args.checkpoint)
    tasks = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for position in (0, 1):
            selected = choose_opponent(pool, rng)
            tasks.append(
                {
                    "seed": seed,
                    "position": position,
                    "opponent": selected,
                    "sampling_seed": args.seed * 1_000_003 + seed * 2 + position,
                    "episode_steps": args.episode_steps,
                    "margin_scale": args.margin_scale,
                    "win_bonus": args.win_bonus,
                    "reward_clip": args.reward_clip,
                    "output": str(output),
                }
            )
    workers = max(1, int(args.workers))
    if workers == 1:
        initialize_worker(args.checkpoint, args.temperature)
        entries = [collect_game(task) for task in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            workers,
            initializer=initialize_worker,
            initargs=(args.checkpoint, args.temperature),
        ) as pool:
            entries = list(pool.imap(collect_game, tasks))
    for entry in entries:
        print(
            f"seed={entry['seed']} position={entry['position']} opponent={entry['opponent']} "
            f"margin={entry['margin']:.0f} rl_reward={entry['terminal_reward']:.4f} "
            f"steps={entry['steps']}"
        )
    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "temperature": args.temperature,
        "opponent_pool": [{"agent": name, "weight": weight} for name, weight in pool],
        "trajectories": entries,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(entries)} trajectories to {output.resolve()}")


if __name__ == "__main__":
    main()
