#!/usr/bin/env python3
"""Generate fresh on-policy states for iterative AWR fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kaggle_environments import make

from kaggriculture_agent.inference import DynamicAgent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--opponent",
        action="append",
        default=[],
        help="Built-in agent or .py entry point; repeat to create an opponent pool.",
    )
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--opponent-temperature",
        type=float,
        default=0.0,
        help="Temperature for the current-checkpoint opponent selected with --opponent self.",
    )
    parser.add_argument("--output", default="data/raw/selfplay")
    args = parser.parse_args()
    output = Path(args.output)
    replay_dir = output / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    policy = DynamicAgent(args.checkpoint, temperature=args.temperature)
    opponents = args.opponent or ["self", "starter"]
    self_opponent = None
    if "self" in opponents:
        self_opponent = DynamicAgent(args.checkpoint, temperature=args.opponent_temperature)
    entries = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        for position in (0, 1):
            opponent = opponents[(seed + position) % len(opponents)]
            opponent_agent = self_opponent if opponent == "self" else opponent
            agents = [opponent_agent, opponent_agent]
            agents[position] = policy
            env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=False)
            env.run(agents)
            mine = float(env.steps[-1][position].reward or 0.0)
            theirs = float(env.steps[-1][1 - position].reward or 0.0)
            episode_id = seed * 10 + position
            replay_path = replay_dir / f"selfplay-{episode_id}-replay.json"
            replay_path.write_text(json.dumps(env.toJSON(), separators=(",", ":")), encoding="utf-8")
            entries.append(
                {
                    "episode_id": episode_id,
                    "replay_path": str(replay_path.relative_to(output)),
                    "expert_indices": [position],
                    "rank_by_player": {
                        str(position): 1 if mine > theirs else 2,
                    },
                    "opponent": opponent,
                    "reward_by_player": {str(position): mine},
                }
            )
            print(
                f"seed={seed} position={position} opponent={opponent} "
                f"reward={mine} opponent_reward={theirs}"
            )
    (output / "manifest.json").write_text(
        json.dumps({"competition": "kaggriculture-selfplay", "replays": entries}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
