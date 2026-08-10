#!/usr/bin/env python3
"""Convert public replay JSON into episode-split safetensors shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file
from tqdm import tqdm

from kaggriculture_agent.codec import (
    encode_action,
    encode_observation,
    legal_market_ops,
    legal_unit_op_matrix,
)
from kaggriculture_agent.constants import MAX_MARKET_ORDERS


def episode_split(episode_id: int, validation_fraction: float, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "little") / 2**64
    return "validation" if value < validation_fraction else "train"


class ShardWriter:
    def __init__(self, root: Path, split: str, samples_per_shard: int) -> None:
        self.directory = root / split
        self.directory.mkdir(parents=True, exist_ok=True)
        self.samples_per_shard = samples_per_shard
        self.buffer: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.shards: list[dict[str, Any]] = []
        self.sample_count = 0

    def add(self, sample: dict[str, np.ndarray | float | int]) -> None:
        for key, value in sample.items():
            array = np.asarray(value)
            self.buffer[key].append(torch.from_numpy(array.copy()) if array.ndim else torch.tensor(value))
        if len(next(iter(self.buffer.values()))) >= self.samples_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        shard_index = len(self.shards)
        filename = f"shard-{shard_index:05d}.safetensors"
        tensors = {key: torch.stack(values).contiguous() for key, values in self.buffer.items()}
        count = next(iter(tensors.values())).shape[0]
        save_file(tensors, str(self.directory / filename))
        self.shards.append({"file": filename, "samples": count})
        self.sample_count += count
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        manifest = {"sample_count": self.sample_count, "shards": self.shards}
        (self.directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def resolve_replay(entry: dict[str, Any], manifest_path: Path) -> Path:
    path = Path(entry["replay_path"])
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def build(
    config_path: Path,
    manifest_override: str | None = None,
    output_override: str | None = None,
) -> None:
    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)["dataset"]
    manifest_path = Path(manifest_override or config["manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = Path(output_override or config["output_dir"])
    writers = {
        split: ShardWriter(output, split, int(config["samples_per_shard"]))
        for split in ("train", "validation")
    }
    rng = random.Random(int(config["seed"]))
    entries = list(manifest.get("replays", []))
    rng.shuffle(entries)
    for entry in tqdm(entries, desc="replays"):
        episode_id = int(entry["episode_id"])
        replay_path = resolve_replay(entry, manifest_path)
        if not replay_path.exists():
            tqdm.write(f"skip missing {replay_path}")
            continue
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        steps = replay.get("steps", [])
        if len(steps) < 2:
            continue
        expert_indices = list(entry.get("expert_indices", []))
        if config.get("include_all_done_players", False):
            expert_indices = [
                index for index, state in enumerate(steps[-1]) if state.get("status") == "DONE"
            ]
        split = episode_split(episode_id, float(config["validation_fraction"]), int(config["seed"]))
        for player in expert_indices:
            final_reward = float(steps[-1][player].get("reward") or 0.0)
            if final_reward < float(config.get("minimum_reward", 0.0)):
                continue
            if str(config.get("value_target", "absolute")).lower() == "margin":
                opponent_rewards = [
                    float(state.get("reward") or 0.0)
                    for index, state in enumerate(steps[-1])
                    if index != player
                ]
                opponent_reward = max(opponent_rewards, default=0.0)
                margin = final_reward - opponent_reward
                result = 1.0 if margin > 0 else -1.0 if margin < 0 else 0.0
                value_target = margin / float(config.get("value_scale", 50_000.0))
                value_target += float(config.get("win_bonus", 0.25)) * result
                value_target = float(
                    np.clip(
                        value_target,
                        -float(config.get("value_clip", 5.0)),
                        float(config.get("value_clip", 5.0)),
                    )
                )
            else:
                value_target = final_reward / float(config.get("value_scale", 200_000.0))
            rank = float(entry.get("rank_by_player", {}).get(str(player), 10))
            sample_weight = max(0.25, 1.25 - 0.05 * rank)
            for turn in range(len(steps) - 1):
                observation = steps[turn][player].get("observation")
                action = steps[turn + 1][player].get("action")
                if not observation or not action:
                    continue
                encoded = encode_observation(observation)
                unit_count = int(encoded["unit_mask"].sum())
                targets = encode_action(action, unit_count)
                sample: dict[str, Any] = {
                    "board": encoded["board"],
                    "global": encoded["global"].astype(np.float16),
                    "units": encoded["units"].astype(np.float16),
                    "unit_mask": encoded["unit_mask"],
                    "unit_legal": legal_unit_op_matrix(observation),
                    "market_legal": np.repeat(
                        legal_market_ops(observation)[None, :], MAX_MARKET_ORDERS, axis=0
                    ),
                    "value_target": np.float32(value_target),
                    "sample_weight": np.float32(sample_weight),
                    "episode_id": np.int64(episode_id),
                }
                for key, value in targets.items():
                    sample[key] = value.astype(np.int16) if value.dtype == np.int64 else value
                writers[split].add(sample)
    for writer in writers.values():
        writer.close()
        print(f"{writer.directory}: {writer.sample_count} samples in {len(writer.shards)} shards")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dataset.toml")
    parser.add_argument("--manifest", help="Override dataset.manifest from the TOML config.")
    parser.add_argument("--output-dir", help="Override dataset.output_dir from the TOML config.")
    args = parser.parse_args()
    build(Path(args.config), args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
