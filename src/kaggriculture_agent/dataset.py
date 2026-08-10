"""Streaming safetensors shards without keeping the full replay corpus in RAM."""

from __future__ import annotations

import json
import multiprocessing
import random
from pathlib import Path
from typing import Iterator

import torch
from safetensors.torch import load_file
from torch.utils.data import IterableDataset, get_worker_info


class ShardedReplayDataset(IterableDataset):
    def __init__(self, directory: str | Path, shuffle: bool, seed: int = 0) -> None:
        super().__init__()
        self.directory = Path(directory)
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = multiprocessing.Value("i", 0)
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.shards = [self.directory / row["file"] for row in manifest["shards"]]
        self.sample_count = int(manifest["sample_count"])

    def set_epoch(self, epoch: int) -> None:
        self._epoch.value = epoch

    def __len__(self) -> int:
        return self.sample_count

    def _partition(self, shards: list[Path]) -> list[Path]:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        consumer_id = rank * worker_count + worker_id
        consumers = world * worker_count
        return shards[consumer_id::consumers]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rng = random.Random(self.seed + self._epoch.value * 1_000_003)
        shards = list(self.shards)
        if self.shuffle:
            rng.shuffle(shards)
        for shard_path in self._partition(shards):
            tensors = load_file(str(shard_path), device="cpu")
            count = next(iter(tensors.values())).shape[0]
            indices = list(range(count))
            if self.shuffle:
                rng.shuffle(indices)
            for index in indices:
                yield {name: tensor[index] for name, tensor in tensors.items()}
