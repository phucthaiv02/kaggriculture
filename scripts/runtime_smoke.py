#!/usr/bin/env python3
"""Load and execute a packaged agent without training dependencies."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tarfile
import tempfile
from pathlib import Path


def observation() -> dict:
    empty = [[None if x < 5 and y < 5 else "LOCKED" for x in range(10)] for y in range(10)]
    farms = [
        {
            "money": 3000,
            "tiles": [row[:] for row in empty],
            "farmer": [4, 4],
            "hands": [],
            "unlocked_quadrants": ["NW"],
            "hires_today": 0,
        }
        for _ in range(2)
    ]
    return {
        "player": 0,
        "day": 0,
        "hour": 0,
        "farms": farms,
        "private": {
            "shed": {"WHEAT": 0, "COW": 0, "FERTILIZER": 0},
            "seeds": {"WHEAT": 2, "CARROT": 0, "TOMATO": 0, "STRAWBERRY": 0, "MELON": 0},
            "inventories": [{}],
        },
        "market": {"prices": {}, "inventory": {}},
        "town": {"unlocked_shops": []},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    args = parser.parse_args()
    archive_path = Path(args.archive).resolve()
    with tempfile.TemporaryDirectory(prefix="runtime-v163-") as temporary:
        root = Path(temporary)
        with tarfile.open(archive_path, "r:gz") as archive:
            if any(Path(member.name).is_absolute() or ".." in Path(member.name).parts for member in archive):
                raise SystemExit("Unsafe archive path")
            archive.extractall(root)
        sys.path.insert(0, str(root))
        spec = importlib.util.spec_from_file_location("submission_main", root / "main.py")
        if spec is None or spec.loader is None:
            raise SystemExit("Cannot import main.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        action = module.agent(observation())
        if not isinstance(action, dict) or set(action) != {"farmer", "hands", "market"}:
            raise SystemExit(f"Invalid action: {action!r}")
        import numpy
        import torch

        print(f"PASS: python={sys.version.split()[0]} numpy={numpy.__version__} torch={torch.__version__}")
        print(action)


if __name__ == "__main__":
    main()
