#!/usr/bin/env python3
"""Merge expert and self-play manifests while resolving replay paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merged = []
    seen = set()
    for manifest_name in args.manifest:
        manifest_path = Path(manifest_name).resolve()
        content = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in content.get("replays", []):
            key = (entry.get("episode_id"), tuple(entry.get("expert_indices", [])))
            if key in seen:
                continue
            seen.add(key)
            row = dict(entry)
            replay_path = Path(row["replay_path"])
            if not replay_path.is_absolute():
                replay_path = (manifest_path.parent / replay_path).resolve()
            row["replay_path"] = str(replay_path)
            merged.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"competition": "merged", "replays": merged}, indent=2), encoding="utf-8")
    print(f"Merged {len(merged)} entries into {output.resolve()}")


if __name__ == "__main__":
    main()
