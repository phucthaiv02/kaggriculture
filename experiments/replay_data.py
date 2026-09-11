"""Shared JSON handling for collected and locally generated replays."""
from __future__ import annotations

import json
from pathlib import Path


def read_replay(path: Path) -> dict:
    replay = json.loads(path.read_text(encoding="utf-8"))
    # Some exported API responses wrap the environment JSON in `replay`.
    if isinstance(replay, dict) and "replay" in replay:
        replay = replay["replay"]
    if isinstance(replay, str):
        replay = json.loads(replay)
    if not isinstance(replay, dict) or replay.get("name") != "kaggriculture":
        raise ValueError("expected a Kaggriculture replay")
    steps = replay.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        raise ValueError("replay needs an initial frame and at least one action frame")
    if any(not isinstance(frame, list) or len(frame) != 2 or
           any(not isinstance(state, dict) for state in frame) for frame in steps):
        raise ValueError("expected two player states per frame")
    return replay


def write_json(path: Path, value) -> None:
    """Replace only after the complete JSON has been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
