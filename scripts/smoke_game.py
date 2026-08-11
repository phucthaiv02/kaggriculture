#!/usr/bin/env python3
"""Run a full local season, validate the strategic layout, and save a replay."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from kaggle_environments import make

from kaggriculture_agent.heuristic import ANIMAL_LAYOUT, CROPS, CROP_LAYOUT, HeuristicAgent, NE_CROP_POSITIONS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--html", type=Path)
    args = parser.parse_args()

    env = make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "weedSpawnChance": 0, "seed": 42},
        debug=True,
    )
    agent = HeuristicAgent()
    env.run([agent, "pass"])

    peak_assets: Counter[str] = Counter()
    peak_crop_tiles = 0
    nw_crop_tiles_at_day_start: dict[int, int] = {}
    ne_crop_tiles_at_day_start: dict[int, int] = {}
    peak_ne_crop_tiles = 0
    for step in env.steps:
        observation = step[0].observation
        if not getattr(observation, "farms", None):
            continue
        current_assets: Counter[str] = Counter()
        crop_positions: set[tuple[int, int]] = set()
        for y, row in enumerate(observation.farms[0]["tiles"]):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict):
                    continue
                if tile.get("crop"):
                    current_assets[tile["crop"]] += 1
                    crop_positions.add((x, y))
                if tile.get("animal"):
                    current_assets[tile["animal"]] += 1
        peak_assets |= current_assets
        crop_tiles = sum(current_assets[crop] for crop in CROPS)
        peak_crop_tiles = max(peak_crop_tiles, crop_tiles)
        peak_ne_crop_tiles = max(peak_ne_crop_tiles, len(crop_positions & set(NE_CROP_POSITIONS)))
        if observation.hour == 0 and 1 <= observation.day <= 25:
            nw_crop_tiles_at_day_start[observation.day] = len(crop_positions & set(CROP_LAYOUT))
        if observation.hour == 0:
            ne_crop_tiles_at_day_start[observation.day] = len(crop_positions & set(NE_CROP_POSITIONS))

    final = env.steps[-1][0]
    assert peak_crop_tiles == len(CROP_LAYOUT) + len(NE_CROP_POSITIONS), peak_crop_tiles
    assert peak_ne_crop_tiles == len(NE_CROP_POSITIONS), peak_ne_crop_tiles
    assert nw_crop_tiles_at_day_start == {
        day: len(CROP_LAYOUT) for day in range(1, 26)
    }, nw_crop_tiles_at_day_start
    expansion_day = min(agent.expansion_choices)
    assert all(
        ne_crop_tiles_at_day_start[day] == len(NE_CROP_POSITIONS)
        for day in range(expansion_day + 1, 30)
    ), ne_crop_tiles_at_day_start
    assert all(peak_assets[animal] == count for animal, count in Counter(ANIMAL_LAYOUT.values()).items())
    assert final.observation.farms[0]["unlocked_quadrants"] == ["NW", "NE"]
    max_hands = max(len(step[0].observation.farms[0].get("hands", [])) for step in env.steps[1:])
    assert max_hands <= 16

    if args.replay:
        args.replay.parent.mkdir(parents=True, exist_ok=True)
        with args.replay.open("w", encoding="utf-8") as replay_file:
            json.dump(env.toJSON(), replay_file)
    if args.html:
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(env.render(mode="html"), encoding="utf-8")

    choices = {day: selected for day, (selected, _projections) in agent.crop_choices.items()}
    expansion_choices = {
        day: selected for day, (selected, _projections) in agent.expansion_choices.items()
    }
    print(
        f"PASS: crop_choices={choices} expansion_choices={expansion_choices} "
        f"peak_crop_tiles={peak_crop_tiles} "
        f"max_hands={max_hands} reward={final.reward}"
    )
    if args.replay:
        print(f"Replay: {args.replay.resolve()} ({args.replay.stat().st_size} bytes)")
    if args.html:
        print(f"HTML: {args.html.resolve()} ({args.html.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
