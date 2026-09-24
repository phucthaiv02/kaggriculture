"""Reproducible benchmark for agent-v5 against the pinned agent-v3 baseline.

This module deliberately does not change planning behavior.  It records the
metrics we care about before agent-v5 experiments start:

* final cash / opponent margin on fixed A.ipynb seeds,
* plant -> WEED and placed-animal -> empty-structure lifecycle failures,
* producer replacement gaps after a producer disappears,
* planner decision/candidate volume, and
* decision latency against the 0.75 second budget.

The GitHub workflow runs this exact file once from agent-v5 and once from a
worktree at the pinned agent-v3 commit, so later planner changes are compared
against the same harness and opponent bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kaggle_environments import make

from agents.expansion_agent import make_agent
from experiments.benchmark_agent import benchmark as latency_benchmark
from experiments.play_match import configuration, resolve_opponent


DEFAULT_SEEDS = (1, 7)
DECISION_LIMIT = 0.75


def _is_plant(tile):
    return isinstance(tile, dict) and tile.get("kind") == "PLANT"


def _is_weed(tile):
    return tile == "WEED" or (isinstance(tile, dict) and tile.get("kind") == "WEED")


def _animal(tile):
    return tile.get("animal") if isinstance(tile, dict) else None


def _is_producer(tile):
    return _is_plant(tile) or bool(_animal(tile))


def replay_health(replay, player=0):
    """Return behavior metrics observable from the replay only.

    A replacement gap starts when a live producer disappears and ends when a
    producer appears on that tile again.  It is deliberately descriptive, not
    a correctness gate: an idle tile can be economically intentional near the
    end of the season.  The distribution lets v5 be compared with the exact
    same baseline instead of inventing a target-specific threshold.
    """
    steps = replay.get("steps", [])
    if len(steps) < 2:
        return {
            "plant_to_weed": 0,
            "animal_escapes": 0,
            "turnover_gap_events": 0,
            "turnover_gap_mean_steps": 0.0,
            "turnover_gap_max_steps": 0,
            "open_turnover_gaps_at_end": 0,
        }

    plant_to_weed = 0
    animal_escapes = 0
    open_gaps = {}
    completed_gaps = []

    for frame in range(1, len(steps)):
        before = steps[frame - 1][0]["observation"]["farms"][player]["tiles"]
        after = steps[frame][0]["observation"]["farms"][player]["tiles"]
        for y, row in enumerate(before):
            for x, old in enumerate(row):
                new = after[y][x]
                pos = (x, y)

                if _is_plant(old) and _is_weed(new):
                    plant_to_weed += 1

                old_animal = _animal(old)
                if old_animal and not _animal(new):
                    # DIG cannot remove a structure containing an animal in the
                    # engine, so this transition is an escape/loss, not culling.
                    animal_escapes += 1

                if pos in open_gaps:
                    if _is_producer(new):
                        completed_gaps.append(open_gaps.pop(pos))
                    else:
                        open_gaps[pos] += 1
                elif _is_producer(old) and not _is_producer(new):
                    # Do not call a PLANT -> WEED failure a normal turnover;
                    # it is already accounted for separately above.
                    if not _is_weed(new):
                        open_gaps[pos] = 1

    gaps = completed_gaps + list(open_gaps.values())
    return {
        "plant_to_weed": plant_to_weed,
        "animal_escapes": animal_escapes,
        "turnover_gap_events": len(gaps),
        "turnover_gap_mean_steps": (sum(gaps) / len(gaps)) if gaps else 0.0,
        "turnover_gap_max_steps": max(gaps, default=0),
        "open_turnover_gaps_at_end": len(open_gaps),
    }


def _planner_metrics(decisions):
    candidate_counts = [len(row.get("candidates", ())) for row in decisions]
    return {
        "decisions": len(decisions),
        "candidates_evaluated": sum(candidate_counts),
        "max_candidates_per_decision": max(candidate_counts, default=0),
        "no_profitable_decisions": sum(
            row.get("reason") == "no_profitable_candidate" for row in decisions
        ),
    }


def play(opponent, seed, replay_dir=None):
    opponent_agent, opponent_name, opponent_meta = resolve_opponent(str(opponent))
    decisions = []
    current = make_agent(29, seed=seed, decision_log=decisions)
    cfg = dict(configuration(seed), actTimeout=DECISION_LIMIT)
    env = make("kaggriculture", configuration=cfg, debug=False)
    env.run([current, opponent_agent])

    replay = env.toJSON()
    final = env.steps[-1]
    result = {
        "seed": seed,
        "opponent": opponent_name,
        "current_cash": float(final[0].reward),
        "opponent_cash": float(final[1].reward),
        "margin": float(final[0].reward) - float(final[1].reward),
        "statuses": [state.status for state in final],
        "frames": len(env.steps),
        "health": replay_health(replay, 0),
        "planner": _planner_metrics(decisions),
        **opponent_meta,
    }

    if replay_dir is not None:
        replay_dir = Path(replay_dir)
        replay_dir.mkdir(parents=True, exist_ok=True)
        path = replay_dir / f"seed-{seed}.json"
        path.write_text(json.dumps(replay), encoding="utf-8")
        result["replay"] = str(path)
    return result


def latency(seed=1):
    timings = latency_benchmark(seed=seed, opponent="self", timeout=DECISION_LIMIT)
    values = sorted(row[1] for row in timings)
    p95 = values[int((len(values) - 1) * .95)] if values else 0.0
    worst = max(timings, key=lambda row: row[1], default=(0, 0.0, 0, 0))
    by_tiles = {}
    for tiles in sorted({row[0] for row in timings}):
        rows = [row for row in timings if row[0] == tiles]
        ordered = sorted(row[1] for row in rows)
        local_worst = max(rows, key=lambda row: row[1])
        by_tiles[str(tiles)] = {
            "actions": len(rows),
            "p95_seconds": ordered[int((len(ordered) - 1) * .95)],
            "max_seconds": local_worst[1],
            "max_day": local_worst[2],
            "max_hour": local_worst[3],
            "over_limit": sum(row[1] > DECISION_LIMIT for row in rows),
        }
    return {
        "limit_seconds": DECISION_LIMIT,
        "actions": len(timings),
        "p95_seconds": p95,
        "max_seconds": worst[1],
        "max_day": worst[2],
        "max_hour": worst[3],
        "over_limit": sum(row[1] > DECISION_LIMIT for row in timings),
        "by_tiles": by_tiles,
    }


def run(opponent, seeds=DEFAULT_SEEDS, output=None, replay_dir=None):
    opponent_path = Path(opponent).expanduser().resolve() if str(opponent) not in {"pass", "random"} else None
    report = {
        "schema": 1,
        "decision_limit_seconds": DECISION_LIMIT,
        "seeds": list(seeds),
        "opponent": str(opponent_path or opponent),
        "opponent_sha256": (
            hashlib.sha256(opponent_path.read_bytes()).hexdigest()
            if opponent_path is not None else None
        ),
        "matches": [play(opponent_path or opponent, seed, replay_dir) for seed in seeds],
        "latency": latency(seed=seeds[0]),
    }
    matches = report["matches"]
    report["aggregate"] = {
        "average_cash": sum(row["current_cash"] for row in matches) / len(matches),
        "average_margin": sum(row["margin"] for row in matches) / len(matches),
        "plant_to_weed": sum(row["health"]["plant_to_weed"] for row in matches),
        "animal_escapes": sum(row["health"]["animal_escapes"] for row in matches),
        "candidates_evaluated": sum(row["planner"]["candidates_evaluated"] for row in matches),
    }

    text = json.dumps(report, indent=2)
    print(text, flush=True)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", default="public/A.ipynb")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay-dir", type=Path)
    args = parser.parse_args()
    report = run(args.opponent, tuple(args.seeds), args.output, args.replay_dir)

    bad = [row for row in report["matches"] if row["statuses"] != ["DONE", "DONE"]]
    if bad:
        raise SystemExit(f"{len(bad)} benchmark matches did not finish normally")
    if report["latency"]["over_limit"]:
        raise SystemExit(
            f"{report['latency']['over_limit']} decisions exceeded "
            f"{DECISION_LIMIT:.2f}s"
        )


if __name__ == "__main__":
    main()
