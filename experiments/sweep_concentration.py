"""Sweep concentration planner penalty against a fixed opponent.

This experiment deliberately changes only planner.TARGET_CONCENTRATION_PENALTY
between matches, keeping the opening, planner version, seed, and opponent fixed.
It reports score plus labor/route diagnostics so a lower concentration score is
not accepted merely because it happens to win one match.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as game

import agents.planner as planner
from agents.expansion_agent import make_agent
from experiments.play_match import END_DAY, configuration, resolve_opponent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPPONENT = ROOT / "public" / "A.ipynb"
DEFAULT_PENALTIES = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)


def _producer_counts(obs, player=0):
    counts = Counter()
    for row in obs["farms"][player]["tiles"]:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            name = tile.get("animal") or (
                tile.get("crop") if tile.get("kind") == "PLANT" else None
            )
            if name:
                counts[name] += 1
    return dict(sorted(counts.items()))


def _diagnostics(env, player=0):
    max_hands_by_day = defaultdict(int)
    hires_by_day = defaultdict(int)
    drops = 0

    for states in env.steps:
        state = states[player]
        obs = state.observation
        day = int(obs["day"])
        farm = obs["farms"][player]
        max_hands_by_day[day] = max(max_hands_by_day[day], len(farm["hands"]))
        hires_by_day[day] = max(hires_by_day[day], int(farm.get("hires_today", 0)))

        action = state.action or {}
        operations = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
        drops += sum(bool(op) and op[0] == "DROP" for op in operations)

    hire_cost = sum(
        sum(game._hire_cost(index) for index in range(count))
        for count in hires_by_day.values()
    )
    max_hands = max(max_hands_by_day.values(), default=0)
    days_14_plus = sum(count >= 14 for count in max_hands_by_day.values())
    days_16 = sum(count >= 16 for count in max_hands_by_day.values())

    final_obs = env.steps[-1][player].observation
    return {
        "hire_cost": hire_cost,
        "max_hands": max_hands,
        "days_14_plus_hands": days_14_plus,
        "days_16_hands": days_16,
        "drops": drops,
        "final_producers": _producer_counts(final_obs, player),
    }


def run_one(opponent, seed, penalty):
    opponent_agent, opponent_name, _meta = resolve_opponent(str(opponent))
    planner.TARGET_CONCENTRATION_PENALTY = float(penalty)
    current = make_agent(
        END_DAY - 1,
        seed=seed,
        opening_version="melon_v2",
        planner_version="concentration",
    )
    env = make("kaggriculture", configuration=configuration(seed), debug=False)
    env.run([current, opponent_agent])

    final = env.steps[-1]
    row = {
        "seed": seed,
        "opponent": opponent_name,
        "penalty": float(penalty),
        "current_cash": float(final[0].reward),
        "opponent_cash": float(final[1].reward),
        "margin": float(final[0].reward) - float(final[1].reward),
        "statuses": [state.status for state in final],
        **_diagnostics(env, 0),
    }
    if row["statuses"] != ["DONE", "DONE"]:
        raise RuntimeError(f"match failed: {row}")
    print("CONCENTRATION_RESULT " + json.dumps(row, sort_keys=True), flush=True)
    return row


def run_sweep(opponent=DEFAULT_OPPONENT, seeds=(1,), penalties=DEFAULT_PENALTIES):
    rows = [
        run_one(opponent, seed, penalty)
        for seed in seeds
        for penalty in penalties
    ]
    summaries = []
    for penalty in penalties:
        group = [row for row in rows if row["penalty"] == float(penalty)]
        summaries.append({
            "penalty": float(penalty),
            "matches": len(group),
            "mean_cash": mean(row["current_cash"] for row in group),
            "mean_margin": mean(row["margin"] for row in group),
            "mean_hire_cost": mean(row["hire_cost"] for row in group),
            "mean_days_14_plus_hands": mean(row["days_14_plus_hands"] for row in group),
            "mean_days_16_hands": mean(row["days_16_hands"] for row in group),
            "mean_drops": mean(row["drops"] for row in group),
        })
    summaries.sort(key=lambda row: (row["mean_cash"], row["mean_margin"]), reverse=True)
    print("CONCENTRATION_SUMMARY " + json.dumps(summaries, sort_keys=True), flush=True)
    return rows, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", type=Path, default=DEFAULT_OPPONENT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--penalties", type=float, nargs="+", default=list(DEFAULT_PENALTIES))
    args = parser.parse_args()
    run_sweep(args.opponent, tuple(args.seeds), tuple(args.penalties))


if __name__ == "__main__":
    main()
