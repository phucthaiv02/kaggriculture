"""Trace planner scores and compare target decisions with real materialization."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from copy import copy
from pathlib import Path

from kaggle_environments import make

from agents import planner
from agents.expansion_agent import make_agent
from agents.forecast import Production
from experiments.play_match import configuration, resolve_opponent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "replays"


def _slug(value):
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return slug or "opponent"


def _first_flow_day(flow, field):
    values = getattr(flow, field)
    return min(
        (
            day
            for day, units in values.items()
            if (sum(units.values()) if hasattr(units, "values") else bool(units))
        ),
        default=None,
    )


def _tracer(records):
    def traced(market, baseline, candidates, counts, labor=None, position=(4, 4)):
        candidates = list(candidates)
        results = planner.evaluate_targets(
            market, baseline, candidates, labor, position
        )
        profitable = [result for result in results if result.profit > 0]
        best = max(
            profitable,
            key=lambda result: (
                result.profit,
                -counts[result.choice[0]],
                result.choice,
            ),
            default=None,
        )

        no_rival = copy(market)
        no_rival.external = Production()
        detail = []
        for result in results:
            sales = sum(result.output.sales.values(), Counter())
            inputs = sum(result.output.inputs.values(), Counter())
            static = (
                sum(
                    (sales[product] - inputs[product])
                    * market.price(product, market.inventory.get(product, 0))
                    for product in sales.keys() | inputs.keys()
                )
                - result.capital_cost
                - result.labor_cost
            )
            detail.append(
                {
                    "name": result.choice[0],
                    "fertilize": result.choice[1],
                    "static_profit": static,
                    "solo_profit": (
                        no_rival.value(result.output)
                        - result.capital_cost
                        - result.labor_cost
                    ),
                    "no_rival_profit": (
                        no_rival.marginal_profit(
                            baseline, result.output, result.capital_cost
                        )
                        - result.labor_cost
                    ),
                    # This is still a forecast, just with visible rival supply
                    # included. The old "actual_profit" label was misleading.
                    "forecast_profit": result.profit,
                    "first_sale_day": _first_flow_day(result.output, "sales"),
                    "first_input_day": _first_flow_day(result.output, "inputs"),
                    "baseline_sales": dict(
                        sum(baseline.sales.values(), Counter())
                    ),
                    "rival_sales": dict(
                        sum(market.external.sales.values(), Counter())
                    ),
                }
            )

        records.append(
            {
                "day": market.day,
                "hour": market.hour,
                "end_day": market.end_day,
                "position": list(position),
                "selected": best.choice[0] if best else None,
                "fertilize": best.choice[1] if best else None,
                "breakdown": detail,
                "inventory": dict(market.inventory),
                "shops": dict(market.shop_demand),
                "excluded": [
                    {
                        "name": name,
                        "reason": (
                            "first_yield_after_horizon"
                            if market.day + planner._first_yield_age(name)
                            > market.end_day
                            else "tile_candidate_restriction"
                        ),
                    }
                    for name in (*planner.CROPS, *planner.ANIMALS)
                    if name not in {result.choice[0] for result in results}
                ],
                "scores": [
                    {
                        "name": result.choice[0],
                        "fertilize": result.choice[1],
                        "cash": result.market_cash,
                        "capital": result.capital_cost,
                        "labor": result.labor_cost,
                        "profit": result.profit,
                        "first_sale_day": _first_flow_day(
                            result.output, "sales"
                        ),
                        "sales": dict(
                            sum(result.output.sales.values(), Counter())
                        ),
                        "inputs": dict(
                            sum(result.output.inputs.values(), Counter())
                        ),
                    }
                    for result in results
                ],
            }
        )
        return (best.choice, best.output) if best else (None, None)

    return traced


def _planting_trace(env):
    plantings = []
    daily = []
    previous = {}
    for step in env.steps:
        obs = step[0].observation
        current = {}
        counts = Counter()
        for y, row in enumerate(obs.farms[0]["tiles"]):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict):
                    continue
                name = tile.get("animal") or tile.get("crop")
                if not name:
                    continue
                counts[name] += 1
                identity = (
                    name,
                    tile.get("planted_day"),
                    tile.get("placed_day"),
                )
                current[x, y] = identity
                if previous.get((x, y)) != identity:
                    plantings.append(
                        {
                            "day": obs.day,
                            "hour": obs.hour,
                            "position": [x, y],
                            "name": name,
                            "planted_day": tile.get("planted_day"),
                            "placed_day": tile.get("placed_day"),
                        }
                    )
        previous = current
        if obs.hour == 0:
            daily.append({"day": obs.day, "counts": dict(counts)})
    return plantings, daily


def _annotate_materialization(records, plantings):
    by_position = {}
    for event in plantings:
        by_position.setdefault(tuple(event["position"]), []).append(event)

    status_counts = Counter()
    for record in records:
        selected = record["selected"]
        if not selected:
            record["materialization"] = {"status": "no_target"}
            status_counts["no_target"] += 1
            continue

        decision_step = record["day"] * 24 + record["hour"]
        match = next(
            (
                event
                for event in by_position.get(tuple(record["position"]), ())
                if event["name"] == selected
                and event["day"] * 24 + event["hour"] >= decision_step
            ),
            None,
        )
        if match is None:
            record["materialization"] = {"status": "not_materialized"}
            status_counts["not_materialized"] += 1
            continue

        delay_days = match["day"] - record["day"]
        status = "same_day" if delay_days == 0 else "delayed"
        record["materialization"] = {
            "status": status,
            "day": match["day"],
            "hour": match["hour"],
            "delay_days": delay_days,
        }
        status_counts[status] += 1
    return status_counts


def run(opponent_spec, seed=1, output_dir=None):
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    original = planner._choose
    opponent, opponent_name, _ = resolve_opponent(str(opponent_spec))
    env = make(
        "kaggriculture",
        configuration=configuration(seed),
        debug=False,
    )
    planner._choose = _tracer(records)
    try:
        env.run([make_agent(29, seed=seed), opponent])
    finally:
        planner._choose = original

    plantings, daily = _planting_trace(env)
    materialization = _annotate_materialization(records, plantings)

    final_state = env.steps[-1]
    current_cash = float(final_state[0].reward)
    opponent_cash = float(final_state[1].reward)
    final_counts = Counter(
        tile.get("animal") or tile.get("crop")
        for row in final_state[0].observation.farms[0]["tiles"]
        for tile in row
        if isinstance(tile, dict)
        and (tile.get("animal") or tile.get("crop"))
    )

    summary = {
        "seed": seed,
        "opponent": opponent_name,
        "current_cash": current_cash,
        "opponent_cash": opponent_cash,
        "margin": current_cash - opponent_cash,
        "picks": dict(
            Counter(record["selected"] or "NONE" for record in records)
        ),
        "materialization": dict(materialization),
        "plantings": dict(Counter(event["name"] for event in plantings)),
        "final": dict(final_counts),
        "daily": daily,
        "planting_events": plantings,
    }

    stem = f"target_analysis_{_slug(opponent_name)}_seed{seed}"
    scores_path = output_dir / f"{stem}.json"
    breakdown_path = output_dir / f"{stem}.breakdown.json"
    summary_path = output_dir / f"{stem}.summary.json"

    scores_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    breakdown_path.write_text(
        json.dumps(
            [
                {
                    key: record[key]
                    for key in (
                        "day",
                        "hour",
                        "position",
                        "selected",
                        "fertilize",
                        "breakdown",
                        "materialization",
                    )
                }
                for record in records
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Scores: {scores_path}")
    print(f"Breakdown: {breakdown_path}")
    print(f"Summary: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "opponent",
        help="pass, random, or a path to a .py/.ipynb opponent",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.opponent, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
