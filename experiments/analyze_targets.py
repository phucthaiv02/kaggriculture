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
            if (
                sum(units.values())
                if hasattr(units, "values")
                else bool(units)
            )
        ),
        default=None,
    )


def _tracer(records):
    def traced(
        market,
        baseline,
        candidates,
        counts,
        labor=None,
        position=(4, 4),
    ):
        candidates = list(candidates)
        results = planner.evaluate_targets(
            market, baseline, candidates, labor, position
        )
        profitable = [
            result for result in results if result.profit > 0
        ]
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
                    * market.price(
                        product,
                        market.inventory.get(product, 0),
                    )
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
                            baseline,
                            result.output,
                            result.capital_cost,
                        )
                        - result.labor_cost
                    ),
                    # This is a forecast with visible rival supply included,
                    # not realized profit from the replay.
                    "forecast_profit": result.profit,
                    "first_sale_day": _first_flow_day(
                        result.output, "sales"
                    ),
                    "first_input_day": _first_flow_day(
                        result.output, "inputs"
                    ),
                    "baseline_sales": dict(
                        sum(baseline.sales.values(), Counter())
                    ),
                    "rival_sales": dict(
                        sum(
                            market.external.sales.values(),
                            Counter(),
                        )
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
                            if (
                                market.day
                                + planner._first_yield_age(name)
                                > market.end_day
                            )
                            else "tile_candidate_restriction"
                        ),
                    }
                    for name in (*planner.CROPS, *planner.ANIMALS)
                    if name
                    not in {
                        result.choice[0] for result in results
                    }
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
                            sum(
                                result.output.sales.values(),
                                Counter(),
                            )
                        ),
                        "inputs": dict(
                            sum(
                                result.output.inputs.values(),
                                Counter(),
                            )
                        ),
                    }
                    for result in results
                ],
            }
        )
        return (
            (best.choice, best.output)
            if best
            else (None, None)
        )

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
                            "planted_day": tile.get(
                                "planted_day"
                            ),
                            "placed_day": tile.get(
                                "placed_day"
                            ),
                        }
                    )

        previous = current
        if obs.hour == 0:
            daily.append(
                {"day": obs.day, "counts": dict(counts)}
            )

    return plantings, daily


def _step(item):
    return int(item["day"]) * 24 + int(item["hour"])


def _matching_planting(events, selected, start, stop):
    return next(
        (
            event
            for event in events
            if event["name"] == selected
            and start <= _step(event) < stop
        ),
        None,
    )


def _annotate_materialization(records, plantings):
    """Classify decisions without matching through later target changes.

    The old analyzer searched arbitrarily far into the replay for the same
    product at the same position. That made a target selected on day 8 look
    "delayed 12 days" even when it was replaced by another target on day 9
    and only happened to return on day 20.

    Decision-level status:
      - same_day / delayed: materialized before the next decision at this tile
      - continued: same target was selected again before materialization
      - superseded: a different target replaced it first
      - not_materialized: final target episode never materialized
      - no_target: planner deliberately selected nothing

    Episode-level counts collapse consecutive identical decisions so a target
    held across several mornings is measured once from its first decision.
    """
    events_by_position = {}
    for event in plantings:
        events_by_position.setdefault(
            tuple(event["position"]), []
        ).append(event)
    for events in events_by_position.values():
        events.sort(key=_step)

    record_indices = {}
    for index, record in enumerate(records):
        record_indices.setdefault(
            tuple(record["position"]), []
        ).append(index)
    for indices in record_indices.values():
        indices.sort(key=lambda i: _step(records[i]))

    decision_counts = Counter()
    episode_counts = Counter()
    episodes = []

    for position, indices in record_indices.items():
        events = events_by_position.get(position, [])

        # Individual decisions: stop matching at the next planner decision.
        for offset, index in enumerate(indices):
            record = records[index]
            selected = record["selected"]
            if not selected:
                record["materialization"] = {
                    "status": "no_target"
                }
                decision_counts["no_target"] += 1
                continue

            start = _step(record)
            next_index = (
                indices[offset + 1]
                if offset + 1 < len(indices)
                else None
            )
            stop = (
                _step(records[next_index])
                if next_index is not None
                else 10**12
            )
            match = _matching_planting(
                events, selected, start, stop
            )

            if match is not None:
                delay_days = match["day"] - record["day"]
                status = (
                    "same_day"
                    if delay_days == 0
                    else "delayed"
                )
                record["materialization"] = {
                    "status": status,
                    "day": match["day"],
                    "hour": match["hour"],
                    "delay_days": delay_days,
                }
            elif next_index is None:
                status = "not_materialized"
                record["materialization"] = {
                    "status": status
                }
            else:
                next_record = records[next_index]
                same_target = (
                    next_record["selected"] == selected
                    and next_record.get("fertilize")
                    == record.get("fertilize")
                )
                if same_target:
                    status = "continued"
                    record["materialization"] = {
                        "status": status,
                        "continued_to_day": next_record["day"],
                        "continued_to_hour": next_record["hour"],
                    }
                else:
                    status = "superseded"
                    record["materialization"] = {
                        "status": status,
                        "next_selected": next_record[
                            "selected"
                        ],
                        "next_fertilize": next_record.get(
                            "fertilize"
                        ),
                        "next_day": next_record["day"],
                        "next_hour": next_record["hour"],
                    }

            decision_counts[status] += 1

        # Target episodes: consecutive identical decisions are one intent.
        offset = 0
        while offset < len(indices):
            first_index = indices[offset]
            first = records[first_index]
            selected = first["selected"]
            fertilize = first.get("fertilize")
            end_offset = offset + 1

            while end_offset < len(indices):
                candidate = records[indices[end_offset]]
                if (
                    candidate["selected"],
                    candidate.get("fertilize"),
                ) != (selected, fertilize):
                    break
                end_offset += 1

            next_index = (
                indices[end_offset]
                if end_offset < len(indices)
                else None
            )
            stop = (
                _step(records[next_index])
                if next_index is not None
                else 10**12
            )
            start = _step(first)

            episode = {
                "position": list(position),
                "selected": selected,
                "fertilize": fertilize,
                "start_day": first["day"],
                "start_hour": first["hour"],
                "decision_count": end_offset - offset,
            }

            if not selected:
                status = "no_target"
            else:
                match = _matching_planting(
                    events, selected, start, stop
                )
                if match is None:
                    status = "not_materialized"
                else:
                    delay_days = (
                        match["day"] - first["day"]
                    )
                    status = (
                        "same_day"
                        if delay_days == 0
                        else "delayed"
                    )
                    episode.update(
                        {
                            "materialized_day": match[
                                "day"
                            ],
                            "materialized_hour": match[
                                "hour"
                            ],
                            "delay_days": delay_days,
                        }
                    )

            episode["status"] = status
            episode_counts[status] += 1
            episodes.append(episode)
            offset = end_offset

    episodes.sort(
        key=lambda episode: (
            episode["start_day"],
            episode["start_hour"],
            episode["position"][1],
            episode["position"][0],
        )
    )
    return decision_counts, episode_counts, episodes


def run(opponent_spec, seed=1, output_dir=None):
    output_dir = Path(
        output_dir or DEFAULT_OUTPUT_DIR
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    original = planner._choose
    opponent, opponent_name, _ = resolve_opponent(
        str(opponent_spec)
    )
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
    (
        decision_materialization,
        episode_materialization,
        episodes,
    ) = _annotate_materialization(records, plantings)

    final_state = env.steps[-1]
    current_cash = float(final_state[0].reward)
    opponent_cash = float(final_state[1].reward)
    final_counts = Counter(
        tile.get("animal") or tile.get("crop")
        for row in final_state[0].observation.farms[0][
            "tiles"
        ]
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
            Counter(
                record["selected"] or "NONE"
                for record in records
            )
        ),
        "materialization": {
            "decisions": dict(decision_materialization),
            "episodes": dict(episode_materialization),
        },
        "plantings": dict(
            Counter(event["name"] for event in plantings)
        ),
        "final": dict(final_counts),
        "daily": daily,
        "planting_events": plantings,
    }

    stem = (
        f"target_analysis_{_slug(opponent_name)}_seed{seed}"
    )
    scores_path = output_dir / f"{stem}.json"
    breakdown_path = (
        output_dir / f"{stem}.breakdown.json"
    )
    summary_path = output_dir / f"{stem}.summary.json"
    episodes_path = output_dir / f"{stem}.episodes.json"

    scores_path.write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
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
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    episodes_path.write_text(
        json.dumps(episodes, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Scores: {scores_path}")
    print(f"Breakdown: {breakdown_path}")
    print(f"Episodes: {episodes_path}")
    print(f"Summary: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "opponent",
        help=(
            "pass, random, or a path to a .py/.ipynb "
            "opponent"
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()
    run(args.opponent, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
