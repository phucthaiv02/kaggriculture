"""Play the v2 optimizer agent and save replay artifacts."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from kaggle_environments import make

from agents.optimizer_agent import make_agent
from experiments.play_match import (
    DEFAULT_OUTPUT_DIR,
    END_DAY,
    _slug,
    configuration,
    resolve_opponent,
)


def run(opponent="pass", seed=1, output_dir=None):
    opponent_agent, opponent_name, opponent_meta = resolve_opponent(str(opponent))
    current = make_agent(END_DAY - 1, seed=seed)

    env = make("kaggriculture", configuration=configuration(seed), debug=False)
    env.run([current, opponent_agent])

    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    final = env.steps[-1]
    stem = f"optimizer_v2_vs_{_slug(opponent_name)}_seed{seed}"
    html_path = output_dir / f"{stem}.html"
    replay_path = output_dir / f"{stem}.json"
    result_path = output_dir / f"{stem}.result.json"

    html_path.write_text(
        env.render(mode="html", width=1200, height=800),
        encoding="utf-8",
    )
    replay_path.write_text(json.dumps(env.toJSON()), encoding="utf-8")

    current_cash = float(final[0].reward)
    opponent_cash = float(final[1].reward)
    result = {
        "seed": seed,
        "opponent": opponent_name,
        "current_cash": current_cash,
        "opponent_cash": opponent_cash,
        "margin": current_cash - opponent_cash,
        "statuses": [state.status for state in final],
        "frames": len(env.steps),
        "engine": version("kaggle-environments"),
        "configuration": dict(env.configuration),
        "invariant_failures": current.debug_state["invariant_failures"],
        "html": str(html_path),
        "replay_json": str(replay_path),
        **opponent_meta,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)

    if result["statuses"] != ["DONE", "DONE"]:
        raise RuntimeError(f"game did not finish normally: {result['statuses']}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--opponent",
        default="pass",
        help="pass, random, or a path to a .py/.ipynb agent (default: pass)",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.opponent, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
