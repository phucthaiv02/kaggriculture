"""Run the v2 agent without changing the production replay script."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from kaggle_environments import make

from agents.agent_v2 import make_agent
from experiments.play_match import ROOT, configuration, resolve_opponent

DEFAULT_OUTPUT_DIR = ROOT / "replays"
END_DAY = 30


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in value).strip("_") or "opponent"


def run(opponent="pass", seed=1, output_dir=None):
    opponent_agent, opponent_name, opponent_meta = resolve_opponent(str(opponent))
    current = make_agent(END_DAY - 1, seed=seed)
    env = make("kaggriculture", configuration=configuration(seed), debug=False)
    env.run([current, opponent_agent])

    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / f"v2_vs_{_slug(opponent_name)}_seed{seed}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    final = env.steps[-1]
    result = {
        "seed": seed,
        "opponent": opponent_name,
        "current_cash": float(final[0].reward),
        "opponent_cash": float(final[1].reward),
        "margin": float(final[0].reward) - float(final[1].reward),
        "statuses": [state.status for state in final],
        "frames": len(env.steps),
        "engine": version("kaggle-environments"),
        "configuration": dict(env.configuration),
        **opponent_meta,
    }
    (run_dir / "replay.json").write_text(json.dumps(env.toJSON()), encoding="utf-8")
    (run_dir / "replay.html").write_text(env.render(mode="html", width=1200, height=800), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if result["statuses"] != ["DONE", "DONE"]:
        raise RuntimeError(f"game did not finish normally: {result['statuses']}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", default="pass")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.opponent, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
