"""Play the production agent against a selectable opponent and save a replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from importlib.metadata import version
from pathlib import Path

from kaggle_environments import make

from agents.expansion_agent import make_agent
from agents.opening_book import OPENING_VERSIONS
from experiments.crop_schedules import pass_agent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "replays"
END_DAY = 30


def configuration(seed):
    return {
        "episodeSteps": END_DAY * 24,
        "boardSize": 10,
        "startingMoney": 3000.0,
        "maxMarketOrdersPerTurn": 10,
        "turnsPerDay": 24,
        "shedCapacity": 100,
        "weedSpawnChance": 0.0,
        "townShopUnlockInterval": 3,
        "townShopSellInterval": 4,
        "townCenterSellInterval": 24,
        "seed": seed,
    }


def notebook_agent(notebook: Path):
    """Load the agent written by a notebook's single ``%%writefile main.py`` cell."""
    document = json.loads(notebook.read_text(encoding="utf-8"))
    sources = []
    for cell in document.get("cells", []):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and source.startswith("%%writefile main.py\n"):
            sources.append(source)

    if len(sources) != 1:
        raise ValueError("expected exactly one %%writefile main.py code cell")

    source = sources[0].split("\n", 1)[1]
    namespace = {"__name__": "notebook_opponent"}
    exec(compile(source, f"{notebook}:main.py", "exec"), namespace)

    agent = namespace.get("agent")
    if not callable(agent):
        raise ValueError("main.py cell must define a callable named 'agent'")
    return agent, hashlib.sha256(source.encode()).hexdigest()


def resolve_opponent(spec: str):
    """Resolve ``pass``, ``random``, a Python agent, or a Kaggle notebook."""
    if spec == "pass":
        return pass_agent, "pass", {}
    if spec == "random":
        return "random", "random", {}

    path = Path(spec).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    if path.suffix == ".ipynb":
        agent, source_hash = notebook_agent(path)
        return agent, path.stem, {
            "opponent_file": str(path),
            "opponent_sha256": source_hash,
        }
    if path.suffix == ".py":
        return str(path), path.stem, {
            "opponent_file": str(path),
            "opponent_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    raise ValueError("opponent must be 'pass', 'random', or a .py/.ipynb file")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return slug or "opponent"


def run(
    opponent="pass", seed=1, output_dir=None, opening_version="classic",
    planner_version="concentration",
):
    opponent_agent, opponent_name, opponent_meta = resolve_opponent(str(opponent))
    current = make_agent(
        END_DAY - 1, seed=seed, opening_version=opening_version,
        planner_version=planner_version,
    )

    env = make("kaggriculture", configuration=configuration(seed), debug=False)
    env.run([current, opponent_agent])

    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    final = env.steps[-1]
    stem = f"current_vs_{_slug(opponent_name)}_seed{seed}"
    if opening_version != "classic":
        stem += f"_{opening_version}"
    if planner_version != "concentration":
        stem += f"_{planner_version}"
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
        "opening_version": opening_version,
        "planner_version": planner_version,
        "opponent": opponent_name,
        "current_cash": current_cash,
        "opponent_cash": opponent_cash,
        "margin": current_cash - opponent_cash,
        "statuses": [state.status for state in final],
        "frames": len(env.steps),
        "engine": version("kaggle-environments"),
        "configuration": dict(env.configuration),
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
    parser.add_argument("--opening", choices=tuple(OPENING_VERSIONS), default="classic")
    parser.add_argument(
        "--planner", choices=("concentration", "mirror_v2"),
        default="concentration",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(
        args.opponent, args.seed, args.output_dir,
        opening_version=args.opening, planner_version=args.planner,
    )


if __name__ == "__main__":
    main()
