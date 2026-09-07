"""Play the current agent against an agent embedded in a Kaggle notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from kaggle_environments import make

from agents.expansion_agent import make_agent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "replays"


def public_agent(notebook: Path):
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
    namespace = {"__name__": "public_solution_agent"}
    exec(compile(source, f"{notebook}:main.py", "exec"), namespace)

    agent = namespace.get("agent")
    if not callable(agent):
        raise ValueError("main.py cell must define a callable named 'agent'")
    return agent, hashlib.sha256(source.encode()).hexdigest()


def run(notebook, seed=1, seats=(0, 1), output_dir=None):
    notebook = Path(notebook).expanduser().resolve()
    if not notebook.is_file():
        raise FileNotFoundError(notebook)

    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seat in seats:
        opponent, source_hash = public_agent(notebook)
        current = make_agent(seed=seed)
        players = [current, opponent] if seat == 0 else [opponent, current]
        config = {
            "episodeSteps": 720,
            "townCenterSellInterval": 24,
            "farmHandCostMult": 1,
            "seed": seed,
        }
        env = make("kaggriculture", configuration=config, debug=False)
        env.run(players)

        final = env.steps[-1]
        stem = f"current_vs_public_seed{seed}_seat{seat}"
        html_path = output_dir / f"{stem}.html"
        replay_path = output_dir / f"{stem}.json"
        html_path.write_text(
            env.render(mode="html", width=1200, height=800),
            encoding="utf-8",
        )
        replay_path.write_text(json.dumps(env.toJSON()), encoding="utf-8")

        row = {
            "seed": seed,
            "current_seat": seat,
            "current_cash": final[seat].reward,
            "public_cash": final[1 - seat].reward,
            "margin": final[seat].reward - final[1 - seat].reward,
            "statuses": [state.status for state in final],
            "frames": len(env.steps),
            "engine": version("kaggle-environments"),
            "public_source_sha256": source_hash,
            "configuration": dict(env.configuration),
            "html": str(html_path),
            "replay_json": str(replay_path),
        }
        results.append(row)
        (output_dir / "public_match_results.json").write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(row), flush=True)

        if row["statuses"] != ["DONE", "DONE"]:
            raise RuntimeError(f"game did not finish normally: {row['statuses']}")

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path, help="Notebook containing %%writefile main.py")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seats", type=int, nargs="+", choices=(0, 1), default=[0, 1])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.notebook, args.seed, args.seats, args.output_dir)


if __name__ == "__main__":
    main()
