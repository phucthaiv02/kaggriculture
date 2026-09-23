"""Round-robin leaderboard for Kaggriculture agents.

Examples:
    python -m experiments.agent_leaderboard
    python -m experiments.agent_leaderboard --public-dir path/to/public

Every pair plays the same randomly generated, persisted seed. Results are
content-addressed, so a second invocation only runs matches whose agent code or
configuration changed.
All ``.py`` and ``.ipynb`` files in ``public/`` are discovered automatically;
the production agent from ``agents/`` is always included as ``current``.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import secrets
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Callable

from kaggle_environments import make

from agents.expansion_agent import make_agent
from experiments.crop_schedules import pass_agent
from experiments.play_match import configuration, notebook_agent


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "leaderboard"
DEFAULT_PUBLIC_DIR = ROOT / "public"
CACHE_SCHEMA = 1
SEED_STATE_FILE = "run.json"


@dataclass(frozen=True)
class AgentSpec:
    label: str
    fingerprint: str
    factory: Callable[[int], object]
    source: str | None = None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _production_fingerprint() -> str:
    """Hash every production-agent module, not only its small entry point."""
    digest = hashlib.sha256()
    for path in sorted((ROOT / "agents").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resolve_agent(value: str) -> AgentSpec:
    """Resolve ``[LABEL=]current|pass|random|PATH`` into a fresh-agent factory."""
    if "=" in value:
        label, source = value.split("=", 1)
        if not label.strip():
            raise ValueError(f"empty agent label in {value!r}")
        label = label.strip()
    else:
        source = value
        label = source if source in {"current", "pass", "random"} else Path(source).stem

    if source == "current":
        return AgentSpec(label, "current:" + _production_fingerprint(),
                         lambda seed: make_agent(29, seed=seed), source)
    if source == "pass":
        return AgentSpec(label, "builtin:pass:v1", lambda seed: pass_agent, source)
    if source == "random":
        return AgentSpec(label, "builtin:random:v1", lambda seed: "random", source)

    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    content_hash = _sha256(path.read_bytes())
    if path.suffix.lower() == ".py":
        return AgentSpec(label, "python:" + content_hash, lambda seed: str(path), str(path))
    if path.suffix.lower() == ".ipynb":
        # Re-execute the notebook for every game so module-level state cannot leak.
        def load_notebook(seed):
            agent, _ = notebook_agent(path)
            return agent
        return AgentSpec(label, "notebook:" + content_hash, load_notebook, str(path))
    raise ValueError("agent must be current, pass, random, or a .py/.ipynb file")


def discover_public_agents(public_dir=DEFAULT_PUBLIC_DIR) -> list[str]:
    """Return supported agent files directly inside ``public_dir``."""
    public_dir = Path(public_dir).expanduser().resolve()
    if not public_dir.is_dir():
        raise FileNotFoundError(public_dir)
    return [str(path) for path in sorted(public_dir.iterdir(), key=lambda item: item.name)
            if path.is_file() and path.suffix.lower() in {".py", ".ipynb"}]


def _write_run_state(output_dir: Path, state: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / SEED_STATE_FILE
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare_run(output_dir: Path, agents: list[AgentSpec], requested=None):
    """Resume an interrupted run, otherwise create one with a fresh seed."""
    path = output_dir / SEED_STATE_FILE
    state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
        "schema": 2, "runs": []}
    # Migrate the original single-seed format as an interrupted first run.
    if "runs" not in state:
        state = {"schema": 2, "runs": [{
            "id": 1, "seed": int(state["seed"]), "status": "running",
            "agents": [], "results": [],
        }]}
    runs = state["runs"]
    if runs and runs[-1]["status"] == "running":
        current = runs[-1]
        if requested is not None and requested != current["seed"]:
            raise ValueError(
                f"interrupted run must resume seed {current['seed']}, not {requested}"
            )
    else:
        used = {item["seed"] for item in runs}
        seed = requested if requested is not None else secrets.randbits(128)
        while seed in used:
            seed = secrets.randbits(128)
        current = {"id": len(runs) + 1, "seed": seed, "status": "running",
                   "agents": [], "results": []}
        runs.append(current)

    roster = {agent.label: agent.fingerprint for agent in agents}
    # Agent additions join the interrupted run. Removed or edited agents have
    # stale results discarded so every retained match represents this roster.
    current["results"] = [result for result in current.get("results", [])
                          if all(roster.get(name) == fingerprint for name, fingerprint
                                 in zip(result["players"], result["fingerprints"]))]
    current["agents"] = [{"label": agent.label, "fingerprint": agent.fingerprint}
                         for agent in agents]
    _write_run_state(output_dir, state)
    return state, current


def _record_result(current: dict, result: dict) -> None:
    pair = frozenset(result["fingerprints"])
    current["results"] = [old for old in current["results"]
                          if frozenset(old["fingerprints"]) != pair]
    current["results"].append(result)


def _match_key(left: AgentSpec, right: AgentSpec, seed: int, config: dict) -> str:
    payload = {
        "schema": CACHE_SCHEMA,
        "agents": sorted((left.fingerprint, right.fingerprint)),
        "seed": seed,
        "configuration": config,
        "engine": version("kaggle-environments"),
        # Seats alternate deterministically and are therefore part of this schema.
        "seat_rule": "sorted-fingerprint-seed-parity-v1",
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return _sha256(packed)


def _seat_order(left: AgentSpec, right: AgentSpec, seed: int):
    ordered = sorted((left, right), key=lambda item: (item.fingerprint, item.label))
    return ordered if seed % 2 == 0 else ordered[::-1]


def play_cached(left: AgentSpec, right: AgentSpec, seed: int, cache_dir: Path,
                force: bool = False) -> tuple[dict, bool]:
    config = configuration(seed)
    key = _match_key(left, right, seed, config)
    cache_path = cache_dir / f"{key}.json"
    if cache_path.is_file() and not force:
        result = json.loads(cache_path.read_text(encoding="utf-8"))
        # Labels are presentation only and deliberately not part of the cache key.
        # Remap them so renaming an agent can still reuse its existing matches.
        labels = {left.fingerprint: left.label, right.fingerprint: right.label}
        result["players"] = [labels[item] for item in result["fingerprints"]]
        return result, True

    player0, player1 = _seat_order(left, right, seed)
    env = make("kaggriculture", configuration=config, debug=False)
    env.run([player0.factory(seed), player1.factory(seed)])
    final = env.steps[-1]
    statuses = [state.status for state in final]
    if statuses != ["DONE", "DONE"]:
        raise RuntimeError(f"seed {seed} did not finish normally: {statuses}")
    rewards = [float(state.reward) for state in final]
    result = {
        "schema": CACHE_SCHEMA,
        "seed": seed,
        "players": [player0.label, player1.label],
        "fingerprints": [player0.fingerprint, player1.fingerprint],
        "rewards": rewards,
        "statuses": statuses,
        "frames": len(env.steps),
        "engine": version("kaggle-environments"),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(cache_path)
    return result, False


def _process_match(left_data, right_data, seed, cache_dir, force):
    """Resolve agents inside a child process and play one isolated match."""
    left = resolve_agent(left_data["source"])
    right = resolve_agent(right_data["source"])
    left = AgentSpec(left_data["label"], left.fingerprint, left.factory, left.source)
    right = AgentSpec(right_data["label"], right.fingerprint, right.factory, right.source)
    return play_cached(left, right, seed, Path(cache_dir), force)


def standings(agents: list[AgentSpec], results: list[dict]) -> list[dict]:
    rows = {agent.label: {"agent": agent.label, "played": 0, "wins": 0,
                          "draws": 0, "losses": 0, "points": 0.0,
                          "reward": 0.0, "margin": 0.0} for agent in agents}
    for match in results:
        a, b = match["players"]
        ra, rb = match["rewards"]
        rows[a]["played"] += 1
        rows[b]["played"] += 1
        rows[a]["reward"] += ra
        rows[b]["reward"] += rb
        rows[a]["margin"] += ra - rb
        rows[b]["margin"] += rb - ra
        if ra > rb:
            outcomes = ((a, "wins", 1.0), (b, "losses", 0.0))
        elif rb > ra:
            outcomes = ((b, "wins", 1.0), (a, "losses", 0.0))
        else:
            outcomes = ((a, "draws", .5), (b, "draws", .5))
        for name, field, points in outcomes:
            rows[name][field] += 1
            rows[name]["points"] += points
    table = list(rows.values())
    for row in table:
        games = row["played"] or 1
        row["score"] = row["points"] / games
        row["average_reward"] = row.pop("reward") / games
        row["average_margin"] = row.pop("margin") / games
    return sorted(table, key=lambda row: (-row["score"], -row["average_margin"],
                                          -row["average_reward"], row["agent"]))


def render_html(table: list[dict], seed: int, cached: int, total: int,
                run_number=1, cumulative=None) -> str:
    body = []
    for rank, row in enumerate(table, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))
        classes = "podium" if rank <= 3 else ""
        if row["agent"] == "current":
            classes += " current"
        margin_class = "positive" if row["average_margin"] >= 0 else "negative"
        body.append(f'<tr class="{classes.strip()}">' + "".join((
            f'<td><span class="rank">{medal}</span></td>',
            f'<td><span class="agent-name">{html.escape(row["agent"])}</span>'
            f'{"<small>YOUR AGENT</small>" if row["agent"] == "current" else ""}</td>',
            f"<td>{row['played']}</td>", f"<td>{row['wins']}</td>",
            f"<td>{row['draws']}</td>", f"<td>{row['losses']}</td>",
            f'<td><div class="score"><span style="width:{row["score"]:.1%}"></span>'
            f'<b>{row["score"]:.1%}</b></div></td>',
            f"<td>{row['average_reward']:,.2f}</td>",
            f'<td class="{margin_class}">{row["average_margin"]:+,.2f}</td>')) + "</tr>")
    cumulative = total if cumulative is None else cumulative
    leader = html.escape(table[0]["agent"]) if table else "—"
    cache_rate = cached / total if total else 0
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Kaggriculture agent leaderboard</title><style>
:root {{ font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #17352a }}
* {{ box-sizing: border-box }}
body {{ margin: 0; min-height: 100vh; padding: 3rem 1.25rem; background:
radial-gradient(circle at 10% 0%, #f9d97688, transparent 28rem),
radial-gradient(circle at 90% 10%, #71d49c66, transparent 30rem),
linear-gradient(145deg, #f8fbf3, #e6f3e9); background-attachment: fixed }}
.wrap {{ max-width: 1180px; margin: auto }}
.hero {{ position: relative; overflow: hidden; color: white; padding: 2.4rem 2.6rem;
border-radius: 28px; background: linear-gradient(120deg, #123f32, #19714a 58%, #45a85f);
box-shadow: 0 22px 55px #18583b38 }}
.hero:after {{ content: "🌾"; position: absolute; right: 2.2rem; top: -.7rem;
font-size: 9rem; opacity: .16; transform: rotate(8deg) }}
.eyebrow {{ margin: 0 0 .55rem; color: #c9f4d5; font-size: .78rem; font-weight: 800;
letter-spacing: .16em; text-transform: uppercase }}
h1 {{ margin: 0; font-size: clamp(2rem, 5vw, 3.8rem); letter-spacing: -.055em; line-height: 1 }}
.seed {{ margin: 1rem 0 0; color: #d8efe0; font-family: ui-monospace, monospace;
font-size: .76rem; overflow-wrap: anywhere }}
.cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.2rem 0 }}
.card {{ padding: 1.15rem 1.3rem; border: 1px solid #ffffffaa; border-radius: 18px;
background: #ffffffbf; backdrop-filter: blur(12px); box-shadow: 0 8px 25px #315f4420 }}
.card span {{ display: block; color: #678076; font-size: .72rem; font-weight: 800;
letter-spacing: .08em; text-transform: uppercase }}
.card strong {{ display: block; margin-top: .3rem; color: #173f30; font-size: 1.45rem }}
.table-shell {{ overflow: hidden; border: 1px solid #fff; border-radius: 22px;
background: #ffffffd9; backdrop-filter: blur(14px); box-shadow: 0 18px 50px #315f4426 }}
table {{ border-collapse: collapse; width: 100% }}
th,td {{ padding: .95rem 1rem; border-bottom: 1px solid #dce9df; text-align: right;
white-space: nowrap }}
th {{ color: #668075; background: #edf6ef; font-size: .68rem; letter-spacing: .08em;
text-transform: uppercase }}
th:nth-child(2),td:nth-child(2) {{ text-align: left }}
tbody tr {{ transition: background .18s, transform .18s }}
tbody tr:hover {{ background: #edf9f0 }} tbody tr:last-child td {{ border-bottom: 0 }}
tbody tr.podium {{ background: linear-gradient(90deg, #fff9df80, transparent 65%) }}
tbody tr.current {{ box-shadow: inset 4px 0 #2ea65e; background-color: #e9f8ed }}
.rank {{ display: inline-grid; min-width: 2rem; min-height: 2rem; place-items: center;
border-radius: 10px; background: #edf4ee; font-weight: 800 }}
.agent-name {{ font-weight: 800; color: #173f30 }}
small {{ margin-left: .55rem; padding: .22rem .45rem; border-radius: 999px; color: #13703f;
background: #c9f3d5; font-size: .58rem; font-weight: 900; letter-spacing: .06em }}
.score {{ position: relative; width: 112px; height: 27px; overflow: hidden; border-radius: 9px;
background: #e4eee6; text-align: center }}
.score span {{ position: absolute; inset: 0 auto 0 0; background: linear-gradient(90deg,#71d18d,#f3cc56) }}
.score b {{ position: relative; line-height: 27px; font-size: .78rem }}
.positive {{ color: #16824b; font-weight: 800 }} .negative {{ color: #c34c4c; font-weight: 800 }}
@media (max-width: 760px) {{ body {{ padding: 1rem }} .hero {{ padding: 1.7rem }}
.cards {{ grid-template-columns: repeat(2, 1fr) }} .table-shell {{ overflow-x: auto }} }}
</style></head><body><main class="wrap"><header class="hero">
<p class="eyebrow">Autonomous farming tournament</p><h1>Agent Leaderboard</h1>
<p class="seed">Run #{run_number} · Seed {seed}</p></header>
<section class="cards"><div class="card"><span>Current leader</span><strong>{leader}</strong></div>
<div class="card"><span>Agents</span><strong>{len(table)}</strong></div>
<div class="card"><span>Total matches</span><strong>{cumulative}</strong></div>
<div class="card"><span>Cache hit</span><strong>{cache_rate:.0%}</strong></div></section>
<div class="table-shell"><table><thead><tr><th>#</th><th>Agent</th><th>Played</th><th>Wins</th><th>Draws</th>
<th>Losses</th><th>Score</th><th>Avg reward</th><th>Avg margin</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></div></main></body></html>"""


def render_saved(output_dir=DEFAULT_OUTPUT_DIR) -> Path:
    """Regenerate HTML from checkpointed runs without starting new matches."""
    output_dir = Path(output_dir).expanduser().resolve()
    state = json.loads((output_dir / SEED_STATE_FILE).read_text(encoding="utf-8"))
    runs = state.get("runs", [])
    if not runs:
        raise ValueError("no saved leaderboard runs to render")
    all_results = [result for saved_run in runs for result in saved_run.get("results", [])]
    labels = list(dict.fromkeys(name for result in all_results for name in result["players"]))
    table = standings([AgentSpec(label, label, lambda seed: None) for label in labels],
                      all_results)
    latest = runs[-1]
    (output_dir / "standings.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    html_path = output_dir / "leaderboard.html"
    current_total = len(latest.get("results", []))
    html_path.write_text(render_html(table, latest["seed"], current_total, current_total,
                                     latest["id"], len(all_results)), encoding="utf-8")
    return html_path


def run(values: list[str] | None = None, output_dir=DEFAULT_OUTPUT_DIR, seed=None,
        force=False, public_dir=DEFAULT_PUBLIC_DIR, rerun_current=False,
        workers=4) -> Path:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    output_dir = Path(output_dir).expanduser().resolve()
    values = discover_public_agents(public_dir) if values is None else values
    agents = [resolve_agent(value) for value in values]
    production = resolve_agent("current")
    if all(agent.fingerprint != production.fingerprint for agent in agents):
        agents.insert(0, production)
    if len(agents) < 2:
        raise ValueError(f"no agent .py/.ipynb files found in {Path(public_dir)}")
    if len({agent.label for agent in agents}) != len(agents):
        raise ValueError("public agent filenames must be unique without their extensions")
    if len({agent.fingerprint for agent in agents}) != len(agents):
        raise ValueError("each agent must have different content")
    state, current_run = prepare_run(output_dir, agents, seed)
    seed = current_run["seed"]
    results, cache_hits = [], 0
    total = len(agents) * (len(agents) - 1) // 2
    jobs = []
    for i, left in enumerate(agents):
        for right in agents[i + 1:]:
            rerun_pair = force or (rerun_current and "current" in {left.label, right.label})
            jobs.append((left, right, rerun_pair))
    if workers == 1:
        completed_jobs = ((left, right, *play_cached(
            left, right, seed, output_dir / "cache", rerun_pair,
        )) for left, right, rerun_pair in jobs)
    else:
        if any(agent.source is None for agent in agents):
            raise ValueError("parallel workers require resolvable agent sources")
        executor = ProcessPoolExecutor(max_workers=min(workers, total))
        pending = {
            executor.submit(
                _process_match,
                {"label": left.label, "source": left.source},
                {"label": right.label, "source": right.source},
                seed, str(output_dir / "cache"), rerun_pair,
            ): (left, right)
            for left, right, rerun_pair in jobs
        }
        completed_jobs = ((pending[future][0], pending[future][1], *future.result())
                          for future in as_completed(pending))
    try:
        for completed, (left, right, result, cached) in enumerate(completed_jobs, 1):
            results.append(result)
            _record_result(current_run, result)
            _write_run_state(output_dir, state)
            cache_hits += int(cached)
            print(f"[{completed}/{total}] {left.label} vs {right.label}, seed={seed}"
                  f" ({'cached' if cached else 'played'})", flush=True)
    finally:
        if workers != 1:
            executor.shutdown(wait=True, cancel_futures=True)
    current_run["status"] = "completed"
    _write_run_state(output_dir, state)
    all_results = [result for saved_run in state["runs"]
                   for result in saved_run.get("results", [])]
    all_labels = list(dict.fromkeys(
        [agent.label for agent in agents]
        + [name for result in all_results for name in result["players"]]
    ))
    table_agents = [AgentSpec(label, label, lambda seed: None) for label in all_labels]
    table = standings(table_agents, all_results)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "standings.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    html_path = output_dir / "leaderboard.html"
    html_path.write_text(render_html(table, seed, cache_hits, total,
                                     current_run["id"], len(all_results)), encoding="utf-8")
    print(f"Run #{current_run['id']} completed with seed {seed}")
    print(f"Leaderboard: {html_path}")
    return html_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dir", type=Path, default=DEFAULT_PUBLIC_DIR,
                        help="directory containing public .py/.ipynb agents")
    parser.add_argument("--seed", type=int,
                        help="seed for a new leaderboard (normally generated randomly)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="ignore cached match results")
    parser.add_argument("--rerun-current", action="store_true",
                        help="rerun only matches involving the current agent")
    parser.add_argument("--workers", type=int, default=4,
                        help="number of isolated match processes (default: 4)")
    parser.add_argument("--render-only", action="store_true",
                        help="refresh HTML from saved runs without playing matches")
    args = parser.parse_args()
    if args.render_only:
        print(f"Leaderboard: {render_saved(args.output_dir)}")
        return
    run(None, args.output_dir, args.seed, args.force, args.public_dir,
        args.rerun_current, args.workers)


if __name__ == "__main__":
    main()
