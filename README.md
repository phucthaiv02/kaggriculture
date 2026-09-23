# Kaggriculture Agent

Strategy agents, experiments, and regression tests for Kaggle's
**Kaggriculture** farming simulation.

The production agent manages crops, animals, labor, land expansion, and market
orders to maximize end-of-season cash.

See [`RULES.md`](RULES.md) for the agent flow, strategy, scheduling invariants,
and rules for changing the code.

## Repository layout

- `RULES.md` — current agent flow, strategy, and maintenance rules.
- `agents/` — production planning, scheduling, forecasting, labor, selling,
  and farm-task logic.
- `experiments/` — executable checks and analysis scripts for strategy and
  game economics.
- `tests/` — regression and integration tests for production behavior.
- `docs/README.md` — detailed game rules and mechanics.
- `docs/AGENTS.md` — agent API, local-running notes, and replay/log commands.
- `replays/` — local replay artifacts created by experiments or Kaggle tools;
  intentionally ignored by Git.

## Setup

Create a virtual environment and install Kaggriculture plus the test runner:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U kaggle-environments pytest
```

Some standalone analysis scripts also use NumPy or Matplotlib:

```bash
python -m pip install -U numpy matplotlib
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Tests

Run the regression suite from the repository root:

```bash
pytest -q
```

## Useful experiments

Run experiment scripts as modules from the repository root so local packages
resolve consistently.

Play the production agent and save an HTML replay plus JSON match data:

```bash
# Passive opponent (default)
python -m experiments.play_match

# Built-in random opponent
python -m experiments.play_match --opponent random

# Python-file opponent
python -m experiments.play_match --opponent path/to/agent.py

# Agent embedded in a notebook's %%writefile main.py cell
python -m experiments.play_match --opponent path/to/notebook.ipynb
```

The production agent always runs as player 0. Match outputs are written under
`replays/` by default.

Create a round-robin HTML leaderboard. Every pair plays once using the same
randomly generated seed:

```bash
python -m experiments.agent_leaderboard
```

Matches run concurrently in four isolated processes by default. Process
isolation is required because the game runner captures global stdout/stderr and
is not thread-safe. Adjust concurrency based
on available CPU and memory with `--workers 8`, or use `--workers 1` for
sequential execution.

The experiment automatically loads every `.py` and `.ipynb` file directly under
`public/`; each agent is named after its filename without the extension. The
production agent from `agents/` is also included as `current`. Use `--public-dir`
to scan a different directory. Each invocation starts a run with a new random
seed and records its seed, roster, status, and match results in
`leaderboard/run.json`. If a run is interrupted, the next invocation resumes
that run with its original seed; a new seed is generated only after it completes.
Results are cached by agent content, engine configuration, and seed under
`leaderboard/cache/`, and the HTML standings aggregate all completed runs. When
a new public agent is added during an interrupted run, only its missing matches
are played. Pass `--rerun-current` to replay matches involving `current`, or
`--force` to replay every pair in the active run.

Compare scheduling while freezing the baseline target trace and successful
PLANT/PLACE/BUILD/DIG dates (save a copy of `agents/*.py` before editing):

```bash
python -m experiments.scheduler_comparison path/to/baseline/replay.json \
  --baseline-source path/to/saved_agents --output replays/scheduler_comparison/new
python -m experiments.benchmark_agent --opponent self --timeout 0.75
```

Keep the original `result.json` beside the baseline replay: it contains the
seed that the engine may omit from the replay configuration. The comparison
reports calendar mismatches and latency, and saves JSON and HTML replays.
Its `--baseline` option uses the original greedy packing algorithm with the
current buyer/seller, for a scheduling-only control. Routing is a bounded
heuristic; a passing comparison is not a proof of a global optimum.

Generate two offline comparison reports from a replay JSON:

```bash
python -m experiments.sales_report replays/current_vs_pass_seed1.json
```

This writes three files next to the replay:

- `sales_analysis.html`: executed sales, average prices, daily cash, and market costs.
- `operations_analysis.html`: daily successful hires, crop deaths that become weeds,
  disappearing animals, productive tile counts, and worker actions for both players.
  MOVE groups the four directions; action counts distinguish attempts from state changes.
  Productive tiles contain a crop or animal; empty structures and unlocked land are
  listed separately. Random weeds after harvesting or digging do not count as crop deaths.
- `sales_analysis.json`: all financial and operations data, including loss events
  with player, day, step, tile, producer, and cause.

Both HTML files embed their charts and work offline. Day indices start at 0;
end-of-day losses belong to the day just completed. A partial final day is labeled.
With `--output path/custom.html`, the second HTML is `path/custom.operations.html`.

Validate the crop and animal maintenance schedules:

```bash
python -m experiments.crop_schedules
python -m experiments.animal_yields
```

Run the larger economics analyses when needed:

```bash
python -m experiments.hands_by_producer
python -m experiments.shop_price_monte_carlo
```

Generated replay and analysis artifacts are ignored so experiments do not
accidentally bloat the repository.

## Documentation

Start with [`docs/README.md`](docs/README.md) for game mechanics, then see
[`docs/AGENTS.md`](docs/AGENTS.md) for the agent interface and Kaggle tooling.

## License

See [`LICENSE`](LICENSE).
