# Kaggriculture Agent

Strategy agents, experiments, and regression tests for Kaggle's
**Kaggriculture** farming simulation.

The production agent manages crops, animals, labor, land expansion, and market
orders to maximize end-of-season cash.

New PLACE/PLANT targets maximize net profit over a shared window of
`min(16, end_day - day)` elapsed days, including harvests on the final day.
Candidates whose first yield falls beyond this window are excluded. Only
scheduled harvests within the window contribute revenue. Crops are replanted
when at least one subsequent scheduled harvest fits, with every seed charged.
Profit includes purchase, feed, fertilizer,
additional labor, and market price impact, without profit-per-day normalization.

## Repository layout

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
