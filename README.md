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
python -m pip install kaggle-environments==1.32.7 pytest
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

# Four-day melon opening
python -m experiments.play_match --opening melon_v2 --seed 1

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

## Opening versions

Select the four-day melon opening with
`make_agent(opening_version="melon_v2")` from `agents.expansion_agent`.
The default `classic` opening remains available. Days below are one-indexed:

- Day 1: plant 9 WHEAT and 12 MELON; place 2 COW and 2 SHEEP.
- Day 2: collect fertilizer, return it to the shed for sale, buy feed, and feed animals.
- Day 3: harvest two WHEAT tiles early, repeat fertilizer refinancing, and add one COW. Keep the second cleared tile empty.
- Day 4: repeat fertilizer refinancing and add one SHEEP on the remaining tile.
- Day 5: hand control to the planner before buying replacement seeds. The 7 mature WHEAT tiles are repriced against all producer types; only seeds for the resulting targets are bought. Existing live MELON and animals remain protected through their current cycles. Fertilizer sales fund same-day purchases; existing workers place the COW on Day 3 and the SHEEP on Day 4. No extra hands are hired to wait for these purchases.

Select the mirrored-opponent planner with
`make_agent(planner_version="mirror_v2")`. For every candidate it assumes the
opponent starts one equivalent producer and sends both projected outputs
through the market-price forecast. The original concentration planner remains
available as `planner_version="concentration"` and is still the default.
