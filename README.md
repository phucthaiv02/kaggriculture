# Kaggriculture Agent

Strategy agents, experiments, and tests for the Kaggle **Kaggriculture** farming simulation.

The goal is to build an agent that manages crops, animals, labor, land expansion, and market orders to maximize end-of-season cash.

## Repository layout

- `agents/` — production agent logic: planning, scheduling, forecasting, labor, selling, and farm tasks.
- `experiments/` — simulations and analysis scripts used to validate strategies and economics.
- `tests/` — regression and integration tests for the agent logic.
- `docs/README.md` — detailed game rules and mechanics.
- `docs/AGENTS.md` — agent API, local-running notes, and replay/log commands.
- `replays/` — local/downloaded replay artifacts; intentionally not tracked by Git.

## Setup

Create a virtual environment and install the runtime dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

For development and tests:

```bash
python -m pip install -r requirements-dev.txt
```

> On Windows, activate the environment with `.venv\\Scripts\\activate`.

## Run the tests

```bash
pytest -q
```

## Try a local simulation

For example, render the opening-book strategy to an HTML replay:

```bash
python experiments/render_opening_book.py
```

Generated replay/analysis HTML files and downloaded replay JSON files are ignored by Git so local experiments do not bloat the repository.

## Documentation

Start with [`docs/README.md`](docs/README.md) for the game mechanics, then see [`docs/AGENTS.md`](docs/AGENTS.md) for the agent interface and Kaggle tooling.

## License

See [`LICENSE`](LICENSE).
