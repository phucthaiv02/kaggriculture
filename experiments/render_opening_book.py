"""Render a deterministic replay of the production expansion agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from kaggle_environments import make

from agents.expansion_agent import make_agent
from experiments.crop_schedules import pass_agent


END_DAY = 30


def configuration(seed):
    return {
        "episodeSteps": (END_DAY + 1) * 24,
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


def run(output=None, seed=1):
    env = make("kaggriculture", configuration=configuration(seed), debug=False)
    env.run([make_agent(END_DAY, seed=seed), pass_agent])

    output_path = Path(output or ROOT / "agent_replay_opening_book.html").resolve()
    output_path.write_text(
        env.render(mode="html", width=1200, height=800), encoding="utf-8"
    )
    print(f"Final cash: {float(env.steps[-1][0].reward):.0f}")
    print(f"Replay HTML: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    run(args.output, args.seed)
