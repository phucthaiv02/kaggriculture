"""Measure decision latency against the real interpreter, without timeout fallback.

Run: python -m experiments.benchmark_agent --seed 1 --opponent self
The report counts decisions exceeding actTimeout; it does not simulate the
runner's shared overage budget. Imports and opponent time are excluded.
"""

import argparse
from time import perf_counter

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as game

from agents.expansion_agent import make_agent
from experiments.crop_schedules import pass_agent
from experiments.play_match import configuration


def benchmark(seed=1, opponent="pass", timeout=.75):
    config = dict(configuration(seed), actTimeout=timeout)
    env = make("kaggriculture", configuration=config, debug=False)
    current = make_agent()
    other = make_agent() if opponent == "self" else pass_agent
    state, timings = env.state, []
    limit = env.configuration.actTimeout
    for step in range(config["episodeSteps"] - 1):
        obs = state[0].observation
        day, hour = obs.day, obs.hour
        tiles = 25 * len(obs.farms[0]["unlocked_quadrants"])
        start = perf_counter()
        action = current(obs, config)
        elapsed = perf_counter() - start
        timings.append((tiles, elapsed, day, hour))
        state[0].action = action
        state[1].action = (other(state[1].observation, config) if opponent == "self"
                           else other(state[1].observation))
        state = game.interpreter(state, env)
        state[0].observation.step = step + 1
    print(f"seed={seed}, opponent={opponent}, actTimeout={limit}s")
    for tiles in sorted({row[0] for row in timings}):
        rows = [row for row in timings if row[0] == tiles]
        worst = max(rows, key=lambda row: row[1])
        p95 = sorted(row[1] for row in rows)[int((len(rows) - 1) * .95)]
        exceeded = sum(row[1] > limit for row in rows)
        print(f"{tiles} tiles: {len(rows)} actions, p95={p95:.3f}s, "
              f"max={worst[1]:.3f}s (day={worst[2]}, hour={worst[3]}), "
              f"over limit={exceeded}")
    print(f"Final reward: {state[0].reward}")
    return timings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--opponent", choices=("pass", "self"), default="pass")
    parser.add_argument("--timeout", type=float, default=.75)
    args = parser.parse_args()
    benchmark(args.seed, args.opponent, args.timeout)
