"""Fast one-seed, no-opponent policy evaluator for the day-16 experiment.

The evaluator aggregates deterministic farm events by day.  It models crop and
animal yields, crop-to-crop/animal rotations, feed, fertilizer, hiring, town
demand, and the environment's unit-by-unit dynamic market price.  It deliberately
does not model worker paths or simultaneous-action edge cases; finalists must be
verified in the real environment later.
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official_game
from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS as ENV_ANIMALS,
    MARKET_I0,
    MARKET_PARAMS,
    SHOPS,
    market_price,
)

try:
    from experiments.day16_allocation import (
        ANIMALS,
        CROPS,
        END_DAY,
        FERT_DAYS,
        MAX_DAY,
        ONGOING,
        Policy,
        _successor,
        candidates,
        configuration,
        make_policy_agent,
        pass_agent,
    )
except ModuleNotFoundError:
    from day16_allocation import (
        ANIMALS,
        CROPS,
        END_DAY,
        FERT_DAYS,
        MAX_DAY,
        ONGOING,
        Policy,
        _successor,
        candidates,
        configuration,
        make_policy_agent,
        pass_agent,
    )


STARTING_MONEY = 3000.0
SEED_COST = {"WHEAT": 10, "CARROT": 20, "MELON": 80, "TOMATO": 50, "STRAWBERRY": 100}
ANIMAL_COST = {name: data["cost"] for name, data in ENV_ANIMALS.items()}
PRODUCT = {name: data["product"] for name, data in ENV_ANIMALS.items()}
FIRST_YIELD = {name: data["first_yield_day"] for name, data in ENV_ANIMALS.items()}
INTERVAL = {name: data["interval"] for name, data in ENV_ANIMALS.items()}
MAX_HELD = {name: data["max_held"] for name, data in ENV_ANIMALS.items()}
CROP_YIELD = {"WHEAT": 4, "CARROT": 3, "MELON": 6}

@dataclass
class Tile:
    target: str | None
    kind: str | None = None
    planted_day: int = -1
    placed_day: int = -1
    care_bank: int = 0
    fertilized_until: int = -1
    fertilized_once: bool = False


@dataclass
class FastResult:
    final_cash: float
    net_profit: float
    feasible_actions: bool
    peak_action_ratio: float
    sales: dict[str, int]
    purchases: dict[str, int]
    final_land: dict[str, int]
    shops: tuple[str, ...]


@dataclass
class ExactResult:
    final_cash: float
    net_profit: float
    feasible_actions: bool
    peak_action_ratio: float
    sales: dict[str, int]
    purchases: dict[str, int]
    final_land: dict[str, int]
    shops: tuple[str, ...]


class FastFarm:
    def __init__(self, policy: Policy):
        self.policy = policy
        self.cash = STARTING_MONEY
        self.inventory = {item: MARKET_I0 for item in MARKET_PARAMS}
        self.wheat = 0
        self.fertilizer = 0
        self.sales = {item: 0 for item in MARKET_PARAMS}
        self.purchases = {item: 0 for item in MARKET_PARAMS}
        self.tiles = [Tile(name) for name, count in policy.allocation for _ in range(count)]
        self.shops = []
        self.unused_weeds = 0
        self.opponent_weeds = 0
        self.feasible_actions = True
        self.peak_action_ratio = 0.0

    @property
    def prices(self):
        return {item: market_price(item, amount) for item, amount in self.inventory.items()}

    def sell(self, item, amount):
        for _ in range(int(amount)):
            price = market_price(item, self.inventory[item])
            self.cash += price
            self.sales[item] += 1
            if price > 1:
                self.inventory[item] += 1

    def buy_wheat(self, amount):
        bought = 0
        for _ in range(int(amount)):
            price = market_price("WHEAT", self.inventory["WHEAT"] - 1)
            if self.cash < price:
                break
            self.inventory["WHEAT"] -= 1
            self.cash -= price
            self.wheat += 1
            self.purchases["WHEAT"] += 1
            bought += 1
        return bought

    def town_demand(self, day):
        # Six shop-consumption ticks and one town-center tick per day.
        for shop in self.shops:
            products = SHOPS[shop]
            multiplier = 2 if len(products) == 1 else 1
            for item in products:
                self.inventory[item] -= 6 * multiplier
        for item in self.inventory:
            if item != "FERTILIZER":
                self.inventory[item] -= 1

    def end_day_rng(self, day):
        """Reproduce seed-1 weed draws and the resulting shop draw."""
        rng = random.Random((1 * 1_000_003) ^ day)
        unused_empty = 25 - len(self.tiles) - self.unused_weeds
        slot_empty = [tile for tile in self.tiles if tile.kind is None]
        for _ in range(unused_empty):
            if rng.random() < 0.005:
                self.unused_weeds += 1
        for tile in slot_empty:
            if rng.random() < 0.005:
                tile.kind = "WEED"
        for _ in range(25 - self.opponent_weeds):
            if rng.random() < 0.005:
                self.opponent_weeds += 1
        next_day = day + 1
        if next_day > 0 and next_day % 3 == 0 and len(self.shops) < 8:
            self.shops.append(rng.choice(sorted(SHOPS)))

    def hire(self):
        fib = [1, 1]
        while len(fib) < self.policy.hands:
            fib.append(fib[-1] + fib[-2])
        cost = sum(fib[:self.policy.hands])
        if self.cash < cost:
            return False
        self.cash -= cost
        return True

    def acquire(self, tile: Tile, day):
        target = tile.target
        dig = 1 if tile.kind == "WEED" else 0
        if target in CROPS:
            cost = SEED_COST[target]
            if self.cash >= cost:
                self.cash -= cost
                tile.kind = target
                tile.planted_day = day
                tile.fertilized_until = -1
                tile.fertilized_once = False
                return dig + 2  # optional DIG + PLANT + WATER
        elif target in ANIMALS:
            # Animal, first feed, structure and placement must all be funded.
            wheat_price = market_price("WHEAT", self.inventory["WHEAT"] - 1) if not self.wheat else 0
            if self.cash >= ANIMAL_COST[target] + wheat_price:
                self.cash -= ANIMAL_COST[target]
                if not self.wheat:
                    self.buy_wheat(1)
                self.wheat -= 1
                tile.kind = target
                tile.placed_day = day
                tile.care_bank = 1
                return dig + 4  # optional DIG + BUILD + PLACE + FEED + CARE
        return 0

    def allocate_fertilizer(self, day):
        if self.policy.fertilizer == "sell" or not self.fertilizer:
            return 0
        actions = 0
        for tile in self.tiles:
            if not self.fertilizer or tile.kind not in CROPS:
                break
            age = day - tile.planted_day
            if age not in FERT_DAYS[tile.kind]:
                continue
            extra_units = 1 if tile.kind == "CARROT" else 2
            worth_using = self.policy.fertilizer == "use" or (
                extra_units * market_price(tile.kind, self.inventory[tile.kind])
                > market_price("FERTILIZER", self.inventory["FERTILIZER"])
            )
            if worth_using:
                self.fertilizer -= 1
                tile.fertilized_until = day + 3
                tile.fertilized_once = True
                actions += 1
        return actions

    def step_day(self, day):
        self.town_demand(day)
        if not self.hire():
            self.feasible_actions = False
            return

        action_count = 0
        action_count += self.allocate_fertilizer(day)
        if self.fertilizer:
            self.sell("FERTILIZER", self.fertilizer)
            self.fertilizer = 0

        # Existing animals are refinanced from yesterday's fertilizer if feed
        # wheat is absent. Every animal is fed and cared for before production.
        live_animals = [tile for tile in self.tiles if tile.kind in ANIMALS]
        if self.wheat < len(live_animals):
            self.buy_wheat(len(live_animals) - self.wheat)
        fed = min(self.wheat, len(live_animals))
        self.wheat -= fed
        if fed < len(live_animals):
            self.feasible_actions = False
        outputs = {item: 0 for item in MARKET_PARAMS}
        for tile in live_animals[:fed]:
            tile.care_bank += 1
            action_count += 3  # FEED + CARE + COLLECT_FERTILIZER
            age = day - tile.placed_day
            animal = tile.kind
            if age >= FIRST_YIELD[animal] and (age - FIRST_YIELD[animal]) % INTERVAL[animal] == 0:
                units = min(MAX_HELD[animal], 1 + tile.care_bank)
                outputs[PRODUCT[animal]] += units
                tile.care_bank = 0
                action_count += 1

        # Select and pre-fund replacements before harvest. This mirrors the
        # real agent: proceeds from a harvest cannot buy an animal that was
        # already needed for HARVEST -> BUILD -> PLACE on that same visit.
        planned = {}
        due = [tile for tile in self.tiles if tile.kind in CROP_YIELD and day - tile.planted_day >= MAX_DAY[tile.kind]]
        for tile in due:
            successor = _successor(self.policy, tile.kind, day, self.prices)
            tile.target = successor
            if successor in CROPS and self.cash >= SEED_COST[successor]:
                self.cash -= SEED_COST[successor]
                planned[id(tile)] = successor
            elif successor in ANIMALS:
                wheat_price = 0 if self.wheat else market_price("WHEAT", self.inventory["WHEAT"] - 1)
                if self.cash >= ANIMAL_COST[successor] + wheat_price:
                    self.cash -= ANIMAL_COST[successor]
                    if not self.wheat:
                        self.buy_wheat(1)
                    self.wheat -= 1
                    planned[id(tile)] = successor

        # Buy/plant still-empty initial or previously transitioned slots. The ordering is
        # part of the policy because it changes what scarce cash funds first.
        empty = [tile for tile in self.tiles if tile.kind in (None, "WEED") and tile.target]
        empty.sort(key=lambda tile: (tile.target not in ANIMALS) if self.policy.animals_first else (tile.target in ANIMALS))
        for tile in empty:
            action_count += self.acquire(tile, day)

        # Daily watering, production, harvest, sale, and target transition.
        harvested = []
        for tile in self.tiles:
            if tile.kind not in CROPS:
                continue
            crop = tile.kind
            age = day - tile.planted_day
            action_count += 1  # WATER every day
            if crop in ONGOING:
                first = 8 if crop == "TOMATO" else 10
                interval = 1 if crop == "TOMATO" else 2
                if age >= first and (age-first) % interval == 0 and (age-first)//interval < 4:
                    outputs[crop] += 2 if tile.fertilized_until >= day else 1
                    action_count += 1
            elif age >= MAX_DAY[crop]:
                units = CROP_YIELD[crop]
                if tile.fertilized_once and crop in {"WHEAT", "CARROT"}:
                    units += 2 if crop == "WHEAT" else 1
                outputs[crop] += units
                action_count += 1
                harvested.append((tile, crop))

        for item, amount in outputs.items():
            if amount:
                if item == "WHEAT":
                    self.wheat += amount
                else:
                    self.sell(item, amount)

        for tile, crop in harvested:
            tile.kind = None
            replacement = planned.get(id(tile))
            if replacement in CROPS:
                tile.kind = replacement
                tile.planted_day = day
                tile.fertilized_until = -1
                tile.fertilized_once = False
                action_count += 2
            elif replacement in ANIMALS:
                tile.kind = replacement
                tile.placed_day = day
                tile.care_bank = 1
                action_count += 4

        target_animals = sum(tile.target in ANIMALS for tile in self.tiles)
        sellable_wheat = max(0, self.wheat - target_animals)
        if sellable_wheat:
            self.sell("WHEAT", sellable_wheat)
            self.wheat -= sellable_wheat

        # Fertilizer becomes available at end of day for tomorrow's cash-flow.
        self.fertilizer += len([tile for tile in self.tiles if tile.kind in ANIMALS])

        # Aggregate capacity check: 21 usable turns per worker allows for two
        # preparation turns and one unit of routing overhead per active tile.
        active = sum(tile.kind is not None for tile in self.tiles)
        required = action_count + active
        capacity = (self.policy.hands + 1) * 21
        self.peak_action_ratio = max(self.peak_action_ratio, required / capacity)
        if required > capacity:
            self.feasible_actions = False
        self.end_day_rng(day)

    def run(self):
        for day in range(END_DAY + 1):
            self.step_day(day)
        # End-day fertilizer exists but cannot improve any more yield; liquidate.
        if self.fertilizer:
            self.sell("FERTILIZER", self.fertilizer)
            self.fertilizer = 0
        land = {}
        for tile in self.tiles:
            name = tile.kind or "EMPTY"
            land[name] = land.get(name, 0) + 1
        return FastResult(
            round(self.cash, 2), round(self.cash - STARTING_MONEY, 2), self.feasible_actions,
            round(self.peak_action_ratio, 4), {k: v for k, v in self.sales.items() if v},
            {k: v for k, v in self.purchases.items() if v}, land, tuple(self.shops),
        )


def estimate(policy):
    """Daily aggregate estimate; never use this to build a final list."""
    return FastFarm(policy).run()


_WORKER_ENV = None


def evaluate(policy):
    """Exact headless evaluation using the official interpreter.

    This manually steps the same environment used by ``env.run``. It omits the
    runner wrapper and rendering, but preserves all 408 turns, validation,
    market ordering, RNG, and observations.
    """
    global _WORKER_ENV
    if _WORKER_ENV is None:
        _WORKER_ENV = make("kaggriculture", configuration=configuration(1), debug=False)
    else:
        _WORKER_ENV.reset(2)
    env = _WORKER_ENV
    agents = (make_policy_agent(policy, 1), pass_agent)
    # State is already structified by reset. Calling the game's official
    # interpreter directly avoids repeated schema/struct conversions and
    # snapshot storage while executing identical game transitions.
    state = env.state
    for step in range(int(env.configuration.episodeSteps) - 1):
        for player in range(2):
            state[player].action = agents[player](state[player].observation)
        state = official_game.interpreter(state, env)
        state[0].observation.step = step + 1
    env.state = state
    final = state[0]
    land = {}
    for row in final.observation.farms[0].tiles:
        for tile in row:
            if not isinstance(tile, dict) or tile.get("kind") == "WEED":
                continue
            name = tile.get("animal") or tile.get("crop") or tile.get("kind")
            land[name] = land.get(name, 0) + 1
    cash = float(final.reward)
    return ExactResult(
        final_cash=cash,
        net_profit=cash - STARTING_MONEY,
        feasible_actions=True,
        peak_action_ratio=0.0,
        sales={},
        purchases={},
        final_land=land,
        shops=tuple(final.observation.town.unlocked_shops),
    )


def policy_record(policy, result):
    return {
        "final_cash": result.final_cash,
        "net_profit": result.net_profit,
        "feasible_actions": result.feasible_actions,
        "peak_action_ratio": result.peak_action_ratio,
        "hands": policy.hands,
        "rotation": policy.rotation,
        "fertilizer": policy.fertilizer,
        "animals_first": policy.animals_first,
        "allocation": dict(policy.allocation),
        "sales": result.sales,
        "purchases": result.purchases,
        "final_land": result.final_land,
        "shops": result.shops,
    }


def run_all(min_profit=25_000, output=None, workers=8, limit=0):
    policies = candidates(0)
    if limit:
        policies = policies[:limit]
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, (policy, result) in enumerate(zip(policies, pool.map(evaluate, policies, chunksize=1)), 1):
            if result.net_profit > min_profit:
                records.append(policy_record(policy, result))
            if index % 100 == 0 or index == len(policies):
                print(f"progress={index}/{len(policies)} qualifying={len(records)}", flush=True)
    records.sort(key=lambda row: (row["net_profit"], row["final_cash"], -row["hands"]), reverse=True)
    if output:
        path = Path(output)
        path.write_text(json.dumps({
            "seed": 1,
            "opponent": None,
            "evaluator": "official_interpreter_headless_v1",
            "verified_in_real_environment": True,
            "starting_money": STARTING_MONEY,
            "threshold_net_profit": min_profit,
            "policies_evaluated": len(policies),
            "final_list_count": len(records),
            "policies": records,
        }, indent=2), encoding="utf-8")
    return len(policies), records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-profit", type=float, default=25_000)
    parser.add_argument("--output", default="experiments/fast_final_list_seed1.json")
    parser.add_argument("--show", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates every generated policy")
    args = parser.parse_args()
    evaluated, records = run_all(args.min_profit, args.output, args.workers, args.limit)
    print(f"evaluated={evaluated} final_list={len(records)} output={Path(args.output).resolve()}")
    for index, row in enumerate(records[:args.show], 1):
        print(
            f"{index:>3}. profit={row['net_profit']:>8.0f} cash={row['final_cash']:>8.0f} "
            f"hands={row['hands']} rotation={row['rotation']:<10} fertilizer={row['fertilizer']:<5} "
            f"first={row['animals_first']} allocation={row['allocation']} final={row['final_land']}"
        )


if __name__ == "__main__":
    main()
