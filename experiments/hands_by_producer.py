"""Break-even/capacity analysis for hiring hands, across all 5 crops AND all
3 animals, grounded in the real Kaggriculture mechanics. Supersedes
hands_by_crop.py (crops only).

CROPS (unfertilized only -- see that limitation noted below):
  A harvest-day tile that's immediately replanted needs FOUR actions in that
  one visit: WATER (old crop's last bonus day, if it has one) -> HARVEST ->
  PLANT (new) -> WATER (new). The last WATER is not optional: kaggriculture
  sets a freshly planted tile's `consecutive_unwatered = 1` at creation
  ("planting day counts as unwatered"), and a tile dies once
  `consecutive_unwatered >= 2` at end of day -- skip the same-day re-water
  and the replant dies before the day ends. That also means the harvest day
  *is* the next cycle's day 0 (same calendar day), so a tile kept in
  continuous rotation repeats every `harvest_day` days, not
  `harvest_day + 1`. Verified against experiments/crop_schedules.py's CASES
  (asserted against the real interpreter) and kaggriculture.py's own
  _daily_refresh_plants.

ANIMALS: an animal never needs replacing (no expiry) as long as it's fed at
least every other day (`consecutive_unfed >= 2` -> it escapes, structure
stays). Steady-state daily routine, matching the specified real sequence:
  every day:      FEED -> CARE -> COLLECT_FERTILIZER
  production day: same, + HARVEST
CARE only pays off if paired with FEED the same day (banks a bonus consumed
on the next production tick); COLLECT_FERTILIZER is available every single
day regardless of feed/care. Verified against kaggriculture.py's
_daily_refresh_animals (consecutive_unfed bookkeeping, pending_care_bonus
accrual, fertilizer_available reset).

Real inputs, not derived by this script:
  - Per-crop water/harvest-day schedule and yield: experiments/crop_schedules.py
    CASES (verified against the real interpreter).
  - Seed cost, animal cost/interval/max_held: kaggle_environments' CROPS/ANIMALS.
  - Sale price: kaggle_environments' MARKET_PARAMS base price (assumes trading
    stays near the market's I0 equilibrium -- ignores the price-impact curve
    a large operation would trigger; see shop_price_monte_carlo.py for that).
  - Feed cost: WHEAT's own base price used as a proxy (opportunity cost of a
    wheat unit not sold), not an actual BUY_PRODUCT quote.
  - Hire cost: Fibonacci-per-day (agents/scheduler.py, matches the engine's
    own _hire_cost).
  - Tile-servicing capacity per hand count: agents/scheduler.py's own
    bin-packing (_pack) -- the same routing/budget logic the production
    agent uses, not a new formula.

Known gaps (stated, not hidden):
  - Crops: only unfertilized schedules modeled. Fertilizer changes an ongoing
    crop's harvest cadence (crop_schedules.py's fertilized TOMATO/STRAWBERRY
    cases show 2 harvests, not 1) in a way this script doesn't generalize.
  - Animals: one-time setup cost (BUILD_COOP/PASTURE, BUY_ANIMAL, PLACE) is
    NOT amortized into profit/tile/day -- only the steady-state recurring
    loop is priced. This makes animals look better the longer you run them.
  - No market-glut price impact: at the tile counts this script reaches (up
    to 100), a real farm's own selling would move prices measurably,
    especially for FERTILIZER (T=200) and low-base-price crops. Treat the
    absolute coin numbers as optimistic; the relative ranking is the useful
    part.
  - TOMATO/STRAWBERRY (ongoing crops) are assumed to need replanting after
    one harvest, inferred from kaggriculture.py's max_lifespan_step logic
    plus CASES not showing a second unfertilized harvest -- not independently
    confirmed with a real, longer interpreter run.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS, CROPS, MARKET_PARAMS

from agents.scheduler import FARMER_BUDGET, HAND_BUDGET, SHED, _pack, predicted_hand_starts
from agents.farm_tasks import Task
from experiments.crop_schedules import CASES

MAX_HANDS_CHECKED = 20
BOARD_TILES = 100  # 10x10 default board, all 4 quadrants bought
WHEAT_FEED_COST = MARKET_PARAMS["WHEAT"]["base"]  # proxy cost of 1 feed unit

CROP_ORDER = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON"]
ANIMAL_ORDER = ["GOOSE", "COW", "SHEEP"]
PRODUCER_ORDER = CROP_ORDER + ANIMAL_ORDER

STYLE = {
    "WHEAT": ("#eda100", "-"),
    "CARROT": ("#eb6834", "-"),
    "TOMATO": ("#e34948", "-"),
    "STRAWBERRY": ("#e87ba4", "-"),
    "MELON": ("#1baf7a", "-"),
    "GOOSE": ("#4a3aa7", "--"),
    "COW": ("#2a78d6", "--"),
    "SHEEP": ("#52514e", "--"),
}


def _fib(n: int) -> int:
    """fib(0)=1, fib(1)=1, fib(2)=2... matches the engine's _fib exactly."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def crop_profile(crop: str) -> dict:
    """Real per-cycle economics and action load for continuous unfertilized
    rotation of one crop, derived from verified schedule + engine constants."""
    schedule = CASES[(crop, False)]
    water_days = schedule["water"]
    harvest_day = max(schedule["harvest"])  # exactly one, for every unfertilized case
    yield_units = schedule["expected_yields"][0]
    period = harvest_day  # harvest day doubles as next cycle's day 0

    middle_actions = sum(1 for age in range(1, harvest_day) if age in water_days)
    merged_actions = (1 if harvest_day in water_days else 0) + 3  # +HARVEST+PLANT+WATER
    actions_per_period = middle_actions + merged_actions

    revenue_per_cycle = yield_units * MARKET_PARAMS[crop]["base"]
    cost_per_cycle = CROPS[crop]["seed"]  # one replant per cycle in steady state

    return {
        "name": crop,
        "kind": "crop",
        "water_days": water_days,
        "harvest_day": harvest_day,
        "yield_units": yield_units,
        "period": period,
        "actions_per_tile_per_day": actions_per_period / period,
        "profit_per_tile_per_day": (revenue_per_cycle - cost_per_cycle) / period,
    }


def animal_profile(animal: str) -> dict:
    """Real steady-state economics for continuous FEED+CARE+COLLECT_FERTILIZER
    (+HARVEST on production days), derived from verified engine constants.
    Excludes the one-time BUILD/BUY_ANIMAL/PLACE setup cost."""
    a = ANIMALS[animal]
    interval = a["interval"]
    per_tick_yield = min(a["max_held"], 1 + interval)  # base(1) + full care bonus(interval)
    product_price = MARKET_PARAMS[a["product"]]["base"]
    fert_price = MARKET_PARAMS["FERTILIZER"]["base"]

    revenue_per_day = fert_price + per_tick_yield * product_price / interval
    cost_per_day = WHEAT_FEED_COST  # 1 WHEAT fed every day
    actions_per_tile_per_day = 3 + 1 / interval  # FEED+CARE+COLLECT_FERTILIZER daily, +HARVEST every `interval` days

    return {
        "name": animal,
        "kind": "animal",
        "interval": interval,
        "per_tick_yield": per_tick_yield,
        "period": interval,
        "actions_per_tile_per_day": actions_per_tile_per_day,
        "profit_per_tile_per_day": revenue_per_day - cost_per_day,
    }


def profile_for(name: str) -> dict:
    return crop_profile(name) if name in CROP_ORDER else animal_profile(name)


def build_tasks_crop(profile: dict, k: int) -> list[Task]:
    crop = profile["name"]
    water_days, harvest_day, yield_units, period = (
        profile["water_days"], profile["harvest_day"], profile["yield_units"], profile["period"]
    )
    positions = [(x, y) for y in range(10) for x in range(10)][:k]
    tasks = []
    for index, position in enumerate(positions):
        age = index % period  # age in 1..period-1, or 0 standing for the merged slot
        actions: list = []
        needs: Counter = Counter()
        sells: Counter = Counter()
        if age == 0:
            if harvest_day in water_days:
                actions.append(["WATER"])
            actions += [["HARVEST"], ["PLANT", crop], ["WATER"]]
            needs["SEED"] += 1
            sells[crop] += yield_units
        elif age in water_days:
            actions.append(["WATER"])
        if not actions:
            continue
        tasks.append(Task(position=position, actions=actions, needs=needs, sells=sells))
    return tasks


def build_tasks_animal(profile: dict, k: int) -> list[Task]:
    animal = profile["name"]
    interval, per_tick_yield = profile["interval"], profile["per_tick_yield"]
    product = ANIMALS[animal]["product"]
    positions = [(x, y) for y in range(10) for x in range(10)][:k]
    tasks = []
    for index, position in enumerate(positions):
        actions = [["FEED"], ["CARE"], ["COLLECT_FERTILIZER"]]
        sells: Counter = Counter({"FERTILIZER": 1})
        if index % interval == 0:
            actions.append(["HARVEST"])
            sells[product] += per_tick_yield
        tasks.append(Task(position=position, actions=actions, needs=Counter({"WHEAT": 1}), sells=sells))
    return tasks


def build_tasks(profile: dict, k: int) -> list[Task]:
    return build_tasks_crop(profile, k) if profile["kind"] == "crop" else build_tasks_animal(profile, k)


def fits(profile: dict, hand_count: int, k: int) -> bool:
    tasks = build_tasks(profile, k)
    starts = [tuple(SHED)] + predicted_hand_starts(SHED, (), hand_count)
    budgets = [FARMER_BUDGET] + [HAND_BUDGET] * hand_count
    _, unassigned = _pack(tasks, starts, budgets)
    return not unassigned


def max_tiles_for_hands(profile: dict, hand_count: int, tile_cap: int = BOARD_TILES) -> int:
    if not fits(profile, hand_count, 1):
        return 0
    lo, hi = 1, tile_cap
    while hi < tile_cap and fits(profile, hand_count, hi):
        lo, hi = hi, min(hi * 2, tile_cap)
    if fits(profile, hand_count, tile_cap):
        return tile_cap
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(profile, hand_count, mid):
            lo = mid
        else:
            hi = mid
    return lo


def calculate(profile: dict) -> list[dict]:
    capacities = [max_tiles_for_hands(profile, h) for h in range(0, MAX_HANDS_CHECKED + 1)]
    rows = []
    for hands in range(1, MAX_HANDS_CHECKED + 1):
        marginal_cost = _fib(hands - 1)
        marginal_tiles = capacities[hands] - capacities[hands - 1]
        breakeven = (marginal_cost / marginal_tiles) if marginal_tiles > 0 else float("inf")
        rows.append(
            {
                "name": profile["name"],
                "hands": hands,
                "max_tiles": capacities[hands],
                "marginal_tiles": marginal_tiles,
                "marginal_cost": marginal_cost,
                "breakeven_profit_per_tile_per_day": breakeven,
            }
        )
    return rows


def optimal_hands(rows: list[dict], profile: dict) -> tuple[int, float]:
    """Hand count maximizing net coin/day (revenue at that many hands' tile
    capacity, minus the cumulative Fibonacci cost of hiring them all) -- not
    just the largest hand whose own marginal break-even clears, since a
    temporarily-bad hand can still be worth hiring through if a later, cheap
    batch of tiles more than makes up for it in the cumulative total."""
    profit = profile["profit_per_tile_per_day"]
    cumulative_cost = 0
    best_hands, best_net = 0, 0.0
    for row in rows:
        cumulative_cost += row["marginal_cost"]
        net = row["max_tiles"] * profit - cumulative_cost
        if net > best_net:
            best_hands, best_net = row["hands"], net
    return best_hands, best_net


def net_profit_series(rows: list[dict], profile: dict) -> list[dict]:
    """Net coin/day at every hand count 0..MAX_HANDS_CHECKED: tile-capacity
    revenue at that hand count, minus the cumulative Fibonacci hire cost."""
    profit = profile["profit_per_tile_per_day"]
    series = [{"hands": 0, "max_tiles": 0, "cumulative_cost": 0, "net": 0.0}]
    cumulative_cost = 0
    for row in rows:
        cumulative_cost += row["marginal_cost"]
        net = row["max_tiles"] * profit - cumulative_cost
        series.append(
            {
                "hands": row["hands"],
                "max_tiles": row["max_tiles"],
                "cumulative_cost": cumulative_cost,
                "net": net,
            }
        )
    return series


def save_net_csv(all_series: dict[str, list[dict]], output: Path) -> None:
    import csv

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["producer", "hands", "max_tiles", "cumulative_hire_cost", "net_coin_per_day"])
        for name in PRODUCER_ORDER:
            for point in all_series[name]:
                writer.writerow([
                    name, point["hands"], point["max_tiles"],
                    round(point["cumulative_cost"], 2), round(point["net"], 2),
                ])


def save_net_chart(all_series: dict[str, list[dict]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 8))
    for name in PRODUCER_ORDER:
        color, linestyle = STYLE[name]
        hands = [p["hands"] for p in all_series[name]]
        net = [p["net"] for p in all_series[name]]
        ax.plot(hands, net, marker="o", markersize=4, linewidth=2.2,
                color=color, linestyle=linestyle, label=name.title())
    ax.axhline(0, color="black", linewidth=1)
    ax.set(
        title="Net coin/day (lai/lo) by number of hands hired, per producer\n"
              "(solid = crop, dashed = animal; below the black line = loss)",
        xlabel="Number of hands",
        ylabel="Net coins / day",
        xticks=range(0, MAX_HANDS_CHECKED + 1, 2),
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_capacity_chart(all_rows: dict[str, list[dict]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7))
    for name in PRODUCER_ORDER:
        rows = all_rows[name]
        color, linestyle = STYLE[name]
        hands = [0] + [r["hands"] for r in rows]
        tiles = [0] + [r["max_tiles"] for r in rows]
        ax.plot(hands, tiles, marker="o", markersize=4, linewidth=2.2,
                color=color, linestyle=linestyle, label=name.title())
    ax.axhline(BOARD_TILES, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.annotate(f"Full board: {BOARD_TILES} tiles", (0, BOARD_TILES),
                xytext=(4, 6), textcoords="offset points", fontsize=9, alpha=0.6)
    ax.set(
        title="Real capacity: max tiles serviceable by number of hands\n"
              "(solid = crop, dashed = animal; agents/scheduler.py's actual routing/budget packing)",
        xlabel="Number of hands",
        ylabel="Max tiles kept in continuous production",
        xticks=range(0, MAX_HANDS_CHECKED + 1, 2),
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_breakeven_chart(all_rows: dict[str, list[dict]], profiles: dict[str, dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 8))
    y_values = [
        r["breakeven_profit_per_tile_per_day"]
        for rows in all_rows.values() for r in rows
        if r["breakeven_profit_per_tile_per_day"] != float("inf")
    ]
    y_max = max(y_values) * 1.3

    for name in PRODUCER_ORDER:
        rows = all_rows[name]
        color, linestyle = STYLE[name]
        hands = [r["hands"] for r in rows]
        breakeven = [
            r["breakeven_profit_per_tile_per_day"]
            if r["breakeven_profit_per_tile_per_day"] != float("inf") else None
            for r in rows
        ]
        ax.plot(hands, breakeven, marker="o", markersize=4, linewidth=2,
                color=color, linestyle=linestyle, label=name.title())
        ax.axhline(profiles[name]["profit_per_tile_per_day"], color=color, linewidth=1, linestyle=":")
        impossible = [r["hands"] for r in rows if r["marginal_tiles"] == 0]
        if impossible:
            ax.scatter(impossible, [y_max] * len(impossible), marker="x", color=color, zorder=5)

    ax.set_yscale("log")
    ax.set(
        title="Break-even profit/tile/day to justify hand N, vs. each producer's own real profit/tile/day\n"
              "(solid line = break-even needed; dotted = that producer's own profit; x = board full, impossible)",
        xlabel="N-th hand hired today",
        ylabel="Coins / tile / day (log scale)",
        xticks=range(1, MAX_HANDS_CHECKED + 1),
        ylim=(min(y_values) * 0.6, y_max * 1.4),
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    profiles = {name: profile_for(name) for name in PRODUCER_ORDER}
    all_rows = {name: calculate(profiles[name]) for name in PRODUCER_ORDER}

    print(f"{'producer':<11} {'actions/tile/day':>16} {'profit/tile/day':>16} {'optimal hands':>14} {'net coin/day':>13}")
    for name in PRODUCER_ORDER:
        profile = profiles[name]
        hands, net = optimal_hands(all_rows[name], profile)
        print(
            f"{name.title():<11} {profile['actions_per_tile_per_day']:>16.3f} "
            f"{profile['profit_per_tile_per_day']:>16.2f} {hands:>14d} {net:>13,.0f}"
        )

    all_series = {name: net_profit_series(all_rows[name], profiles[name]) for name in PRODUCER_ORDER}
    print(f"\n{'hands':>5}  " + "  ".join(f"{name.title():>11}" for name in PRODUCER_ORDER))
    for h in range(0, MAX_HANDS_CHECKED + 1):
        row = "  ".join(f"{all_series[name][h]['net']:>11,.0f}" for name in PRODUCER_ORDER)
        print(f"{h:>5}  {row}")

    capacity_path = output_dir / "hands_capacity_by_producer.png"
    breakeven_path = output_dir / "hands_breakeven_by_producer.png"
    net_chart_path = output_dir / "hands_net_by_producer.png"
    net_csv_path = output_dir / "hands_net_by_producer.csv"
    save_capacity_chart(all_rows, capacity_path)
    save_breakeven_chart(all_rows, profiles, breakeven_path)
    save_net_chart(all_series, net_chart_path)
    save_net_csv(all_series, net_csv_path)
    print(f"\nChart (capacity): {capacity_path}")
    print(f"Chart (break-even): {breakeven_path}")
    print(f"Chart (net lai/lo): {net_chart_path}")
    print(f"CSV (net lai/lo): {net_csv_path}")


if __name__ == "__main__":
    main()
