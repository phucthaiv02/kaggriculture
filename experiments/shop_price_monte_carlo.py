"""Monte Carlo analysis of real random town-shop behavior over 1M seasons.

Both players PASS. Town center demand is excluded. The eight normal shop slots
unlock on days 3, 6, ..., 24 and are sampled uniformly
with replacement, exactly matching the game's documented random process.

This intentionally samples the distribution directly with NumPy rather than
running the 719-turn interpreter one million times. Inventory depletion uses
the exact unlock timing and final prices use the official ``market_price``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from kaggle_environments.envs.kaggriculture import kaggriculture as game


DEFAULT_SEASONS = 1_000_000
DAYS = 30
PRODUCTS = [p for p in game.PRODUCTS if p != "FERTILIZER"]
SHOPS = sorted(game.SHOPS)
UNLOCK_DAYS = np.arange(3, 25, 3, dtype=np.int16)
# A shop first consumes at hour 0 after it unlocks: 6 ticks per remaining day.
SLOT_TICKS = (DAYS - UNLOCK_DAYS) * 6
CROP_ECONOMICS = {
    "WHEAT": {"yield": 4, "cycle_days": 4, "cost": 10},
    "CARROT": {"yield": 3, "cycle_days": 3, "cost": 20},
    "TOMATO": {"yield": 4, "cycle_days": 11, "cost": 50},
    "STRAWBERRY": {"yield": 4, "cycle_days": 16, "cost": 100},
    "MELON": {"yield": 6, "cycle_days": 10, "cost": 80},
}


def prices_from_demand(product: str, demand: np.ndarray) -> np.ndarray:
    """Map demand to official rounded prices, evaluating each unique value once."""
    unique, inverse = np.unique(demand, return_inverse=True)
    prices = np.array(
        [game.market_price(product, game.MARKET_PARAMS[product]["I0"] - int(x)) for x in unique],
        dtype=np.int32,
    )
    return prices[inverse]


def histogram(values: np.ndarray, max_bins: int = 80) -> list[dict]:
    lo, hi = int(values.min()), int(values.max())
    if hi == lo:
        return [{"from": lo, "to": hi, "count": int(values.size)}]
    edges = np.linspace(lo, hi + 1, min(max_bins, hi - lo + 1) + 1, dtype=np.int32)
    edges = np.unique(edges)
    counts, edges = np.histogram(values, bins=edges)
    return [
        {"from": int(edges[i]), "to": int(edges[i + 1] - 1), "count": int(count)}
        for i, count in enumerate(counts)
        if count
    ]


def compact_summary(values: np.ndarray) -> dict:
    """Fast exact order-statistic summary for integer prices."""
    frequency = np.bincount(values)
    support = np.flatnonzero(frequency)
    cumulative = np.cumsum(frequency)

    def percentile(q: float) -> int:
        # Nearest-rank is stable and differs negligibly from interpolated
        # quantiles at one million samples.
        target = max(1, int(np.ceil(q * values.size)))
        return int(np.searchsorted(cumulative, target))

    return {
        "mean": float(values.mean()),
        "p05": percentile(0.05),
        "p25": percentile(0.25),
        "p50": percentile(0.50),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "min": int(support[0]),
        "max": int(support[-1]),
    }


def scaled_summary(integer_values: np.ndarray, scale: float) -> dict:
    """Summarize fixed-point values and return them in original units."""
    summary = compact_summary(integer_values)
    return {key: (value / scale if key != "day" else value) for key, value in summary.items()}


def simulate(seasons: int, random_seed: int) -> dict:
    rng = np.random.default_rng(random_seed)
    draws = rng.integers(0, len(SHOPS), size=(seasons, len(UNLOCK_DAYS)), dtype=np.uint8)

    counts = np.stack([(draws == i).sum(axis=1) for i in range(len(SHOPS))], axis=1)
    unique_shop_count = (counts > 0).sum(axis=1)
    max_duplicate_count = counts.max(axis=1)

    # Encode the unordered multiset in base 9; each digit is one shop count.
    powers = (9 ** np.arange(len(SHOPS), dtype=np.int64))[None, :]
    combo_codes = (counts.astype(np.int64) * powers).sum(axis=1)
    codes, combo_freq = np.unique(combo_codes, return_counts=True)
    top_idx = np.argsort(combo_freq)[-15:][::-1]
    combinations = []
    for idx in top_idx:
        code = int(codes[idx])
        composition = {}
        for shop in SHOPS:
            n, code = code % 9, code // 9
            if n:
                composition[shop] = n
        combinations.append(
            {"shops": composition, "count": int(combo_freq[idx]), "probability": float(combo_freq[idx] / seasons)}
        )

    product_results = {}
    conditional = {}
    total_daily_prices = np.zeros((DAYS + 1, seasons), dtype=np.int32)
    wheat_daily_prices = np.zeros((DAYS + 1, seasons), dtype=np.int32)
    crop_profit_stats = {crop: [] for crop in CROP_ECONOMICS}
    animal_profit_stats = {animal: [] for animal in game.ANIMALS}
    for product in PRODUCTS:
        # Only random shops contribute demand; town center is deliberately
        # excluded. One-product shops consume twice on every tick.
        demand = np.zeros(seasons, dtype=np.int32)
        for shop_id, shop in enumerate(SHOPS):
            if product not in game.SHOPS[shop]:
                continue
            multiplier = 2 if len(game.SHOPS[shop]) == 1 else 1
            demand += ((draws == shop_id) * SLOT_TICKS).sum(axis=1, dtype=np.int32) * multiplier
        prices = prices_from_demand(product, demand)
        q = np.quantile(prices, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        product_results[product] = {
            "base_price": int(game.MARKET_PARAMS[product]["base"]),
            "mean": float(prices.mean()),
            "min": int(prices.min()),
            "max": int(prices.max()),
            "quantiles": dict(zip(["p01", "p05", "p25", "p50", "p75", "p95", "p99"], map(float, q))),
            "demand_mean": float(demand.mean()),
            "histogram": histogram(prices),
            "daily": [],
        }

        # Distribution through time. At the end of day k, a shop unlocked on
        # day d has run 6*(k-d) ticks. Day 0 is the untouched initial market.
        for day in range(DAYS + 1):
            daily_demand = np.zeros(seasons, dtype=np.int32)
            for slot, unlock_day in enumerate(UNLOCK_DAYS):
                elapsed = max(0, day - int(unlock_day))
                if not elapsed:
                    continue
                for shop_id, shop in enumerate(SHOPS):
                    if product not in game.SHOPS[shop]:
                        continue
                    multiplier = 2 if len(game.SHOPS[shop]) == 1 else 1
                    daily_demand += (draws[:, slot] == shop_id) * (elapsed * 6 * multiplier)
            daily_prices = prices_from_demand(product, daily_demand)
            total_daily_prices[day] += daily_prices
            price_summary = compact_summary(daily_prices)
            product_results[product]["daily"].append({"day": day, **price_summary})
            if product == "WHEAT":
                wheat_daily_prices[day] = daily_prices
            if product in CROP_ECONOMICS:
                economics = CROP_ECONOMICS[product]
                numerator = economics["yield"] * daily_prices - economics["cost"]
                crop_profit_stats[product].append(
                    {"day": day, **scaled_summary(numerator, economics["cycle_days"])}
                )
            for animal, economics in game.ANIMALS.items():
                if economics["product"] != product:
                    continue
                # Fixed-point forms preserve exact 1/2 and 1/3 rates while
                # allowing fast integer percentile calculation.
                if animal == "GOOSE":
                    profit_scaled = 2 * daily_prices - wheat_daily_prices[day] - 10
                    scale = 1
                elif animal == "COW":
                    profit_scaled = 9 * daily_prices - 6 * wheat_daily_prices[day] - 80
                    scale = 6
                else:  # SHEEP
                    profit_scaled = 4 * daily_prices - 3 * wheat_daily_prices[day] - 50
                    scale = 3
                animal_profit_stats[animal].append(
                    {"day": day, **scaled_summary(profit_scaled, scale)}
                )
        conditional[product] = {}
        for shop_id, shop in enumerate(SHOPS):
            present = counts[:, shop_id] > 0
            mean_present = float(prices[present].mean())
            mean_absent = float(prices[~present].mean())
            conditional[product][shop] = {
                "present": mean_present,
                "absent": mean_absent,
                "lift": mean_present - mean_absent,
            }

    shop_stats = {}
    for i, shop in enumerate(SHOPS):
        distribution = np.bincount(counts[:, i], minlength=9)
        shop_stats[shop] = {
            "appearance_probability": float((counts[:, i] > 0).mean()),
            "mean_instances": float(counts[:, i].mean()),
            "count_distribution": [int(x) for x in distribution],
        }

    crop_profit_daily = []
    animal_profit_daily = []
    for day in range(DAYS + 1):
        crop_row = {"day": day}
        for crop, economics in CROP_ECONOMICS.items():
            price = product_results[crop]["daily"][day]["mean"]
            crop_row[crop] = (economics["yield"] * price - economics["cost"]) / economics["cycle_days"]
        crop_profit_daily.append(crop_row)

        wheat_price = product_results["WHEAT"]["daily"][day]["mean"]
        animal_row = {"day": day}
        for animal, economics in game.ANIMALS.items():
            product_price = product_results[economics["product"]]["daily"][day]["mean"]
            units_per_day = (economics["interval"] + 1) / economics["interval"]
            animal_row[animal] = (
                units_per_day * product_price
                - wheat_price
                - economics["cost"] / DAYS
            )
        animal_profit_daily.append(animal_row)

    total_daily = []
    total_variation_daily = []
    cumulative_total_variation = []
    cumulative_variation = np.zeros(seasons, dtype=np.int32)
    previous_mean = None
    for day in range(DAYS + 1):
        point = {"day": day, **compact_summary(total_daily_prices[day])}
        point["mean_daily_change"] = 0.0 if previous_mean is None else point["mean"] - previous_mean
        previous_mean = point["mean"]
        total_daily.append(point)
        variation = (
            np.zeros(seasons, dtype=np.int32)
            if day == 0
            else np.abs(total_daily_prices[day] - total_daily_prices[day - 1])
        )
        total_variation_daily.append({"day": day, **compact_summary(variation)})
        cumulative_variation += variation
        cumulative_total_variation.append({"day": day, **compact_summary(cumulative_variation)})

    return {
        "meta": {
            "seasons": seasons,
            "random_seed": random_seed,
            "players": 2,
            "action": "PASS",
            "days": DAYS,
            "unlock_days": UNLOCK_DAYS.tolist(),
            "slot_consumption_ticks": SLOT_TICKS.tolist(),
            "sampling": "Direct Monte Carlo of uniform-with-replacement shop draws; exact official price function.",
        },
        "shops": SHOPS,
        "products": PRODUCTS,
        "shop_products": game.SHOPS,
        "shop_stats": shop_stats,
        "unique_shop_distribution": [int(x) for x in np.bincount(unique_shop_count, minlength=9)],
        "max_duplicate_distribution": [int(x) for x in np.bincount(max_duplicate_count, minlength=9)],
        "products_stats": product_results,
        "total_price_daily": total_daily,
        "total_variation_daily": total_variation_daily,
        "cumulative_total_variation": cumulative_total_variation,
        "crop_profit_daily": crop_profit_daily,
        "animal_profit_daily": animal_profit_daily,
        "crop_profit_stats": crop_profit_stats,
        "animal_profit_stats": animal_profit_stats,
        "profit_assumptions": {
            "crops": CROP_ECONOMICS,
            "animals": "CARE steady-state yield; subtract 1 wheat/day and amortize purchase cost over 30 days; fertilizer excluded.",
        },
        "conditional_lift": conditional,
        "top_combinations": combinations,
    }


HTML = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Tổng biến động giá tích lũy</title><style>
:root{--bg:#f5f1e7;--card:#fffdf8;--ink:#18251c;--muted:#68736b;--line:#d9d5c8;--green:#246c49}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:30px 24px 50px}h1{font-size:clamp(30px,4vw,52px);margin:0 0 8px}h2{font-size:30px;margin:0 0 8px}p{color:var(--muted)}.panel{background:var(--card);border:1px solid var(--line);border-radius:15px;padding:24px;margin-bottom:20px}.summary{font-size:18px;margin:18px 0}.chart{width:100%;height:680px;display:block}.profit-chart{width:100%;height:560px;display:block}.grid{stroke:#dfdbcf}.axis{fill:var(--muted);font-size:14px}.band95{fill:#b8d99e;opacity:.38}.band50{fill:#69a66f;opacity:.42}.median{fill:none;stroke:var(--green);stroke-width:4}.meanline{fill:none;stroke:#e07a39;stroke-width:3;stroke-dasharray:8 6}.legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:10px;color:var(--muted)}.swatch{display:inline-block;width:20px;border-top:4px solid;margin-right:6px;vertical-align:middle}@media(max-width:700px){main{padding:16px}.panel{padding:14px}.chart,.profit-chart{height:560px}}
</style></head><body><main><section class="panel"><h1>Tổng biến động giá tích lũy</h1><p>Đến ngày d: Σ từ ngày 1→d Σ sản phẩm |giá hôm nay − giá hôm trước| · 1.000.000 seed · hai player PASS · không town center.</p><div id="summary" class="summary"></div><svg id="timeline" class="chart" viewBox="0 0 1200 680"></svg><div class="legend">P05–P95 · P25–P75 · <span style="color:#246c49">median</span> · <span style="color:#e07a39">trung bình</span></div></section>
<section class="panel"><h2>CROP — lợi nhuận/ngày theo giá</h2><p>(yield tối ưu không fertilizer × giá − seed cost) ÷ cycle days. Mỗi loại gồm median, dải đậm P25–P75 và dải nhạt P05–P95.</p><svg id="cropProfit" class="profit-chart" viewBox="0 0 1200 560"></svg><div id="cropLegend" class="legend"></div></section>
<section class="panel"><h2>ANIMAL — lợi nhuận/ngày theo giá</h2><p>CARE steady-state yield × giá sản phẩm − giá 1 wheat/ngày − animal cost/30 ngày. Mỗi loại gồm median, P25–P75 và P05–P95; không tính fertilizer.</p><svg id="animalProfit" class="profit-chart" viewBox="0 0 1200 560"></svg><div id="animalLegend" class="legend"></div></section></main><script>
const D=__DATA__,rows=D.cumulative_total_variation,fmt=x=>new Intl.NumberFormat('vi-VN').format(x);function area(top,bottom,x,y){return rows.map((r,i)=>`${i?'L':'M'}${x(r.day)} ${y(r[top])}`).join(' ')+' '+[...rows].reverse().map(r=>`L${x(r.day)} ${y(r[bottom])}`).join(' ')+' Z'}function trend(key,x,y){return rows.map((r,i)=>`${i?'L':'M'}${x(r.day)} ${y(r[key])}`).join(' ')}const hi=Math.max(...rows.map(r=>r.p95)),ymax=hi*1.08,x=d=>90+d/30*1070,y=v=>600-v/ymax*530;let out='';for(let i=0;i<7;i++){const yy=70+i*88.33,v=Math.round(ymax-i*ymax/6);out+=`<line class="grid" x1="90" y1="${yy}" x2="1160" y2="${yy}"/><text class="axis" x="78" y="${yy+5}" text-anchor="end">$${fmt(v)}</text>`}for(const d of [0,3,6,9,12,15,18,21,24,27,30])out+=`<text class="axis" x="${x(d)}" y="635" text-anchor="middle">D${d}</text>`;out+=`<path class="band95" d="${area('p95','p05',x,y)}"/><path class="band50" d="${area('p75','p25',x,y)}"/><path class="median" d="${trend('p50',x,y)}"/><path class="meanline" d="${trend('mean',x,y)}"/>`;document.querySelector('#timeline').innerHTML=out;const last=rows.at(-1);document.querySelector('#summary').innerHTML=`Ngày 30 — trung bình <b>$${fmt(Math.round(last.mean))}</b> · median <b>$${fmt(last.p50)}</b> · P05–P95 <b>$${fmt(last.p05)}–$${fmt(last.p95)}</b>`;
const colors=['#246c49','#df7b3c','#386cb0','#9a4d9e','#b59b22'];function seriesArea(rows,top,bottom,x,y){return rows.map((r,i)=>`${i?'L':'M'}${x(r.day)} ${y(r[top])}`).join(' ')+' '+[...rows].reverse().map(r=>`L${x(r.day)} ${y(r[bottom])}`).join(' ')+' Z'}function bandChart(stats,keys,svgId,legendId){const vals=keys.flatMap(k=>stats[k].flatMap(r=>[r.p05,r.p95])),lo=Math.min(0,...vals),hi=Math.max(...vals),pad=Math.max(1,(hi-lo)*.08),ymin=lo-pad,ymax=hi+pad,x=d=>90+d/30*1070,y=v=>490-(v-ymin)/(ymax-ymin)*430;let s='';for(let i=0;i<6;i++){const yy=60+i*86,v=Math.round(ymax-i*(ymax-ymin)/5);s+=`<line class="grid" x1="90" y1="${yy}" x2="1160" y2="${yy}"/><text class="axis" x="78" y="${yy+5}" text-anchor="end">$${v}</text>`}for(const d of [0,3,6,9,12,15,18,21,24,27,30])s+=`<text class="axis" x="${x(d)}" y="525" text-anchor="middle">D${d}</text>`;keys.forEach((k,i)=>{const rows=stats[k],c=colors[i],median=rows.map((r,j)=>`${j?'L':'M'}${x(r.day)} ${y(r.p50)}`).join(' ');s+=`<path d="${seriesArea(rows,'p95','p05',x,y)}" fill="${c}" opacity=".09"/><path d="${seriesArea(rows,'p75','p25',x,y)}" fill="${c}" opacity=".20"/><path d="${median}" fill="none" stroke="${c}" stroke-width="3"><title>${k} median</title></path>`});document.querySelector('#'+svgId).innerHTML=s;document.querySelector('#'+legendId).innerHTML=keys.map((k,i)=>`<span><i class="swatch" style="border-color:${colors[i]}"></i>${k}</span>`).join('')}bandChart(D.crop_profit_stats,['WHEAT','CARROT','TOMATO','STRAWBERRY','MELON'],'cropProfit','cropLegend');bandChart(D.animal_profit_stats,['GOOSE','COW','SHEEP'],'animalProfit','animalLegend');
</script></body></html>"""


def render(results: dict, output: Path) -> None:
    visual = {
        "meta": results["meta"],
        "cumulative_total_variation": results["cumulative_total_variation"],
        "crop_profit_stats": results["crop_profit_stats"],
        "animal_profit_stats": results["animal_profit_stats"],
    }
    payload = json.dumps(visual, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    output.write_text(HTML.replace("__DATA__", payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", "--seeds", dest="seasons", type=int, default=DEFAULT_SEASONS)
    parser.add_argument("--random-seed", type=int, default=20260903)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "shop_price_monte_carlo.html")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.seasons < 1:
        parser.error("--seasons must be positive")
    results = simulate(args.seasons, args.random_seed)
    render(results, args.output)
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Simulated {args.seasons:,} seeds -> {args.output}")
    print(f"P(all 8 shop types) = {results['unique_shop_distribution'][8] / args.seasons:.4%}")
    for product in PRODUCTS:
        s = results["products_stats"][product]
        print(f"{product:<11} mean=${s['mean']:.2f} p05=${s['quantiles']['p05']:.0f} p50=${s['quantiles']['p50']:.0f} p95=${s['quantiles']['p95']:.0f}")


if __name__ == "__main__":
    main()
