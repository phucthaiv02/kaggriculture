"""Reconstruct executed sales from a Kaggriculture replay and export offline HTML."""
from __future__ import annotations

import argparse
from copy import deepcopy
import html
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from kaggle_environments.envs.kaggriculture import kaggriculture as game
from kaggle_environments.utils import structify


def daily_cash(replay: dict, player: int) -> list[dict]:
    """Last post-action cash in each action day, including a partial final day."""
    turns = max(1, int(replay["configuration"].get("turnsPerDay", 24)))
    days = {}
    for frame in range(1, len(replay["steps"])):
        previous = replay["steps"][frame - 1][0]["observation"]
        current = replay["steps"][frame][0]["observation"]
        tick = previous.get("step", frame - 1)
        day = tick // turns
        days[day] = {"day": day, "step": tick,
                     "cash": current["farms"][player]["money"],
                     "complete_day": (tick + 1) % turns == 0}
    return [days[day] for day in sorted(days)]


def analyze(replay: dict) -> dict:
    """Use the installed engine's action/market rules, recording successful units only."""
    steps = replay.get("steps", [])
    if len(steps) < 2 or replay.get("name") != "kaggriculture":
        raise ValueError("expected a Kaggriculture replay with at least two frames")
    cfg = replay["configuration"]
    turns = max(1, int(cfg.get("turnsPerDay", 24)))
    board = int(cfg.get("boardSize", 10))
    capacity = int(cfg.get("shedCapacity", 100))
    events = []
    costs = []
    commit = game._commit_unit
    hire, buy_land = game._do_hire, game._do_buy_land
    for frame in range(1, len(steps)):
        previous, current = steps[frame - 1], steps[frame]
        state = structify(deepcopy(previous))
        tick = previous[0]["observation"].get("step", frame - 1)
        day = tick // turns
        farms = state[0].observation.farms
        for player, s in enumerate(state):
            s.action = current[player].get("action") or {}
            action = s.action if isinstance(s.action, dict) else {}
            farmer = action.get("farmer", ["PASS"])
            hands = action.get("hands", [])
            hands = hands if isinstance(hands, list) else []
            units = [farmer, *hands]
            demand = {}
            for a in units:
                if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT":
                    demand[a[1]] = demand.get(a[1], 0) + 1
            seeds = s.observation.private.get("seeds", {})
            blocked = {crop for crop, count in demand.items() if count > seeds.get(crop, 0)}
            for unit, a in enumerate(units):
                if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT" and a[1] in blocked:
                    a = ["PASS"]
                game._apply_unit_action(farms[player], s.observation.private, unit, a,
                                        board, day, turns, capacity)

        def record(op, item, price, farm, private, market, *args, **kwargs):
            ok = commit(op, item, price, farm, private, market, *args, **kwargs)
            if ok and op == "SELL":
                player = next(i for i, f in enumerate(farms) if f is farm)
                events.append({"player": player, "step": tick, "day": day,
                               "product": item, "quantity": 1, "revenue": price})
            elif ok:
                costs.append({"player": next(i for i, f in enumerate(farms) if f is farm),
                              "step": tick, "category": op, "item": item, "amount": price})
            return ok

        def atomic(fn, category):
            def wrapped(farm, *args, **kwargs):
                before = farm["money"]
                result = fn(farm, *args, **kwargs)
                spent = before - farm["money"]
                if spent:
                    costs.append({"player": next(i for i, f in enumerate(farms) if f is farm),
                                  "step": tick, "category": category, "item": category, "amount": spent})
                return result
            return wrapped

        with (patch.object(game, "_commit_unit", record),
              patch.object(game, "_do_hire", atomic(hire, "HIRE")),
              patch.object(game, "_do_buy_land", atomic(buy_land, "BUY_LAND"))):
            game._process_market(state, SimpleNamespace(configuration=structify(cfg)))
        # Later phases do not change money. Detect engine/replay incompatibility.
        expected = current[0]["observation"]["farms"]
        if any(abs(f["money"] - expected[i]["money"]) > 1e-6 for i, f in enumerate(farms)):
            raise ValueError(f"cash mismatch at frame {frame}: {[f["money"] for f in farms]} != {[f["money"] for f in expected]}; use the engine version that created the replay")

    last_step = steps[-2][0]["observation"].get("step", len(steps) - 2)
    rows = []
    for player in range(len(steps[0])):
        for product in game.PRODUCTS:
            timeline = [{"step": d, "quantity": 0, "revenue": 0} for d in range(last_step + 1)]
            for event in events:
                if event["player"] == player and event["product"] == product:
                    row = timeline[event["step"]]
                    row["quantity"] += event["quantity"]
                    row["revenue"] += event["revenue"]
            quantity = revenue = 0
            for row in timeline:
                quantity += row["quantity"]
                revenue += row["revenue"]
                row.update(average_price=row["revenue"] / row["quantity"] if row["quantity"] else None,
                           cumulative_quantity=quantity,
                           cumulative_average_price=revenue / quantity if quantity else None)
            rows.append({"player": player, "product": product, "quantity": quantity,
                         "revenue": revenue, "average_price": revenue / quantity if quantity else None,
                         "timeline": timeline})
    finances = []
    for player in range(len(steps[0])):
        timeline = [{"step": tick, "cost": 0, "revenue": 0} for tick in range(last_step + 1)]
        for event in events:
            if event["player"] == player:
                timeline[event["step"]]["revenue"] += event["revenue"]
        breakdown = {}
        for event in costs:
            if event["player"] == player:
                timeline[event["step"]]["cost"] += event["amount"]
                key = event["category"] + ":" + event["item"]
                breakdown[key] = breakdown.get(key, 0) + event["amount"]
        total_cost = total_revenue = 0
        for row in timeline:
            total_cost += row["cost"]
            total_revenue += row["revenue"]
            row.update(cumulative_cost=total_cost, cumulative_revenue=total_revenue,
                       net_cashflow=total_revenue - total_cost)
        initial_cash = steps[0][0]["observation"]["farms"][player]["money"]
        final_cash = steps[-1][0]["observation"]["farms"][player]["money"]
        if abs(initial_cash + total_revenue - total_cost - final_cash) > 1e-6:
            raise ValueError(f"unaccounted cash flow for player {player}")
        finances.append({"player": player, "cost": total_cost, "revenue": total_revenue,
                         "net_cashflow": total_revenue - total_cost, "initial_cash": initial_cash,
                         "final_cash": final_cash, "breakdown": breakdown, "timeline": timeline,
                         "cash_daily": daily_cash(replay, player)})
    return {"rows": rows, "events": events, "cost_events": costs,
            "finances": finances, "turns_per_day": turns}


def render(report: dict, source: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indexed = {(r["product"], r["player"]): r for r in report["rows"]}
    products = list(dict.fromkeys(r["product"] for r in report["rows"]))
    if {r["player"] for r in report["rows"]} != {0, 1}:
        raise ValueError("comparison report requires players 0 and 1")

    def number(value, signed=False):
        if value is None:
            return "—"
        return f"{value:+,.2f}" if signed else f"{value:,.2f}"

    table_rows = []
    sections = []
    metrics = [("quantity", "Sản lượng bán theo step", "Đơn vị"),
               ("average_price", "Giá bán trung bình theo step", "Tiền game / đơn vị"),
               ("cumulative_quantity", "Tổng sản lượng bán lũy kế", "Đơn vị"),
               ("cumulative_average_price", "Giá bán trung bình lũy kế", "Tiền game / đơn vị")]
    for product in products:
        rows = [indexed[product, player] for player in (0, 1)]
        left, right = rows
        cells = [f'<a href="#product-{html.escape(product, quote=True)}">{html.escape(product)}</a>']
        for key in ("quantity", "average_price", "revenue"):
            v0, v1 = left[key], right[key]
            delta = v0 - v1 if v0 is not None and v1 is not None else None
            cells.extend([number(v0), number(v1), number(delta, signed=True)])
        table_rows.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')

        fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
        for ax, (key, title, ylabel) in zip(axes.flat, metrics):
            for player, row in enumerate(rows):
                ax.plot([d["step"] for d in row["timeline"]],
                        [float("nan") if d[key] is None else d[key] for d in row["timeline"]],
                        label=f"Player {player}", color=("#2563eb", "#ea580c")[player],
                        linestyle=("-", "--")[player],
                        marker="." if key == "average_price" else None, linewidth=1.8)
            ax.set(title=title, xlabel="Step (bắt đầu từ 0)", ylabel=ylabel)
            ax.grid(alpha=.2)
            ax.legend(fontsize=9)
            if not any(r["quantity"] for r in rows):
                ax.text(.5, .5, "Chưa có giao dịch bán", transform=ax.transAxes, ha="center")
        svg = StringIO()
        fig.savefig(svg, format="svg")
        plt.close(fig)
        details = ' · '.join(
            f'Player {r["player"]}: <b>{r["quantity"]:,}</b> đơn vị, '
            f'giá TB <b>{number(r["average_price"])}</b>' for r in rows)
        sections.append(f'<section id="product-{html.escape(product, quote=True)}">'
                        f'<h2>{html.escape(product)}</h2><p>{details}</p>'
                        f'{svg.getvalue()[svg.getvalue().index("<svg"):]}</section>')
    table = ('<section><h2>So sánh tổng kết theo sản phẩm</h2>'
             '<p>Chênh lệch = Player 0 − Player 1. Giá TB và chênh lệch giá để trống nếu chưa có giao dịch bán.</p>'
             '<div class="table-scroll"><table><thead><tr><th rowspan="2">Sản phẩm</th>'
             '<th colspan="3">Tổng lượng bán</th><th colspan="3">Giá bán TB</th>'
             '<th colspan="3">Doanh thu</th></tr><tr>'
             + '<th>P0</th><th>P1</th><th>Chênh lệch</th>' * 3
             + '</tr></thead><tbody>' + ''.join(table_rows) + '</tbody></table></div></section>')
    finances = sorted(report["finances"], key=lambda r: r["player"])
    cost_labels = {"BUY_SEED": "Hạt giống", "BUY_PRODUCT": "Mua sản phẩm",
                   "BUY_ANIMAL": "Con vật", "HIRE": "Thuê nhân công", "BUY_LAND": "Mua đất"}
    cost_rows = []
    def comparison_row(label, values):
        return ('<tr><td>' + html.escape(label) + '</td>'
                + ''.join(f'<td>{number(v)}</td>' for v in values)
                + f'<td>{number(values[0] - values[1], signed=True)}</td></tr>')
    for key, label in [("revenue", "Tổng doanh thu bán"), ("cost", "Tổng chi phí thực trả"),
                       ("net_cashflow", "Doanh thu − chi phí"), ("initial_cash", "Tiền ban đầu"),
                       ("final_cash", "Tiền cuối trận")]:
        cost_rows.append(comparison_row(label, [f[key] for f in finances]))
    for category, label in cost_labels.items():
        values = [sum(v for k, v in f["breakdown"].items() if k.split(":")[0] == category) for f in finances]
        cost_rows.append(comparison_row(label + " (tổng)", values))
        if category.startswith("BUY_") and category != "BUY_LAND":
            keys = sorted({k for f in finances for k in f["breakdown"] if k.split(":")[0] == category})
            for key in keys:
                cost_rows.append(comparison_row(label + " / " + key.split(":")[1],
                                               [f["breakdown"].get(key, 0) for f in finances]))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
    for ax, (key, title) in zip(axes.flat, [("cost", "Chi phí thực trả theo step"),
                                            ("cumulative_cost", "Chi phí lũy kế"),
                                            ("cumulative_revenue", "Doanh thu bán lũy kế"),
                                            ("net_cashflow", "Doanh thu − chi phí lũy kế")]):
        for player, finance in enumerate(finances):
            ax.plot([r["step"] for r in finance["timeline"]], [r[key] for r in finance["timeline"]],
                    label=f"Player {player}", color=("#2563eb", "#ea580c")[player],
                    linestyle=("-", "--")[player], linewidth=1.8)
        ax.set(title=title, xlabel="Step (bắt đầu từ 0)", ylabel="Tiền game")
        ax.grid(alpha=.2)
        ax.legend()
    svg = StringIO()
    fig.savefig(svg, format="svg")
    plt.close(fig)
    cost_section = ('<section><h2>Chi phí và dòng tiền</h2>'
                    '<p>Chi phí là tiền đã chi cho hạt giống, sản phẩm, con vật, nhân công và đất. '
                    'Nhân công và đất là chi phí dùng chung, chưa phân bổ cho từng sản phẩm. '
                    'Doanh thu − chi phí là dòng tiền ròng, không định giá tồn kho hoặc tài sản cuối trận.</p>'
                    '<div class="table-scroll"><table><thead><tr><th>Khoản mục</th><th>Player 0</th>'
                    '<th>Player 1</th><th>Chênh lệch P0 − P1</th></tr></thead><tbody>'
                    + ''.join(cost_rows) + '</tbody></table></div>'
                    + svg.getvalue()[svg.getvalue().index("<svg"):] + '</section>')
    fig, ax = plt.subplots(figsize=(14, 5), layout="constrained")
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    for player, finance in enumerate(finances):
        points = finance["cash_daily"]
        ax.plot([r["day"] for r in points], [r["cash"] for r in points],
                label=f"Player {player}", color=("#2563eb", "#ea580c")[player],
                linestyle=("-", "--")[player], marker="o", markersize=4, linewidth=2)
    ax.set(title="Cash cuối ngày — so sánh hai player", xlabel="Ngày trong game (bắt đầu từ 0)",
           ylabel="Tiền mặt (tiền game)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(alpha=.2)
    ax.legend()
    svg = StringIO()
    fig.savefig(svg, format="svg")
    plt.close(fig)
    cash_section = ('<section id="cash-daily"><h2>Cash theo ngày</h2>'
                    '<p>Số dư tiền mặt sau step cuối cùng của mỗi ngày, lấy trực tiếp từ replay. '
                    'Ngày bắt đầu từ 0; nếu trận kết thúc giữa ngày, dùng số dư tại thời điểm kết thúc. '
                    'Số dư đã bao gồm tiền ban đầu, doanh thu và toàn bộ chi phí thực trả.</p>'
                    + svg.getvalue()[svg.getvalue().index("<svg"):] + '</section>')
    return ('<!doctype html><html lang="vi"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<title>So sánh bán hàng — Player 0 và Player 1</title><style>'
            'body{font:16px/1.6 system-ui;margin:0;background:#f3f6f2;color:#183325}'
            'main{max-width:1400px;margin:auto;padding:24px}section{background:white;padding:24px;margin:24px 0;border-radius:12px}'
            '.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;white-space:nowrap}'
            'th,td{text-align:right;padding:8px;border-bottom:1px solid #ddd}'
            'th:first-child,td:first-child{text-align:left}svg{width:100%;height:auto}'
            'a{color:#2563eb}.p0{color:#2563eb}.p1{color:#ea580c}</style><main>'
            f'<h1>So sánh bán hàng giữa hai player</h1><p>Replay: {html.escape(source)}</p>'
            '<p><b class="p0">Player 0: xanh, nét liền</b> · <b class="p1">Player 1: cam, nét đứt</b>. '
            'Mỗi sản phẩm có biểu đồ chung cho hai player, cùng trục step và cùng thang đo.</p>'
            '<p>Giá bán TB = tổng doanh thu / tổng đơn vị đã bán. Chỉ tính lệnh SELL khớp thành công; '
            'không tính mua hàng hoặc tiêu thụ của thị trấn. Step không bán có giá TB trống. '
            'Player 0 là agent hiện tại; Player 1 là đối thủ trong replay từ play_match.</p>'
            + cash_section + cost_section + table + ''.join(sections) + '</main></html>')


def run(replay_path: Path, output: Path | None = None) -> Path:
    report = analyze(json.loads(replay_path.read_text(encoding="utf-8")))
    output = output or replay_path.with_suffix(".sales.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(report, replay_path.name), encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _project_command(replay: Path, output: Path | None):
    """Use the same project environment as local replay generation.

    Comparing prefixes, rather than executable symlinks, distinguishes a venv
    from system Python even when both executables resolve to the same binary.
    Absolute artifact paths preserve the caller's working-directory semantics.
    """
    root = Path(__file__).resolve().parents[1]
    environment = root / ".venv"
    python = environment / "bin" / "python"
    if not python.is_file() or Path(sys.prefix).resolve() == environment.resolve():
        return None
    command = [str(python), "-m", "experiments.sales_report", str(replay.resolve())]
    if output is not None:
        command += ["--output", str(output.resolve())]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path, help="JSON replay exported by play_match")
    parser.add_argument("--output", type=Path, help="HTML output (default: <replay>.sales.html)")
    args = parser.parse_args()
    command = _project_command(args.replay, args.output)
    if command is not None:
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1])
        raise SystemExit(result.returncode)
    try:
        print(run(args.replay, args.output))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"sales_report: {exc}\n")


if __name__ == "__main__":
    main()
