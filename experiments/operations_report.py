"""Collect farm operations during sales replay reconstruction and render HTML."""
from collections import Counter
import html
from io import StringIO

MOVES = {"NORTH", "SOUTH", "EAST", "WEST"}
ACTIONS = ("MOVE", "FEED", "CARE", "HARVEST", "WATER", "PLANT", "PLACE",
           "FERTILIZE", "COLLECT_FERTILIZER", "PICKUP", "DROP", "DIG",
           "BUILD_COOP", "BUILD_PASTURE", "PASS", "INVALID")


class Operations:
    def __init__(self, players, turns):
        self.players = players
        self.turns = turns
        self.days = {}
        self.losses = []

    def day(self, player, tick):
        day = tick // self.turns
        return self.days.setdefault((player, day), {
            "player": player, "day": day, "hires": 0, "crop_deaths": 0,
            "animal_losses": 0, "actions": Counter(), "effective_actions": Counter(),
            "complete_day": False,
        })

    def action(self, player, tick, command, changed):
        op = command[0] if isinstance(command, list) and command and isinstance(command[0], str) else "INVALID"
        op = "MOVE" if op in MOVES else op
        row = self.day(player, tick)
        row["actions"][op] += 1
        if changed:
            row["effective_actions"][op] += 1

    def finish_turn(self, farms, observed, tick):
        """Compare post-worker/market tiles to the recorded post-refresh board.

        A harvested/dug tile is already empty here, so a subsequent random
        weed is not a crop death. Newly planted crops dying this turn count.
        """
        for player, farm in enumerate(farms):
            row = self.day(player, tick)
            row["complete_day"] = (tick + 1) % self.turns == 0
            tiles = [tile for line in observed[player]["tiles"] for tile in line]
            row["unlocked_tiles"] = sum(tile != "LOCKED" for tile in tiles)
            row["crop_tiles"] = sum(isinstance(tile, dict) and tile.get("kind") == "PLANT" for tile in tiles)
            row["animal_tiles"] = sum(isinstance(tile, dict) and bool(tile.get("animal")) for tile in tiles)
            row["used_tiles"] = row["crop_tiles"] + row["animal_tiles"]
            row["empty_structures"] = sum(isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE") and not tile.get("animal") for tile in tiles)
            for y, tiles in enumerate(farm["tiles"]):
                for x, before in enumerate(tiles):
                    if not isinstance(before, dict):
                        continue
                    after = observed[player]["tiles"][y][x]
                    after = after if isinstance(after, dict) else {}
                    if before.get("kind") == "PLANT" and after.get("kind") == "WEED":
                        metric, producer = "crop_deaths", before["crop"]
                        lifespan = before.get("max_lifespan_step", -1)
                        decays_now = lifespan >= 0 and tick >= lifespan and (tick - lifespan) % 2 == 0
                        if decays_now and before.get("yield_units", 0) <= 1:
                            cause = "decay"
                        elif row["complete_day"] and not before.get("watered_today") and before.get("consecutive_unwatered", 0) >= 1:
                            cause = "unwatered"
                        else:
                            cause = "unknown"
                    elif before.get("animal") and after.get("animal") != before["animal"]:
                        metric, producer = "animal_losses", before["animal"]
                        cause = "unfed" if row["complete_day"] and not before.get("fed_today") and before.get("consecutive_unfed", 0) >= 1 else "unknown"
                    else:
                        continue
                    row[metric] += 1
                    self.losses.append({"player": player, "day": tick // self.turns,
                                        "step": tick, "position": [x, y], "producer": producer,
                                        "metric": metric, "cause": cause})

    def report(self):
        daily = sorted(self.days.values(), key=lambda row: (row["player"], row["day"]))
        totals = []
        for player in range(self.players):
            rows = [row for row in daily if row["player"] == player]
            totals.append({
                "player": player,
                **{key: sum(row[key] for row in rows) for key in ("hires", "crop_deaths", "animal_losses")},
                **{key: dict(sum((row[key] for row in rows), Counter()))
                   for key in ("actions", "effective_actions")},
                "used_tiles_final": rows[-1]["used_tiles"] if rows else 0,
            })
        return {"daily": daily, "totals": totals, "loss_events": self.losses,
                "turns_per_day": self.turns}


def render(report, source):
    """Standalone offline charts, comparisons and auditable loss events."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    totals = report["totals"]
    if [row["player"] for row in totals] != [0, 1]:
        raise ValueError("comparison report requires players 0 and 1")
    colors = ("#2563eb", "#ea580c")
    days = sorted({row["day"] for row in report["daily"]})
    indexed = {(row["player"], row["day"]): row for row in report["daily"]}

    def svg(fig):
        output = StringIO()
        fig.savefig(output, format="svg")
        plt.close(fig)
        text = output.getvalue()
        return text[text.index("<svg"):]

    def comparison(label, values):
        return f"<tr><td>{html.escape(label)}</td><td>{values[0]:,}</td><td>{values[1]:,}</td><td>{values[0] - values[1]:+,}</td></tr>"

    def table(rows):
        return '<div class="scroll"><table><thead><tr><th>Chỉ số</th><th>Player 0</th><th>Player 1</th><th>P0 − P1</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'

    def daily_plot(ax, key, title, action=None):
        for player in (0, 1):
            values = [indexed[player, day][key].get(action, 0) if action else indexed[player, day][key] for day in days]
            ax.bar([day + (-.2 if player == 0 else .2) for day in days], values,
                   width=.38, color=colors[player], label=f"Player {player}")
        ax.set(title=title, xlabel="Ngày (index từ 0)", ylabel="Số ô" if "tiles" in key else "Số lượt")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=.2)
        ax.legend()

    metrics = (("hires", "Hand thuê thành công"), ("crop_deaths", "CROP chết → WEED"),
               ("animal_losses", "ANIMAL biến mất"))
    summary = table([comparison(label, [row[key] for row in totals]) for key, label in metrics])
    summary += table([comparison("Ô đang sản xuất cuối replay", [row["used_tiles_final"] for row in totals])])
    tile_metrics = (("used_tiles", "Ô đang sản xuất"), ("crop_tiles", "Ô CROP"),
                    ("animal_tiles", "Ô ANIMAL"), ("empty_structures", "Chuồng/pasture trống"),
                    ("unlocked_tiles", "Ô đã mở khóa"))
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), layout="constrained")
    for ax, (key, label) in zip(axes, (*metrics, tile_metrics[0])):
        daily_plot(ax, key, label + " theo ngày")
    overview = svg(fig)
    observed = {op for row in totals for op in row["actions"]}
    actions = list(ACTIONS[:-1]) + sorted(observed - set(ACTIONS[:-1]))
    action_rows = []
    for op in actions:
        values = [row["actions"].get(op, 0) for row in totals]
        effective = [row["effective_actions"].get(op, 0) for row in totals]
        action_rows.append(comparison(op + " / gửi lệnh", values))
        action_rows.append(comparison(op + " / có thay đổi", effective))
    fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")
    for player in (0, 1):
        ax.barh([i + (-.2 if player == 0 else .2) for i in range(len(actions))],
                [totals[player]["actions"].get(op, 0) for op in actions],
                height=.38, label=f"Player {player}", color=colors[player])
    ax.set_yticks(range(len(actions)), actions)
    ax.invert_yaxis()
    ax.set(title="Tổng action của farmer + hands", xlabel="Số lượt gửi lệnh")
    ax.grid(axis="x", alpha=.2)
    ax.legend()
    action_chart = svg(fig)
    fig, axes = plt.subplots((len(actions) + 1) // 2, 2, figsize=(14, 3 * ((len(actions) + 1) // 2)), layout="constrained")
    for ax, op in zip(axes.flat, actions):
        daily_plot(ax, "actions", op, action=op)
    for ax in list(axes.flat)[len(actions):]:
        ax.set_visible(False)
    daily_actions = svg(fig)
    reasons = {"decay": "Hết vòng đời / decay", "unwatered": "Thiếu nước",
               "unfed": "Thiếu thức ăn", "unknown": "Không xác định"}
    loss_rows = ''.join(
        '<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in (
            event["day"], event["step"], f'Player {event["player"]}',
            tuple(event["position"]), event["producer"], reasons[event["cause"]])) + '</tr>'
        for event in report["loss_events"]
    ) or '<tr><td colspan="6">Không có sự kiện mất cây/con vật.</td></tr>'
    daily_rows = []
    for day in days:
        for key, label in (*metrics, *tile_metrics):
            partial = " (chưa đủ ngày)" if not indexed[0, day]["complete_day"] else ""
            daily_rows.append(comparison(f"Ngày {day}{partial} / {label}", [indexed[p, day][key] for p in (0, 1)]))
    return f'''<!doctype html><html lang="vi"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vận hành nông trại — Player 0 và Player 1</title>
<style>body{{font:16px/1.6 system-ui;margin:0;background:#f3f6f2;color:#183325}}
main{{max-width:1400px;margin:auto;padding:24px}}section{{background:white;padding:24px;margin:24px 0;border-radius:12px}}
svg{{width:100%;height:auto}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}
th,td{{padding:8px;text-align:right;border-bottom:1px solid #ddd}}th:first-child,td:first-child{{text-align:left}}
.p0{{color:{colors[0]}}}.p1{{color:{colors[1]}}}a{{color:#2563eb}}</style><main>
<h1>Vận hành nông trại</h1><p>Replay: {html.escape(source)}</p>
<p><b class="p0">Player 0: xanh</b> · <b class="p1">Player 1: cam</b> · Chênh lệch = P0 − P1.</p>
<section><h2>Cách đọc số liệu</h2><ul>
<li>Hand: số lần HIRE thành công trong ngày, không gồm farmer; tính cả thuê ở lượt cuối ngày và thuê miễn phí.</li>
<li>CROP chết: ô PLANT sau action chuyển thành WEED sau decay/refresh; không tính cỏ ngẫu nhiên trên ô trống, sau DIG hoặc thu hoạch.</li>
<li>ANIMAL biến mất: con vật còn trên ô sau action nhưng không còn trong observation tiếp theo. Không tính con vật còn trong shed/inventory.</li>
<li>Ô đang sản xuất = ô CROP + ô có ANIMAL ở observation cuối ngày (hoặc cuối replay nếu ngày chưa đủ). Không gồm WEED, ô trống, chuồng/pasture trống và đất khóa. Đây là số ô tại thời điểm chốt, không phải tổng lượt dùng ô.</li>
<li>Action: farmer + hand đã tồn tại đầu lượt; MOVE gộp NORTH/SOUTH/EAST/WEST. Thiếu action của worker được tính PASS; bỏ action gửi cho hand chưa tồn tại.</li>
<li>“Có thay đổi” nghĩa là action làm thay đổi farm/private khi chạy lại engine; “gửi lệnh” gồm cả no-op. PASS không có thay đổi. Market orders không nằm trong bảng action.</li>
<li>Ngày tính theo lượt gây sự kiện, bắt đầu từ 0; refresh cuối ngày thuộc ngày vừa kết thúc. Ngày cuối có thể chưa đủ lượt.</li>
</ul></section><section><h2>Tổng kết hai player</h2>{summary}{overview}</section>
<section><h2>Action: tổng trận</h2>{action_chart}{table(action_rows)}</section>
<section><h2>Action theo từng ngày</h2>{daily_actions}</section>
<section><h2>Số liệu từng ngày</h2>{table(daily_rows)}</section>
<section><h2>Chi tiết cây chết / con vật biến mất</h2><div class="scroll"><table><thead>
<tr><th>Ngày</th><th>Step</th><th>Player</th><th>Ô (x, y)</th><th>Loại</th><th>Nguyên nhân</th></tr>
</thead><tbody>{loss_rows}</tbody></table></div></section></main></html>'''
