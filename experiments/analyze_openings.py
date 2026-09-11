"""Find repeated opening sequences using visible farm changes in replay JSON."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

from experiments.replay_data import read_replay, write_json


def farm_at(frame, player):
    for state in frame:
        farms = state.get("observation", {}).get("farms")
        if farms is not None:
            return farms[player]
    raise ValueError("replay frame has no public farms")


def tile_name(tile):
    if not isinstance(tile, dict):
        return None
    if tile.get("kind") == "PLANT":
        return f"PLANT:{tile['crop']}"
    if tile.get("animal"):
        return f"PLACE:{tile['animal']}"
    if tile.get("kind") in ("COOP", "PASTURE"):
        return f"BUILD:{tile['kind']}"
    return None


def visible_events(before, after):
    """Changes visible after the turn; does not infer successful market fills."""
    events = Counter()
    for y, row in enumerate(after["tiles"]):
        for x, tile in enumerate(row):
            old = before["tiles"][y][x]
            name = tile_name(tile)
            replanted = (name and name.startswith("PLANT:") and isinstance(old, dict)
                         and old.get("planted_day") != tile.get("planted_day"))
            if name and (name != tile_name(old) or replanted):
                events[name] += 1
    hires = after.get("hires_today", len(after.get("hands", []))) - before.get(
        "hires_today", len(before.get("hands", [])))
    if hires > 0:
        events["HIRE"] = hires
    for quadrant in set(after.get("unlocked_quadrants", [])) - set(before.get("unlocked_quadrants", [])):
        events[f"BUY_LAND:{quadrant}"] += 1
    return events


def submitted_orders(action):
    counts = Counter()
    if not isinstance(action, dict):
        return counts
    for order in action.get("market", []) or []:
        if not isinstance(order, list) or not order or not isinstance(order[0], str):
            continue
        key = order[0] + (":" + str(order[1]) if len(order) > 1 else "")
        quantity = order[2] if len(order) > 2 else 1
        if isinstance(quantity, (float, int)) and not isinstance(quantity, bool) and math.isfinite(quantity) and quantity > 0:
            counts[key] += quantity
    return counts


def cohort_info(replay):
    configuration = {k: v for k, v in replay["configuration"].items()
                     if k not in ("seed", "actTimeout", "runTimeout")}
    info = {"version": replay.get("version"), "module_version": replay.get("module_version"),
            "configuration": configuration}
    key = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()[:12]
    return key, info


def extract_opening(replay, player, days=10):
    if days < 1 or player not in (0, 1):
        raise ValueError("days must be positive and player must be 0 or 1")
    turns = int(replay["configuration"].get("turnsPerDay", 24))
    if turns < 1:
        raise ValueError("turnsPerDay must be positive")
    steps = replay["steps"]
    events, daily, sequence, first_seen = [], {}, [], {}
    total_visible, total_submitted = Counter(), Counter()
    ticks = []
    for frame in range(1, len(steps)):
        observation = steps[frame - 1][0].get("observation", {})
        tick = int(observation.get("step", frame - 1))
        if tick >= days * turns:
            break
        if tick != frame - 1:
            raise ValueError("opening requires contiguous replay frames beginning at step 0")
        ticks.append(tick)
        before, after = farm_at(steps[frame - 1], player), farm_at(steps[frame], player)
        changes = visible_events(before, after)
        orders = submitted_orders(steps[frame][player].get("action"))
        total_visible.update(changes)
        total_submitted.update(orders)
        if changes:
            # One token per turn: simultaneous unit actions have no invented ordering.
            token = " + ".join(sorted(changes))
            sequence.append(token)
            events.append({"step": tick, "day": tick // turns, "hour": tick % turns,
                           "changes": dict(changes), "token": token})
            for name in changes:
                first_seen.setdefault(name, tick)
        day = tick // turns
        record = daily.setdefault(day, {"day": day, "visible_changes": Counter(),
                                        "submitted_market_quantities": Counter()})
        record["visible_changes"].update(changes)
        record["submitted_market_quantities"].update(orders)
        record.update({"last_step": tick, "complete_day": (tick + 1) % turns == 0,
                       "cash": after["money"],
                       "assets": dict(Counter(name for row in after["tiles"]
                                              for tile in row if (name := tile_name(tile)))),
                       "land": after.get("unlocked_quadrants", [])})
    final = steps[-1]
    rewards = [s.get("reward") for s in final]
    valid_result = (all(s.get("status") == "DONE" for s in final) and
                    all(isinstance(r, (int, float)) and math.isfinite(r) for r in rewards))
    outcome = None
    if valid_result:
        outcome = "win" if rewards[player] > rewards[1 - player] else (
            "loss" if rewards[player] < rewards[1 - player] else "draw")
    key, info = cohort_info(replay)
    return {"player": player, "cohort": key, "cohort_info": info,
            "opening_complete": len(ticks) == days * turns,
            "opening_steps": len(ticks), "opening_days": days,
            "sequence": sequence, "events": events, "first_seen": first_seen,
            "visible_changes": dict(total_visible),
            "submitted_market_quantities": dict(total_submitted),
            "daily": list(daily.values()), "outcome": outcome,
            "final_reward": rewards[player] if valid_result else None,
            "final_margin": rewards[player] - rewards[1 - player] if valid_result else None,
            "statuses": [s.get("status") for s in final]}


def outcome_stats(rows):
    scored = [r for r in rows if r["outcome"] is not None]
    return {"samples": len(rows), "episodes": len({r["episode_id"] for r in rows}),
            "teams": len({r["team_id"] for r in rows}), "scored_samples": len(scored),
            "wins": sum(r["outcome"] == "win" for r in scored),
            "draws": sum(r["outcome"] == "draw" for r in scored),
            "win_rate": mean(r["outcome"] == "win" for r in scored) if scored else None,
            "mean_final_margin": mean(r["final_margin"] for r in scored) if scored else None}


def mine_patterns(rows, min_support=2, max_length=4):
    """Per-opening support, never occurrence counts; mine within each config/version."""
    if min_support < 1 or max_length < 1:
        raise ValueError("pattern limits must be positive")
    cohorts = defaultdict(list)
    for row in rows:
        if row["opening_complete"]:
            cohorts[row["cohort"]].append(row)
    results = []
    for cohort, samples in sorted(cohorts.items()):
        memberships = defaultdict(list)
        for row in samples:
            sequence = row["sequence"]
            found = set()
            for size in range(1, min(max_length, len(sequence)) + 1):
                found.add(("prefix", tuple(sequence[:size])))
                if size >= 2:
                    for start in range(len(sequence) - size + 1):
                        found.add(("motif", tuple(sequence[start:start + size])))
            for pattern in found:
                memberships[pattern].append(row)
        for (kind, pattern), matching in memberships.items():
            if len(matching) < min_support:
                continue
            results.append({"cohort": cohort, "kind": kind, "pattern": list(pattern),
                            "support": len(matching), "support_fraction": len(matching) / len(samples),
                            **outcome_stats(matching),
                            "examples": [{"episode_id": r["episode_id"], "player": r["player"],
                                          "team": r["team_name"]} for r in matching[:5]]})
    return sorted(results, key=lambda p: (-p["support"], p["cohort"], p["kind"], p["pattern"]))


def opening_families(rows, min_support=2):
    """Exact visible-event timing/count signatures at multiple window lengths."""
    groups = defaultdict(list)
    for row in rows:
        turns = int(row["cohort_info"]["configuration"].get("turnsPerDay", 24))
        for days in sorted({3, 5, 7, 10, row["opening_days"]}):
            if row["opening_steps"] < days * turns:
                continue
            signature = [(event["step"], sorted(event["changes"].items()))
                         for event in row["events"] if event["step"] < days * turns]
            digest = hashlib.sha256(json.dumps(signature).encode()).hexdigest()[:16]
            groups[(row["cohort"], days, digest)].append(row)
    return sorted([{"cohort": cohort, "days": days, "signature": digest,
                    **outcome_stats(samples),
                    "team_names": sorted({s["team_name"] for s in samples}),
                    "examples": [{"episode_id": s["episode_id"], "player": s["player"]}
                                 for s in samples[:5]]}
                   for (cohort, days, digest), samples in groups.items()
                   if len(samples) >= min_support],
                  key=lambda group: (group["days"], -group["samples"], group["signature"]))


def analyze_inputs(path, days=10, all_players=False):
    manifest_path = path / "manifest.json" if path.is_dir() else path
    rows, skipped, seen = [], [], set()
    if manifest_path.name == "manifest.json" and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        inputs = [(manifest_path.parent / item["file"], episode_id, item)
                  for episode_id, item in manifest["episodes"].items()]
    else:
        paths = sorted(path.glob("*.json")) if path.is_dir() else [path]
        inputs = [(p, p.stem, {}) for p in paths
                  if not p.name.endswith((".result.json", ".sales.json"))]
    for replay_path, episode_id, metadata in inputs:
        try:
            digest = hashlib.sha256(replay_path.read_bytes()).hexdigest()
            if metadata.get("sha256") and digest != metadata["sha256"]:
                raise ValueError("replay checksum does not match manifest")
            replay = read_replay(replay_path)
            agents = {int(a["index"]): a for a in metadata.get("metadata", {}).get("agents", [])}
            players = range(2) if all_players or not metadata else metadata["target_players"]
            pending = []
            for player in players:
                identity = (digest, player)
                if identity in seen:
                    continue
                agent = agents.get(player, {})
                row = extract_opening(replay, player, days)
                row.update({"episode_id": str(episode_id), "file": str(replay_path),
                            "team_id": agent.get("teamId", f"local-player-{player}"),
                            "team_name": agent.get("teamName", f"Local player {player}"),
                            "submission_id": agent.get("submissionId")})
                pending.append((identity, row))
            for identity, row in pending:
                seen.add(identity)
                rows.append(row)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            skipped.append({"file": str(replay_path), "error": str(exc)})
    return rows, skipped


def export_report(rows, skipped, output, min_support=2, max_length=4):
    patterns = mine_patterns(rows, min_support, max_length)
    groups = defaultdict(list)
    for row in rows:
        if row["opening_complete"]:
            groups[(row["cohort"], row["team_id"])].append(row)
    teams = []
    for (cohort, team_id), samples in groups.items():
        milestones = defaultdict(list)
        for sample in samples:
            for event, step in sample["first_seen"].items():
                milestones[event].append(step)
        checkpoints = {}
        for window in sorted({3, 5, 7, 10, samples[0]["opening_days"]}):
            day_rows = [day for sample in samples for day in sample["daily"]
                        if day["day"] == window - 1 and day["complete_day"]]
            if day_rows:
                assets = sorted({asset for day in day_rows for asset in day["assets"]})
                checkpoints[str(window)] = {"samples": len(day_rows),
                    "median_cash": median(day["cash"] for day in day_rows),
                    "median_assets": {asset: median(day["assets"].get(asset, 0) for day in day_rows)
                                      for asset in assets},
                    "median_land_quadrants": median(len(day["land"]) for day in day_rows)}
        teams.append({"cohort": cohort, "team_id": team_id, "team_name": samples[0]["team_name"],
                      **outcome_stats(samples),
                      "checkpoints": checkpoints,
                      "first_seen": {event: {"samples": len(values), "median_step": median(values)}
                                     for event, values in sorted(milestones.items())}})
    report = {"schema_version": 1, "samples": len(rows),
              "complete_openings": sum(r["opening_complete"] for r in rows),
              "cohorts": {r["cohort"]: r["cohort_info"] for r in rows},
              "parameters": {"min_support": min_support, "max_length": max_length},
              "teams": teams, "patterns": patterns,
              "families": opening_families(rows, min_support), "skipped": skipped}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "openings.json", rows)
    write_json(output / "patterns.json", report)
    fields = ["episode_id", "player", "team_id", "team_name", "submission_id", "cohort",
              "opening_complete", "opening_steps", "outcome", "final_reward", "final_margin",
              "sequence", "first_seen", "visible_changes", "submitted_market_quantities"]
    with (output / "openings.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if isinstance(row[key], (dict, list))
                             else row[key] for key in fields})
    lines = ["# Opening patterns", "", f"{len(rows)} player openings; "
             f"{report['complete_openings']} complete; {len(skipped)} files skipped.", "",
             "Patterns use visible planting, structures, animal placement, hires and land changes. "
             "Each turn is one unordered bundle; quantities, coordinates and idle/movement turns are omitted. "
             "Motifs are contiguous in that event sequence, not necessarily in game time. "
             "Submitted market quantities are requests, not confirmed fills. "
             "Changes that disappear within a turn (including hires at day rollover) are not recoverable here.", "",
             "Support counts each player opening once. Partial openings are excluded. "
             "Only DONE/DONE games with numeric rewards enter win rates; draws count as non-wins. "
             "Results are descriptive, not evidence that an opening causes wins. "
             "The two players of one match are correlated samples. Configurations/versions are kept separate.", ""]
    def safe(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    for cohort in report["cohorts"]:
        lines += [f"## Configuration {cohort}", "", "| Team | Openings | Scored | Win rate |",
                  "|---|---:|---:|---:|"]
        for team in teams:
            if team["cohort"] == cohort:
                rate = f"{team['win_rate']:.1%}" if team["win_rate"] is not None else "—"
                lines.append(f"| {safe(team['team_name'])} | {team['samples']} | {team['scored_samples']} | {rate} |")
        for kind in ("prefix", "motif"):
            lines += ["", f"### Frequent {kind} sequences", "",
                      "| Pattern | Support | Share | Teams | Win rate |", "|---|---:|---:|---:|---:|"]
            for pattern in [p for p in patterns if p["cohort"] == cohort and p["kind"] == kind][:20]:
                rate = f"{pattern['win_rate']:.1%}" if pattern["win_rate"] is not None else "—"
                lines.append(f"| {safe(' → '.join(pattern['pattern']))} | {pattern['support']} | "
                             f"{pattern['support_fraction']:.1%} | {pattern['teams']} | {rate} |")
        lines += ["", "### Production checkpoints (end of day count, medians)", "",
                  "| Team | Days elapsed | Cash | Land quadrants | Assets |", "|---|---:|---:|---:|---|"]
        for team in teams:
            if team["cohort"] != cohort:
                continue
            for window, checkpoint in team["checkpoints"].items():
                assets = ", ".join(f"{key}={value:g}" for key, value in checkpoint["median_assets"].items())
                lines.append(f"| {safe(team['team_name'])} | {window} | {checkpoint['median_cash']:,.0f} | "
                             f"{checkpoint['median_land_quadrants']:g} | {safe(assets)} |")
        lines += ["", "### Repeated timed openings", "",
                  "These match visible event types, counts and steps, excluding coordinates, "
                  "movement, maintenance and market orders. They do not prove identical agents.", "",
                  "| Days | Signature | Openings | Teams |", "|---:|---|---:|---|"]
        for family in report["families"]:
            if family["cohort"] == cohort:
                lines.append(f"| {family['days']} | {family['signature']} | {family['samples']} | "
                             f"{safe(', '.join(family['team_names']))} |")
        lines.append("")
    if skipped:
        lines += ["## Skipped inputs", ""] + [f"- {safe(s['file'])}: {safe(s['error'])}" for s in skipped]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="collector directory, manifest.json, replay JSON, or replay directory")
    parser.add_argument("--days", type=int, default=10, help="opening window: days 0..N-1 (default: 10)")
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=4)
    parser.add_argument("--all-players", action="store_true", help="include opponents as well as selected top submissions")
    parser.add_argument("--output-dir", type=Path, default=Path("replays/opening_analysis"))
    args = parser.parse_args()
    if min(args.days, args.min_support, args.max_length) < 1:
        parser.error("days and pattern limits must be positive")
    if not args.input.exists():
        parser.error(f"input not found: {args.input}")
    try:
        rows, skipped = analyze_inputs(args.input, args.days, args.all_players)
        report = export_report(rows, skipped, args.output_dir, args.min_support, args.max_length)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Opening analysis failed: {exc}\n")
    print(f"{report['complete_openings']}/{len(rows)} complete openings; "
          f"{len(report['patterns'])} patterns; {len(skipped)} skipped files. "
          f"Report: {args.output_dir / 'summary.md'}")
    if skipped or not report["complete_openings"]:
        parser.exit(1)


if __name__ == "__main__":
    main()
