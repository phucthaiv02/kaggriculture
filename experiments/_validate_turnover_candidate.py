from pathlib import Path


path = Path("agents/expansion_agent.py")
text = path.read_text()

old = "from agents.planner import SEASON_END_DAY, plan_targets\n"
new = old + "from agents.schedules import cycle_finished\n"
if text.count(old) != 1:
    raise SystemExit(f"planner import count={text.count(old)}")
text = text.replace(old, new, 1)

old = '''        if hour == 0 or new_positions:
            state["opening_active"] = opening_governs(obs, targets, positions)
            if not state["opening_active"]:
                state["pending_targets"].update(new_positions)
                if hour == 0 and not state["pending_targets"]:
                    unrealized_committed = (
                        state["committed_targets"] - physical_positions
                    )
                    state["pending_targets"].update(
                        active_position_set - unrealized_committed
                    )
        pinned_animals = reconcile_animals(obs, targets)
'''
new = '''        if hour == 0 or new_positions:
            state["opening_active"] = opening_governs(obs, targets, positions)
            if not state["opening_active"]:
                state["pending_targets"].update(new_positions)
                if hour == 0 and not state["pending_targets"]:
                    unrealized_committed = (
                        state["committed_targets"] - physical_positions
                    )
                    state["pending_targets"].update(
                        active_position_set - unrealized_committed
                    )

        # When a crop cycle ends, the standing target is already the planned
        # successor unless it is absent or can no longer start in the horizon.
        # Resolve that dependency before build_tasks sees _pending_targets;
        # do not reprice a valid target merely because HARVEST happens today.
        turnover_positions = set()
        unresolved_turnover = set()
        if hour == 0 and not state["opening_active"]:
            for y, row in enumerate(farm["tiles"]):
                for x, tile in enumerate(row):
                    if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
                        continue
                    crop = tile["crop"]
                    if not cycle_finished(crop, day - tile["planted_day"], tile):
                        continue
                    position = (x, y)
                    turnover_positions.add(position)
                    current = targets.get(position)
                    if current and can_start_today(current[0], obs):
                        state["pending_targets"].discard(position)
                    else:
                        unresolved_turnover.add(position)
                        state["pending_targets"].add(position)

        pinned_animals = reconcile_animals(obs, targets)
'''
if text.count(old) != 1:
    raise SystemExit(f"turnover scan anchor count={text.count(old)}")
text = text.replace(old, new, 1)

old = '''        obs["_committed_targets"] = set(state["committed_targets"])
        if hour == 0 and not state["opening_active"] and state["pending_targets"]:
            pending_before = set(state["pending_targets"])
            state["pending_targets"] = set(plan_targets(
                obs, targets, positions, effective_end,
                max_positions=TARGETS_PER_DAY,
                replan_positions=state["pending_targets"],
            ))
'''
new = '''        obs["_committed_targets"] = set(state["committed_targets"])
        if hour == 0 and not state["opening_active"] and unresolved_turnover:
            still_unresolved = set(plan_targets(
                obs, targets, positions, effective_end,
                max_positions=len(unresolved_turnover),
                replan_positions=unresolved_turnover,
            ))
            resolved_turnover = unresolved_turnover - still_unresolved
            state["pending_targets"].difference_update(resolved_turnover)
            state["pending_targets"].update(still_unresolved)

        if hour == 0 and not state["opening_active"] and state["pending_targets"]:
            pending_before = set(state["pending_targets"])
            state["pending_targets"] = set(plan_targets(
                obs, targets, positions, effective_end,
                max_positions=TARGETS_PER_DAY,
                replan_positions=state["pending_targets"],
            ))
'''
if text.count(old) != 1:
    raise SystemExit(f"turnover planning anchor count={text.count(old)}")
text = text.replace(old, new, 1)

path.write_text(text)
