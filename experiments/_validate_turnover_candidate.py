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

        # A producer that ends its crop cycle today needs its successor chosen
        # before the daily task graph is built. This is a dependency of the
        # HARVEST task, not an action priority: every such position is repriced
        # at hour 0 outside the normal investment batching limit.
        turnover_positions = set()
        if hour == 0 and not state["opening_active"]:
            for y, row in enumerate(farm["tiles"]):
                for x, tile in enumerate(row):
                    if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
                        continue
                    crop = tile["crop"]
                    if cycle_finished(crop, day - tile["planted_day"], tile):
                        turnover_positions.add((x, y))
            state["pending_targets"].update(turnover_positions)

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
        if hour == 0 and not state["opening_active"] and turnover_positions:
            unresolved_turnover = set(plan_targets(
                obs, targets, positions, effective_end,
                max_positions=len(turnover_positions),
                replan_positions=turnover_positions,
            ))
            resolved_turnover = turnover_positions - unresolved_turnover
            state["pending_targets"].difference_update(resolved_turnover)
            state["pending_targets"].update(unresolved_turnover)

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
