from pathlib import Path

# Apply the resource-safe post-opening scheduler candidate first.
base = Path("experiments/_validate_scheduler_resource_safe.py").read_text()
exec(compile(base, "_validate_scheduler_resource_safe.py", "exec"), {})

# A daily schedule may not switch execution regimes in the middle of the day.
# If hour 0 was governed by the opening book, opening exceptions remain valid
# for that whole schedule. A mid-day land unlock is deferred; handoff happens
# when the next day's hour-0 plan is built.
path = Path("agents/expansion_agent.py")
text = path.read_text()
old = '''        if hour > 0 and new_positions:
            state["deferred_expansion_positions"].update(new_positions)
        if hour == 0 or new_positions:
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
'''
new = '''        if hour > 0 and new_positions:
            state["deferred_expansion_positions"].update(new_positions)
            # New land is next-day planning input.  Do not change the regime
            # or mutate the running daily schedule because it appeared mid-day.
            if not state["opening_active"]:
                state["pending_targets"].update(new_positions)
        if hour == 0:
            state["opening_active"] = opening_governs(obs, targets, positions)
            if not state["opening_active"]:
                state["pending_targets"].update(new_positions)
                if not state["pending_targets"]:
                    unrealized_committed = (
                        state["committed_targets"] - physical_positions
                    )
                    state["pending_targets"].update(
                        active_position_set - unrealized_committed
                    )
'''
if text.count(old) != 1:
    raise SystemExit(f"daily regime anchor count={text.count(old)}")
text = text.replace(old, new, 1)
path.write_text(text)
