from pathlib import Path

path = Path("agents/expansion_agent.py")
text = path.read_text()
old = '''        frozen_targets = {
            position: targets.get(position)
            for position in state["frozen_positions"]
            if position in targets
        }
        replanned = build_tasks(
            obs, frozen_targets,
            prioritize_fertilizer_drop=state["opening_active"],
        )
'''
new = '''        # Intraday reconciliation may reroute unfinished work, but it must
        # never retarget a position that was admitted by the morning plan.
        # Global targets/pending_targets continue evolving for future mornings;
        # today's executable commitment lives in daily_targets until rollover.
        frozen_targets = {
            position: state["daily_targets"].get(position)
            for position in state["frozen_positions"]
            if position in state["daily_targets"]
        }
        replan_obs = dict(obs)
        replan_obs["_pending_targets"] = (
            set(obs.get("_pending_targets", ())) - state["frozen_positions"]
        )
        replanned = build_tasks(
            replan_obs, frozen_targets,
            prioritize_fertilizer_drop=state["opening_active"],
        )
'''
assert text.count(old) == 1, text.count(old)
path.write_text(text.replace(old, new))
