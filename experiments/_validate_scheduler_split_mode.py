from pathlib import Path

# Build a post-opening scheduler candidate without touching the legacy scheduler.
legacy = Path("agents/scheduler.py")
post = Path("agents/scheduler_post.py")
post.write_text(legacy.read_text())

# Remove only action-label ordering/rescue behavior. Keep the legacy packing,
# head-count search, terminal accounting and route scoring otherwise intact.
patcher = Path("experiments/_validate_scheduler_surgical.py").read_text()
exec(compile(patcher, "_validate_scheduler_surgical.py", "exec"), {})

# Expansion uses the legacy scheduler only while the opening controller governs.
# Every post-opening call, including a global replan, routes through scheduler_post.
path = Path("agents/expansion_agent.py")
text = path.read_text()
anchor = (
    "from agents.scheduler import MAX_HANDS, build_queues, hands_needed, nearest_shed, route\n"
)
insert = anchor + (
    "from agents.scheduler_post import (\n"
    "    build_queues as post_build_queues,\n"
    "    hands_needed as post_hands_needed,\n"
    ")\n"
)
if text.count(anchor) != 1:
    raise SystemExit(f"scheduler import anchor count={text.count(anchor)}")
text = text.replace(anchor, insert, 1)

# Rewrite only call sites; imports are unaffected because they do not contain '('.
text = text.replace("hands_needed(", "_daily_hands_needed(")
text = text.replace("build_queues(", "_daily_build_queues(")

state_anchor = "    opening_governs = make_opening_controller()\n"
helpers = state_anchor + '''\n    def _daily_hands_needed(*args, **kwargs):\n        scheduler = hands_needed if state["opening_active"] else post_hands_needed\n        return scheduler(*args, **kwargs)\n\n    def _daily_build_queues(*args, **kwargs):\n        scheduler = build_queues if state["opening_active"] else post_build_queues\n        return scheduler(*args, **kwargs)\n'''
if text.count(state_anchor) != 1:
    raise SystemExit(f"opening controller anchor count={text.count(state_anchor)}")
text = text.replace(state_anchor, helpers, 1)

path.write_text(text)
