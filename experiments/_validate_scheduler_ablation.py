import os
from pathlib import Path

variant = os.environ["SCHEDULER_ABLATION"]
path = Path("agents/scheduler_post.py")
text = Path("agents/scheduler.py").read_text()

if variant in {"no_animal_rescue", "all_minimal"}:
    old = '''    if best[1] and any(_is_animal_service(task) for task in tasks):
        grouped = _pack_greedy(
            tasks, worker_starts, budgets, shed_access, group_animal=True
        )
        if not grouped[1]:
            return grouped

'''
    if text.count(old) != 1:
        raise SystemExit(f"animal rescue anchor count={text.count(old)}")
    text = text.replace(old, "", 1)

if variant in {"no_timesensitive_cost", "all_minimal"}:
    old = '''                    # Time-sensitive work must finish early for survival,
                    # crop expiry and same-day sales. For ordinary work,
                    # minimize extra travel and pickups instead.
                    cost = projected if time_sensitive and not variant else projected - lengths[worker]
'''
    new = '''                    # Route choice uses incremental queue cost only.
                    cost = projected - lengths[worker]
'''
    if text.count(old) != 1:
        raise SystemExit(f"time-sensitive cost anchor count={text.count(old)}")
    text = text.replace(old, new, 1)

if variant in {"no_route_priority", "all_minimal"}:
    old = '''                # Buckets are already priority-sorted; only the two new
                # neighbours can violate the invariant after insertion.
                if insertion and priorities[id(bucket[insertion - 1])] > priority:
                    continue
                if insertion < len(bucket) and priority > priorities[id(bucket[insertion])]:
                    continue
'''
    if text.count(old) != 1:
        raise SystemExit(f"route-priority anchor count={text.count(old)}")
    text = text.replace(old, "", 1)

if variant in {"no_feed_order", "all_minimal"}:
    old = '''                    for insertion in range(len(bucket)+1):
                        # Insert after fixed work so planting and harvest are not delayed.
                        if any(not service(t) for t in bucket[insertion:]):
                            continue
                        merged = bucket[:insertion] + [task] + bucket[insertion:]
'''
    new = '''                    for insertion in range(len(bucket)+1):
                        merged = bucket[:insertion] + [task] + bucket[insertion:]
'''
    if text.count(old) != 1:
        raise SystemExit(f"feed-order anchor count={text.count(old)}")
    text = text.replace(old, new, 1)

path.write_text(text)

# Dispatch legacy scheduler while opening_book governs, candidate scheduler after handoff.
agent = Path("agents/expansion_agent.py")
a = agent.read_text()
anchor = "from agents.scheduler import MAX_HANDS, build_queues, hands_needed, nearest_shed, route\n"
insert = anchor + (
    "from agents.scheduler_post import (\n"
    "    build_queues as post_build_queues,\n"
    "    hands_needed as post_hands_needed,\n"
    ")\n"
)
if a.count(anchor) != 1:
    raise SystemExit(f"scheduler import anchor count={a.count(anchor)}")
a = a.replace(anchor, insert, 1)
a = a.replace("hands_needed(", "_daily_hands_needed(")
a = a.replace("build_queues(", "_daily_build_queues(")
state_anchor = "    opening_governs = make_opening_controller()\n"
helpers = state_anchor + '''\n    def _daily_hands_needed(*args, **kwargs):\n        scheduler = hands_needed if state["opening_active"] else post_hands_needed\n        return scheduler(*args, **kwargs)\n\n    def _daily_build_queues(*args, **kwargs):\n        scheduler = build_queues if state["opening_active"] else post_build_queues\n        return scheduler(*args, **kwargs)\n'''
if a.count(state_anchor) != 1:
    raise SystemExit(f"opening controller anchor count={a.count(state_anchor)}")
a = a.replace(state_anchor, helpers, 1)
agent.write_text(a)
