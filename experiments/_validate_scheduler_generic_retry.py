from pathlib import Path

scheduler = Path("agents/scheduler_post.py")
text = Path("agents/scheduler.py").read_text()

# 1) The alternative pack is global/spatial, not animal-specific.
old = '''    # Keep the production greedy ordering exactly unchanged unless this is the
    # explicit hand-count rescue. Opening and already-valid schedules therefore
    # cannot move because of the relaxed animal grouping.
    if group_animal:
        ordered = sorted(
            tasks,
            key=lambda task: (
                priorities[id(task)],
                min(
                    abs(start[0] - task.position[0]) + abs(start[1] - task.position[1])
                    for start in worker_starts
                ),
                task.position[1],
                task.position[0],
                not task.animal_harvest,
                -len(task.actions),
            ),
        )
'''
new = '''    # A second deterministic spatial packing order is available when the
    # default packing cannot fit the complete workload at this headcount.
    # It depends only on geometry/task size, never on action labels.
    if group_animal:
        ordered = sorted(
            tasks,
            key=lambda task: (
                priorities[id(task)],
                min(
                    abs(start[0] - task.position[0]) + abs(start[1] - task.position[1])
                    for start in worker_starts
                ),
                task.position[1],
                task.position[0],
                -len(task.actions),
            ),
        )
'''
if text.count(old) != 1:
    raise SystemExit(f"spatial ordering anchor count={text.count(old)}")
text = text.replace(old, new, 1)

# 2) Remove same-pen HARVEST/FEED action-label constraints from this alternative.
old = '''        task_is_service = group_animal and _is_animal_service(task)
        for worker, bucket in enumerate(buckets):
            paired_harvest = None
            paired_service = None
            if task_is_service:
                for index, queued in enumerate(bucket):
                    if queued.position != task.position:
                        continue
                    if queued.animal_harvest:
                        paired_harvest = index
                    elif _is_animal_service(queued):
                        paired_service = index
            for insertion in range(len(bucket) + 1):
'''
new = '''        for worker, bucket in enumerate(buckets):
            for insertion in range(len(bucket) + 1):
'''
if text.count(old) != 1:
    raise SystemExit(f"paired-service setup anchor count={text.count(old)}")
text = text.replace(old, new, 1)
old = '''                # In rescue mode, tasks at the same pen may share a visit but
                # explicit HARVEST stays before that pen's FEED/CARE work.
                if task_is_service:
                    if task.animal_harvest and paired_service is not None and insertion > paired_service:
                        continue
                    if not task.animal_harvest and paired_harvest is not None and insertion <= paired_harvest:
                        continue
'''
if text.count(old) != 1:
    raise SystemExit(f"paired-service ordering anchor count={text.count(old)}")
text = text.replace(old, "", 1)

# 3) Try the alternative for any incomplete global packing, not because an
# animal-labelled task exists.
old = '''    if best[1] and any(_is_animal_service(task) for task in tasks):
        grouped = _pack_greedy(
            tasks, worker_starts, budgets, shed_access, group_animal=True
        )
        if not grouped[1]:
            return grouped
'''
new = '''    if best[1]:
        spatial = _pack_greedy(
            tasks, worker_starts, budgets, shed_access, group_animal=True
        )
        if not spatial[1]:
            return spatial
'''
if text.count(old) != 1:
    raise SystemExit(f"global retry anchor count={text.count(old)}")
text = text.replace(old, new, 1)

# 4) These two action-label ordering mechanisms were independently replayed at
# exactly the 91,362 baseline and can be removed now.
old = '''                    # Time-sensitive work must finish early for survival,
                    # crop expiry and same-day sales. For ordinary work,
                    # minimize extra travel and pickups instead.
                    cost = projected if time_sensitive and not variant else projected - lengths[worker]
'''
new = '''                    cost = projected - lengths[worker]
'''
if text.count(old) != 1:
    raise SystemExit(f"time-sensitive anchor count={text.count(old)}")
text = text.replace(old, new, 1)
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

scheduler.write_text(text)

# Dispatch opening to untouched legacy scheduler; post-opening to candidate.
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
    raise SystemExit(f"import anchor count={a.count(anchor)}")
a = a.replace(anchor, insert, 1)
a = a.replace("hands_needed(", "_daily_hands_needed(")
a = a.replace("build_queues(", "_daily_build_queues(")
state_anchor = "    opening_governs = make_opening_controller()\n"
helpers = state_anchor + '''\n    def _daily_hands_needed(*args, **kwargs):\n        fn = hands_needed if state["opening_active"] else post_hands_needed\n        return fn(*args, **kwargs)\n\n    def _daily_build_queues(*args, **kwargs):\n        fn = build_queues if state["opening_active"] else post_build_queues\n        return fn(*args, **kwargs)\n'''
if a.count(state_anchor) != 1:
    raise SystemExit(f"controller anchor count={a.count(state_anchor)}")
a = a.replace(state_anchor, helpers, 1)
agent.write_text(a)
