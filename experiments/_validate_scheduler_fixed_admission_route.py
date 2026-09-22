from pathlib import Path

# Start from the validated generic-retry candidate: opening stays legacy;
# post-opening has no time-sensitive/feed ordering and no animal-specific rescue.
base = Path("experiments/_validate_scheduler_generic_retry.py").read_text()
exec(compile(base, "_validate_scheduler_generic_retry.py", "exec"), {})

path = Path("agents/scheduler_post.py")
text = path.read_text()

# Route a FIXED admitted task set with all tasks equal from the router's point
# of view.  mandatory remains meaningful only in the preceding admission step.
anchor = "\ndef build_queues(\n"
helper = r'''

def _route_admitted_without_class_order(tasks, starts, budgets, shed_access=SHED_ACCESS):
    """Route a fixed workload without mandatory/optional execution ordering.

    Clone only the admission class bit.  All task actions, dependencies, value,
    inventory needs, cashout constraints and positions are unchanged.  The
    generic bounded pack search then optimizes geometry for the already chosen
    workload.
    """
    import copy

    clones = []
    originals = {}
    for task in tasks:
        clone = copy.copy(task)
        clone.mandatory = None
        clones.append(clone)
        originals[id(clone)] = task

    clone_buckets, clone_missing = _pack(clones, starts, budgets, shed_access)
    buckets = [
        [originals[id(task)] for task in bucket]
        for bucket in clone_buckets
    ]
    missing = [originals[id(task)] for task in clone_missing]
    return buckets, missing
'''
if text.count(anchor) != 1:
    raise SystemExit(f"build_queues anchor count={text.count(anchor)}")
text = text.replace(anchor, helper + anchor, 1)

old = '''    buckets, unassigned = _pack(tasks, starts, budgets, shed_access)\n\n    demand = sum(t.needs.get("WHEAT", 0) for t in tasks)\n'''
new = '''    # Stage 1: choose the feasible/economic workload exactly as the stable
    # scheduler does.  This diagnostic deliberately freezes admission so the
    # effect of ROUTE ordering can be measured independently.
    admitted_buckets, unassigned = _pack(tasks, starts, budgets, shed_access)
    admitted = [task for bucket in admitted_buckets for task in bucket]

    # Stage 2: route that fixed set without mandatory/optional action ordering.
    buckets, route_missing = _route_admitted_without_class_order(
        admitted, starts, budgets, shed_access
    )
    if route_missing:
        known = {id(task) for task in unassigned}
        unassigned.extend(task for task in route_missing if id(task) not in known)

    demand = sum(t.needs.get("WHEAT", 0) for t in tasks)\n'''
if text.count(old) != 1:
    raise SystemExit(f"build assignment anchor count={text.count(old)}")
text = text.replace(old, new, 1)

path.write_text(text)
