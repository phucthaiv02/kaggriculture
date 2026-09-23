from pathlib import Path

path = Path('agents/scheduler.py')
text = path.read_text()

anchor = '''def _pack(\n    tasks,\n    worker_starts,\n    budgets,\n    shed_access=SHED_ACCESS,\n    optimize_routes=True,\n):\n'''
if anchor not in text:
    raise SystemExit('pack anchor not found')
helper = '''def _full_assignment_step_lower_bound(tasks, worker_starts):\n    \"\"\"Necessary total-step bound for assigning every task.\n\n    Any complete schedule must execute every task action and at least one\n    PICKUP for each distinct required item.  Its worker routes must also\n    connect every distinct task position to at least one worker start.  The\n    multi-source Manhattan MST is a lower bound on that route length: treating\n    every worker start as connected to a virtual root for free can only make\n    the network cheaper than real worker routes.\n\n    If this bound already exceeds aggregate worker budgets, no packing variant\n    can possibly assign every task.  In that case _pack historically returns\n    the default greedy result after spending time on variants that all fail,\n    so callers may skip those variants without changing the result.\n    \"\"\"\n    if not tasks:\n        return 0\n    actions = sum(len(task.actions) for task in tasks)\n    pickups = len({\n        item for task in tasks for item, amount in task.needs.items()\n        if amount > 0\n    })\n    remaining = set(task.position for task in tasks)\n    if not remaining:\n        return actions + pickups\n\n    starts = tuple(worker_starts)\n    # _pack always has at least the farmer, but keep the helper total for\n    # isolated callers/tests.  With no starts, no non-empty assignment exists.\n    if not starts:\n        return float('inf')\n\n    def distance(a, b):\n        return abs(a[0] - b[0]) + abs(a[1] - b[1])\n\n    best = {\n        position: min(distance(position, start) for start in starts)\n        for position in remaining\n    }\n    travel = 0\n    while best:\n        position, edge = min(\n            best.items(), key=lambda item: (item[1], item[0][1], item[0][0])\n        )\n        travel += edge\n        del best[position]\n        for other in tuple(best):\n            step = distance(position, other)\n            if step < best[other]:\n                best[other] = step\n    return actions + pickups + travel\n\n\n''' + anchor
text = text.replace(anchor, helper, 1)
old = '''    best = _pack_greedy(tasks, worker_starts, budgets, shed_access)\n    if not tasks or sum(len(t.actions) for t in tasks) > sum(budgets):\n        return best\n'''
new = '''    best = _pack_greedy(tasks, worker_starts, budgets, shed_access)\n    if not tasks or _full_assignment_step_lower_bound(tasks, worker_starts) > sum(budgets):\n        return best\n'''
if old not in text:
    raise SystemExit('legacy pack lower-bound anchor not found')
text = text.replace(old, new, 1)
path.write_text(text)
