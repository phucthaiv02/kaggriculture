"""Runner-only ablation: remove execution-order priorities, keep stable admission economics."""

from pathlib import Path

path = Path("agents/scheduler_post.py")
text = path.read_text()

start = text.index("def _priority(task):")
end = text.index("def _bucket_budget", start)
text = text[:start] + '''def _admission_class(task):
    """Hard inclusion class only; never an execution-order priority."""
    return task.mandatory is False


''' + text[end:]

start = text.index("def _pack_greedy(")
end = text.index("def _pack(", start)
text = text[:start] + '''def _pack_greedy(
    tasks,
    worker_starts,
    budgets,
    shed_access=SHED_ACCESS,
    variant=0,
):
    """Pack required work first, then optional work, without action ordering.

    Required-vs-optional is an admission constraint only. Once a task is
    admitted to a worker bucket, its insertion point is chosen solely by route
    cost and capacity. Action labels, urgency, crop/animal type and hour never
    affect execution order.
    """
    buckets = [[] for _ in worker_starts]
    sheds = {}

    def shed_for(position):
        if position not in sheds:
            sheds[position] = nearest_shed(position, shed_access)
        return sheds[position]

    terminal_day = any(task.terminal_day or task.cashout for task in tasks)

    def distance_from_start(task):
        return min(
            abs(start[0] - task.position[0]) + abs(start[1] - task.position[1])
            for start in worker_starts
        )

    def admission_key(task):
        distance = distance_from_start(task)
        capacity = distance + len(task.actions) + len(task.needs)
        if task.cashout:
            shed = shed_for(task.position)
            capacity += (
                abs(task.position[0] - shed[0])
                + abs(task.position[1] - shed[1])
                + 1
            )
        value_density = (
            -task.value / max(1, capacity)
            if task.mandatory is False
            else 0
        )
        x, y = task.position
        if variant == 1:
            geometry = (-distance - len(task.actions), y, x)
        elif variant == 2:
            geometry = (x, y)
        elif variant == 3:
            geometry = (y, x)
        else:
            geometry = (distance, -len(task.actions), y, x)
        return _admission_class(task), value_density, geometry

    ordered = sorted(tasks, key=admission_key)
    unassigned = []
    lengths = [0] * len(worker_starts)

    for task in ordered:
        best = None
        for worker, bucket in enumerate(buckets):
            for insertion in range(len(bucket) + 1):
                candidate_bucket = bucket[:insertion] + [task] + bucket[insertion:]
                projected = _task_length(
                    worker_starts[worker], candidate_bucket, shed_for
                )
                if projected > _bucket_budget(
                    budgets[worker], candidate_bucket, terminal_day
                ):
                    continue
                candidate = (
                    projected - lengths[worker],
                    projected,
                    worker,
                    insertion,
                )
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            unassigned.append(task)
            continue
        _, projected, worker, insertion = best
        buckets[worker].insert(insertion, task)
        lengths[worker] = projected

    return buckets, unassigned


''' + text[end:]

start = text.index("def _pack(")
end = text.index("def hands_needed(", start)
text = text[:start] + '''def _pack(tasks, worker_starts, budgets, shed_access=SHED_ACCESS):
    """Try deterministic geometry variants without changing task classes."""
    best = _pack_greedy(tasks, worker_starts, budgets, shed_access)
    if not tasks or sum(len(t.actions) for t in tasks) > sum(budgets):
        return best

    terminal_day = any(task.terminal_day or task.cashout for task in tasks)

    def score(result):
        missing_required = sum(
            task.mandatory is not False for task in result[1]
        )
        missing_optional_value = sum(
            task.value for task in result[1] if task.mandatory is False
        )
        stranded, travel = 0, 0
        for start, budget, bucket in zip(worker_starts, budgets, result[0]):
            if not bucket:
                continue
            length = _task_length(
                start, bucket, lambda p: nearest_shed(p, shed_access)
            )
            end = bucket[-1].position
            shed = nearest_shed(end, shed_access)
            home = abs(end[0] - shed[0]) + abs(end[1] - shed[1]) + 1
            effective_budget = _bucket_budget(budget, bucket, terminal_day)
            if not _cashout_needed(bucket) and length + home > effective_budget:
                stranded += sum(sum(task.sells.values()) for task in bucket)
            travel += length
        return missing_required, missing_optional_value, stranded, travel

    best_score = score(best)
    for variant in (1, 2, 3):
        candidate = _pack_greedy(
            tasks, worker_starts, budgets, shed_access, variant
        )
        candidate_score = score(candidate)
        if candidate_score < best_score:
            best, best_score = candidate, candidate_score
    return best


''' + text[end:]

old = '''                    for insertion in range(len(bucket)+1):
                        # Insert after fixed work so planting and harvest are not delayed.
                        if any(not service(t) for t in bucket[insertion:]):
                            continue
                        merged = bucket[:insertion] + [task] + bucket[insertion:]
'''
new = '''                    for insertion in range(len(bucket)+1):
                        # Resource consolidation may insert anywhere that lowers
                        # route cost; action class does not constrain ordering.
                        merged = bucket[:insertion] + [task] + bucket[insertion:]
'''
if text.count(old) != 1:
    raise SystemExit(f"feed insertion replacement count={text.count(old)}")
text = text.replace(old, new, 1)

for forbidden in (
    "def _priority",
    "def _rescue_priority",
    "group_animal",
    "Time-sensitive work must finish early",
    "planting and harvest are not delayed",
):
    if forbidden in text:
        raise SystemExit(f"forbidden priority mechanism remains: {forbidden}")

path.write_text(text)
