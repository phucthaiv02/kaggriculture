from pathlib import Path

path = Path("agents/scheduler_post.py")
text = path.read_text()

start = text.index("_ANIMAL_SERVICE_OPS =")
end = text.index("\ndef hands_needed(")

replacement = r'''
def _admission_class(task):
    """Required work is admitted before optional work; this is not route order."""
    return task.mandatory is False


def _bucket_budget(base_budget, bucket, terminal_day):
    """Return the executable worker budget, including terminal deadlines."""
    if not terminal_day:
        return base_budget
    return max(0, base_budget - 1 - int(_cashout_needed(bucket)))


def _pack_greedy(
    tasks,
    worker_starts,
    budgets,
    shed_access=SHED_ACCESS,
    variant=0,
):
    """Legacy exact-length packer without action-label ordering.

    Required tasks are admitted first so feasibility is a hard constraint.
    Inside a worker route, insertion is governed only by complete-route cost,
    geometry and the task's own action dependencies.
    """
    buckets = [[] for _ in worker_starts]

    sheds = {}
    def shed_for(position):
        if position not in sheds:
            sheds[position] = nearest_shed(position, shed_access)
        return sheds[position]

    terminal_day = any(task.terminal_day or task.cashout for task in tasks)

    def distance(task):
        return min(
            abs(start[0] - task.position[0]) + abs(start[1] - task.position[1])
            for start in worker_starts
        )

    if variant == 1:
        ordered = sorted(
            tasks,
            key=lambda task: (
                _admission_class(task),
                -distance(task) - len(task.actions),
                task.position[1], task.position[0],
            ),
        )
    elif variant == 2:
        ordered = sorted(
            tasks,
            key=lambda task: (
                _admission_class(task),
                task.position[0], task.position[1],
            ),
        )
    elif variant == 3:
        ordered = sorted(
            tasks,
            key=lambda task: (
                _admission_class(task),
                task.position[1], task.position[0],
            ),
        )
    else:
        ordered = sorted(
            tasks,
            key=lambda task: (
                _admission_class(task),
                distance(task),
                -len(task.actions),
                task.position[1], task.position[0],
            ),
        )

    # Keep the legacy optional value-density admission without imposing any
    # execution order on already admitted tasks.
    def admission_key(task):
        capacity = distance(task) + len(task.actions) + len(task.needs)
        if task.cashout:
            shed = shed_for(task.position)
            capacity += (
                abs(task.position[0] - shed[0])
                + abs(task.position[1] - shed[1]) + 1
            )
        value_density = (
            -task.value / max(1, capacity)
            if task.mandatory is False else 0
        )
        return _admission_class(task), value_density

    ordered.sort(key=admission_key)

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
                delta = projected - lengths[worker]
                candidate = (delta, projected, worker, insertion)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            unassigned.append(task)
            continue
        _, projected, worker, insertion = best
        buckets[worker].insert(insertion, task)
        lengths[worker] = projected
    return buckets, unassigned


def _pack(tasks, worker_starts, budgets, shed_access=SHED_ACCESS):
    """Try the legacy bounded geometry variants, with no rescue priority mode."""
    best = _pack_greedy(tasks, worker_starts, budgets, shed_access)
    if not tasks or sum(len(t.actions) for t in tasks) > sum(budgets):
        return best

    terminal_day = any(task.terminal_day or task.cashout for task in tasks)

    def score(result):
        if result[1]:
            return (float("inf"), float("inf"))
        stranded, travel = 0, 0
        for start, budget, bucket in zip(worker_starts, budgets, result[0]):
            if not bucket:
                continue
            length = _task_length(
                start, bucket, lambda p: nearest_shed(p, shed_access)
            )
            end = bucket[-1].position
            shed = nearest_shed(end, shed_access)
            home = abs(end[0]-shed[0]) + abs(end[1]-shed[1]) + 1
            effective_budget = _bucket_budget(
                budget, bucket, terminal_day
            )
            if not _cashout_needed(bucket) and length + home > effective_budget:
                stranded += sum(sum(task.sells.values()) for task in bucket)
            travel += length
        return stranded, travel

    best_score = score(best)
    for variant in (1, 2, 3):
        candidate = _pack_greedy(
            tasks, worker_starts, budgets, shed_access, variant
        )
        candidate_score = score(candidate)
        if candidate_score < best_score:
            best, best_score = candidate, candidate_score
    return best
'''

text = text[:start] + replacement + text[end:]

# Feed consolidation may optimize pickup logistics, but it may not force FEED
# behind PLANT/HARVEST. Remove only that ordering restriction; keep the legacy
# bounded rebalancer and its route/capacity checks.
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
    raise SystemExit(f"feed ordering anchor count={text.count(old)}")
text = text.replace(old, new, 1)

path.write_text(text)
