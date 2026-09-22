from pathlib import Path

path = Path("agents/scheduler.py")
text = path.read_text()

old = '''                delta = projected - lengths[worker]
                candidate = (delta, projected, worker, insertion)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            unassigned.append(task)
            continue
        _, projected, worker, insertion = best
'''
new = '''                delta = projected - lengths[worker]
                makespan = max(
                    projected,
                    *(length for index, length in enumerate(lengths) if index != worker),
                )
                candidate = (makespan, delta, projected, worker, insertion)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            unassigned.append(task)
            continue
        _, _, projected, worker, insertion = best
'''
if text.count(old) != 1:
    raise SystemExit(f"required insertion anchor count={text.count(old)}")
text = text.replace(old, new, 1)

old = '''    best = None
    for worker, bucket in enumerate(buckets):
        baseline = _task_length(worker_starts[worker], bucket, shed_for) if bucket else 0
        for insertion in range(len(bucket) + 1):
            candidate_bucket = bucket[:insertion] + [task] + bucket[insertion:]
            projected = _task_length(
                worker_starts[worker], candidate_bucket, shed_for
            )
            if projected > _bucket_budget(
                budgets[worker], candidate_bucket, terminal_day
            ):
                continue
            delta = projected - baseline
            candidate = (delta, projected, worker, insertion)
            if best is None or candidate < best:
                best = candidate
    return best
'''
new = '''    current_lengths = [
        _task_length(start, bucket, shed_for) if bucket else 0
        for start, bucket in zip(worker_starts, buckets)
    ]
    best = None
    for worker, bucket in enumerate(buckets):
        baseline = current_lengths[worker]
        for insertion in range(len(bucket) + 1):
            candidate_bucket = bucket[:insertion] + [task] + bucket[insertion:]
            projected = _task_length(
                worker_starts[worker], candidate_bucket, shed_for
            )
            if projected > _bucket_budget(
                budgets[worker], candidate_bucket, terminal_day
            ):
                continue
            delta = projected - baseline
            makespan = max(
                projected,
                *(length for index, length in enumerate(current_lengths) if index != worker),
            )
            candidate = (makespan, delta, projected, worker, insertion)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return None
    _, delta, projected, worker, insertion = best
    return delta, projected, worker, insertion
'''
if text.count(old) != 1:
    raise SystemExit(f"optional insertion anchor count={text.count(old)}")
text = text.replace(old, new, 1)

old = '''    total_steps = sum(
        _task_length(start, bucket, shed_for) if bucket else 0
        for start, bucket in zip(worker_starts, buckets)
    )
    return missing_required, missing_optional_value, total_steps
'''
new = '''    route_lengths = [
        _task_length(start, bucket, shed_for) if bucket else 0
        for start, bucket in zip(worker_starts, buckets)
    ]
    makespan = max(route_lengths, default=0)
    total_steps = sum(route_lengths)
    return missing_required, missing_optional_value, makespan, total_steps
'''
if text.count(old) != 1:
    raise SystemExit(f"packing score anchor count={text.count(old)}")
text = text.replace(old, new, 1)

path.write_text(text)
