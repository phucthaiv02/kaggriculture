from pathlib import Path

path = Path("agents/scheduler.py")
text = path.read_text()

old = '''                delta = projected - lengths[worker]
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
new = '''                delta = projected - lengths[worker]
                candidate_lengths = list(lengths)
                candidate_lengths[worker] = projected
                # Preserve locality while there is ample time.  Once a route
                # consumes most of its available day, use already-available
                # parallel capacity to create slack.  This is horizon pressure,
                # not an action-class priority.
                pressure = sum(
                    max(0, length * 4 - budget * 3)
                    for length, budget in zip(candidate_lengths, budgets)
                )
                candidate = (pressure, delta, projected, worker, insertion)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            unassigned.append(task)
            continue
        _, _, projected, worker, insertion = best
'''
if text.count(old) != 1:
    raise SystemExit(f"required makespan anchor count={text.count(old)}")
text = text.replace(old, new, 1)

old = '''            delta = projected - baseline
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
'''
new = '''            delta = projected - baseline
            candidate_lengths = list(current_lengths)
            candidate_lengths[worker] = projected
            pressure = sum(
                max(0, length * 4 - budget * 3)
                for length, budget in zip(candidate_lengths, budgets)
            )
            candidate = (pressure, delta, projected, worker, insertion)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return None
    _, delta, projected, worker, insertion = best
'''
if text.count(old) != 1:
    raise SystemExit(f"optional makespan anchor count={text.count(old)}")
text = text.replace(old, new, 1)

old = '''    route_lengths = [
        _task_length(start, bucket, shed_for) if bucket else 0
        for start, bucket in zip(worker_starts, buckets)
    ]
    makespan = max(route_lengths, default=0)
    total_steps = sum(route_lengths)
    return missing_required, missing_optional_value, makespan, total_steps
'''
new = '''    route_lengths = [
        _task_length(start, bucket, shed_for) if bucket else 0
        for start, bucket in zip(worker_starts, buckets)
    ]
    pressure = sum(
        max(0, length * 4 - budget * 3)
        for length, budget in zip(route_lengths, budgets)
    )
    total_steps = sum(route_lengths)
    makespan = max(route_lengths, default=0)
    return missing_required, missing_optional_value, pressure, total_steps, makespan
'''
if text.count(old) != 1:
    raise SystemExit(f"packing score anchor count={text.count(old)}")
text = text.replace(old, new, 1)

path.write_text(text)
