from pathlib import Path

def replace_between(text, start, end, replacement):
    i = text.index(start)
    j = text.index(end, i)
    return text[:i] + replacement.rstrip() + "\n\n" + text[j:]

scheduler = Path("agents/scheduler.py")
text = scheduler.read_text()
pack_greedy = 'def _pack_greedy(\n    tasks,\n    worker_starts,\n    budgets,\n    shed_access=SHED_ACCESS,\n    variant=0,\n    group_animal=False,\n):\n    """Greedy bin-pack using exact route length with an O(1) common fast path.\n\n    Normal production buckets have no forced mid-route DROP/cashout. For those\n    buckets, inserting one task only changes two Manhattan edges, action count\n    and (at most) the initial shed pickup set, so recomputing the whole route\n    for every candidate insertion is unnecessary. Opening refinance/cashout\n    tasks retain the exact legacy projection as a bounded slow path.\n    """\n    buckets = [[] for _ in worker_starts]\n\n    sheds = {}\n    def shed_for(position):\n        if position not in sheds:\n            sheds[position] = nearest_shed(position, shed_access)\n        return sheds[position]\n\n    def distance(a, b):\n        return abs(a[0] - b[0]) + abs(a[1] - b[1])\n\n    priority_for = _rescue_priority if group_animal else _priority\n    priorities = {id(task): priority_for(task) for task in tasks}\n    terminal_day = any(task.terminal_day or task.cashout for task in tasks)\n\n    if group_animal:\n        ordered = sorted(\n            tasks,\n            key=lambda task: (\n                priorities[id(task)],\n                min(distance(start, task.position) for start in worker_starts),\n                task.position[1],\n                task.position[0],\n                not task.animal_harvest,\n                -len(task.actions),\n            ),\n        )\n    else:\n        ordered = sorted(\n            tasks,\n            key=lambda task: (\n                priorities[id(task)],\n                min(distance(start, task.position) for start in worker_starts),\n                -len(task.actions),\n                task.position[1],\n                task.position[0],\n            ),\n        )\n    if variant:\n        def alternative(task):\n            x, y = task.position\n            dist = min(distance((x, y), start) for start in worker_starts)\n            geometry = ((-dist - len(task.actions), y, x) if variant == 1\n                        else (x, y) if variant == 2 else (y, x))\n            return priorities[id(task)], geometry\n        ordered = sorted(tasks, key=alternative)\n\n    def admission_key(task):\n        dist = min(distance(start, task.position) for start in worker_starts)\n        capacity = dist + len(task.actions) + len(task.needs)\n        if task.cashout:\n            shed = shed_for(task.position)\n            capacity += distance(task.position, shed) + 1\n        return priorities[id(task)], -task.value / max(1, capacity) if task.mandatory is False else 0\n\n    ordered.sort(key=admission_key)\n    unassigned = []\n    lengths = [0] * len(worker_starts)\n\n    # Metadata for the common no-mid-route-DROP case.\n    # path_lengths excludes the initial worker->shed leg and PICKUP ops.\n    fast_bucket = [True] * len(worker_starts)\n    path_lengths = [0] * len(worker_starts)\n    action_steps = [0] * len(worker_starts)\n    need_items = [set() for _ in worker_starts]\n\n    for task in ordered:\n        best = None\n        best_fast = None\n        priority = priorities[id(task)]\n        time_sensitive = task.immediate_drop or task.cashout\n        task_is_service = group_animal and _is_animal_service(task)\n        task_need_items = {\n            item for item, amount in task.needs.items() if amount > 0\n        }\n        task_fast = not task.immediate_drop and not task.cashout\n\n        for worker, bucket in enumerate(buckets):\n            paired_harvest = None\n            paired_service = None\n            if task_is_service:\n                for index, queued in enumerate(bucket):\n                    if queued.position != task.position:\n                        continue\n                    if queued.animal_harvest:\n                        paired_harvest = index\n                    elif _is_animal_service(queued):\n                        paired_service = index\n\n            use_fast = fast_bucket[worker] and task_fast\n            if use_fast:\n                old_items = need_items[worker]\n                new_pickups = len(old_items | task_need_items)\n                old_origin = shed_for(worker_starts[worker]) if old_items else worker_starts[worker]\n                new_origin = shed_for(worker_starts[worker]) if new_pickups else worker_starts[worker]\n                base_path = path_lengths[worker]\n                if bucket and old_origin != new_origin:\n                    first = bucket[0].position\n                    base_path += distance(new_origin, first) - distance(old_origin, first)\n                prefix = distance(worker_starts[worker], new_origin) if new_pickups else 0\n                effective_budget = max(0, budgets[worker] - int(terminal_day))\n\n            for insertion in range(len(bucket) + 1):\n                if insertion and priorities[id(bucket[insertion - 1])] > priority:\n                    continue\n                if insertion < len(bucket) and priority > priorities[id(bucket[insertion])]:\n                    continue\n                if task_is_service:\n                    if task.animal_harvest and paired_service is not None and insertion > paired_service:\n                        continue\n                    if not task.animal_harvest and paired_harvest is not None and insertion <= paired_harvest:\n                        continue\n\n                fast_detail = None\n                if use_fast:\n                    previous = new_origin if insertion == 0 else bucket[insertion - 1].position\n                    candidate_path = base_path\n                    if insertion < len(bucket):\n                        following = bucket[insertion].position\n                        candidate_path += (\n                            distance(previous, task.position)\n                            + distance(task.position, following)\n                            - distance(previous, following)\n                        )\n                    else:\n                        candidate_path += distance(previous, task.position)\n                    projected = (\n                        prefix + new_pickups + candidate_path\n                        + action_steps[worker] + len(task.actions)\n                    )\n                    fits = projected <= effective_budget\n                    fast_detail = candidate_path\n                else:\n                    candidate_bucket = bucket[:insertion] + [task] + bucket[insertion:]\n                    projected = _task_length(worker_starts[worker], candidate_bucket, shed_for)\n                    fits = projected <= _bucket_budget(\n                        budgets[worker], candidate_bucket, terminal_day\n                    )\n\n                if fits:\n                    cost = projected if time_sensitive and not variant else projected - lengths[worker]\n                    rank = (cost, projected, worker, insertion)\n                    if best is None or rank < best:\n                        best = rank\n                        best_fast = fast_detail if use_fast else None\n\n        if best is not None:\n            _, projected, worker, insertion = best\n            buckets[worker].insert(insertion, task)\n            lengths[worker] = projected\n            if best_fast is not None:\n                path_lengths[worker] = best_fast\n                action_steps[worker] += len(task.actions)\n                need_items[worker].update(task_need_items)\n            else:\n                fast_bucket[worker] = False\n        else:\n            unassigned.append(task)\n    return buckets, unassigned'
pack = 'def _pack(\n    tasks,\n    worker_starts,\n    budgets,\n    shed_access=SHED_ACCESS,\n    optimize_routes=True,\n):\n    """Pack tasks, separating feasibility search from route polishing.\n\n    hands_needed only needs a feasible assignment; spending three extra full\n    packing passes to improve an already-feasible route made morning planning\n    scale badly with farm size. Final queue construction may still request\n    route polishing, but stops as soon as no sellable output is stranded.\n    """\n    best = _pack_greedy(tasks, worker_starts, budgets, shed_access)\n    if not tasks or sum(len(t.actions) for t in tasks) > sum(budgets):\n        return best\n\n    if best[1] and any(_is_animal_service(task) for task in tasks):\n        grouped = _pack_greedy(\n            tasks, worker_starts, budgets, shed_access, group_animal=True\n        )\n        if not grouped[1]:\n            return grouped\n\n    if not optimize_routes:\n        if not best[1]:\n            return best\n        for variant in (1, 2, 3):\n            candidate = _pack_greedy(\n                tasks, worker_starts, budgets, shed_access, variant\n            )\n            if not candidate[1]:\n                return candidate\n        return best\n\n    terminal_day = any(task.terminal_day or task.cashout for task in tasks)\n    def score(result):\n        if result[1]:\n            return (float("inf"), float("inf"))\n        stranded, travel = 0, 0\n        for start, budget, bucket in zip(worker_starts, budgets, result[0]):\n            if not bucket:\n                continue\n            length = _task_length(start, bucket, lambda p: nearest_shed(p, shed_access))\n            end = bucket[-1].position\n            shed = nearest_shed(end, shed_access)\n            home = abs(end[0]-shed[0]) + abs(end[1]-shed[1]) + 1\n            effective_budget = _bucket_budget(budget, bucket, terminal_day)\n            if not _cashout_needed(bucket) and length + home > effective_budget:\n                stranded += sum(sum(task.sells.values()) for task in bucket)\n            travel += length\n        return stranded, travel\n\n    best_score = score(best)\n    # Once all assigned workers can return sellable output, extra full repacks\n    # only optimize travel distance and are not worth risking the 1s act budget.\n    if not best[1] and best_score[0] == 0:\n        return best\n\n    for variant in (1, 2, 3):\n        candidate = _pack_greedy(tasks, worker_starts, budgets, shed_access, variant)\n        candidate_score = score(candidate)\n        if candidate_score < best_score:\n            best, best_score = candidate, candidate_score\n            if best_score[0] == 0:\n                break\n    return best'
hands = 'def hands_needed(\n    tasks,\n    farmer_start,\n    existing_hand_starts=(),\n    shed_access=SHED_ACCESS,\n    pending_hand_budget=HAND_BUDGET,\n    existing_hand_budget=None,\n    max_hands=MAX_HANDS,\n    marginal_hire_costs=None,\n):\n    """Find the fewest/economically useful hands without polishing each route.\n\n    Feasibility sizing is cheaper than final route construction. With economic\n    sizing, larger Fibonacci-priced headcounts stop being evaluated once their\n    cumulative extra hire cost exceeds all optional value that could possibly\n    be recovered from the first fully-mandatory schedule.\n    """\n    minimum = min(len(existing_hand_starts), max_hands)\n    existing_budget = HAND_BUDGET if existing_hand_budget is None else existing_hand_budget\n    required = tasks if marginal_hire_costs is None else [\n        task for task in tasks if task.mandatory is not False\n        and len(task.actions) <= max(existing_budget, pending_hand_budget)\n    ]\n    mandatory_steps = sum(len(task.actions) for task in required)\n    mandatory_steps += len({\n        item for task in required for item, amount in task.needs.items() if amount > 0\n    })\n    candidates = []\n    first_required_count = None\n    max_optional_recovery = 0.0\n\n    for count in range(minimum, max_hands + 1):\n        if marginal_hire_costs is not None and first_required_count is not None:\n            extra_cost = sum(\n                marginal_hire_costs[\n                    first_required_count - minimum:count - minimum\n                ]\n            )\n            if extra_cost > max_optional_recovery:\n                break\n\n        starts = [tuple(farmer_start)] + predicted_hand_starts(\n            farmer_start, existing_hand_starts, count\n        )\n        budgets = [existing_budget]\n        budgets += [existing_budget] * min(count, len(existing_hand_starts))\n        budgets += [pending_hand_budget] * max(0, count - len(existing_hand_starts))\n        if count < max_hands and sum(budgets) < mandatory_steps:\n            continue\n\n        _, unassigned = _pack(\n            tasks, starts, budgets, shed_access, optimize_routes=False\n        )\n        if marginal_hire_costs is None and not unassigned:\n            return count, []\n\n        candidates.append((count, unassigned))\n        if marginal_hire_costs is not None and first_required_count is None:\n            missing_required = sum(\n                task.mandatory is not False for task in unassigned\n            )\n            if missing_required == 0:\n                first_required_count = count\n                max_optional_recovery = sum(\n                    max(0.0, task.value)\n                    for task in unassigned if task.mandatory is False\n                )\n\n    if marginal_hire_costs is None:\n        return max_hands, unassigned\n\n    required_missing = [\n        sum(task.mandatory is not False for task in missing)\n        for _, missing in candidates\n    ]\n    best_required = min(required_missing)\n    base_index = required_missing.index(best_required)\n    chosen_count, chosen_missing = candidates[base_index]\n\n    base_count = chosen_count\n    best_net = -sum(task.value for task in chosen_missing if task.mandatory is False)\n    for index in range(base_index + 1, len(candidates)):\n        count, missing = candidates[index]\n        if required_missing[index] != best_required:\n            continue\n        lost_value = sum(task.value for task in missing if task.mandatory is False)\n        price = sum(marginal_hire_costs[base_count - minimum:count - minimum])\n        net = -lost_value - price\n        if net > best_net:\n            chosen_count, chosen_missing, best_net = count, missing, net\n    return chosen_count, chosen_missing'
text = replace_between(text, "def _pack_greedy(", "def _pack(", pack_greedy)
text = replace_between(text, "def _pack(", "def hands_needed(", pack)
text = replace_between(text, "def hands_needed(", "def _rebalance_feed(", hands)
scheduler.write_text(text)

agent = Path("agents/expansion_agent.py")
text = agent.read_text()
old = """            hand_target, _dropped = hands_needed(
                tasks,
                tuple(farm["farmer"]),
                tuple(map(tuple, farm["hands"])),
                _open_shed_access(farm),
                marginal_hire_costs=(None if state["opening_active"] and day < effective_end else
                    [_hire_costs(farm, n + 1) - _hire_costs(farm, n)
                     for n in range(MAX_HANDS - len(farm["hands"]))]),
            )
"""
new = old.replace("hand_target, _dropped =", "hand_target, rejected =")
assert text.count(old) == 1, text.count(old)
text = text.replace(old, new, 1)

old = """            preliminary, rejected = build_queues(
                tasks, tuple(farm["farmer"]), hand_target,
                tuple(map(tuple, farm["hands"])), _open_shed_access(farm),
            )
            rejected_ids = {id(task) for task in rejected}
"""
new = """            # hands_needed already ran the exact feasibility pack at the chosen
            # headcount. Repacking the same tasks here was a duplicate
            # superlinear morning pass used only to recover the same rejected set.
            rejected_ids = {id(task) for task in rejected}
"""
assert text.count(old) == 1, text.count(old)
text = text.replace(old, new, 1)

old = """            mandatory_tasks = [task for task in tasks if task.mandatory is not False]
            mandatory_hand_target, _mandatory_missing = hands_needed(
                mandatory_tasks,
                tuple(farm["farmer"]), tuple(map(tuple, farm["hands"])),
                _open_shed_access(farm),
            )
"""
new = """            mandatory_tasks = [task for task in tasks if task.mandatory is not False]
            if len(mandatory_tasks) == len(tasks):
                mandatory_hand_target = hand_target
            else:
                mandatory_hand_target, _mandatory_missing = hands_needed(
                    mandatory_tasks,
                    tuple(farm["farmer"]), tuple(map(tuple, farm["hands"])),
                    _open_shed_access(farm),
                )
"""
assert text.count(old) == 1, text.count(old)
text = text.replace(old, new, 1)


old = """            desired_hands, _dropped = hands_needed(
                tasks,
                tuple(farm["farmer"]),
                existing_hands,
                shed_access,
                pending_hand_budget=pending_budget,
                existing_hand_budget=remaining_budget,
                marginal_hire_costs=(None if state["opening_active"] and day < effective_end else
                    [_hire_costs(farm, n + 1) - _hire_costs(farm, n)
                     for n in range(MAX_HANDS - len(existing_hands))]),
            )
            hand_count = len(existing_hands)
            state["hand_target"] = desired_hands
"""
new = """            # Optional labor was already economically sized and ordered at
            # hour 0. Re-running the full economic hand search after the
            # morning market queue cannot add an optional worker here; this
            # phase only schedules workers that actually arrived. If mandatory
            # work still does not fit, the emergency path below sizes the exact
            # missing survival capacity.
            hand_count = len(existing_hands)
            state["hand_target"] = hand_count
"""
assert text.count(old) == 1, text.count(old)
text = text.replace(old, new, 1)
agent.write_text(text)

forecast = Path("agents/forecast.py")
text = forecast.read_text()
old = """    def add(self, other, position=None):
        for day, units in other.sales.items():
            self.sales[day].update(units)
        for day, units in other.inputs.items():
            self.inputs[day].update(units)
        for day, visits in other.visits.items():
            self.visits[day].extend((position if p is None else p, n, inputs, goods)
                                   for p, n, inputs, goods in visits)
"""
new = """    def add(self, other, position=None, include_visits=True):
        for day, units in other.sales.items():
            self.sales[day].update(units)
        for day, units in other.inputs.items():
            self.inputs[day].update(units)
        if include_visits:
            for day, visits in other.visits.items():
                self.visits[day].extend((position if p is None else p, n, inputs, goods)
                                       for p, n, inputs, goods in visits)
"""
assert text.count(old) == 1, text.count(old)
text = text.replace(old, new, 1)

old = """        self.price = lru_cache(maxsize=None)(lambda product, stock: game.market_price(product, stock, params))
        self.trade = lru_cache(maxsize=None)(self._trade)
"""
new = """        self.price = lru_cache(maxsize=None)(lambda product, stock: game.market_price(product, stock, params))
        self.trade = lru_cache(maxsize=None)(self._trade)
        # Candidate Production objects are reused across every tile repriced in
        # one morning. Cache their touched markets once instead of rescanning
        # up to sixteen days of sales/inputs for every tile.
        self._candidate_products = {}
"""
assert text.count(old) == 1, text.count(old)
text = text.replace(old, new, 1)

old = """        touched = set()
        for field in (candidate.sales, candidate.inputs):
            for units in field.values():
                touched.update(product for product, amount in units.items() if amount)
        products = tuple(product for product in game.PRODUCTS if product in touched)
        if not products:
            return 0.0
"""
new = """        candidate_id = id(candidate)
        products = self._candidate_products.get(candidate_id)
        if products is None:
            touched = set()
            for field in (candidate.sales, candidate.inputs):
                for units in field.values():
                    touched.update(product for product, amount in units.items() if amount)
            products = tuple(product for product in game.PRODUCTS if product in touched)
            self._candidate_products[candidate_id] = products
        if not products:
            return 0.0
"""
assert text.count(old) == 1, text.count(old)
text = text.replace(old, new, 1)
forecast.write_text(text)

planner = Path("agents/planner.py")
text = planner.read_text()
old = """def evaluate_targets(
    market, baseline, candidates, labor=None, position=(4, 4)
):
"""
new = """def evaluate_targets(
    market, baseline, candidates, labor=None, position=(4, 4), *, flows_scoped=False
):
"""
assert text.count(old) == 1, text.count(old)
text = text.replace(old, new, 1)

old = """    scoped_baseline = scoped(baseline)
    baseline_values = {}
    for choice, output, cost in candidates:
        if market.day + _first_yield_age(choice[0]) > end:
            continue
        output = scoped(output)
        market_cash = scoped_market.marginal_value(
            scoped_baseline, output, baseline_values
        )
"""
new = """    scoped_baseline = baseline if flows_scoped else scoped(baseline)
    baseline_values = {}
    for choice, output, cost in candidates:
        if market.day + _first_yield_age(choice[0]) > end:
            continue
        if not flows_scoped:
            output = scoped(output)
        market_cash = scoped_market.marginal_value(
            scoped_baseline, output, baseline_values
        )
"""
assert text.count(old) == 1, text.count(old)
text = text.replace(old, new, 1)

old = """def _choose(
    market, baseline, candidates, counts, labor=None, position=(4, 4), *, current=None,
    audit=None,
):
    rows = evaluate_targets(market, baseline, candidates, labor, position)
"""
new = """def _choose(
    market, baseline, candidates, counts, labor=None, position=(4, 4), *, current=None,
    audit=None, flows_scoped=False,
):
    rows = evaluate_targets(
        market, baseline, candidates, labor, position, flows_scoped=flows_scoped
    )
"""
assert text.count(old) == 1, text.count(old)
text = text.replace(old, new, 1)

for old, new in (
    ("                destination.add(future, (x, y))",
     "                destination.add(future, (x, y), include_visits=False)"),
    ("            baseline.add(_rotation(*target, day, end_day)[0], position)",
     "            baseline.add(_rotation(*target, day, end_day)[0], position, include_visits=False)"),
    ("            baseline.add(output, position)",
     "            baseline.add(output, position, include_visits=False)"),
):
    assert text.count(old) == 1, (old, text.count(old))
    text = text.replace(old, new, 1)

old = """        choice, output = _choose(
            market, baseline, allowed, counts, position=position, current=current,
            audit=audit,
        )
"""
new = """        choice, output = _choose(
            market, baseline, allowed, counts, position=position, current=current,
            audit=audit, flows_scoped=True,
        )
"""
assert text.count(old) == 1, text.count(old)
text = text.replace(old, new, 1)
planner.write_text(text)
