from pathlib import Path


path = Path("agents/expansion_agent.py")
text = path.read_text()

anchor = text.index('            state["daily_targets"] = daily_targets')
start = text.index('            hand_target, _dropped = hands_needed(', anchor)
end_marker = (
    '            state["morning_market_queue"] = list(sales) + list(orders)\n'
    '            return _dispatch_morning_market(farm)\n'
)
end = text.index(end_marker, start) + len(end_marker)

replacement = '''            if state["opening_active"]:
                # Keep the opening book exactly on its historical bootstrap path.
                hand_target, _dropped = hands_needed(
                    tasks,
                    tuple(farm["farmer"]),
                    tuple(map(tuple, farm["hands"])),
                    _open_shed_access(farm),
                    marginal_hire_costs=(None if day < effective_end else
                        [_hire_costs(farm, n + 1) - _hire_costs(farm, n)
                         for n in range(MAX_HANDS - len(farm["hands"]))]),
                )
                mandatory_tasks = [task for task in tasks if task.mandatory is not False]
                mandatory_hand_target, _mandatory_missing = hands_needed(
                    mandatory_tasks,
                    tuple(farm["farmer"]), tuple(map(tuple, farm["hands"])),
                    _open_shed_access(farm),
                )
                state.update(
                    day=day, hand_target=hand_target,
                    mandatory_hand_target=mandatory_hand_target,
                    plans=[], reserved={}, emergency_hires=0,
                    frozen_positions=set(), unassigned=[], plan_frozen=False,
                )
                preliminary, rejected = build_queues(
                    tasks, tuple(farm["farmer"]), hand_target,
                    tuple(map(tuple, farm["hands"])), _open_shed_access(farm),
                )
                rejected_ids = {id(task) for task in rejected}
                investment_task_positions = {
                    task.position for task in tasks
                    if any(op[0] in ("PLANT", "PLACE") for op in task.actions)
                }
                state["purchase_positions"] = {
                    task.position for task in tasks
                    if id(task) not in rejected_ids
                    and task.position in investment_task_positions
                }
                state["investment_backlog"].update(
                    task.position for task in rejected
                    if task.position in investment_task_positions
                )
                purchase_targets = {
                    p: t for p, t in daily_targets.items()
                    if p in state["purchase_positions"]
                }
                assigned_tasks = [task for task in tasks if id(task) not in rejected_ids]
                reservations = reserved_items(assigned_tasks)
                reservations["WHEAT"] = max(
                    reservations.get("WHEAT", 0),
                    feed_wheat_reserve(obs, purchase_targets, positions),
                )
                non_wheat_reservations = dict(reservations)
                non_wheat_reservations["WHEAT"] = obs["private"]["shed"].get("WHEAT", 0)
                non_wheat_sales = sell_orders(obs, non_wheat_reservations)
                reservations = _reserve_feed_for_affordable_animals(
                    obs, farm, purchase_targets, reservations,
                    non_wheat_sales, 0,
                )
                sales = sell_orders(obs, reservations)
                orders = _hire_and_buy_orders(
                    obs, farm, purchase_targets, hand_target, pending_sales=sales,
                    replant_same_crop=True, reserve_hire_budget=False,
                    mandatory_hand_target=mandatory_hand_target,
                )
                return {
                    "farmer": ["PASS"], "hands": [["PASS"] for _ in farm["hands"]],
                    "market": (sales + orders)[:MARKET_ORDER_CAP],
                }

            # Post-opening, solve admission and labor against the worker time
            # that remains after the complete morning market schedule. Target
            # selection is untouched; this chooses only which selected work is
            # admitted today and how many hands that schedule can justify.
            existing_starts = tuple(map(tuple, farm["hands"]))
            shed_access = _open_shed_access(farm)
            existing_count = len(existing_starts)

            def morning_candidate(worker_budget):
                mandatory_tasks = [task for task in tasks if task.mandatory is not False]

                mandatory_hand_target = MAX_HANDS
                mandatory_missing = mandatory_tasks
                for count in range(existing_count, MAX_HANDS + 1):
                    _mandatory_plans, missing = build_queues(
                        mandatory_tasks, tuple(farm["farmer"]), count,
                        existing_starts, shed_access,
                        pending_hand_budget=worker_budget,
                        existing_hand_budget=worker_budget,
                    )
                    if not missing:
                        mandatory_hand_target = count
                        mandatory_missing = []
                        break
                    mandatory_missing = missing

                base_hires = max(0, mandatory_hand_target - existing_count)
                base_hire_cost = _hire_costs(farm, base_hires)
                best = None
                for count in range(mandatory_hand_target, MAX_HANDS + 1):
                    plans, missing = build_queues(
                        tasks, tuple(farm["farmer"]), count,
                        existing_starts, shed_access,
                        pending_hand_budget=worker_budget,
                        existing_hand_budget=worker_budget,
                    )
                    if any(task.mandatory is not False for task in missing):
                        continue
                    lost_optional = sum(
                        task.value for task in missing if task.mandatory is False
                    )
                    hires = max(0, count - existing_count)
                    incremental_hire_cost = _hire_costs(farm, hires) - base_hire_cost
                    net = -lost_optional - incremental_hire_cost
                    key = (net, -count)
                    if best is None or key > best[0]:
                        best = (key, count, plans, missing)

                if best is None:
                    count = mandatory_hand_target
                    plans, missing = build_queues(
                        tasks, tuple(farm["farmer"]), count,
                        existing_starts, shed_access,
                        pending_hand_budget=worker_budget,
                        existing_hand_budget=worker_budget,
                    )
                else:
                    _, count, plans, missing = best

                rejected_ids = {id(task) for task in missing}
                investment_task_positions = {
                    task.position for task in tasks
                    if any(op[0] in ("PLANT", "PLACE") for op in task.actions)
                }
                purchase_positions = {
                    task.position for task in tasks
                    if id(task) not in rejected_ids
                    and task.position in investment_task_positions
                }
                rejected_investments = {
                    task.position for task in missing
                    if task.position in investment_task_positions
                }
                purchase_targets = {
                    p: t for p, t in daily_targets.items()
                    if p in purchase_positions
                }
                assigned_tasks = [task for task in tasks if id(task) not in rejected_ids]
                reservations = reserved_items(assigned_tasks)
                reservations["WHEAT"] = max(
                    reservations.get("WHEAT", 0),
                    feed_wheat_reserve(obs, purchase_targets, positions),
                )
                non_wheat_reservations = dict(reservations)
                non_wheat_reservations["WHEAT"] = obs["private"]["shed"].get("WHEAT", 0)
                non_wheat_sales = sell_orders(obs, non_wheat_reservations)
                protected_hires = max(0, mandatory_hand_target - existing_count)
                reservations = _reserve_feed_for_affordable_animals(
                    obs, farm, purchase_targets, reservations,
                    non_wheat_sales, protected_hires,
                )
                sales = sell_orders(obs, reservations)
                orders = _hire_and_buy_orders(
                    obs, farm, purchase_targets, count, pending_sales=sales,
                    replant_same_crop=True, reserve_hire_budget=True,
                    mandatory_hand_target=mandatory_hand_target,
                )
                return {
                    "hand_target": count,
                    "mandatory_hand_target": mandatory_hand_target,
                    "mandatory_missing": mandatory_missing,
                    "plans": plans,
                    "rejected": missing,
                    "purchase_positions": purchase_positions,
                    "rejected_investments": rejected_investments,
                    "sales": sales,
                    "orders": orders,
                }

            worker_budget = 23
            candidate = morning_candidate(worker_budget)
            for _ in range(23):
                market_count = len(candidate["sales"]) + len(candidate["orders"])
                market_turns = max(
                    1, (market_count + MARKET_ORDER_CAP - 1) // MARKET_ORDER_CAP
                )
                next_budget = max(0, 24 - market_turns)
                if next_budget >= worker_budget:
                    break
                worker_budget = next_budget
                candidate = morning_candidate(worker_budget)

            state.update(
                day=day,
                hand_target=candidate["hand_target"],
                mandatory_hand_target=candidate["mandatory_hand_target"],
                plans=[], reserved={}, emergency_hires=0,
                frozen_positions=set(), unassigned=[], plan_frozen=False,
            )
            state["purchase_positions"] = set(candidate["purchase_positions"])
            state["investment_backlog"].update(candidate["rejected_investments"])
            state["morning_market_queue"] = (
                list(candidate["sales"]) + list(candidate["orders"])
            )
            return _dispatch_morning_market(farm)
'''
text = text[:start] + replacement + text[end:]

# Finalization after market must never resize labor from a shorter remaining
# horizon; it only rebuilds routes from observed resources.
old = '''            remaining_budget = max(0, 24 - hour)
            pending_hand_budget = max(0, 23 - hour)
            scheduling_pending_budget = 22 if state["opening_active"] else pending_hand_budget
            desired_hands, _dropped = hands_needed(
                tasks,
                tuple(farm["farmer"]),
                existing_hands,
                shed_access,
                pending_hand_budget=scheduling_pending_budget,
                marginal_hire_costs=(None if state["opening_active"] and day < effective_end else
                    [_hire_costs(farm, n + 1) - _hire_costs(farm, n)
                     for n in range(MAX_HANDS - len(existing_hands))]),
            )
            hand_count = len(existing_hands)
            state["hand_target"] = desired_hands

            # Post-opening waits for every planned worker before freezing a
            # complete daily schedule. Opening retains its bootstrap behavior.
            if not state["opening_active"] and desired_hands > hand_count:
                affordable = _affordable_hires(
                    farm, desired_hands - hand_count, farm["money"]
                )
                if affordable:
                    state["morning_market_queue"] = [["HIRE"] for _ in range(affordable)]
                    return _dispatch_morning_market(farm)

            queue_kwargs = {
                "pending_hand_budget": scheduling_pending_budget,
                "available_wheat": obs["private"]["shed"].get("WHEAT", 0),
            }
            if not state["opening_active"]:
                queue_kwargs["existing_hand_budget"] = remaining_budget
'''
new = '''            remaining_budget = max(0, 24 - hour)
            hand_count = len(existing_hands)
            if state["opening_active"]:
                scheduling_pending_budget = 22
                desired_hands, _dropped = hands_needed(
                    tasks,
                    tuple(farm["farmer"]),
                    existing_hands,
                    shed_access,
                    pending_hand_budget=scheduling_pending_budget,
                    marginal_hire_costs=(None if day < effective_end else
                        [_hire_costs(farm, n + 1) - _hire_costs(farm, n)
                         for n in range(MAX_HANDS - len(existing_hands))]),
                )
                state["hand_target"] = desired_hands
                queue_kwargs = {
                    "pending_hand_budget": scheduling_pending_budget,
                    "available_wheat": obs["private"]["shed"].get("WHEAT", 0),
                }
            else:
                scheduling_pending_budget = remaining_budget
                queue_kwargs = {
                    "pending_hand_budget": remaining_budget,
                    "existing_hand_budget": remaining_budget,
                    "available_wheat": obs["private"]["shed"].get("WHEAT", 0),
                }
'''
if text.count(old) != 1:
    raise SystemExit(f"finalization replacement count={text.count(old)}")
text = text.replace(old, new)

old = '''            if mandatory_unassigned:
                required_hands, _ = hands_needed(
                    mandatory_tasks,
                    tuple(farm["farmer"]),
                    existing_hands,
                    shed_access,
                    pending_hand_budget=scheduling_pending_budget,
                )
                needed = max(0, required_hands - hand_count)
                if state["opening_active"]:
                    # Historical opening exception: execute the safe partial
                    # mandatory plan while asking the engine for missing hands,
                    # then rebuild next observation. This is intentionally not
                    # the post-opening scheduling contract.
                    state["emergency_hires"] = needed
                else:
                    affordable = _affordable_hires(farm, needed, farm["money"])
                    if affordable:
                        state["morning_market_queue"] = [["HIRE"] for _ in range(affordable)]
                        state["plans"] = []
                        state["unassigned"] = tasks
                        state["reserved"] = {}
                        return _dispatch_morning_market(farm)
                    # No partial rescue execution post-opening. If the complete
                    # mandatory schedule is infeasible with observed resources,
                    # expose the planning failure instead of masking it.
                    state["plans"] = []
                    state["unassigned"] = tasks
                    state["reserved"] = {}
                    state["plan_frozen"] = False
                    return {
                        "farmer": ["PASS"],
                        "hands": [["PASS"] for _ in farm["hands"]],
                        "market": [],
                    }
            else:
                state["emergency_hires"] = 0
'''
new = '''            if mandatory_unassigned and state["opening_active"]:
                required_hands, _ = hands_needed(
                    mandatory_tasks,
                    tuple(farm["farmer"]),
                    existing_hands,
                    shed_access,
                    pending_hand_budget=scheduling_pending_budget,
                )
                state["emergency_hires"] = max(0, required_hands - hand_count)
            else:
                state["emergency_hires"] = 0
'''
if text.count(old) != 1:
    raise SystemExit(f"mandatory replacement count={text.count(old)}")
path.write_text(text.replace(old, new))
