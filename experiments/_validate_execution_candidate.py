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
                # Opening is an intentional bootstrap exception. Keep the
                # historical behavior byte-for-byte in spirit: size labor on
                # the full opening day and let its fixed book own sequencing.
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

            # Post-opening market turns, labor and optional admissions are one
            # scheduling problem. Enumerate bounded (market turns, hand count)
            # schedules instead of creating a feedback loop where extra market
            # latency causes more hires which cause still more latency.
            existing_starts = tuple(map(tuple, farm["hands"]))
            shed_access = _open_shed_access(farm)
            existing_count = len(existing_starts)
            mandatory_tasks = [task for task in tasks if task.mandatory is not False]
            optional_tasks = [task for task in tasks if task.mandatory is False]
            optional_value = sum(task.value for task in optional_tasks)
            investment_task_positions = {
                task.position for task in tasks
                if any(op[0] in ("PLANT", "PLACE") for op in task.actions)
            }

            def market_plan(source_tasks, missing, hand_count, mandatory_count):
                missing_ids = {id(task) for task in missing}
                purchase_positions = {
                    task.position for task in source_tasks
                    if id(task) not in missing_ids
                    and task.position in investment_task_positions
                }
                purchase_targets = {
                    p: t for p, t in daily_targets.items()
                    if p in purchase_positions
                }
                assigned = [task for task in source_tasks if id(task) not in missing_ids]
                reservations = reserved_items(assigned)
                reservations["WHEAT"] = max(
                    reservations.get("WHEAT", 0),
                    feed_wheat_reserve(obs, purchase_targets, positions),
                )
                non_wheat_reservations = dict(reservations)
                non_wheat_reservations["WHEAT"] = obs["private"]["shed"].get("WHEAT", 0)
                non_wheat_sales = sell_orders(obs, non_wheat_reservations)
                protected_hires = max(0, mandatory_count - existing_count)
                reservations = _reserve_feed_for_affordable_animals(
                    obs, farm, purchase_targets, reservations,
                    non_wheat_sales, protected_hires,
                )
                sales = sell_orders(obs, reservations)
                orders = _hire_and_buy_orders(
                    obs, farm, purchase_targets, hand_count, pending_sales=sales,
                    replant_same_crop=True, reserve_hire_budget=True,
                    mandatory_hand_target=mandatory_count,
                )
                return sales, orders, purchase_positions

            best = None
            # The daily target batch bounds investment orders; six market turns
            # is safely above the reachable queue here while keeping the search
            # cheap enough for the per-decision limit.
            for market_turns in range(1, 7):
                worker_budget = max(0, 24 - market_turns)

                mandatory_count = None
                mandatory_plan = None
                for count in range(existing_count, MAX_HANDS + 1):
                    _plans, missing = build_queues(
                        mandatory_tasks,
                        tuple(farm["farmer"]),
                        count,
                        existing_starts,
                        shed_access,
                        pending_hand_budget=worker_budget,
                        existing_hand_budget=worker_budget,
                    )
                    if not missing:
                        baseline_sales, baseline_orders, baseline_positions = market_plan(
                            mandatory_tasks, [], count, count
                        )
                        if len(baseline_sales) + len(baseline_orders) <= market_turns * MARKET_ORDER_CAP:
                            mandatory_count = count
                            mandatory_plan = (
                                _plans, [], baseline_sales, baseline_orders,
                                baseline_positions,
                            )
                            break
                if mandatory_count is None:
                    continue

                # Mandatory-only is always a valid candidate. Optional work is
                # admitted only when the complete resulting market queue still
                # fits the same assumed market-turn budget.
                plans, missing, sales, orders, purchase_positions = mandatory_plan
                all_optional_missing = list(optional_tasks)
                base_score = -_hire_costs(
                    farm, max(0, mandatory_count - existing_count)
                )
                candidate = (
                    base_score,
                    -market_turns,
                    -mandatory_count,
                    mandatory_count,
                    mandatory_count,
                    plans,
                    all_optional_missing,
                    sales,
                    orders,
                    purchase_positions,
                )
                if best is None or candidate[:3] > best[:3]:
                    best = candidate

                for count in range(mandatory_count, MAX_HANDS + 1):
                    plans, missing = build_queues(
                        tasks,
                        tuple(farm["farmer"]),
                        count,
                        existing_starts,
                        shed_access,
                        pending_hand_budget=worker_budget,
                        existing_hand_budget=worker_budget,
                    )
                    if any(task.mandatory is not False for task in missing):
                        continue
                    sales, orders, purchase_positions = market_plan(
                        tasks, missing, count, mandatory_count
                    )
                    if len(sales) + len(orders) > market_turns * MARKET_ORDER_CAP:
                        continue
                    lost_optional = sum(
                        task.value for task in missing if task.mandatory is False
                    )
                    admitted_value = optional_value - lost_optional
                    hire_cost = _hire_costs(
                        farm, max(0, count - existing_count)
                    )
                    score = admitted_value - hire_cost
                    candidate = (
                        score,
                        -market_turns,
                        -count,
                        count,
                        mandatory_count,
                        plans,
                        missing,
                        sales,
                        orders,
                        purchase_positions,
                    )
                    if best is None or candidate[:3] > best[:3]:
                        best = candidate

            if best is None:
                # This should be unreachable with the bounded daily task set;
                # keep failure explicit rather than inventing runtime actions.
                state.update(
                    day=day, plans=[], reserved={}, emergency_hires=0,
                    frozen_positions=set(), unassigned=list(tasks), plan_frozen=False,
                    purchase_positions=set(), hand_target=existing_count,
                    mandatory_hand_target=existing_count,
                )
                return {
                    "farmer": ["PASS"],
                    "hands": [["PASS"] for _ in farm["hands"]],
                    "market": [],
                }

            (_score, _neg_turns, _neg_count, hand_target,
             mandatory_hand_target, _plans, rejected, sales, orders,
             purchase_positions) = best
            rejected_ids = {id(task) for task in rejected}
            state.update(
                day=day,
                hand_target=hand_target,
                mandatory_hand_target=mandatory_hand_target,
                plans=[], reserved={}, emergency_hires=0,
                frozen_positions=set(), unassigned=[], plan_frozen=False,
            )
            state["purchase_positions"] = set(purchase_positions)
            state["investment_backlog"].update(
                task.position for task in tasks
                if id(task) in rejected_ids
                and task.position in investment_task_positions
            )
            state["morning_market_queue"] = list(sales) + list(orders)
            return _dispatch_morning_market(farm)
'''
text = text[:start] + replacement + text[end:]

# Once the morning market plan is chosen, observation may change inventory and
# worker positions but must not launch a second HAND-sizing cycle. Rebuild the
# remaining routes from the actual state and actual remaining time only.
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
