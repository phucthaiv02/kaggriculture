from pathlib import Path


path = Path("agents/expansion_agent.py")
text = path.read_text()

# This patcher is applied after the validation workflow restores the exact
# pre-fix expansion_agent.py. Keep the experiment intentionally small.
text = text.replace(
    "BOARD_SIZE = 10\n",
    '''BOARD_SIZE = 10
MARKET_ORDER_CAP = 10


def _pop_market_batch(queue, cap=MARKET_ORDER_CAP):
    batch = list(queue[:cap])
    del queue[:cap]
    return batch


def _strip_partial_animal_builds(tasks, opening_active=False):
    """Post-opening BUILD and PLACE are one investment commitment."""
    if opening_active:
        return tasks
    cleaned = []
    for task in tasks:
        operations = [operation[0] for operation in task.actions]
        if any(op.startswith("BUILD_") for op in operations) and "PLACE" not in operations:
            task.actions = [
                operation for operation in task.actions
                if not operation[0].startswith("BUILD_")
            ]
        if task.actions:
            cleaned.append(task)
    return cleaned

''',
    1,
)

text = text.replace(
    '        "emergency_hires": 0,\n    }',
    '        "emergency_hires": 0,\n        "morning_market_queue": [],\n    }',
    1,
)

opening_anchor = "    opening_governs = make_opening_controller()\n"
text = text.replace(
    opening_anchor,
    opening_anchor + '''

    def _dispatch_morning_market(farm):
        return {
            "farmer": ["PASS"],
            "hands": [["PASS"] for _ in farm["hands"]],
            "market": _pop_market_batch(state["morning_market_queue"]),
        }
''',
    1,
)

# Global replans and final morning builds must never materialize a standalone
# animal structure post-opening.
text = text.replace(
    '''        replanned = build_tasks(
            obs, frozen_targets,
            prioritize_fertilizer_drop=state["opening_active"],
        )
''',
    '''        replanned = build_tasks(
            obs, frozen_targets,
            prioritize_fertilizer_drop=state["opening_active"],
        )
        replanned = _strip_partial_animal_builds(
            replanned, opening_active=state["opening_active"]
        )
''',
    1,
)

# Preserve the opening path. Post-opening, only genuinely required non-HIRE
# overflow survives the engine's ten-order first batch. Optional HIREs that
# would create an extra market turn were never admitted to the final market
# schedule, so this is not truncating an already-chosen order plan.
old = '''            protected_hires = (
                0 if state["opening_active"] else
                max(0, mandatory_hand_target - len(farm["hands"]))
            )
            reservations = _reserve_feed_for_affordable_animals(
                obs, farm, purchase_targets, reservations,
                non_wheat_sales, protected_hires,
            )
            sales = sell_orders(obs, reservations)
            if not state["opening_active"]:
                # Reserve market-order slots for every mandatory hire. The
                # replay regression had 3-5 SELLs consume the ten-order cap,
                # silently dropping survival hands before hour 1.
                sales = sales[:max(0, 10 - protected_hires)]
            orders = _hire_and_buy_orders(
                obs, farm, purchase_targets, hand_target, pending_sales=sales,
                replant_same_crop=True,
                reserve_hire_budget=not state["opening_active"],
                mandatory_hand_target=mandatory_hand_target,
            )
            return {
                "farmer": ["PASS"], "hands": [["PASS"] for _ in farm["hands"]],
                "market": (sales + orders)[:10],
            }
'''
new = '''            base_reservations = dict(reservations)
            protected_hires = (
                0 if state["opening_active"] else
                max(0, mandatory_hand_target - len(farm["hands"]))
            )

            def planned_market(required_hands):
                protected = max(0, required_hands - len(farm["hands"]))
                guarded = _reserve_feed_for_affordable_animals(
                    obs, farm, purchase_targets, base_reservations,
                    non_wheat_sales, protected,
                )
                planned_sales = sell_orders(obs, guarded)
                planned_sales = planned_sales[:max(0, MARKET_ORDER_CAP - protected)]
                planned_orders = _hire_and_buy_orders(
                    obs, farm, purchase_targets,
                    max(hand_target, required_hands), pending_sales=planned_sales,
                    replant_same_crop=True, reserve_hire_budget=True,
                    mandatory_hand_target=required_hands,
                )
                full = list(planned_sales) + list(planned_orders)
                required_tail = [
                    order for order in full[MARKET_ORDER_CAP:]
                    if order[0] != "HIRE"
                ]
                turns = 1 + (
                    (len(required_tail) + MARKET_ORDER_CAP - 1) // MARKET_ORDER_CAP
                    if required_tail else 0
                )
                return planned_sales, planned_orders, required_tail, turns

            if state["opening_active"]:
                reservations = _reserve_feed_for_affordable_animals(
                    obs, farm, purchase_targets, base_reservations,
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

            required_hands = mandatory_hand_target
            # Required input overflow consumes worker turns. Size only mandatory
            # capacity against that known latency; never re-size optional labor.
            for _ in range(4):
                sales, orders, required_tail, market_turns = planned_market(required_hands)
                worker_budget = max(0, 24 - market_turns)
                next_required = required_hands
                for count in range(len(farm["hands"]), MAX_HANDS + 1):
                    _plans, missing = build_queues(
                        mandatory_tasks,
                        tuple(farm["farmer"]),
                        count,
                        tuple(map(tuple, farm["hands"])),
                        _open_shed_access(farm),
                        pending_hand_budget=worker_budget,
                        existing_hand_budget=worker_budget,
                    )
                    if not missing:
                        next_required = count
                        break
                if next_required <= required_hands:
                    break
                required_hands = next_required

            sales, orders, required_tail, _market_turns = planned_market(required_hands)
            state["mandatory_hand_target"] = required_hands
            state["hand_target"] = max(hand_target, required_hands)
            full = list(sales) + list(orders)
            state["morning_market_queue"] = list(required_tail)
            return {
                "farmer": ["PASS"],
                "hands": [["PASS"] for _ in farm["hands"]],
                "market": full[:MARKET_ORDER_CAP],
            }
'''
if text.count(old) != 1:
    raise SystemExit(f"market replacement count={text.count(old)}")
text = text.replace(old, new)

# Dispatch only the required overflow selected at hour 0. All workers remain
# idle during this morning market phase, exactly as the day plan assumed.
anchor = '''        if state["day"] == day and not state["plan_frozen"]:
'''
text = text.replace(
    anchor,
    '''        if (not state["opening_active"] and state["day"] == day
                and state["morning_market_queue"]):
            return _dispatch_morning_market(farm)

''' + anchor,
    1,
)

# Finalize the post-opening route from observed resources and the actual time
# left. Do not launch a second economic HAND-sizing cycle after market latency.
old = '''            tasks = build_tasks(
                obs,
                state["daily_targets"],
                prioritize_fertilizer_drop=state["opening_active"],
            )
'''
new = old + '''            tasks = _strip_partial_animal_builds(
                tasks, opening_active=state["opening_active"]
            )
'''
if text.count(old) < 1:
    raise SystemExit("missing final build_tasks block")
text = text.replace(old, new, 1)

old = '''            desired_hands, _dropped = hands_needed(
                tasks,
                tuple(farm["farmer"]),
                existing_hands,
                shed_access,
                pending_hand_budget=22,
                marginal_hire_costs=(None if state["opening_active"] and day < effective_end else
                    [_hire_costs(farm, n + 1) - _hire_costs(farm, n)
                     for n in range(MAX_HANDS - len(existing_hands))]),
            )
            hand_count = len(existing_hands)
            state["hand_target"] = desired_hands
            plans, unassigned = build_queues(
                tasks,
                tuple(farm["farmer"]),
                hand_count,
                existing_hands,
                shed_access,
                pending_hand_budget=22,
                available_wheat=obs["private"]["shed"].get("WHEAT", 0),
            )
'''
new = '''            hand_count = len(existing_hands)
            if state["opening_active"]:
                desired_hands, _dropped = hands_needed(
                    tasks,
                    tuple(farm["farmer"]),
                    existing_hands,
                    shed_access,
                    pending_hand_budget=22,
                    marginal_hire_costs=(None if day < effective_end else
                        [_hire_costs(farm, n + 1) - _hire_costs(farm, n)
                         for n in range(MAX_HANDS - len(existing_hands))]),
                )
                state["hand_target"] = desired_hands
                queue_kwargs = {"pending_hand_budget": 22}
            else:
                remaining_budget = max(0, 24 - hour)
                queue_kwargs = {
                    "pending_hand_budget": remaining_budget,
                    "existing_hand_budget": remaining_budget,
                }
            plans, unassigned = build_queues(
                tasks,
                tuple(farm["farmer"]),
                hand_count,
                existing_hands,
                shed_access,
                available_wheat=obs["private"]["shed"].get("WHEAT", 0),
                **queue_kwargs,
            )
'''
if text.count(old) != 1:
    raise SystemExit(f"final sizing replacement count={text.count(old)}")
text = text.replace(old, new)

# The hour-0 mandatory sizing already incorporated required market latency.
# Keep the opening-only emergency path, but post-opening never invents another
# HIRE cycle here.
old = '''            if mandatory_unassigned:
                required_hands, _ = hands_needed(
                    mandatory_tasks,
                    tuple(farm["farmer"]),
                    existing_hands,
                    shed_access,
                    pending_hand_budget=22,
                )
                state["emergency_hires"] = max(0, required_hands - hand_count)
            else:
                state["emergency_hires"] = 0
'''
new = '''            if mandatory_unassigned and state["opening_active"]:
                required_hands, _ = hands_needed(
                    mandatory_tasks,
                    tuple(farm["farmer"]),
                    existing_hands,
                    shed_access,
                    pending_hand_budget=22,
                )
                state["emergency_hires"] = max(0, required_hands - hand_count)
            else:
                state["emergency_hires"] = 0
'''
if text.count(old) != 1:
    raise SystemExit(f"emergency replacement count={text.count(old)}")
text = text.replace(old, new)

path.write_text(text)
