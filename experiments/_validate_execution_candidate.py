from pathlib import Path


path = Path("agents/expansion_agent.py")
text = path.read_text()

# Validation workflow restores the exact f3c820e baseline before this patcher.
# Preserve baseline admission. Only when an already-admitted animal investment
# overflows the ten-order market cap do we make market latency part of that
# fixed plan and buy enough labor once for the fixed task set.
text = text.replace(
    "BOARD_SIZE = 10\n",
    '''BOARD_SIZE = 10
MARKET_ORDER_CAP = 10


def _pop_market_batch(queue, cap=MARKET_ORDER_CAP):
    batch = list(queue[:cap])
    del queue[:cap]
    return batch


def _strip_partial_animal_builds(tasks, opening_active=False):
    """Post-opening, BUILD and PLACE are one atomic investment task."""
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
    '        "emergency_hires": 0,\n        "morning_market_queue": [],\n        "market_delayed_plan": False,\n    }',
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

old = '''        replanned = build_tasks(
            obs, frozen_targets,
            prioritize_fertilizer_drop=state["opening_active"],
        )
'''
new = old + '''        replanned = _strip_partial_animal_builds(
            replanned, opening_active=state["opening_active"]
        )
'''
if text.count(old) != 1:
    raise SystemExit(f"global build replacement count={text.count(old)}")
text = text.replace(old, new, 1)

old = '''            orders = _hire_and_buy_orders(
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
new = '''            orders = _hire_and_buy_orders(
                obs, farm, purchase_targets, hand_target, pending_sales=sales,
                replant_same_crop=True,
                reserve_hire_budget=not state["opening_active"],
                mandatory_hand_target=mandatory_hand_target,
            )
            full_market = list(sales) + list(orders)
            state["market_delayed_plan"] = False
            if not state["opening_active"] and any(
                order[0] in ("BUY_ANIMAL", "BUY_PRODUCT")
                for order in full_market[MARKET_ORDER_CAP:]
            ):
                # The task set is already fixed by baseline admission. Account
                # for the market turns it requires, but never reopen admission
                # or add new targets. Increase labor monotonically until that
                # same fixed task set fits the remaining worker horizon.
                candidate_hands = hand_target
                while candidate_hands <= MAX_HANDS:
                    candidate_orders = _hire_and_buy_orders(
                        obs, farm, purchase_targets, candidate_hands,
                        pending_sales=sales, replant_same_crop=True,
                        reserve_hire_budget=True,
                        mandatory_hand_target=mandatory_hand_target,
                    )
                    candidate_market = list(sales) + list(candidate_orders)
                    market_turns = max(
                        1,
                        (len(candidate_market) + MARKET_ORDER_CAP - 1)
                        // MARKET_ORDER_CAP,
                    )
                    worker_budget = max(0, 24 - market_turns)
                    _candidate_plans, missing = build_queues(
                        assigned_tasks,
                        tuple(farm["farmer"]),
                        candidate_hands,
                        tuple(map(tuple, farm["hands"])),
                        _open_shed_access(farm),
                        pending_hand_budget=worker_budget,
                        existing_hand_budget=worker_budget,
                    )
                    if not missing:
                        hand_target = candidate_hands
                        state["hand_target"] = candidate_hands
                        orders = candidate_orders
                        full_market = candidate_market
                        state["market_delayed_plan"] = market_turns > 1
                        break
                    candidate_hands += 1

                # Once an animal investment has made this a multi-turn morning
                # schedule, retain every order belonging to that fixed plan.
                state["morning_market_queue"] = list(full_market[MARKET_ORDER_CAP:])
            return {
                "farmer": ["PASS"], "hands": [["PASS"] for _ in farm["hands"]],
                "market": full_market[:MARKET_ORDER_CAP],
            }
'''
if text.count(old) != 1:
    raise SystemExit(f"market replacement count={text.count(old)}")
text = text.replace(old, new, 1)

anchor = '''        if state["day"] == day and not state["plan_frozen"]:
'''
if text.count(anchor) != 1:
    raise SystemExit(f"finalization anchor count={text.count(anchor)}")
text = text.replace(
    anchor,
    '''        if (not state["opening_active"] and state["day"] == day
                and state["morning_market_queue"]):
            return _dispatch_morning_market(farm)

''' + anchor,
    1,
)

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
if text.count(old) != 1:
    raise SystemExit(f"daily build replacement count={text.count(old)}")
text = text.replace(old, new, 1)

# A delayed fixed market plan has already paid for enough hands under this
# exact remaining horizon. Rebuild from observed resources using that horizon;
# do not call a second labor-sizing loop or create rescue actions.
old = '''            plans, unassigned = build_queues(
                tasks,
                tuple(farm["farmer"]),
                hand_count,
                existing_hands,
                shed_access,
                pending_hand_budget=22,
                available_wheat=obs["private"]["shed"].get("WHEAT", 0),
            )
'''
new = '''            route_budget = (
                max(0, 24 - hour)
                if state["market_delayed_plan"] and not state["opening_active"]
                else 22
            )
            plans, unassigned = build_queues(
                tasks,
                tuple(farm["farmer"]),
                hand_count,
                existing_hands,
                shed_access,
                pending_hand_budget=route_budget,
                existing_hand_budget=(
                    route_budget
                    if state["market_delayed_plan"] and not state["opening_active"]
                    else None
                ),
                available_wheat=obs["private"]["shed"].get("WHEAT", 0),
            )
'''
if text.count(old) != 1:
    raise SystemExit(f"primary route replacement count={text.count(old)}")
text = text.replace(old, new, 1)

old = '''                plans, mandatory_unassigned = build_queues(
                    mandatory_tasks,
                    tuple(farm["farmer"]),
                    hand_count,
                    existing_hands,
                    shed_access,
                    pending_hand_budget=22,
                    available_wheat=obs["private"]["shed"].get("WHEAT", 0),
                )
'''
new = '''                plans, mandatory_unassigned = build_queues(
                    mandatory_tasks,
                    tuple(farm["farmer"]),
                    hand_count,
                    existing_hands,
                    shed_access,
                    pending_hand_budget=route_budget,
                    existing_hand_budget=(
                        route_budget
                        if state["market_delayed_plan"] and not state["opening_active"]
                        else None
                    ),
                    available_wheat=obs["private"]["shed"].get("WHEAT", 0),
                )
'''
if text.count(old) != 1:
    raise SystemExit(f"mandatory route replacement count={text.count(old)}")
text = text.replace(old, new, 1)

path.write_text(text)
