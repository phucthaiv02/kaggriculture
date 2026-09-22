from pathlib import Path


path = Path("agents/expansion_agent.py")
text = path.read_text()

# Validation workflow restores the exact f3c820e baseline before this patcher.
# Keep this candidate deliberately narrow: preserve baseline behavior until an
# admitted animal purchase is actually truncated by the 10-order market cap.
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

# A global replan may observe a failed/missing animal purchase. Never leave the
# BUILD half of an invalidated BUILD->PLACE dependency in the rebuilt plan.
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

# Preserve the baseline first market batch exactly. Only animal resources from
# the already-admitted task set survive truncation into the next morning turn;
# seed and optional-HIRE overflow retain baseline behavior and are not expanded.
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
            if not state["opening_active"]:
                state["morning_market_queue"] = [
                    order for order in full_market[MARKET_ORDER_CAP:]
                    if order[0] in ("BUY_ANIMAL", "BUY_PRODUCT")
                ]
            return {
                "farmer": ["PASS"], "hands": [["PASS"] for _ in farm["hands"]],
                "market": full_market[:MARKET_ORDER_CAP],
            }
'''
if text.count(old) != 1:
    raise SystemExit(f"market replacement count={text.count(old)}")
text = text.replace(old, new, 1)

# Complete the selected morning market schedule before workers start. This is
# still a fixed schedule: runtime only pops the stored queue.
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

# After purchases are observed, build the real worker schedule. If an animal
# still is not present, remove only the invalid BUILD half rather than creating
# a permanent empty pasture/coop.
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

path.write_text(text)
