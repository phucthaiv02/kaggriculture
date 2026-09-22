from pathlib import Path


path = Path("agents/expansion_agent.py")
text = path.read_text()

# Validation starts from the exact f3c820e expansion agent. Preserve its
# admission, hand sizing, and route assignment; patch only the confirmed
# animal-input truncation and dependency correctness failures.
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

# Global replans must not resurrect a BUILD-only animal task if a purchase did
# not materialize.
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

# A planned dependency bundle may not start if its required successor action
# cannot execute before day rollover. Runtime does not invent a replacement;
# it marks the fixed plan invalid so the existing global replan rebuilds all
# unfinished work from the observation and actual remaining worker budget.
anchor = '''        return False

    def agent(obs, configuration=None):
'''
guard = '''        remaining_turns = max(0, 24 - int(obs.get("hour", 0)))
        for index, _position in enumerate(positions):
            queue = state["plans"][index].queue
            if not queue:
                continue
            first = queue[0][0]
            required = None
            if first.startswith("BUILD_"):
                required = "PLACE"
            elif first == "PLANT":
                required = "WATER"
            if required is None:
                continue
            # Dependencies inside one tile task are contiguous; any movement,
            # pickup or drop marks the end of this local bundle.
            for offset, queued in enumerate(queue):
                op = queued[0]
                if offset and (op in MOVES or op in ("PICKUP", "DROP")):
                    break
                if op == required:
                    if offset + 1 > remaining_turns:
                        return True
                    break
            else:
                return True
        return False

    def agent(obs, configuration=None):
'''
if text.count(anchor) != 1:
    raise SystemExit(f"plan-invalid anchor count={text.count(anchor)}")
text = text.replace(anchor, guard, 1)

# Preserve the baseline first market batch exactly. Only already-admitted animal
# resources that were truncated by the ten-order cap continue on the next
# morning turn; optional seed/HIRE overflow retains baseline behavior.
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

# Planned market tail is executed before final worker queues are frozen.
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

# Final task generation uses observed inventory. Missing animal input must
# remove the BUILD half as well, never leave an empty permanent structure.
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
