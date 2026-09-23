from pathlib import Path


def replace_once(text, old, new):
    count = text.count(old)
    assert count == 1, (count, old[:160])
    return text.replace(old, new, 1)


scheduler = Path("agents/scheduler.py")
text = scheduler.read_text()
text = replace_once(
    text,
    '''    shed_access=SHED_ACCESS,
    pending_hand_budget=HAND_BUDGET,
    max_hands=MAX_HANDS,
    marginal_hire_costs=None,
):
''',
    '''    shed_access=SHED_ACCESS,
    pending_hand_budget=HAND_BUDGET,
    existing_hand_budget=None,
    max_hands=MAX_HANDS,
    marginal_hire_costs=None,
):
''',
)
text = replace_once(
    text,
    '''    minimum = min(len(existing_hand_starts), max_hands)
    # Every action and each distinct positive supply pickup is unavoidable,
''',
    '''    minimum = min(len(existing_hand_starts), max_hands)
    existing_budget = HAND_BUDGET if existing_hand_budget is None else existing_hand_budget
    # Every action and each distinct positive supply pickup is unavoidable,
''',
)
text = replace_once(
    text,
    '''        task for task in tasks if task.mandatory is not False
        and len(task.actions) <= max(FARMER_BUDGET, HAND_BUDGET, pending_hand_budget)
''',
    '''        task for task in tasks if task.mandatory is not False
        and len(task.actions) <= max(existing_budget, pending_hand_budget)
''',
)
text = replace_once(
    text,
    '''        budgets = [FARMER_BUDGET]
        budgets += [HAND_BUDGET] * min(count, len(existing_hand_starts))
''',
    '''        budgets = [existing_budget]
        budgets += [existing_budget] * min(count, len(existing_hand_starts))
''',
)
scheduler.write_text(text)

agent = Path("agents/expansion_agent.py")
text = agent.read_text()
text = replace_once(
    text,
    '''            existing_hands = tuple(map(tuple, farm["hands"]))
            shed_access = _open_shed_access(farm)
            desired_hands, _dropped = hands_needed(
''',
    '''            existing_hands = tuple(map(tuple, farm["hands"]))
            shed_access = _open_shed_access(farm)
            # Final queues can be built after one or more morning market-only
            # turns. Existing workers can act from this observation through
            # hour 23; a HIRE ordered now first acts on the next observation.
            # Size and pack against those real remaining turns instead of the
            # day-start 23/22 constants, otherwise each bucket can silently
            # strand its final WATER/FEED action at rollover.
            remaining_budget = max(0, 24 - hour)
            pending_budget = max(0, 23 - hour)
            desired_hands, _dropped = hands_needed(
''',
)
text = replace_once(
    text,
    '''                shed_access,
                pending_hand_budget=22,
                marginal_hire_costs=(None if state["opening_active"] and day < effective_end else
''',
    '''                shed_access,
                pending_hand_budget=pending_budget,
                existing_hand_budget=remaining_budget,
                marginal_hire_costs=(None if state["opening_active"] and day < effective_end else
''',
)
# Three build/size calls below use the same real remaining-day capacities.
text = text.replace(
    '''                shed_access,
                pending_hand_budget=22,
                available_wheat=obs["private"]["shed"].get("WHEAT", 0),
''',
    '''                shed_access,
                pending_hand_budget=pending_budget,
                existing_hand_budget=remaining_budget,
                available_wheat=obs["private"]["shed"].get("WHEAT", 0),
''',
)
assert text.count('pending_hand_budget=22') == 1, text.count('pending_hand_budget=22')
# The remaining occurrence is the required_hands call, which also needs exact budgets.
text = text.replace(
    '''                    shed_access,
                    pending_hand_budget=22,
                )
''',
    '''                    shed_access,
                    pending_hand_budget=pending_budget,
                    existing_hand_budget=remaining_budget,
                )
''',
    1,
)
assert 'pending_hand_budget=22' not in text
agent.write_text(text)

# Add a scheduler-level regression proving reduced existing-worker time affects
# hand sizing. Keep it independent from the expansion agent state machine.
tests = Path("tests/test_agents_scheduler.py")
t = tests.read_text()
marker = '''def test_hands_needed_zero_for_no_tasks():
'''
assert marker in t
case = '''def test_hands_needed_respects_reduced_existing_worker_budget():
    task_a = Task(SHED, [["WATER"]] * 5, mandatory=True)
    task_b = Task(SHED, [["WATER"]] * 5, mandatory=True)
    count, dropped = hands_needed(
        [task_a, task_b],
        farmer_start=SHED,
        existing_hand_budget=5,
        pending_hand_budget=5,
    )
    assert count == 1
    assert dropped == []


'''
tests.write_text(t.replace(marker, case + marker, 1))
