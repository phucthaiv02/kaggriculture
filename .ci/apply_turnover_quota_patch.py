from pathlib import Path


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:160]
    return text.replace(old, new, 1)


planner = Path("agents/planner.py")
text = planner.read_text()
text = replace_once(
    text,
    '''    replanning = []
    deferred = []
    external = Production()
''',
    '''    replanning = []
    turnover = []
    deferred = []
    external = Production()
''',
)
text = replace_once(
    text,
    '''            # Preserve a conversion already scheduled by the opening.
            if current and current[0] != tile["crop"] and can_start(current[0], day, end_day):
                continue
        replanning.append(position)

    shed_access = ((4, 4), (5, 4), (4, 5), (5, 5))
    def distance(p):
        return min(abs(p[0] - s[0]) + abs(p[1] - s[1]) for s in shed_access)
    replanning.sort(key=lambda p: (distance(p), p[1], p[0]))
    pending = deferred + (replanning[max_positions:] if max_positions is not None else [])
    if max_positions is not None:
        replanning = replanning[:max_positions]
''',
    '''            # Preserve a conversion already scheduled by the opening.
            if current and current[0] != tile["crop"] and can_start(current[0], day, end_day):
                continue
            turnover.append(position)
            continue
        replanning.append(position)

    shed_access = ((4, 4), (5, 4), (4, 5), (5, 5))
    def distance(p):
        return min(abs(p[0] - s[0]) + abs(p[1] - s[1]) for s in shed_access)
    turnover.sort(key=lambda p: (distance(p), p[1], p[0]))
    replanning.sort(key=lambda p: (distance(p), p[1], p[0]))
    pending = deferred + (replanning[max_positions:] if max_positions is not None else [])
    if max_positions is not None:
        replanning = replanning[:max_positions]
    # Known same-day turnover is lifecycle work, not speculative expansion.
    # Price every such successor this morning even when a cohort is larger
    # than TARGETS_PER_DAY; the batch cap applies only to ordinary investments.
    replanning = turnover + replanning
''',
)
planner.write_text(text)

tests = Path("tests/test_agents_planner.py")
test_text = tests.read_text()
marker = '''def test_plan_targets_replans_a_finished_tile():
'''
assert marker in test_text
new_test = '''def test_same_day_turnover_is_not_limited_by_target_batch():
    positions = [(x, y) for y in range(3) for x in range(4)]
    tiles = {
        position: {
            "kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
            "yield_units": 4, "watered_today": True,
        }
        for position in positions
    }
    obs = make_obs(day=4, tiles=tiles)
    targets = {position: ("WHEAT", False) for position in positions}
    pending = plan_targets(
        obs, targets, positions, end_day=20,
        max_positions=10, replan_positions=set(positions),
    )
    assert pending == []
    assert all(position in targets for position in positions)


'''
test_text = test_text.replace(marker, new_test + marker, 1)
tests.write_text(test_text)
