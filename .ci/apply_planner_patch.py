from pathlib import Path


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:120]
    return text.replace(old, new, 1)


path = Path("agents/planner.py")
text = path.read_text()
text = replace_once(
    text,
    '''from agents.schedules import (
    CROP_FERTILIZE_DAYS, CROP_LAST_AGE, ONGOING_CROPS, cycle_finished,
)
''',
    '''from agents.schedules import (
    CROP_FERTILIZE_DAYS, CROP_LAST_AGE, ONGOING_CROPS, cycle_turns_over_today,
)
''',
)
text = replace_once(
    text,
    '''    replanning = []
    external = Production()
''',
    '''    replanning = []
    deferred = []
    external = Production()
''',
)
text = replace_once(
    text,
    '''        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if not cycle_finished(tile["crop"], day - tile["planted_day"], tile):
                continue
            # Preserve a conversion already scheduled by the opening.
''',
    '''        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if not cycle_turns_over_today(tile["crop"], day - tile["planted_day"], tile):
                # Keep growing crops in the persistent pending set instead of
                # forgetting them after one batch scan. They are repriced on
                # the morning their known lifecycle will free the tile today.
                if replan_positions is not None:
                    deferred.append(position)
                continue
            # Preserve a conversion already scheduled by the opening.
''',
)
text = replace_once(
    text,
    '''    pending = replanning[max_positions:] if max_positions is not None else []
''',
    '''    pending = deferred + (replanning[max_positions:] if max_positions is not None else [])
''',
)
path.write_text(text)
