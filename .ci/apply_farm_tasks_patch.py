from pathlib import Path


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:120]
    return text.replace(old, new, 1)


path = Path("agents/farm_tasks.py")
text = path.read_text()
text = replace_once(
    text,
    '''from agents.schedules import (
    ONGOING_CROPS, cycle_finished, is_maintenance_day, should_care_animal,
    should_feed_animal, should_fertilize_today, animal_feed_end_age, should_harvest_animal,
)
''',
    '''from agents.schedules import (
    ONGOING_CROPS, cycle_turns_over_today, is_maintenance_day, should_care_animal,
    should_feed_animal, should_fertilize_today, animal_feed_end_age, should_harvest_animal,
)
''',
)
text = text.replace("cycle_finished(", "cycle_turns_over_today(")
assert "cycle_finished(" not in text

text = replace_once(
    text,
    '''            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop = tile["crop"]
                if tile.get("yield_units", 0) > 0:
                    actions = [] if tile.get("watered_today") else [["WATER"]]
                    tasks.append(Task(position, actions + [["HARVEST"]], urgent=True,
                                      sells=Counter({crop: tile["yield_units"]}), mandatory=True))
                elif tile.get("consecutive_unwatered", 0) >= 1 and not tile.get("watered_today"):
                    tasks.append(Task(position, [["WATER"]], urgent=True))
                elif cycle_turns_over_today(crop, day - tile["planted_day"], tile):
                    tasks.append(Task(position, [["DIG"]], ends_cycle=True))
''',
    '''            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop = tile["crop"]
                age = day - tile["planted_day"]
                turns_over = cycle_turns_over_today(crop, age, tile)
                if tile.get("yield_units", 0) > 0:
                    if crop in ONGOING_CROPS:
                        scheduled_water = is_maintenance_day(
                            crop, age,
                            tile.get("fertilized_until_day", -1) >= day,
                            None if prioritize_fertilizer_drop else position,
                            tile["planted_day"],
                        )
                        need_water = (
                            not tile.get("watered_today")
                            and (scheduled_water or tile.get("consecutive_unwatered", 0) >= 1)
                        )
                    else:
                        # Preserve the historical one-time-crop cashout bonus.
                        need_water = not tile.get("watered_today")
                    actions = [["WATER"]] if need_water else []
                    actions.append(["HARVEST"])
                    if turns_over:
                        actions.append(["DIG"])
                    tasks.append(Task(
                        position, actions, urgent=True, ends_cycle=turns_over,
                        sells=Counter({crop: tile["yield_units"]}), mandatory=True,
                    ))
                elif turns_over:
                    tasks.append(Task(position, [["DIG"]], ends_cycle=True))
                elif tile.get("consecutive_unwatered", 0) >= 1 and not tile.get("watered_today"):
                    tasks.append(Task(position, [["WATER"]], urgent=True))
''',
)

text = replace_once(
    text,
    '''            if (is_maintenance_day(crop, age, fertilize_commit)
                    or tile.get("consecutive_unwatered", 0) >= 1) and not tile.get("watered_today"):
''',
    '''            if (is_maintenance_day(
                    crop, age, fertilize_commit,
                    None if prioritize_fertilizer_drop else position,
                    tile["planted_day"],
                ) or tile.get("consecutive_unwatered", 0) >= 1) and not tile.get("watered_today"):
''',
)

text = replace_once(
    text,
    '''                if not tile.get("watered_today") and ["WATER"] not in actions:
                    actions.append(["WATER"])
                actions.append(["HARVEST"])
''',
    '''                # Opening keeps its historical water-before-harvest rule.
                # Post-opening, equivalent spatial WATER profiles decide the
                # protective day; harvesting cached output alone must not
                # synchronize every ongoing crop back onto the same day.
                if (prioritize_fertilizer_drop and not tile.get("watered_today")
                        and ["WATER"] not in actions):
                    actions.append(["WATER"])
                actions.append(["HARVEST"])
''',
)

path.write_text(text)
