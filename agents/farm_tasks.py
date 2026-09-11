"""Execution wrapper that staggers yield-equivalent maintenance work.

The core task builder remains frozen in farm_tasks_core.py. This wrapper
post-processes only WATER / COW FEED+CARE timing on live producers, using
engine-validated equivalent schedules from agents.maintenance. Planning,
purchases, reservations and every non-maintenance task keep the core behavior.
"""
from __future__ import annotations

from collections import Counter

from agents import farm_tasks_core as _core
from agents import maintenance as _maintenance
from agents.schedules import (
    ANIMAL_CARE_DAYS,
    ANIMAL_FEED_DAYS,
    CROP_WATER_DAYS,
    animal_maintenance_can_still_pay,
)

# Preserve the old module surface, including private helpers used by debugging
# scripts, then override the maintenance-aware entry points below.
for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)


def _tasks_at(tasks, position):
    return [task for task in tasks if task.position == position]


def _current_water_index(task):
    """Index of WATER for the live crop, never a replacement PLANT's age-0 WATER."""
    for index, action in enumerate(task.actions):
        if action and action[0] in ("PLANT", "PLACE"):
            return None
        if action and action[0] == "WATER":
            return index
    return None


def _remove_current_water(tasks, position):
    for task in list(_tasks_at(tasks, position)):
        index = _current_water_index(task)
        if index is None:
            continue
        task.actions.pop(index)
        if not task.actions and not task.needs and not task.sells:
            tasks.remove(task)
        return


def _ensure_current_water(tasks, position):
    position_tasks = _tasks_at(tasks, position)
    for task in position_tasks:
        if _current_water_index(task) is not None:
            return

    # If the core already has work on this crop, keep FERTILIZE before WATER
    # and keep HARVEST / replacement work after it.
    if position_tasks:
        task = position_tasks[-1]
        insertion = len(task.actions)
        for index, action in enumerate(task.actions):
            if action and action[0] in ("HARVEST", "DIG", "PLANT", "PLACE"):
                insertion = index
                break
        task.actions.insert(insertion, ["WATER"])
        task.urgent = True
        return

    tasks.append(_core.Task(position, [["WATER"]], urgent=True))


def _animal_maintenance_task(tasks, position):
    for task in _tasks_at(tasks, position):
        if any(
            action and action[0] in ("FEED", "CARE", "COLLECT_FERTILIZER")
            for action in task.actions
        ):
            return task
    return None


def _clean_counter(counter):
    for key in list(counter):
        if counter[key] <= 0:
            del counter[key]


def _remove_cow_maintenance(tasks, position, tile, remove_feed, remove_care):
    task = _animal_maintenance_task(tasks, position)
    if task is None:
        return
    if remove_feed:
        for index, action in enumerate(list(task.actions)):
            if action and action[0] == "FEED":
                task.actions.pop(index)
                if not tile.get("fed_today"):
                    task.needs["WHEAT"] -= 1
                    _clean_counter(task.needs)
                break
    if remove_care:
        task.actions[:] = [
            action
            for action in task.actions
            if not (action and action[0] == "CARE")
        ]
    if not task.actions and not task.needs and not task.sells:
        tasks.remove(task)


def _ensure_cow_maintenance(tasks, position, tile, want_feed, want_care):
    if (not want_feed or tile.get("fed_today")) and (
        not want_care or tile.get("cared_today")
    ):
        return

    task = _animal_maintenance_task(tasks, position)
    if task is None:
        task = _core.Task(position, [], Counter(), urgent=True)
        tasks.append(task)

    insert_at = next(
        (
            i
            for i, action in enumerate(task.actions)
            if action and action[0] == "COLLECT_FERTILIZER"
        ),
        len(task.actions),
    )
    existing = {action[0] for action in task.actions if action}
    if want_feed and not tile.get("fed_today") and "FEED" not in existing:
        task.actions.insert(insert_at, ["FEED"])
        insert_at += 1
        task.needs["WHEAT"] += 1
        existing.add("FEED")
    if want_care and not tile.get("cared_today") and "CARE" not in existing:
        task.actions.insert(insert_at, ["CARE"])
    task.urgent = True


def _cow_feed_delta(obs, targets, active_positions, when):
    """Desired staggered COW feed count minus the core schedule's count."""
    farm = obs["farms"][obs["player"]]
    end_day = obs.get("_planning_end_day", _core.SEASON_END_DAY)
    delta = 0
    for position in active_positions:
        target = targets.get(position)
        if not target or target[0] != "COW":
            continue
        x, y = position
        tile = farm["tiles"][y][x]
        if not (isinstance(tile, dict) and tile.get("animal") == "COW"):
            continue
        age = when - tile["placed_day"]
        if not animal_maintenance_can_still_pay("COW", age, when, end_day):
            continue
        baseline = age in ANIMAL_FEED_DAYS["COW"]
        desired = _maintenance.should_feed("COW", age, position)
        delta += int(desired) - int(baseline)
    return delta


def _feed_need_adjustment(
    obs, targets, active_positions, replant_same_crop=False, purchase_mode=False
):
    """Change in WHEAT requirement caused only by staggered live-COW FEED."""
    (
        _seed_demand,
        animal_missing,
        live_animals,
        pending_feed,
        wheat_incoming,
        carried_wheat,
    ) = _core._animal_and_seed_demand(
        obs, targets, active_positions, replant_same_crop=replant_same_crop
    )
    farm = obs["farms"][obs["player"]]
    day = obs["day"]
    tomorrow_feed = _core._tomorrow_feed_need(
        obs, farm, active_positions, animal_missing
    )
    today_delta = _cow_feed_delta(obs, targets, active_positions, day)

    # Match _tomorrow_feed_need's own guards. If the core deliberately does
    # not reserve tomorrow yet, staggering must not invent that reserve either.
    tomorrow_delta = 0  # Core reserve already uses the spatial maintenance phase.

    baseline_now = live_animals + pending_feed
    desired_now = baseline_now + today_delta
    if purchase_mode:
        baseline_total = baseline_now + max(0, tomorrow_feed - wheat_incoming)
        desired_total = desired_now + max(
            0, tomorrow_feed + tomorrow_delta - wheat_incoming
        )
    else:
        # Intraday feed orders must use the same incoming-harvest credit as
        # purchase_orders. Otherwise this earlier order spends the seed
        # budget on a reserve that today's WHEAT harvest already covers.
        # Keep today's feed protected until the harvest actually arrives.
        baseline_total = baseline_now + max(0, tomorrow_feed - wheat_incoming)
        desired_total = desired_now + max(
            0, tomorrow_feed + tomorrow_delta - wheat_incoming
        )
    return (
        desired_total - baseline_total,
        desired_total,
        carried_wheat,
    )


def build_tasks(
    obs,
    targets,
    assume_crop_seeds=False,
    assume_animal_inputs=False,
    prioritize_fertilizer_drop=False,
):
    tasks = _core.build_tasks(
        obs,
        targets,
        assume_crop_seeds=assume_crop_seeds,
        assume_animal_inputs=assume_animal_inputs,
        prioritize_fertilizer_drop=prioritize_fertilizer_drop,
    )

    day = obs["day"]
    end_day = obs.get("_planning_end_day", _core.SEASON_END_DAY)
    if day >= end_day:
        return tasks  # final-day liquidation owns the whole schedule

    farm = obs["farms"][obs["player"]]

    # WATER: move only baseline maintenance actions to a position-specific,
    # equal-yield phase. Forced WATER that the core adds next to HARVEST is
    # retained unless it is exactly the baseline day being shifted away.
    for position, target in targets.items():
        if target is None:
            continue
        x, y = position
        tile = farm["tiles"][y][x]
        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
            continue
        crop = tile["crop"]
        fertilized = bool(target[1])
        age = day - tile["planted_day"]
        baseline = age in CROP_WATER_DAYS[(crop, fertilized)]
        desired = _maintenance.should_water(crop, age, fertilized, position)
        if tile.get("consecutive_unwatered", 0) and not tile.get("watered_today"):
            _ensure_current_water(tasks, position)
        elif baseline and not desired:
            _remove_current_water(tasks, position)
        elif desired and not baseline:
            _ensure_current_water(tasks, position)

    # FEED/CARE: only COW has a useful equal-yield phase. Variant 1 moves the
    # age-3 pair to age 2, leaving PLACE day unchanged and preserving every
    # prefix of milk production. GOOSE/SHEEP remain on their proven minimum.
    for position, target in targets.items():
        if target is None:
            continue
        x, y = position
        tile = farm["tiles"][y][x]
        if not (isinstance(tile, dict) and tile.get("animal") == "COW"):
            continue
        age = day - tile["placed_day"]
        can_pay = animal_maintenance_can_still_pay("COW", age, day, end_day)
        baseline_feed = can_pay and age in ANIMAL_FEED_DAYS["COW"]
        baseline_care = can_pay and age in ANIMAL_CARE_DAYS["COW"]
        desired_feed = can_pay and _maintenance.should_feed("COW", age, position)
        desired_care = can_pay and _maintenance.should_care("COW", age, position)

        if (baseline_feed and not desired_feed) or (
            baseline_care and not desired_care
        ):
            _remove_cow_maintenance(
                tasks,
                position,
                tile,
                remove_feed=baseline_feed and not desired_feed,
                remove_care=baseline_care and not desired_care,
            )
        if (desired_feed and not baseline_feed) or (
            desired_care and not baseline_care
        ):
            _ensure_cow_maintenance(
                tasks,
                position,
                tile,
                want_feed=desired_feed,
                want_care=desired_care,
            )

    # A missed action invalidates the minimum-care schedule's assumptions.
    # Rescue the animal even when today was originally a planned rest day.
    for position, target in targets.items():
        tile = farm["tiles"][position[1]][position[0]]
        if (target and isinstance(tile, dict) and tile.get("animal")
                and tile.get("consecutive_unfed", 0) and not tile.get("fed_today")):
            _ensure_cow_maintenance(tasks, position, tile, True, False)
    return tasks


def feed_wheat_order(obs, targets, active_positions):
    """Today's feed shortfall using the same COW phase as build_tasks."""
    _delta, desired_total, carried_wheat = _feed_need_adjustment(
        obs, targets, active_positions, purchase_mode=False
    )
    shed = obs["private"]["shed"]
    wheat_needed = max(
        0, desired_total - shed.get("WHEAT", 0) - carried_wheat
    )
    return [["BUY_PRODUCT", "WHEAT", wheat_needed]] if wheat_needed else []


def purchase_orders(
    obs, targets, active_positions, available_money=None, available_wheat=None,
    replant_same_crop=False,
):
    """Core purchasing with WHEAT availability shifted by staggered COW demand."""
    delta, _desired_total, _carried_wheat = _feed_need_adjustment(
        obs,
        targets,
        active_positions,
        replant_same_crop=replant_same_crop,
        purchase_mode=True,
    )
    base_wheat = (
        obs["private"]["shed"].get("WHEAT", 0)
        if available_wheat is None
        else available_wheat
    )
    return _core.purchase_orders(
        obs,
        targets,
        active_positions,
        available_money=available_money,
        available_wheat=base_wheat - delta,
        replant_same_crop=replant_same_crop,
    )
