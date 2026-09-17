"""Per-tile lifecycle schedule locking for agent v2.

Experiment code owns the legal schedules. This module only chooses among
verified equivalent variants so that the peak number of required tile actions
on future days is as small as possible. Once chosen, a variant is locked until
that tile starts a new lifecycle (new planted_day / placed_day).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from agents.schedules import should_care_animal, should_feed_animal
from experiments.animal_yields import OPTIMAL_SCHEDULES, find_equivalent_schedules
from experiments.crop_schedules import CASES, find_equivalent_water_schedules

CROPS = ("WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY")
ANIMALS = ("GOOSE", "COW", "SHEEP")
VARIANT_LIMIT = 8
SEASON_DAYS = 30


@dataclass(frozen=True)
class LifecycleSchedule:
    producer: str
    fertilized: bool
    water: frozenset[int] = frozenset()
    fertilize: frozenset[int] = frozenset()
    harvest: frozenset[int] = frozenset()
    feed: frozenset[int] = frozenset()
    care: frozenset[int] = frozenset()
    collect: frozenset[int] = frozenset()

    def action_count(self, age: int) -> int:
        return sum(
            age in days
            for days in (
                self.water,
                self.fertilize,
                self.harvest,
                self.feed,
                self.care,
                self.collect,
            )
        )


@dataclass(frozen=True)
class TileLifecycle:
    token: tuple
    start_day: int
    schedule: LifecycleSchedule


@lru_cache(maxsize=None)
def schedule_variants(producer: str, fertilized: bool = False):
    """Return experiment-backed variants, cheapest-action variants first.

    Animal setup still materializes through the verified placement rules in
    ``farm_tasks``. Therefore an equivalent animal variant is only legal for
    v2 if its age-0 FEED/CARE decisions match those placement-day actions.
    Otherwise the schedule may be max-yield in isolation but starts from a
    different engine state and can let a newly placed animal escape on age 1.
    """
    if producer in CROPS:
        case = CASES[(producer, bool(fertilized))]
        waters = find_equivalent_water_schedules(
            producer, bool(fertilized), limit=VARIANT_LIMIT
        )
        minimum = min(map(len, waters))
        # Never add maintenance actions merely to make the calendar prettier.
        waters = [days for days in waters if len(days) == minimum]
        return tuple(
            LifecycleSchedule(
                producer,
                bool(fertilized),
                water=frozenset(days),
                fertilize=frozenset(case["fertilize"]),
                harvest=frozenset(case["harvest"]),
            )
            for days in waters
        )

    baseline = OPTIMAL_SCHEDULES[producer]
    searched = find_equivalent_schedules(
        producer, days=SEASON_DAYS, limit=VARIANT_LIMIT
    )

    # Include the verified baseline even if the DP's top-N equivalent paths do
    # not happen to contain it. Then keep only variants that begin from the
    # same placement-day state that the real setup job creates.
    raw_candidates = [baseline, *searched]
    expected_feed0 = should_feed_animal(producer, 0)
    expected_care0 = should_care_animal(producer, 0)
    candidates = []
    seen = set()
    for candidate in raw_candidates:
        signature = (
            frozenset(candidate["harvest"]),
            frozenset(candidate["feed"]),
            frozenset(candidate["care"]),
        )
        if signature in seen:
            continue
        seen.add(signature)
        if (0 in candidate["feed"]) != expected_feed0:
            continue
        if (0 in candidate["care"]) != expected_care0:
            continue
        candidates.append(candidate)

    if not candidates:
        # The baseline is derived from the same verified age-relative setup
        # rules, so this is a defensive invariant rather than an expected path.
        candidates = [baseline]

    baseline_collect = frozenset(baseline["collect_fertilizer"])
    return tuple(
        LifecycleSchedule(
            producer,
            False,
            harvest=frozenset(candidate["harvest"]),
            feed=frozenset(candidate["feed"]),
            care=frozenset(candidate["care"]),
            collect=baseline_collect,
        )
        for candidate in candidates[:VARIANT_LIMIT]
    )


class ScheduleBook:
    def __init__(self, end_day=SEASON_DAYS - 1):
        self.end_day = end_day
        self.tiles: dict[tuple[int, int], TileLifecycle] = {}

    def _load(self, exclude=()):
        excluded = set(exclude)
        load = Counter()
        for position, lifecycle in self.tiles.items():
            if position in excluded:
                continue
            for day in range(max(0, lifecycle.start_day), self.end_day + 1):
                age = day - lifecycle.start_day
                load[day] += lifecycle.schedule.action_count(age)
        return load

    def _variant_score(self, variant, start_day, load):
        projected = Counter(load)
        for day in range(start_day, self.end_day + 1):
            projected[day] += variant.action_count(day - start_day)
        peak = max(projected.values(), default=0)
        total = sum(projected.values())
        squared = sum(value * value for value in projected.values())
        return peak, squared, total

    def assign_batch(self, positions, producer, fertilized, start_day, token_by_position):
        """Assign same-day/same-producer starts as one balancing batch.

        Positions do not affect this layer's objective. Routing stays in the
        daily scheduler; this layer only balances the future action calendar.
        """
        positions = sorted(positions)
        variants = schedule_variants(producer, fertilized)
        load = self._load(exclude=positions)
        for position in positions:
            scored = [
                (self._variant_score(variant, start_day, load), index, variant)
                for index, variant in enumerate(variants)
            ]
            _, _, chosen = min(scored, key=lambda row: (row[0], row[1]))
            lifecycle = TileLifecycle(token_by_position[position], start_day, chosen)
            self.tiles[position] = lifecycle
            for day in range(start_day, self.end_day + 1):
                load[day] += chosen.action_count(day - start_day)

    def sync(self, obs, targets):
        """Detect new crop/animal lifecycles and lock a variant for each."""
        farm = obs["farms"][obs["player"]]
        pending = {}
        live_positions = set()
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                if not isinstance(tile, dict):
                    continue
                producer = tile.get("animal")
                start_day = tile.get("placed_day")
                if producer:
                    fertilized = False
                elif tile.get("kind") == "PLANT" and tile.get("crop"):
                    producer = tile["crop"]
                    start_day = tile.get("planted_day")
                    target = targets.get((x, y))
                    fertilized = bool(target and target[0] == producer and target[1])
                else:
                    continue
                if start_day is None:
                    continue
                position = (x, y)
                live_positions.add(position)
                token = (producer, int(start_day), bool(fertilized))
                current = self.tiles.get(position)
                if current and current.token == token:
                    continue
                key = (producer, bool(fertilized), int(start_day))
                pending.setdefault(key, {})[position] = token

        for position in list(self.tiles):
            if position not in live_positions:
                del self.tiles[position]

        for (producer, fertilized, start_day), token_by_position in sorted(pending.items()):
            self.assign_batch(
                token_by_position,
                producer,
                fertilized,
                start_day,
                token_by_position,
            )

    def get(self, position):
        lifecycle = self.tiles.get(tuple(position))
        return lifecycle.schedule if lifecycle else None
