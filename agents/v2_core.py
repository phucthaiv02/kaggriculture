"""Core data model for the v2 optimizer agent.

The old production agent mixes lifecycle policy, route priority and runtime
rescues.  V2 keeps those responsibilities separate:

* experiments define verified lifecycle schedules;
* every tile locks one schedule variant for one lifecycle;
* the daily builder emits neutral atomic TileJobs;
* the scheduler decides only labor and routing.

Only schedules already validated by the repository experiments are loaded
here.  The representation intentionally supports multiple equivalent variants;
new experiment-backed variants can be added without changing the scheduler.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS as ENV_ANIMALS

from agents.schedules import ONGOING_CROPS

CROPS = ("WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY")
ANIMALS = ("GOOSE", "COW", "SHEEP")
BUILD = {"GOOSE": "BUILD_COOP", "COW": "BUILD_PASTURE", "SHEEP": "BUILD_PASTURE"}
STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}


@dataclass(frozen=True)
class ScheduleVariant:
    id: str
    water_days: frozenset[int] = frozenset()
    fertilize_days: frozenset[int] = frozenset()
    harvest_days: frozenset[int] = frozenset()
    feed_days: frozenset[int] = frozenset()
    care_days: frozenset[int] = frozenset()
    collect_days: frozenset[int] = frozenset()
    ongoing: bool = False

    @property
    def last_age(self) -> int:
        days = (
            self.water_days
            | self.fertilize_days
            | self.harvest_days
            | self.feed_days
            | self.care_days
            | self.collect_days
        )
        return max(days) if days else 0

    def maintenance_actions(self, age: int) -> int:
        """Number of tile actions contributed by this lifecycle at ``age``."""
        return sum(
            age in days
            for days in (
                self.water_days,
                self.fertilize_days,
                self.harvest_days,
                self.feed_days,
                self.care_days,
                self.collect_days,
            )
        )


# These tables mirror experiments/crop_schedules.py and
# experiments/animal_yields.py.  Each key maps to a tuple on purpose: the
# lifecycle allocator works with any number of equivalent experiment-backed
# variants and chooses them in a batch to minimize peak future tile actions.
CROP_VARIANTS: dict[tuple[str, bool], tuple[ScheduleVariant, ...]] = {
    ("WHEAT", False): (
        ScheduleVariant("wheat-base", frozenset({0, 2, 3, 4}), harvest_days=frozenset({4})),
    ),
    ("WHEAT", True): (
        ScheduleVariant(
            "wheat-fert-base",
            frozenset({0, 2, 3, 4}),
            frozenset({2}),
            frozenset({4}),
        ),
    ),
    ("CARROT", False): (
        ScheduleVariant("carrot-base", frozenset({0, 2, 3}), harvest_days=frozenset({3})),
    ),
    ("CARROT", True): (
        ScheduleVariant(
            "carrot-fert-base",
            frozenset({0, 2, 3}),
            frozenset({2}),
            frozenset({3}),
        ),
    ),
    ("MELON", False): (
        ScheduleVariant(
            "melon-base",
            frozenset({0, 2, 4, 6, 7, 8, 9, 10}),
            harvest_days=frozenset({10}),
        ),
    ),
    ("MELON", True): (
        ScheduleVariant(
            "melon-fert-base",
            frozenset({0, 2, 4, 6, 8, 9}),
            frozenset({8}),
            frozenset({10}),
        ),
    ),
    ("TOMATO", False): (
        ScheduleVariant(
            "tomato-base",
            frozenset({0, 2, 4, 6, 8, 9}),
            harvest_days=frozenset({11}),
            ongoing=True,
        ),
    ),
    ("TOMATO", True): (
        ScheduleVariant(
            "tomato-fert-base",
            frozenset({0, 2, 4, 5, 7, 8, 9, 10}),
            frozenset({7, 10}),
            frozenset({9, 11}),
            ongoing=True,
        ),
    ),
    ("STRAWBERRY", False): (
        ScheduleVariant(
            "strawberry-base",
            frozenset({0, 2, 4, 6, 8, 10, 12, 14}),
            harvest_days=frozenset({16}),
            ongoing=True,
        ),
    ),
    ("STRAWBERRY", True): (
        ScheduleVariant(
            "strawberry-fert-base",
            frozenset({0, 2, 4, 6, 7, 9, 11, 13, 15}),
            frozenset({9, 13}),
            frozenset({12, 16}),
            ongoing=True,
        ),
    ),
}

ANIMAL_VARIANTS: dict[str, tuple[ScheduleVariant, ...]] = {
    "GOOSE": (
        ScheduleVariant(
            "goose-base",
            harvest_days=frozenset({4, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29}),
            feed_days=frozenset(range(29)),
            care_days=frozenset(range(28)),
            collect_days=frozenset(range(1, 30)),
            ongoing=True,
        ),
    ),
    "COW": (
        ScheduleVariant(
            "cow-base",
            harvest_days=frozenset({8, 12, 16, 21, 25, 29}),
            feed_days=frozenset({1, *range(3, 28)}),
            care_days=frozenset({1, *range(3, 27)}),
            collect_days=frozenset(range(1, 30)),
            ongoing=True,
        ),
    ),
    "SHEEP": (
        ScheduleVariant(
            "sheep-base",
            harvest_days=frozenset({6, 9, 12, 17, 18, 21, 24, 27}),
            feed_days=frozenset(range(28)),
            care_days=frozenset(range(26)),
            collect_days=frozenset(range(1, 30)),
            ongoing=True,
        ),
    ),
}


def variants_for(name: str, fertilize: bool = False) -> tuple[ScheduleVariant, ...]:
    if name in CROPS:
        return CROP_VARIANTS[(name, bool(fertilize))]
    return ANIMAL_VARIANTS[name]


@dataclass(frozen=True)
class LifecycleLock:
    producer: str
    start_day: int
    fertilize: bool
    variant_id: str


@dataclass(frozen=True)
class JobAction:
    op: tuple
    consume: Counter = field(default_factory=Counter)
    produce: Counter = field(default_factory=Counter)

    @classmethod
    def simple(cls, op, consume=None, produce=None):
        return cls(tuple(op), Counter(consume or {}), Counter(produce or {}))


@dataclass(frozen=True)
class TileJob:
    position: tuple[int, int]
    actions: tuple[JobAction, ...]
    key: str
    split_index: int | None = None
    parent_key: str | None = None
    release_from: str | None = None

    @property
    def raw_actions(self) -> list[list]:
        return [list(step.op) for step in self.actions]

    @property
    def tile_action_count(self) -> int:
        return len(self.actions)

    def split(self) -> tuple["TileJob", "TileJob"] | None:
        cut = self.split_index
        if cut is None or cut <= 0 or cut >= len(self.actions):
            return None
        prefix_key = f"{self.key}:prefix"
        suffix_key = f"{self.key}:suffix"
        return (
            TileJob(
                self.position,
                self.actions[:cut],
                prefix_key,
                None,
                parent_key=self.key,
            ),
            TileJob(
                self.position,
                self.actions[cut:],
                suffix_key,
                None,
                parent_key=self.key,
                release_from=prefix_key,
            ),
        )


class LifecycleBook:
    """Persistent, per-agent schedule locks.

    A lock lasts exactly one lifecycle.  New lifecycles are assigned in a
    batch, not one tile at a time, so the result is independent of tile
    iteration order.
    """

    def __init__(self):
        self.locks: dict[tuple[int, int], LifecycleLock] = {}

    def _variant(self, lock: LifecycleLock) -> ScheduleVariant:
        for variant in variants_for(lock.producer, lock.fertilize):
            if variant.id == lock.variant_id:
                return variant
        return variants_for(lock.producer, lock.fertilize)[0]

    def variant_at(self, position: tuple[int, int]) -> ScheduleVariant | None:
        lock = self.locks.get(position)
        return self._variant(lock) if lock else None

    def _calendar(self, exclude: set[tuple[int, int]] | None = None) -> Counter:
        exclude = exclude or set()
        calendar = Counter()
        for position, lock in self.locks.items():
            if position in exclude:
                continue
            variant = self._variant(lock)
            for age in range(variant.last_age + 1):
                actions = variant.maintenance_actions(age)
                if actions:
                    calendar[lock.start_day + age] += actions
        return calendar

    @staticmethod
    def _compositions(total: int, parts: int):
        if parts == 1:
            yield (total,)
            return
        for first in range(total + 1):
            for rest in LifecycleBook._compositions(total - first, parts - 1):
                yield (first, *rest)

    def assign_batch(
        self,
        positions: Iterable[tuple[int, int]],
        producer: str,
        fertilize: bool,
        start_day: int,
    ) -> None:
        positions = tuple(sorted(set(positions), key=lambda p: (p[1], p[0])))
        if not positions:
            return
        options = variants_for(producer, fertilize)
        if len(options) == 1:
            variant = options[0]
            for position in positions:
                self.locks[position] = LifecycleLock(
                    producer, start_day, bool(fertilize), variant.id
                )
            return

        base = self._calendar(exclude=set(positions))
        best_score = None
        best_counts = None
        for counts in self._compositions(len(positions), len(options)):
            trial = Counter(base)
            for count, variant in zip(counts, options):
                if not count:
                    continue
                for age in range(variant.last_age + 1):
                    actions = variant.maintenance_actions(age)
                    if actions:
                        trial[start_day + age] += count * actions
            if trial:
                peak = max(trial.values())
                spread = sum(value * value for value in trial.values())
            else:
                peak = spread = 0
            score = (peak, spread, counts)
            if best_score is None or score < best_score:
                best_score, best_counts = score, counts

        cursor = 0
        for count, variant in zip(best_counts, options):
            for position in positions[cursor:cursor + count]:
                self.locks[position] = LifecycleLock(
                    producer, start_day, bool(fertilize), variant.id
                )
            cursor += count

    def ensure_existing(self, position, producer, start_day, fertilize=False):
        lock = self.locks.get(position)
        if (
            lock is None
            or lock.producer != producer
            or lock.start_day != int(start_day)
            or lock.fertilize != bool(fertilize)
        ):
            # A game loaded mid-lifecycle cannot safely switch to a different
            # equivalent variant because earlier maintenance is unknown.
            variant = variants_for(producer, fertilize)[0]
            self.locks[position] = LifecycleLock(
                producer, int(start_day), bool(fertilize), variant.id
            )
        return self.locks[position]

    def prepare_new(self, position, producer, fertilize, start_day):
        self.assign_batch([position], producer, fertilize, start_day)
        return self.locks[position]

    def reconcile(self, obs, targets):
        day = obs["day"]
        farm = obs["farms"][obs["player"]]
        stale = []
        for position, lock in self.locks.items():
            x, y = position
            tile = farm["tiles"][y][x]
            actual = None
            started = None
            if isinstance(tile, dict):
                if tile.get("animal"):
                    actual, started = tile["animal"], tile.get("placed_day")
                elif tile.get("kind") == "PLANT":
                    actual, started = tile.get("crop"), tile.get("planted_day")
            if actual == lock.producer and started == lock.start_day:
                continue
            target = targets.get(position)
            # Keep a same-day pending lock for a lifecycle whose PLANT/PLACE
            # has been scheduled but has not executed yet.
            if (
                actual is None
                and target
                and target[0] == lock.producer
                and lock.start_day == day
            ):
                continue
            stale.append(position)
        for position in stale:
            self.locks.pop(position, None)

        for position, target in targets.items():
            x, y = position
            tile = farm["tiles"][y][x]
            if not isinstance(tile, dict):
                continue
            if tile.get("animal"):
                name, start = tile["animal"], tile.get("placed_day")
            elif tile.get("kind") == "PLANT":
                name, start = tile.get("crop"), tile.get("planted_day")
            else:
                continue
            fertilize = bool(target and target[0] == name and target[1] and name in CROPS)
            self.ensure_existing(position, name, start, fertilize)


def _step(op, consume=None, produce=None):
    return JobAction.simple(op, consume, produce)


def _harvest_step(tile, producer):
    amount = int(tile.get("yield_units", 0)) if isinstance(tile, dict) else 0
    if producer in ANIMALS:
        product_name = ENV_ANIMALS[producer]["product"]
    else:
        product_name = producer
    return _step(["HARVEST"], produce={product_name: max(0, amount)})


def _setup_actions(name, variant, tile):
    actions: list[JobAction] = []
    if isinstance(tile, dict) and tile.get("kind") == "WEED":
        actions.append(_step(["DIG"]))
        tile = None
    if name in CROPS:
        if tile is not None:
            return actions
        actions.append(_step(["PLANT", name]))
        if 0 in variant.water_days:
            actions.append(_step(["WATER"]))
        return actions

    already_built = isinstance(tile, dict) and tile.get("kind") == STRUCTURE[name]
    if isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE") and not already_built:
        return actions
    if not already_built:
        if tile is not None:
            return actions
        actions.append(_step([BUILD[name]]))
    actions.append(_step(["PLACE", name], consume={name: 1}))
    if 0 in variant.feed_days:
        actions.append(_step(["FEED"], consume={"WHEAT": 1}))
    if 0 in variant.care_days:
        actions.append(_step(["CARE"]))
    return actions


def _split_index(actions: list[JobAction], producer: str) -> int | None:
    names = [step.op[0] for step in actions]
    if producer in CROPS:
        # Safe crop cut: [WATER -> HARVEST] | [PLANT -> WATER].
        if "HARVEST" in names and "PLANT" in names:
            harvest = names.index("HARVEST")
            plant = names.index("PLANT")
            if harvest < plant:
                return harvest + 1
    else:
        # Safe animal cut: [HARVEST -> COLLECT] | [FEED -> CARE].
        maintenance = [i for i, name in enumerate(names) if name in ("FEED", "CARE")]
        prefix = [i for i, name in enumerate(names) if name in ("HARVEST", "COLLECT_FERTILIZER")]
        if maintenance and prefix and max(prefix) < min(maintenance):
            return min(maintenance)
    return None


def _crop_today_actions(tile, variant, age):
    actions: list[JobAction] = []
    ongoing = variant.ongoing
    # Experiments require ongoing crops to HARVEST before the same day's
    # production maintenance. One-time crops WATER before the max-yield harvest.
    if ongoing and age in variant.harvest_days and tile.get("yield_units", 0) > 0:
        actions.append(_harvest_step(tile, tile["crop"]))
    if age in variant.fertilize_days:
        actions.append(_step(["FERTILIZE"], consume={"FERTILIZER": 1}))
    if age in variant.water_days and not tile.get("watered_today"):
        actions.append(_step(["WATER"]))
    if (not ongoing) and age in variant.harvest_days:
        actions.append(_harvest_step(tile, tile["crop"]))
    return actions


def _animal_today_actions(tile, variant, age, name):
    actions: list[JobAction] = []
    if age in variant.harvest_days and tile.get("yield_units", 0) > 0:
        actions.append(_harvest_step(tile, name))
    if age in variant.collect_days and tile.get("fertilizer_available"):
        actions.append(_step(["COLLECT_FERTILIZER"], produce={"FERTILIZER": 1}))
    if age in variant.feed_days and not tile.get("fed_today"):
        actions.append(_step(["FEED"], consume={"WHEAT": 1}))
    if age in variant.care_days and not tile.get("cared_today"):
        actions.append(_step(["CARE"]))
    return actions


def build_jobs(obs, targets, book: LifecycleBook) -> list[TileJob]:
    """Build neutral atomic jobs for the current day.

    There is deliberately no urgency/deadline/priority metadata.  Every emitted
    action is required by the locked lifecycle or by a strategy transition.
    """
    day = obs["day"]
    farm = obs["farms"][obs["player"]]
    tiles = farm["tiles"]
    book.reconcile(obs, targets)

    # Batch-assign all brand-new empty targets first. This is where multiple
    # equivalent experiment schedules are balanced against the farm's future
    # peak tile-action calendar.
    new_groups = defaultdict(list)
    for position, target in targets.items():
        if not target:
            continue
        x, y = position
        tile = tiles[y][x]
        if tile is None or (isinstance(tile, dict) and tile.get("kind") == "WEED"):
            name, fertilize = target
            new_groups[(name, bool(fertilize and name in CROPS))].append(position)
    for (name, fertilize), positions in new_groups.items():
        missing = [position for position in positions if position not in book.locks]
        if missing:
            book.assign_batch(missing, name, fertilize, day)

    jobs: list[TileJob] = []
    for position in sorted(targets, key=lambda p: (p[1], p[0])):
        target = targets.get(position)
        if not target:
            continue
        target_name, target_fertilize = target
        x, y = position
        tile = tiles[y][x]
        actions: list[JobAction] = []
        job_producer = target_name

        if isinstance(tile, dict) and tile.get("animal"):
            name = tile["animal"]
            lock = book.ensure_existing(position, name, tile["placed_day"], False)
            variant = book._variant(lock)
            actions.extend(_animal_today_actions(tile, variant, day - tile["placed_day"], name))
            job_producer = name

        elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop = tile["crop"]
            fertilize = bool(target_name == crop and target_fertilize)
            lock = book.ensure_existing(position, crop, tile["planted_day"], fertilize)
            variant = book._variant(lock)
            age = day - tile["planted_day"]
            actions.extend(_crop_today_actions(tile, variant, age))
            job_producer = crop

            # A one-time crop starts the next lifecycle immediately after its
            # experiment-scheduled max-yield harvest. No DIG is inserted.
            if crop not in ONGOING_CROPS and age in variant.harvest_days and target_name:
                next_fertilize = bool(target_fertilize and target_name in CROPS)
                book.prepare_new(position, target_name, next_fertilize, day)
                next_variant = book.variant_at(position)
                actions.extend(_setup_actions(target_name, next_variant, None))
                job_producer = target_name

        else:
            lock = book.locks.get(position)
            if lock is None or lock.producer != target_name or lock.start_day != day:
                book.prepare_new(
                    position,
                    target_name,
                    bool(target_fertilize and target_name in CROPS),
                    day,
                )
            variant = book.variant_at(position)
            actions.extend(_setup_actions(target_name, variant, tile))

        if not actions:
            continue
        key = f"{day}:{position[0]},{position[1]}"
        jobs.append(
            TileJob(
                position,
                tuple(actions),
                key,
                _split_index(actions, job_producer),
            )
        )
    return jobs


def convert_legacy_tasks(obs, tasks) -> list[TileJob]:
    """Merge opening-book Task objects by tile and convert them to v2 jobs.

    Opening strategy remains authoritative; worker priority flags, immediate
    DROP hints and rescue metadata are intentionally discarded.
    """
    farm = obs["farms"][obs["player"]]
    grouped = defaultdict(list)
    for task in tasks:
        grouped[tuple(task.position)].append(task)

    jobs = []
    for index, position in enumerate(sorted(grouped, key=lambda p: (p[1], p[0]))):
        x, y = position
        tile = farm["tiles"][y][x]
        actions = []
        for task in grouped[position]:
            for raw in task.actions:
                name = raw[0]
                consume = Counter()
                produce = Counter()
                if name == "PLACE" and len(raw) > 1:
                    consume[raw[1]] += int(raw[2]) if len(raw) > 2 else 1
                elif name == "FEED":
                    consume["WHEAT"] += 1
                elif name == "FERTILIZE":
                    consume["FERTILIZER"] += 1
                elif name == "HARVEST" and isinstance(tile, dict):
                    producer = tile.get("animal") or tile.get("crop")
                    if producer:
                        step = _harvest_step(tile, producer)
                        produce.update(step.produce)
                elif name == "COLLECT_FERTILIZER":
                    produce["FERTILIZER"] += 1
                actions.append(_step(raw, consume, produce))

        if not actions:
            continue
        producer = None
        if isinstance(tile, dict):
            producer = tile.get("animal") or tile.get("crop")
        producer = producer or "OPENING"
        jobs.append(
            TileJob(
                position,
                tuple(actions),
                f"{obs['day']}:opening:{index}:{x},{y}",
                _split_index(actions, producer),
            )
        )
    return jobs
