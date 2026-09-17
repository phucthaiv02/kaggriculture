"""Data model for the v2 execution agent.

Every TileJob is required work for the current day. There is no urgency or
priority field. Jobs stay atomic in the main solve; only an approved lifecycle
cut may be used by tail filling.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class TileJob:
    position: tuple[int, int]
    actions: tuple[tuple, ...]
    needs: Counter = field(default_factory=Counter, compare=False)
    produces: Counter = field(default_factory=Counter, compare=False)
    split_after: int | None = None
    split_role: str | None = None

    def split(self) -> tuple["TileJob", "TileJob"] | None:
        if self.split_after is None:
            return None
        prefix = TileJob(
            self.position,
            self.actions[: self.split_after],
            Counter(),
            Counter(self.produces),
            None,
            "prefix",
        )
        suffix = TileJob(
            self.position,
            self.actions[self.split_after :],
            Counter(self.needs),
            Counter(),
            None,
            "suffix",
        )
        return prefix, suffix


def _safe_split(actions: tuple[tuple, ...]) -> int | None:
    names = tuple(action[0] for action in actions)
    for index in range(1, len(names)):
        left, right = names[:index], names[index:]
        if (
            len(left) >= 2
            and left[-2:] == ("WATER", "HARVEST")
            and len(right) >= 2
            and right[:2] == ("PLANT", "WATER")
        ):
            return index
        if (
            len(left) >= 2
            and left[-2:] == ("HARVEST", "COLLECT_FERTILIZER")
            and len(right) >= 2
            and right[:2] == ("FEED", "CARE")
        ):
            return index
    return None


def _normalize_actions(actions: list[tuple]) -> tuple[tuple, ...]:
    names = [action[0] for action in actions]
    animal_ops = {"HARVEST", "COLLECT_FERTILIZER", "FEED", "CARE"}
    if names and set(names).issubset(animal_ops):
        rank = {"HARVEST": 0, "COLLECT_FERTILIZER": 1, "FEED": 2, "CARE": 3}
        actions = sorted(actions, key=lambda action: rank[action[0]])
    return tuple(actions)


@dataclass
class WorkerPlan:
    start: tuple[int, int]
    jobs: list[TileJob]
    queue: list[list]
    pickup: Counter = field(default_factory=Counter)

    @property
    def length(self) -> int:
        return len(self.queue)


@dataclass(frozen=True)
class SupplyPlan:
    pickup_by_worker: tuple[Counter, ...]
    buy_shortfall: Counter


def jobs_from_tasks(tasks: Iterable) -> list[TileJob]:
    """Collapse all same-tile legacy Tasks into one atomic v2 TileJob.

    Legacy urgency/deadline/drop metadata is deliberately ignored. DIG is
    preserved when the task generator explicitly requires it (notably WEED
    recovery). The v2 lifecycle layer separately removes DIG from normal
    HARVEST -> next-lifecycle transitions, where harvesting already frees the
    tile and no DIG rule exists.
    """
    grouped = {}
    order = []
    for task in tasks:
        position = tuple(task.position)
        if position not in grouped:
            grouped[position] = {"actions": [], "needs": Counter(), "produces": Counter()}
            order.append(position)
        entry = grouped[position]
        entry["actions"].extend(
            tuple(action)
            for action in task.actions
            if action
        )
        entry["needs"].update(task.needs)
        entry["produces"].update(task.sells)

    jobs = []
    for position in order:
        entry = grouped[position]
        actions = _normalize_actions(entry["actions"])
        if not actions:
            continue
        jobs.append(
            TileJob(
                position,
                actions,
                +entry["needs"],
                +entry["produces"],
                _safe_split(actions),
            )
        )
    return jobs
