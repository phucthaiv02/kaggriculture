"""Immutable fertilizer plans shared by planner, forecast and daily tasks.

Dynamic planner targets may carry a multi-cycle :class:`FertilizerPlan` so the
market forecast can value every fertilizer event independently.  Daily task
building only needs the first/current cycle.  Opening targets keep their
historic boolean flag and remain fully supported.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class FertilizerPlan:
    """Fertilizer ages for each forecast crop cycle.

    ``cycles[i]`` is the tuple of crop ages at which cycle ``i`` should receive
    FERTILIZE.  The first tuple is the standing commitment for the currently
    planted/newly planted crop; later tuples are forecast assumptions only and
    are reconsidered when that crop cycle actually finishes.
    """

    cycles: tuple[tuple[int, ...], ...] = ()

    def cycle(self, index: int = 0) -> tuple[int, ...]:
        if 0 <= index < len(self.cycles):
            return self.cycles[index]
        return ()

    def __bool__(self) -> bool:
        """Backward-compatible truthiness means current-cycle fertilization."""
        return bool(self.cycle(0))


def cycle_plan(plan, index: int = 0):
    """Return the per-cycle plan while preserving legacy booleans."""
    if isinstance(plan, FertilizerPlan):
        return plan.cycle(index)
    return plan


def plan_ages(plan, index: int = 0) -> tuple[int, ...] | None:
    """Normalize an event plan to ages; ``None`` means legacy boolean True."""
    plan = cycle_plan(plan, index)
    if isinstance(plan, bool):
        return None if plan else ()
    if plan is None:
        return ()
    return tuple(sorted(int(age) for age in plan))


def plan_json(plan):
    """JSON-safe representation used by diagnostics and experiments."""
    if isinstance(plan, FertilizerPlan):
        return [list(cycle) for cycle in plan.cycles]
    if isinstance(plan, bool):
        return plan
    if plan is None:
        return []
    return list(plan)
