"""Estimate daily route/worker costs for comparing production portfolios.

This is a greedy route estimate, not the execution scheduler: it batches
pickups, includes travel and a return/drop for same-day sales, and charges
actual daily Fibonacci hire prices. It does not claim exact future routes.
"""
from functools import lru_cache
from kaggle_environments.envs.kaggriculture import kaggriculture as game
from agents.scheduler import HAND_BUDGET, MAX_HANDS


class LaborForecast:
    def __init__(self, shed_access=((4, 4), (5, 4), (4, 5), (5, 5))):
        self.shed_access = shed_access
        self._location = lru_cache(maxsize=256)(self._location)
        self.daily_cost = lru_cache(maxsize=4096)(self._daily_cost)

    def _location(self, x, y):
        shed = min(self.shed_access, key=lambda s: abs(x-s[0]) + abs(y-s[1]))
        return abs(x-shed[0]) + abs(y-shed[1]), shed

    def _daily_cost(self, tasks):
        # Geometry and input sets are invariant across the greedy insertions.
        # Compute them once instead of inside every worker/task comparison.
        remaining = [(*task, *self._location(task[0], task[1])) for task in tasks]
        remaining = [(x, y, n, frozenset(inputs), goods, back, shed)
                     for x, y, n, inputs, goods, back, shed in remaining]
        workers = 0
        while remaining:
            if workers > MAX_HANDS:
                return float('inf')
            first = min(remaining, key=lambda t: t[5])
            cx, cy = first[6]
            used, supplies, goods = 0, frozenset(), False
            assigned = False
            while remaining:
                best = None
                for index, (x, y, count, inputs, output, back, _) in enumerate(remaining):
                    travel = abs(cx-x) + abs(cy-y)
                    projected = used + travel + count + len(inputs - supplies)
                    if projected + (back + 1 if goods or output else 0) <= HAND_BUDGET:
                        candidate = (travel, projected, index)
                        if best is None or candidate < best:
                            best = candidate
                if best is None:
                    break
                _, used, index = best
                cx, cy, _, inputs, output, _, _ = remaining.pop(index)
                supplies = supplies | inputs
                goods = goods or output
                assigned = True
            if not assigned:
                return float('inf')
            workers += 1
        return sum(game._hire_cost(n) for n in range(max(0, workers - 1)))

    def prepare(self, visits):
        """Merge a baseline once for all candidate targets at one position."""
        days = {}
        for day, entries in visits.items():
            merged = {}
            for position, count, inputs, goods in entries:
                position = self.shed_access[0] if position is None else position
                row = merged.setdefault(position, [0, set(), False])
                row[0] += count
                row[1].update(inputs)
                row[2] |= goods
            days[day] = tuple(sorted((x, y, n, tuple(sorted(inputs)), goods)
                                     for (x, y), (n, inputs, goods) in merged.items()))
        return days

    def prepared_cost(self, days):
        return sum(self.daily_cost(tasks) for tasks in days.values())

    def cost(self, visits):
        return self.prepared_cost(self.prepare(visits))

    def marginal_cost(self, baseline, candidate, position, baseline_cost=None, prepared=None):
        prepared = self.prepare(baseline.visits) if prepared is None else prepared
        combined = dict(prepared)
        for day, entries in candidate.visits.items():
            tasks = prepared.get(day, ())
            # Candidate visits all belong to this tile. Other baseline rows
            # are already canonical and can be reused without merging again.
            x, y = position
            count, inputs, goods = 0, set(), False
            unchanged = []
            for row in tasks:
                if row[:2] == position:
                    count += row[2]
                    inputs.update(row[3])
                    goods |= row[4]
                else:
                    unchanged.append(row)
            for _, n, supplies, output in entries:
                count += n
                inputs.update(supplies)
                goods |= output
            if entries or len(unchanged) != len(tasks):
                unchanged.append((x, y, count, tuple(sorted(inputs)), goods))
            combined[day] = tuple(sorted(unchanged))
        if baseline_cost is None:
            baseline_cost = self.prepared_cost(prepared)
        combined_cost = self.prepared_cost(combined)
        if combined_cost == float('inf'):
            return float('inf')
        # Greedy route packing can improve accidentally when another task
        # changes insertion order. That is not income earned by the target.
        return max(0, combined_cost - baseline_cost)
