"""Estimate daily route/worker costs for comparing production portfolios.

This is a greedy route estimate, not the execution scheduler: it batches
pickups, includes travel and a return/drop for same-day sales, and charges
actual daily Fibonacci hire prices. It does not claim exact future routes.
"""
from collections import defaultdict
from functools import lru_cache
from kaggle_environments.envs.kaggriculture import kaggriculture as game
from agents.scheduler import HAND_BUDGET, MAX_HANDS


class LaborForecast:
    def __init__(self, shed_access=((4, 4), (5, 4), (4, 5), (5, 5))):
        self.shed_access = shed_access
        self.daily_cost = lru_cache(maxsize=4096)(self._daily_cost)

    def _daily_cost(self, tasks):
        remaining = list(tasks)
        workers = 0
        while remaining:
            if workers > MAX_HANDS:
                return float('inf')
            # Start at the shed access nearest the next service cluster.
            first = min(remaining, key=lambda t: min(abs(t[0]-s[0])+abs(t[1]-s[1]) for s in self.shed_access))
            start = min(self.shed_access, key=lambda s: abs(first[0]-s[0])+abs(first[1]-s[1]))
            current, used, supplies, goods = start, 0, set(), False
            assigned = 0
            while remaining:
                possible = []
                for index, (x, y, count, inputs, output) in enumerate(remaining):
                    travel = abs(current[0]-x) + abs(current[1]-y)
                    pickups = len(set(inputs) - supplies)
                    # Pickups are batched at the route's initial shed visit.
                    back = min(abs(x-s[0])+abs(y-s[1]) for s in self.shed_access) if goods or output else 0
                    projected = used + travel + count + pickups
                    if projected + back + int(goods or output) <= HAND_BUDGET:
                        possible.append((travel, projected, index))
                if not possible:
                    break
                _, used, index = min(possible)
                x, y, _, inputs, output = remaining.pop(index)
                supplies.update(inputs)
                goods = goods or output
                current = (x, y)
                assigned += 1
            if not assigned:
                return float('inf')
            workers += 1
        return sum(game._hire_cost(n) for n in range(max(0, workers - 1)))

    def cost(self, visits):
        total = 0
        for entries in visits.values():
            merged = {}
            for position, count, inputs, goods in entries:
                if position is None:
                    position = self.shed_access[0]
                row = merged.setdefault(position, [0, set(), False])
                row[0] += count
                row[1].update(inputs)
                row[2] |= goods
            tasks = tuple(sorted((x, y, n, tuple(sorted(inputs)), goods)
                                 for (x, y), (n, inputs, goods) in merged.items()))
            total += self.daily_cost(tasks)
        return total

    def marginal_cost(self, baseline, candidate, position, baseline_cost=None):
        visits = defaultdict(list, {day: list(v) for day, v in baseline.visits.items()})
        for day, entries in candidate.visits.items():
            visits[day].extend((position, count, inputs, goods) for _, count, inputs, goods in entries)
        if baseline_cost is None:
            baseline_cost = self.cost(baseline.visits)
        combined_cost = self.cost(visits)
        if combined_cost == float('inf'):
            return float('inf')
        # Greedy route packing can improve accidentally when another task
        # changes insertion order. That is not income earned by the target.
        return max(0, combined_cost - baseline_cost)
