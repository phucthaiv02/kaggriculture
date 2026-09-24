"""Install agent-v5 production-path valuation without forking planner.py.

The v3 planner remains the compatibility surface for analysis helpers and
focused unit tests. Only real ``plan_targets`` calls (which pass
``flows_scoped=True`` with a MarketForecast and baseline) replace their old
same-crop rotation candidates with bounded mixed production paths.
"""
from __future__ import annotations

from functools import wraps

from agents import planner
from agents.production_paths import production_path_candidates


_INSTALLED = False


def install():
    global _INSTALLED
    if _INSTALLED:
        return

    original = planner._choose

    @wraps(original)
    def choose_with_paths(
        market, baseline, candidates, counts, labor=None, position=(4, 4), *,
        current=None, audit=None, decision_log=None, decision_step=None,
        flows_scoped=False,
    ):
        # plan_targets already scoped both market and baseline to the common
        # finite horizon. Synthetic/direct _choose tests and legacy analysis
        # helpers keep the original candidate contract untouched.
        if flows_scoped and market is not None and baseline is not None:
            names = {
                row[0][0]
                for row in candidates
                if row and row[0]
            }
            paths = production_path_candidates(
                market, baseline, market.day, market.end_day,
                first_names=names,
            )
            candidates = [row[:3] for row in paths]

        return original(
            market, baseline, candidates, counts, labor, position,
            current=current, audit=audit, decision_log=decision_log,
            decision_step=decision_step, flows_scoped=flows_scoped,
        )

    choose_with_paths._agent_v5_path_choose = True
    planner._choose = choose_with_paths
    _INSTALLED = True
