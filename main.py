"""Kaggle entry point for the deterministic 5x5 heuristic."""

from __future__ import annotations

from typing import Any

from kaggriculture_agent.heuristic import HeuristicAgent

_AGENT = HeuristicAgent()


def agent(obs: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
    return _AGENT(obs, configuration)
