"""Kaggle entry point for the observation-conditioned dynamic policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kaggriculture_agent.inference import DynamicAgent


ROOT = Path(globals().get("__file__", "main.py")).resolve().parent
CHECKPOINT = ROOT / "model.pt"
_AGENT: DynamicAgent | None = None


def agent(obs: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
    global _AGENT
    if _AGENT is None:
        _AGENT = DynamicAgent(CHECKPOINT, device="cpu", temperature=0.0)
    return _AGENT(obs)
