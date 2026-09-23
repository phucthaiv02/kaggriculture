"""Kaggle submission entry point for the production Kaggriculture agent."""

from agents.expansion_agent import make_agent


# Kaggle looks for a module-level callable named ``agent``.
# A fresh worker process imports this module once per match, so the closure can
# safely retain the production agent's per-match planning state.
agent = make_agent()
