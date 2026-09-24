"""Agent package initialization for the agent-v5 branch."""

from agents.v5_schedule_guard import install as _install_schedule_guard
from agents.v5_wiring import install as _install_v5

_install_schedule_guard()
_install_v5()
