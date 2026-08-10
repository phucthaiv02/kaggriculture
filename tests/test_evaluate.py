from __future__ import annotations

import pytest

from scripts.evaluate import distribution, result_metrics, wilson_interval


def test_distribution_includes_tail_and_confidence_metrics() -> None:
    metrics = distribution([1.0, 2.0, 3.0, 4.0])
    assert metrics["mean"] == pytest.approx(2.5)
    assert metrics["cvar10"] == pytest.approx(1.0)
    assert metrics["mean_ci95"][0] < metrics["mean"] < metrics["mean_ci95"][1]


def test_result_metrics_cover_win_draw_loss() -> None:
    rows = [{"result": result} for result in ("win", "win", "draw", "loss")]
    metrics = result_metrics(rows)
    assert metrics["wins"] == 2
    assert metrics["draws"] == 1
    assert metrics["losses"] == 1
    assert metrics["win_rate"] == pytest.approx(0.5)
    lower, upper = metrics["win_rate_wilson95"]
    assert 0.0 <= lower < 0.5 < upper <= 1.0


def test_wilson_interval_handles_empty_input() -> None:
    assert wilson_interval(0, 0) == [0.0, 0.0]
