"""Unit tests for the daily aggregate evaluator."""

from experiments.day16_allocation import Policy
from experiments.fast_evaluator import STARTING_MONEY, evaluate, estimate


def test_single_wheat_policy_is_profitable_without_an_opponent():
    result = evaluate(Policy.from_dict({"WHEAT": 1}, hands=0, rotation="WHEAT"))
    assert result.final_cash > STARTING_MONEY
    assert result.net_profit == result.final_cash - STARTING_MONEY


def test_aggregate_estimate_is_explicitly_separate():
    result = estimate(Policy.from_dict({"WHEAT": 1}, hands=0, rotation="WHEAT"))
    assert result.sales["WHEAT"] > 0
