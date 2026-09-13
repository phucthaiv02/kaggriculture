from collections import Counter

from kaggle_environments.envs.kaggriculture.kaggriculture import MARKET_I0, PRODUCTS, market_price

from agents import planner
from agents.forecast import Production


def make_obs(money):
    board = [[None] * 10 for _ in range(10)]
    inventory = {product: MARKET_I0 for product in PRODUCTS}
    farm = {
        "tiles": board,
        "unlocked_quadrants": ["NW"],
        "farmer": [4, 4],
        "hands": [],
        "money": money,
    }
    return {
        "day": 0,
        "hour": 0,
        "player": 0,
        "farms": [farm, farm],
        "market": {
            "inventory": inventory,
            "prices": {p: market_price(p, n) for p, n in inventory.items()},
        },
        "town": {"unlocked_shops": []},
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
    }


class FakeLabor:
    def __init__(self, *_args, **_kwargs):
        pass

    def cost(self, visits):
        entries = list(visits.get(0, ()))
        return 4 + 3 * bool(entries)

    def marginal_cost(self, *_args, **_kwargs):
        return 3


def candidate():
    output = Production()
    output.sales[4]["WHEAT"] = 4
    output.visits[0].append((None, 1, Counter(), Counter()))
    return (("WHEAT", False), output, planner.SEED_COST["WHEAT"])


def install_isolated_candidate(monkeypatch):
    item = candidate()
    monkeypatch.setattr(planner, "LaborForecast", FakeLabor)
    monkeypatch.setattr(planner, "_daily_candidates", lambda *_args: [item])

    def choose(_market, _baseline, candidates, *_args):
        if not candidates:
            return None, None
        return candidates[0][0], candidates[0][1]

    monkeypatch.setattr(planner, "_choose_daily", choose)


def test_planner_rejects_start_that_cannot_fund_today_labor(monkeypatch):
    install_isolated_candidate(monkeypatch)
    seed = planner.SEED_COST["WHEAT"]
    targets = {}
    planner.plan_targets(make_obs(seed + 4 + 2), targets, [(0, 0)], end_day=29)
    assert targets[(0, 0)] is None


def test_planner_accepts_start_once_today_labor_is_funded(monkeypatch):
    install_isolated_candidate(monkeypatch)
    seed = planner.SEED_COST["WHEAT"]
    targets = {}
    planner.plan_targets(make_obs(seed + 4 + 3), targets, [(0, 0)], end_day=29)
    assert targets[(0, 0)] == ("WHEAT", False)
