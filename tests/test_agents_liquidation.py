from agents.liquidation import liquidate_queues
from agents.scheduler import WorkerPlan


def observation(position, inventory, hour=12):
    return {'day': 29, 'hour': hour, 'step': 29 * 24 + hour, 'player': 0,
            'farms': [{'farmer': position, 'hands': []}],
            'private': {'inventories': [inventory]}}


def test_carried_goods_return_before_the_last_sell_decision():
    obs = observation((0, 0), {'CARROT': 12})
    plans = [WorkerPlan((0, 0), [['WATER']] * 18)]
    liquidate_queues(obs, plans, 718, ((4, 4),))
    assert plans[0].queue == [['WATER']] + [['EAST']] * 4 + [['SOUTH']] * 4 + [['DROP']]
    assert obs['step'] + len(plans[0].queue) == 718  # next decision can SELL


def test_future_harvest_also_reserves_a_return_even_with_empty_inventory():
    obs = observation((4, 4), {})
    plans = [WorkerPlan((4, 4), [['WEST']] * 4 + [['HARVEST']] + [['WATER']] * 10)]
    liquidate_queues(obs, plans, 718, ((4, 4),))
    assert plans[0].queue == [['WEST']] * 4 + [['HARVEST']] + [['EAST']] * 4 + [['DROP']]


def test_existing_drop_is_retained_without_a_second_return():
    obs = observation((4, 4), {'WOOL': 5})
    plans = [WorkerPlan((4, 4), [['DROP'], ['PASS']])]
    liquidate_queues(obs, plans, 718, ((4, 4),))
    assert plans[0].queue == [['DROP'], ['PASS']]
