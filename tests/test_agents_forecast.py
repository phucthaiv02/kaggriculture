from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
import pytest
from kaggle_environments.envs.kaggriculture import kaggriculture as game
from agents.forecast import MarketForecast, Production, production


@pytest.mark.parametrize('crop,fertilize', [(c, f) for c in game.CROPS for f in (False, True)])
def test_crop_forecast_matches_real_interpreter_schedule(crop, fertilize):
    from experiments.crop_schedules import run_case
    trace, _ = run_case(crop, fertilize, seed=1)
    predicted = production(crop, fertilize, 0, 29)
    assert sum(s[crop] for s in predicted.sales.values()) == sum(n for _, _, n in trace)


def test_existing_crop_uses_held_yield_and_does_not_predict_another_cycle():
    tile = game._new_plant('WHEAT', 0, 24)
    tile.update(yield_units=5, watered_today=True)
    snapshot = deepcopy(tile)
    predicted = production('WHEAT', True, 4, 29, tile)
    assert sum(predicted.sales.values(), Counter()) == {'WHEAT': 5}
    assert tile == snapshot


def test_fertilized_melon_does_not_invent_yield_above_cap():
    unfertilized = production('MELON', False, 0, 20)
    fertilized = production('MELON', True, 0, 20)
    assert sum(unfertilized.sales.values(), Counter()) == sum(fertilized.sales.values(), Counter()) == {'MELON': 6}


@pytest.mark.parametrize('animal,expected', [('GOOSE',54), ('COW',36), ('SHEEP',34)])
def test_animal_forecast_matches_verified_season_yields(animal, expected):
    predicted = production(animal, False, 0, 29)
    assert sum(predicted.sales.values(), Counter())[game.ANIMALS[animal]['product']] == expected


def engine_market(inventory, params=None, shops=()):
    """Use actual order quoting/settlement; ample cash/storage isolate pricing."""
    market = {'inventory': {p: inventory.get(p, game.MARKET_I0) for p in game.PRODUCTS},
              'prices': {}, 'params': params}
    farms = [{'money': 10**12}, {'money': 10**12}]
    state = [SimpleNamespace(
        observation=SimpleNamespace(market=market, farms=farms,
                                    private={'shed': {}}, town={'unlocked_shops': list(shops)}),
        action={},
    ) for _ in range(2)]
    env = SimpleNamespace(configuration={'shedCapacity': 10**9})
    return state, env


def settle(state, env, orders, rival_orders=()):
    for player, queue in enumerate((orders, rival_orders)):
        state[player].action = {'market': list(queue)}
        shed = state[player].observation.private['shed']
        for op, product, quantity in queue:
            if op == 'SELL':
                shed[product] = shed.get(product, 0) + quantity
    before = state[0].observation.farms[0]['money']
    game._process_market(state, env)
    return state[0].observation.farms[0]['money'] - before


@pytest.mark.parametrize('product,amount',
                         [(p, 200) for p in game.PRODUCTS] +
                         [(p, -200) for p in ('WHEAT', 'FERTILIZER')])
@pytest.mark.parametrize('stock', [-100, 0, game.MARKET_I0, 100000])
@pytest.mark.parametrize('custom_params', [False, True])
def test_trade_matches_engine_orders(product, amount, stock, custom_params):
    params = deepcopy(game.MARKET_PARAMS) if custom_params else None
    if params:
        for values in params.values():
            values['base'] *= 2
    state, env = engine_market({product: stock}, params)
    forecast = MarketForecast({}, (), 0, 0, params=params)
    cash = settle(state, env, [['SELL' if amount > 0 else 'BUY_PRODUCT', product, abs(amount)]])
    assert forecast.trade(product, stock, amount) == (
        cash, state[0].observation.market['inventory'][product])


def test_buy_then_sell_round_trip_matches_engine():
    state, env = engine_market({'WHEAT': -10})
    forecast = MarketForecast({}, (), 0, 0)
    spent, stock = forecast.trade('WHEAT', -10, -40)
    earned, stock = forecast.trade('WHEAT', stock, 40)
    assert settle(state, env, [['BUY_PRODUCT', 'WHEAT', 40], ['SELL', 'WHEAT', 40]]) == spent + earned == 0
    assert stock == state[0].observation.market['inventory']['WHEAT'] == -10


@pytest.mark.parametrize('own,rival', [
    ({'MELON': 80}, {'MELON': 80}),
    ({'MELON': 80}, {'MELON': 15}),
    ({'MELON': 15}, {'MELON': 80}),
    ({'WHEAT': -20}, {'WHEAT': 35}),
    ({'WHEAT': 35}, {'WHEAT': -20}),
    ({'WHEAT': -20}, {'WHEAT': -35}),
    ({'WHEAT': 5, 'MELON': 80}, {'MELON': 15}),
    ({'MELON': 80}, {}),
    ({}, {'MELON': 80}),
])
@pytest.mark.parametrize('stock', [-100, game.MARKET_I0, 100000])
@pytest.mark.parametrize('custom_params', [False, True])
def test_simultaneous_forecast_matches_engine(own, rival, stock, custom_params):
    params = deepcopy(game.MARKET_PARAMS) if custom_params else None
    if params:
        for values in params.values():
            values['base'] *= 2
    inventory = {p: stock for p in game.PRODUCTS}
    state, env = engine_market(inventory, params)
    flows = [Production(), Production()]
    queues = []
    for flow, amounts in zip(flows, (own, rival)):
        orders = []
        for product in game.PRODUCTS:
            amount = amounts.get(product, 0)
            if amount:
                (flow.sales if amount > 0 else flow.inputs)[0][product] = abs(amount)
                orders.append(['SELL' if amount > 0 else 'BUY_PRODUCT', product, abs(amount)])
        queues.append(orders)
    expected = settle(state, env, *queues)
    forecast = MarketForecast(inventory, (), 0, 0, params=params, external=flows[1])
    stocks = dict(inventory)
    assert forecast._settle_day(stocks, flows[0], 0) == expected
    assert stocks == state[0].observation.market['inventory']


@pytest.mark.parametrize('hour', [0, 3, 4, 23])
def test_daily_cash_matches_engine_demand_and_settlement(hour):
    # Replay the forecast's explicit convention: net inputs with own output,
    # settle both players' order queues together at hour 23.
    shop = next(iter(game.SHOPS))
    shops = [shop, shop]  # Duplicate shops consume independently.
    inventory = {p: -5 for p in game.PRODUCTS}
    flow, external = Production(), Production()
    flow.sales[2].update(WHEAT=5, MELON=80)
    flow.inputs[2].update(WHEAT=12, FERTILIZER=3)
    flow.sales[3].update(WHEAT=20, MELON=10)
    external.sales[2]['MELON'] = 15
    state, env = engine_market(inventory, shops=shops)
    expected = 0
    for step in range(2 * 24 + hour, 4 * 24):
        day = step // 24
        if step % 24 == 23:
            queues = []
            for flows in (flow, external):
                orders = []
                for product in game.PRODUCTS:
                    amount = flows.sales[day][product] - flows.inputs[day][product]
                    if amount:
                        orders.append(['SELL' if amount > 0 else 'BUY_PRODUCT', product, abs(amount)])
                queues.append(orders)
            expected += settle(state, env, *queues)
        game._town_consume(env, state, step)
    assert MarketForecast(inventory, shops, 2, 3, hour, external=external).value(flow) == expected


def test_marginal_profit_matches_two_engine_portfolios():
    inventory = {p: game.MARKET_I0 for p in game.PRODUCTS}
    baseline, candidate = Production(), Production()
    baseline.sales[0]['MELON'] = 100
    candidate.sales[0]['MELON'] = 60
    revenues = []
    for quantity in (100, 160):
        state, env = engine_market(inventory)
        for step in range(23):
            game._town_consume(env, state, step)
        revenues.append(settle(state, env, [['SELL', 'MELON', quantity]]))
    market = MarketForecast(inventory, (), 0, 0)
    assert market.marginal_profit(baseline, candidate, 80) == revenues[1] - revenues[0] - 80


def test_demand_reduces_existing_market_stock_without_new_supply():
    inventory = {p: game.MARKET_I0 for p in game.PRODUCTS}
    flow = Production()
    flow.sales[4]['CARROT'] = 6
    shops = [name for name, products in game.SHOPS.items() if 'CARROT' in products]
    assert MarketForecast(inventory, shops, 0, 4).value(flow) > MarketForecast(inventory, (), 0, 4).value(flow)


def test_late_supply_does_not_depress_an_earlier_harvest():
    inventory = {p: game.MARKET_I0 for p in game.PRODUCTS}
    candidate = Production()
    candidate.sales[4]['WHEAT'] = 6
    late = Production()
    late.sales[15]['WHEAT'] = 100
    with_late_rival = MarketForecast(inventory, (), 0, 20, external=late)
    without_rival = MarketForecast(inventory, (), 0, 20)
    assert with_late_rival.value(candidate) == without_rival.value(candidate)


def test_extra_supply_accounts_for_price_impact_on_existing_crop():
    inventory = {p: game.MARKET_I0 for p in game.PRODUCTS}
    market = MarketForecast(inventory, (), 0, 10)
    baseline = Production()
    baseline.sales[10]['MELON'] = 100
    candidate = Production()
    candidate.sales[10]['MELON'] = 6
    assert market.marginal_profit(baseline, candidate, 80) < market.marginal_profit(Production(), candidate, 80)

@pytest.mark.parametrize('custom_params', [False, True])
def test_sale_budget_matches_engine_for_repeated_orders(custom_params):
    from agents.expansion_agent import _sale_revenue
    params = deepcopy(game.MARKET_PARAMS) if custom_params else None
    if params:
        params['MELON']['base'] *= 3
    inventory = {p: game.MARKET_I0 for p in game.PRODUCTS}
    state, env = engine_market(inventory, params)
    orders = [['SELL', 'MELON', 60], ['SELL', 'MELON', 60]]
    obs = {'market': {'inventory': inventory, 'params': params}}
    snapshot = deepcopy(obs)
    assert _sale_revenue(obs, orders) == settle(state, env, orders)
    assert obs == snapshot


@pytest.mark.parametrize('stock', [-100, 0, game.MARKET_I0])
@pytest.mark.parametrize('shortfall', [0, 1])
@pytest.mark.parametrize('custom_params', [False, True])
def test_animal_budget_reserves_engine_feed_purchase_cost(stock, shortfall, custom_params):
    from agents.farm_tasks import purchase_orders
    params = deepcopy(game.MARKET_PARAMS) if custom_params else None
    if params:
        params['WHEAT']['base'] *= 3
    inventory = {p: game.MARKET_I0 for p in game.PRODUCTS}
    inventory['WHEAT'] = stock
    state, env = engine_market(inventory, params)
    cost = -settle(state, env, [['BUY_ANIMAL', 'GOOSE', 1], ['BUY_PRODUCT', 'WHEAT', 1]])
    farm = {'tiles': [[None] * 10 for _ in range(10)], 'money': cost - shortfall}
    obs = {'day': 0, 'player': 0, 'farms': [farm],
           'private': {'shed': {}, 'seeds': {}, 'inventories': [{}]},
           'market': {'inventory': inventory, 'params': params}}
    orders = purchase_orders(obs, {(0, 0): ('GOOSE', False)}, [(0, 0)])
    assert (['BUY_ANIMAL', 'GOOSE', 1] in orders) == (shortfall == 0)
