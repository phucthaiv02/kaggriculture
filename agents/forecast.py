"""Daily production and marginal market cash flows for target selection.

Production uses the installed game's tile actions and daily refresh functions.
Forecasts assume scheduled care succeeds and harvested goods sell that day;
travel delays, future opponents' decisions and unknown shops are not predicted.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache

from kaggle_environments.envs.kaggriculture import kaggriculture as game
from agents.schedules import (
    CROP_LAST_AGE, ONGOING_CROPS, is_maintenance_day,
    should_fertilize_today, should_feed_animal, should_care_animal,
)


@dataclass
class Production:
    sales: dict = field(default_factory=lambda: defaultdict(Counter))
    inputs: dict = field(default_factory=lambda: defaultdict(Counter))
    visits: dict = field(default_factory=lambda: defaultdict(list))

    def add(self, other, position=None):
        for day, units in other.sales.items():
            self.sales[day].update(units)
        for day, units in other.inputs.items():
            self.inputs[day].update(units)
        for day, visits in other.visits.items():
            self.visits[day].extend((position if p is None else p, n, inputs, goods)
                                   for p, n, inputs, goods in visits)


def production(name, fertilize, day, end_day, tile=None):
    """Remaining output, including held yield, without inventing replants."""
    animal = name in game.ANIMALS
    initial = (game._new_animal(name, day) if animal else game._new_plant(name, day, 24))
    if tile is not None:
        initial.update(deepcopy(tile))
    farm = {'tiles': [[initial]], 'farmer': [0, 0], 'hands': []}
    private = {'inventories': [{}], 'shed': {}, 'seeds': {}}
    result = Production()

    actions = 0
    def act(op, when):
        nonlocal actions
        if op == 'WATER' and farm['tiles'][0][0].get('watered_today'):
            return
        actions += 1
        game._apply_unit_action(farm, private, 0, [op], 1, when, 24)

    for when in range(day, end_day + 1):
        live = farm['tiles'][0][0]
        if not isinstance(live, dict) or live.get('kind') == 'WEED':
            break
        actions = (2 if animal else 1) if tile is None and when == day else 0
        inventory = private['inventories'][0]
        inventory.clear()
        if animal:
            if not live.get('animal'):
                break
            age = when - live['placed_day']
            if live.get('yield_units', 0):
                act('HARVEST', when)
            if live.get('fertilizer_available'):
                act('COLLECT_FERTILIZER', when)
            if should_feed_animal(name, age) and not live.get('fed_today'):
                result.inputs[when]['WHEAT'] += 1
                inventory['WHEAT'] = inventory.get('WHEAT', 0) + 1
                act('FEED', when)
            if should_care_animal(name, age) and not live.get('cared_today'):
                act('CARE', when)
        else:
            age = when - live['planted_day']
            if should_fertilize_today(name, age, fertilize) and live.get('fertilized_until_day', -1) < when + 2:
                result.inputs[when]['FERTILIZER'] += 1
                inventory['FERTILIZER'] = 1
                act('FERTILIZE', when)
            # Match build_tasks: ongoing ready output gets WATER then HARVEST.
            ready = name in ONGOING_CROPS and live.get('yield_units', 0) > 0
            finished = age >= CROP_LAST_AGE[name]
            if is_maintenance_day(name, age, fertilize) or ready or (finished and name not in ONGOING_CROPS):
                act('WATER', when)
            if ready or (finished and name not in ONGOING_CROPS):
                act('HARVEST', when)
        result.sales[when].update({p: n for p, n in inventory.items() if n > 0})
        pickups = set(result.inputs[when])
        if tile is None and when == day and animal:
            pickups.add(name)
        if not animal and age >= CROP_LAST_AGE[name] and name in ONGOING_CROPS:
            actions += 1  # DIG the exhausted plant before a subsequent sowing.
        if actions:
            result.visits[when].append((None, actions, tuple(sorted(pickups)), bool(result.sales[when])))
        if not animal and age >= CROP_LAST_AGE[name]:
            break
        game._daily_refresh_plants(farm, when, 24)
        game._daily_refresh_animals(farm, when)
    return result


class MarketForecast:
    """Requote every unit; price new output and its impact on existing output."""
    def __init__(self, inventory, shops, day, end_day, hour=0, params=None, external=None):
        self.external = external or Production()
        self.inventory = inventory
        self.day, self.end_day, self.hour = day, end_day, hour
        self.shop_demand = Counter()
        for shop in shops:
            products = game.SHOPS[shop]
            for product in products:
                self.shop_demand[product] += 2 if len(products) == 1 else 1
        self.center_products = set(game.TOWN_CENTER_PRODUCTS)
        self.price = lru_cache(maxsize=None)(lambda product, stock: game.market_price(product, stock, params))
        self.trade = lru_cache(maxsize=None)(self._trade)

    def _trade(self, product, stock, amount):
        cash = 0
        for _ in range(abs(amount)):
            price = self.price(product, stock if amount > 0 else stock - 1)
            if amount > 0:
                cash += price
                stock += int(price > 1)  # The engine does not add $1 sales to stock.
            else:
                cash -= price
                stock -= 1
        return cash, stock

    def value(self, flows):
        stocks = dict(self.inventory)
        cash = 0
        previous = self.day * 24 + self.hour - 1
        for when in range(self.day, self.end_day + 1):
            # Forecast delivery by the end of each harvest day. Today's held
            # shed goods are also priced here, consistently in both scenarios.
            step = when * 24 + 23
            shop_ticks = step // 4 - previous // 4
            center_ticks = step // 24 - previous // 24
            for product in stocks:
                stocks[product] -= self.shop_demand[product] * shop_ticks
                if product in self.center_products:
                    stocks[product] -= center_ticks
            previous = step
            rival_sales = self.external.sales.get(when, {})
            rival_inputs = self.external.inputs.get(when, {})
            for product in rival_sales.keys() | rival_inputs.keys():
                _, stocks[product] = self.trade(
                    product, stocks.get(product, 0),
                    rival_sales.get(product, 0) - rival_inputs.get(product, 0),
                )
            sales, inputs = flows.sales.get(when, {}), flows.inputs.get(when, {})
            for product in sales.keys() | inputs.keys():
                # Own harvest used as feed/fertilizer need not be sold then bought.
                amount = sales.get(product, 0) - inputs.get(product, 0)
                delta, stocks[product] = self.trade(product, stocks.get(product, 0), amount)
                cash += delta
        return cash

    def marginal_profit(self, baseline, candidate, fixed_cost, baseline_value=None):
        combined = Production()
        combined.add(baseline)
        combined.add(candidate)
        if baseline_value is None:
            baseline_value = self.value(baseline)
        return self.value(combined) - baseline_value - fixed_cost
