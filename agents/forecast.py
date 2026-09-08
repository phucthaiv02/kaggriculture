"""Daily production and marginal market cash flows for target selection.

Production uses the installed game's tile actions and daily refresh functions.
Forecasts assume scheduled care succeeds and harvested goods sell that day;
travel delays, future opponents' decisions and unknown shops are not predicted.

The exact queue simulator is retained for engine-validation tests. Planner
marginals use a robust rival projection instead: future rival *sales* affect
only the same product, while future rival inputs are not treated as guaranteed
market buys. We can observe a rival producer, but not whether its future feed
or fertilizer comes from the market, its own shed, or its own production; and
we do not know the rival's future cross-product market-order queue positions.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import zip_longest

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
        if tile is None and when == day:
            if animal:
                pickups.add(name)
            else:
                # Seeds are not market products, so they must not appear in
                # Production.inputs. They *are* a real shed pickup for route
                # capacity, though. A synthetic token lets LaborForecast count
                # one batched pickup per crop type without changing economics.
                pickups.add(f'SEED:{name}')
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

    def _value(self, flows, robust_external=False):
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
            if robust_external:
                cash += self._settle_day_robust(stocks, flows, when)
            else:
                cash += self._settle_day(stocks, flows, when)
        return cash

    def value(self, flows):
        """Exact value under the forecast's explicit future queue convention."""
        return self._value(flows, robust_external=False)

    def robust_value(self, flows):
        """Value with only observable/product-local rival effects.

        Rival future sales are projected because visible producers make those
        plausible. Rival future inputs are deliberately ignored: a required
        WHEAT/FERTILIZER unit does not imply a market BUY because the rival may
        already own it or produce it internally. Cross-product queue alignment
        is also ignored because future market-order positions are unobserved.
        """
        return self._value(flows, robust_external=True)

    def _settle_day(self, stocks, flows, when):
        # Exact daily-queue convention retained for engine-validation tests.
        # Daily flows have no observed order queue. Use PRODUCTS order for
        # both players, netting harvested feed/fertilizer before trading.
        queues = []
        for production_flow in (flows, self.external):
            sales = production_flow.sales.get(when, {})
            inputs = production_flow.inputs.get(when, {})
            queues.append([(p, sales.get(p, 0) - inputs.get(p, 0))
                           for p in game.PRODUCTS
                           if sales.get(p, 0) != inputs.get(p, 0)])
        cash = 0
        for orders in zip_longest(*queues):
            if None in orders:
                player = 0 if orders[0] is not None else 1
                product, amount = orders[player]
                delta, stocks[product] = self.trade(product, stocks.get(product, 0), amount)
                if player == 0:
                    cash += delta
                continue
            for unit in range(max(abs(order[1]) for order in orders if order)):
                quoted = []
                for player, order in enumerate(orders):
                    if order is None or unit >= abs(order[1]):
                        continue
                    product, amount = order
                    stock = stocks.get(product, 0)
                    price = self.price(product, stock if amount > 0 else stock - 1)
                    quoted.append((player, product, amount, price))
                # Match the engine: quote both units before committing either.
                for player, product, amount, price in quoted:
                    stocks[product] = stocks.get(product, 0) + (int(price > 1) if amount > 0 else -1)
                    if player == 0:
                        cash += price if amount > 0 else -price
        return cash

    def _settle_day_robust(self, stocks, flows, when):
        """Settle rival pressure per product, without invented rival buys.

        The market has independent stock per product. Pairing a newly inserted
        CARROT order with a rival FERTILIZER order can shift later queue indices
        in the exact convention and accidentally change MILK/WHEAT pricing,
        even though the candidate has no relationship to those products. That
        alignment is unknowable for future turns, so robust planning makes
        rival interaction local to each product instead.
        """
        own_sales = flows.sales.get(when, {})
        own_inputs = flows.inputs.get(when, {})
        rival_sales = self.external.sales.get(when, {})
        cash = 0

        for product in game.PRODUCTS:
            own_amount = own_sales.get(product, 0) - own_inputs.get(product, 0)
            # Do not subtract external.inputs here. A future rival requirement
            # is not evidence of a future market BUY.
            rival_amount = rival_sales.get(product, 0)
            if not own_amount and not rival_amount:
                continue

            for unit in range(max(abs(own_amount), abs(rival_amount))):
                stock = stocks.get(product, 0)
                quoted = []
                if unit < abs(own_amount):
                    price = self.price(product, stock if own_amount > 0 else stock - 1)
                    quoted.append((0, own_amount, price))
                if unit < abs(rival_amount):
                    price = self.price(product, stock if rival_amount > 0 else stock - 1)
                    quoted.append((1, rival_amount, price))

                # Same-product simultaneous units are still quoted from the
                # same pre-commit stock, matching engine simultaneous quoting.
                for player, amount, price in quoted:
                    stocks[product] = stocks.get(product, 0) + (
                        int(price > 1) if amount > 0 else -1
                    )
                    if player == 0:
                        cash += price if amount > 0 else -price
        return cash

    def _has_external_projection(self):
        return any(self.external.sales.values()) or any(self.external.inputs.values())

    def marginal_profit(self, baseline, candidate, fixed_cost, baseline_value=None):
        combined = Production()
        combined.add(baseline)
        combined.add(candidate)

        # Planner calls this with visible rival production. Use the robust
        # projection there, but keep exact engine-equivalent behavior when no
        # rival projection exists (including existing unit tests and helpers).
        if self._has_external_projection():
            return (
                self.robust_value(combined)
                - self.robust_value(baseline)
                - fixed_cost
            )

        if baseline_value is None:
            baseline_value = self.value(baseline)
        return self.value(combined) - baseline_value - fixed_cost
