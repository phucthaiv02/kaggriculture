"""Daily production and marginal market cash flows for target selection.

Production uses the installed game's tile actions and daily refresh functions.
Forecasts assume scheduled care succeeds and harvested goods sell that day;
travel delays and future opponents' decisions are not predicted.

The exact queue simulator is retained for engine-validation tests with the
currently observed town. Planner marginals use a robust rival projection and
integrate unknown future shop unlocks as an exact per-product expectation:
future rival *sales* affect only the same product, future rival inputs are not
treated as guaranteed market buys, and random shop draws are not collapsed
into an average inventory path (important for nonlinear/hinge price curves).
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


# Current engine defaults. The observation exposes already-unlocked shops but
# not configuration/seed, so future shop identity is unknowable to the agent.
SHOP_UNLOCK_INTERVAL = 3
SHOP_SELL_INTERVAL = 4
CENTER_SELL_INTERVAL = 24
MAX_SHOP_INSTANCES = getattr(game, "MAX_SHOP_INSTANCES", 8)


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
        self.shop_count = len(shops)
        for shop in shops:
            products = game.SHOPS[shop]
            for product in products:
                self.shop_demand[product] += 2 if len(products) == 1 else 1

        remaining_unlocks = max(0, MAX_SHOP_INSTANCES - self.shop_count)
        self.future_shop_days = tuple(
            when
            for when in range(day + 1, end_day + 1)
            if when % SHOP_UNLOCK_INTERVAL == 0
        )[:remaining_unlocks]

        self.shop_draw_distribution = {}
        total_shops = len(game.SHOPS)
        for product in game.PRODUCTS:
            increments = Counter()
            for products in game.SHOPS.values():
                increment = 0
                if product in products:
                    increment = 2 if len(products) == 1 else 1
                increments[increment] += 1
            self.shop_draw_distribution[product] = tuple(
                (increment, count / total_shops)
                for increment, count in sorted(increments.items())
            )

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
        """Static-town valuation used by exact engine-equivalence tests."""
        stocks = dict(self.inventory)
        cash = 0
        previous = self.day * 24 + self.hour - 1
        for when in range(self.day, self.end_day + 1):
            # Forecast delivery by the end of each harvest day. Today's held
            # shed goods are also priced here, consistently in both scenarios.
            step = when * 24 + 23
            shop_ticks = step // SHOP_SELL_INTERVAL - previous // SHOP_SELL_INTERVAL
            center_ticks = step // CENTER_SELL_INTERVAL - previous // CENTER_SELL_INTERVAL
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
        """Exact value under the forecast's explicit current-town convention."""
        return self._value(flows, robust_external=False)

    def robust_value(self, flows):
        """Planner value under observable rival pressure and unknown shops.

        Future shop identities are random and hidden. We integrate them exactly
        per product rather than forecasting one shop or applying expected
        demand directly to stock. Product inventories/prices are independent,
        so E[sum(product cash)] == sum(E[product cash]); correlations between
        products induced by a common shop draw therefore do not require a
        joint 8^N enumeration.
        """
        if self.future_shop_days:
            return sum(
                self._expected_product_value(flows, product)
                for product in game.PRODUCTS
            )
        return self._value(flows, robust_external=True)

    def _expected_product_value(self, flows, product):
        """Exact DP over future shop-demand rate for one product.

        State = (market stock, per-shop-tick demand). Each unlock branches only
        over the demand increment that a uniformly drawn shop contributes to
        this product: usually {0,1}, or {0,2} for single-product shops. Paths
        that reach the same state are merged with their probability-weighted
        accumulated cash.
        """
        # state -> [probability, probability-weighted accumulated own cash]
        states = {
            (self.inventory.get(product, 0), self.shop_demand[product]): [1.0, 0.0]
        }
        future_shop_days = set(self.future_shop_days)
        previous = self.day * 24 + self.hour - 1

        for when in range(self.day, self.end_day + 1):
            if when in future_shop_days:
                branched = defaultdict(lambda: [0.0, 0.0])
                for (stock, demand_rate), (probability, weighted_cash) in states.items():
                    for increment, draw_probability in self.shop_draw_distribution[product]:
                        key = (stock, demand_rate + increment)
                        branched[key][0] += probability * draw_probability
                        branched[key][1] += weighted_cash * draw_probability
                states = dict(branched)

            step = when * 24 + 23
            shop_ticks = step // SHOP_SELL_INTERVAL - previous // SHOP_SELL_INTERVAL
            center_ticks = step // CENTER_SELL_INTERVAL - previous // CENTER_SELL_INTERVAL
            own_sales = flows.sales.get(when, {})
            own_inputs = flows.inputs.get(when, {})
            rival_sales = self.external.sales.get(when, {})
            own_amount = own_sales.get(product, 0) - own_inputs.get(product, 0)
            rival_amount = rival_sales.get(product, 0)

            settled = defaultdict(lambda: [0.0, 0.0])
            for (stock, demand_rate), (probability, weighted_cash) in states.items():
                stock -= demand_rate * shop_ticks
                if product in self.center_products:
                    stock -= center_ticks
                delta, stock = self._settle_product_robust(
                    stock, product, own_amount, rival_amount
                )
                key = (stock, demand_rate)
                settled[key][0] += probability
                settled[key][1] += weighted_cash + probability * delta
            states = dict(settled)
            previous = step

        return sum(weighted_cash for _, weighted_cash in states.values())

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

    def _settle_product_robust(self, stock, product, own_amount, rival_amount):
        """Settle one product from one pre-trade stock under robust semantics."""
        cash = 0
        if not own_amount and not rival_amount:
            return cash, stock

        for unit in range(max(abs(own_amount), abs(rival_amount))):
            quoted = []
            if unit < abs(own_amount):
                price = self.price(product, stock if own_amount > 0 else stock - 1)
                quoted.append((0, own_amount, price))
            if unit < abs(rival_amount):
                price = self.price(product, stock if rival_amount > 0 else stock - 1)
                quoted.append((1, rival_amount, price))

            # Same-product simultaneous units are quoted from the same
            # pre-commit stock, matching engine simultaneous quoting.
            for player, amount, price in quoted:
                stock += int(price > 1) if amount > 0 else -1
                if player == 0:
                    cash += price if amount > 0 else -price
        return cash, stock

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
            delta, stocks[product] = self._settle_product_robust(
                stocks.get(product, 0), product, own_amount, rival_amount
            )
            cash += delta
        return cash

    def _has_external_projection(self):
        return any(self.external.sales.values()) or any(self.external.inputs.values())

    def marginal_profit(self, baseline, candidate, fixed_cost, baseline_value=None):
        combined = Production()
        combined.add(baseline)
        combined.add(candidate)

        # Planner calls this with visible rival production. Use the robust
        # projection there, including stochastic future-shop expectation, but
        # keep exact engine-equivalent behavior when no rival projection exists
        # (including legacy helpers and exact unit tests).
        if self._has_external_projection():
            return (
                self.robust_value(combined)
                - self.robust_value(baseline)
                - fixed_cost
            )

        if baseline_value is None:
            baseline_value = self.value(baseline)
        return self.value(combined) - baseline_value - fixed_cost
