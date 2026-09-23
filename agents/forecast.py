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
    should_fertilize_today, should_feed_animal, should_care_animal, should_harvest_animal,
)


@dataclass
class Production:
    sales: dict = field(default_factory=lambda: defaultdict(Counter))
    inputs: dict = field(default_factory=lambda: defaultdict(Counter))
    visits: dict = field(default_factory=lambda: defaultdict(list))

    def add(self, other, position=None, include_visits=True):
        for day, units in other.sales.items():
            self.sales[day].update(units)
        for day, units in other.inputs.items():
            self.inputs[day].update(units)
        if include_visits:
            for day, visits in other.visits.items():
                self.visits[day].extend((position if p is None else p, n, inputs, goods)
                                       for p, n, inputs, goods in visits)


def _simulate_production(name, fertilize, day, end_day, tile=None):
    """Uncached interpreter-backed production simulation."""
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
            last_age = end_day - live['placed_day']
            if should_harvest_animal(name, age, live.get('yield_units', 0), force=when == end_day):
                act('HARVEST', when)
            if live.get('fertilizer_available'):
                act('COLLECT_FERTILIZER', when)
            if should_feed_animal(name, age, last_age) and not live.get('fed_today'):
                result.inputs[when]['WHEAT'] += 1
                inventory['WHEAT'] = inventory.get('WHEAT', 0) + 1
                act('FEED', when)
            if should_care_animal(name, age, last_age) and not live.get('cared_today'):
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
            if live.get('consecutive_unwatered', 0) >= 1 or is_maintenance_day(name, age, fertilize) or ready or (finished and name not in ONGOING_CROPS):
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


def _production_snapshot(result):
    """Compact immutable representation safe to retain in an LRU cache."""
    return (
        tuple((day, tuple(sorted(units.items()))) for day, units in sorted(result.sales.items())),
        tuple((day, tuple(sorted(units.items()))) for day, units in sorted(result.inputs.items())),
        tuple((day, tuple(visits)) for day, visits in sorted(result.visits.items())),
    )


def _restore_production(snapshot):
    result = Production()
    sales, inputs, visits = snapshot
    for day, units in sales:
        result.sales[day].update(dict(units))
    for day, units in inputs:
        result.inputs[day].update(dict(units))
    for day, entries in visits:
        result.visits[day].extend(entries)
    return result


def _fertilize_cache_key(fertilize):
    # Dynamic plans are frozen dataclasses; legacy callers may still pass lists
    # or sets. Normalize only mutable containers, preserving plan semantics.
    if isinstance(fertilize, list):
        return tuple(_fertilize_cache_key(value) for value in fertilize)
    if isinstance(fertilize, set):
        return tuple(sorted(_fertilize_cache_key(value) for value in fertilize))
    if isinstance(fertilize, tuple):
        return tuple(_fertilize_cache_key(value) for value in fertilize)
    return fertilize


def _tile_cache_key(tile):
    if tile is None:
        return None
    # Kaggriculture producer tiles are flat dictionaries of primitive values.
    # Keeping every field in the key is intentionally conservative: a new
    # engine field automatically invalidates sharing instead of risking a stale
    # forecast based on an incomplete hand-picked state signature.
    return tuple(sorted(tile.items()))


@lru_cache(maxsize=4096)
def _cached_production_snapshot(name, fertilize, day, end_day, tile_key):
    tile = None if tile_key is None else dict(tile_key)
    return _production_snapshot(
        _simulate_production(name, fertilize, day, end_day, tile)
    )


def production(name, fertilize, day, end_day, tile=None):
    """Remaining output with exact state-keyed interpreter forecast caching."""
    snapshot = _cached_production_snapshot(
        name, _fertilize_cache_key(fertilize), int(day), int(end_day),
        _tile_cache_key(tile),
    )
    # Return a fresh object to preserve the public function's old ownership
    # semantics even though the expensive simulation result is shared.
    return _restore_production(snapshot)


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
        self.settle_product = lru_cache(maxsize=None)(self._settle_product)
        # Candidate Production objects are reused across every tile repriced in
        # one morning. Cache their touched markets once instead of rescanning
        # up to sixteen days of sales/inputs for every tile.
        self._candidate_products = {}

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

    def value(self, flows, discount=1.0):
        """Forecast total cash; optionally discount each future day's net cash."""
        stocks = dict(self.inventory)
        cash = 0.0
        previous = self.day * 24 + self.hour - 1
        for when in range(self.day, self.end_day + 1):
            # Forecast delivery by the end of each harvest day. Shops consume
            # every 4 turns, so a full default day contributes six separate
            # demand ticks before end-of-day settlement. Today's held shed
            # goods are also priced here, consistently in both scenarios.
            step = when * 24 + 23
            shop_ticks = step // 4 - previous // 4
            center_ticks = step // 24 - previous // 24
            for product in stocks:
                stocks[product] -= self.shop_demand[product] * shop_ticks
                if product in self.center_products:
                    stocks[product] -= center_ticks
            previous = step
            daily_cash = self._settle_day(stocks, flows, when)
            cash += (discount ** (when - self.day)) * daily_cash
        return cash

    def _value_products(self, flows, products, extra=None):
        """Value only the requested independent product markets.

        Product-local settlement means a candidate cannot change cash in a
        product it neither sells nor consumes.  Restricting valuation to the
        touched products is therefore exactly equivalent to subtracting two
        full-portfolio valuations, but much cheaper when many fertilizer event
        plans are compared against the same baseline.
        """
        products = tuple(products)
        if not products:
            return 0.0
        stocks = {product: self.inventory.get(product, 0) for product in products}
        cash = 0.0
        previous = self.day * 24 + self.hour - 1
        for when in range(self.day, self.end_day + 1):
            step = when * 24 + 23
            shop_ticks = step // 4 - previous // 4
            center_ticks = step // 24 - previous // 24
            for product in products:
                stocks[product] -= self.shop_demand[product] * shop_ticks
                if product in self.center_products:
                    stocks[product] -= center_ticks
            previous = step
            cash += self._settle_day(stocks, flows, when, products, extra)
        return cash

    def marginal_value(self, baseline, candidate, baseline_values=None):
        """Exact ``value(base + candidate) - value(base)`` on touched markets."""
        candidate_id = id(candidate)
        products = self._candidate_products.get(candidate_id)
        if products is None:
            touched = set()
            for field in (candidate.sales, candidate.inputs):
                for units in field.values():
                    touched.update(product for product, amount in units.items() if amount)
            products = tuple(product for product in game.PRODUCTS if product in touched)
            self._candidate_products[candidate_id] = products
        if not products:
            return 0.0

        key = products
        if baseline_values is not None and key in baseline_values:
            baseline_value = baseline_values[key]
        else:
            baseline_value = self._value_products(baseline, products)
            if baseline_values is not None:
                baseline_values[key] = baseline_value

        return self._value_products(baseline, products, candidate) - baseline_value

    def _settle_product(self, product, stock, own_amount, rival_amount):
        """Exact one-product settlement transition, cached across candidates."""
        cash = 0
        own_units = abs(own_amount)
        rival_units = abs(rival_amount)
        for unit in range(max(own_units, rival_units)):
            quoted = []
            if unit < own_units:
                price = self.price(product, stock if own_amount > 0 else stock - 1)
                quoted.append((0, own_amount, price))
            if unit < rival_units:
                price = self.price(product, stock if rival_amount > 0 else stock - 1)
                quoted.append((1, rival_amount, price))
            for player, amount, price in quoted:
                stock += int(price > 1) if amount > 0 else -1
                if player == 0:
                    cash += price if amount > 0 else -price
        return cash, stock

    def _settle_day(self, stocks, flows, when, products=None, extra=None):
        """Settle daily market pressure independently for each product.

        The planner knows forecast product flows by day, not either player's
        future market-order queue. Pairing different products by queue index
        makes unrelated targets change each other's value. Instead, preserve
        the stock path per product: same-product own/opponent units are quoted
        simultaneously for that day's pressure, while different products are
        completely independent.
        """
        own_sales = flows.sales.get(when, {})
        own_inputs = flows.inputs.get(when, {})
        extra_sales = extra.sales.get(when, {}) if extra is not None else {}
        extra_inputs = extra.inputs.get(when, {}) if extra is not None else {}
        rival_sales = self.external.sales.get(when, {})
        rival_inputs = self.external.inputs.get(when, {})
        cash = 0

        for product in (game.PRODUCTS if products is None else products):
            own_amount = (
                own_sales.get(product, 0) - own_inputs.get(product, 0)
                + extra_sales.get(product, 0) - extra_inputs.get(product, 0)
            )
            rival_amount = rival_sales.get(product, 0) - rival_inputs.get(product, 0)
            if not own_amount and not rival_amount:
                continue

            product_cash, next_stock = self.settle_product(
                product, stocks.get(product, 0), own_amount, rival_amount
            )
            stocks[product] = next_stock
            cash += product_cash
        return cash

    def marginal_profit(self, baseline, candidate, fixed_cost, baseline_value=None):
        # baseline_value is retained for call compatibility; product-local
        # marginal valuation is exact and avoids full-portfolio rescoring.
        del baseline_value
        return self.marginal_value(baseline, candidate) - fixed_cost
