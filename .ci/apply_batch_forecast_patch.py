from pathlib import Path

forecast = Path('agents/forecast.py')
text = forecast.read_text()
if 'import numpy as np' not in text:
    text = text.replace('from functools import lru_cache\n\n', 'from functools import lru_cache\n\nimport numpy as np\n\n')
anchor = '    def marginal_value(self, baseline, candidate, baseline_values=None):\n'
if anchor not in text:
    raise SystemExit('marginal_value anchor not found')
method = '''    def marginal_values(self, baseline, candidates):
        \"\"\"Exact batch marginal values using vectorized candidate worlds.\"\"\"
        candidates = list(candidates)
        if not candidates:
            return []
        products = tuple(game.PRODUCTS)
        pindex = {name: i for i, name in enumerate(products)}
        days = self.end_day - self.day + 1
        rows = len(candidates) + 1
        nprod = len(products)
        base = np.zeros((days, nprod), dtype=np.int32)
        rival = np.zeros((days, nprod), dtype=np.int32)
        extra = np.zeros((len(candidates), days, nprod), dtype=np.int32)

        def fill(target, flow, sign=1):
            for when, units in flow.items():
                offset = when - self.day
                if not 0 <= offset < days:
                    continue
                for product, amount in units.items():
                    index = pindex.get(product)
                    if index is not None and amount:
                        target[offset, index] += sign * int(amount)

        fill(base, baseline.sales, 1)
        fill(base, baseline.inputs, -1)
        fill(rival, self.external.sales, 1)
        fill(rival, self.external.inputs, -1)
        for row, candidate in enumerate(candidates):
            fill(extra[row], candidate.sales, 1)
            fill(extra[row], candidate.inputs, -1)

        stocks = np.empty((rows, nprod), dtype=np.int64)
        for index, product in enumerate(products):
            stocks[:, index] = int(self.inventory.get(product, 0))
        cash = np.zeros(rows, dtype=np.float64)
        previous = self.day * 24 + self.hour - 1

        def quote(product, stock_values, buying):
            quoted_stock = stock_values - buying.astype(np.int64)
            unique, inverse = np.unique(quoted_stock, return_inverse=True)
            values = np.fromiter(
                (self.price(product, int(stock)) for stock in unique),
                dtype=np.int64, count=len(unique),
            )
            return values[inverse]

        for offset, when in enumerate(range(self.day, self.end_day + 1)):
            step = when * 24 + 23
            shop_ticks = step // 4 - previous // 4
            center_ticks = step // 24 - previous // 24
            previous = step
            for column, product in enumerate(products):
                demand = self.shop_demand[product] * shop_ticks
                if product in self.center_products:
                    demand += center_ticks
                if demand:
                    stocks[:, column] -= demand

                own = np.empty(rows, dtype=np.int64)
                own[0] = base[offset, column]
                own[1:] = own[0] + extra[:, offset, column]
                own_abs = np.abs(own)
                rival_amount = int(rival[offset, column])
                rounds = max(int(own_abs.max()), abs(rival_amount))
                if not rounds:
                    continue
                own_sale = own > 0
                own_buy = own < 0
                rival_sale = rival_amount > 0
                rival_buy = rival_amount < 0
                for unit in range(rounds):
                    before = stocks[:, column].copy()
                    own_active = own_abs > unit
                    if own_active.any():
                        own_prices = quote(product, before, own_buy & own_active)
                        selling = own_active & own_sale
                        buying = own_active & own_buy
                        cash[selling] += own_prices[selling]
                        cash[buying] -= own_prices[buying]
                    else:
                        own_prices = None
                        selling = buying = own_active
                    rival_active = abs(rival_amount) > unit
                    if rival_active:
                        rival_prices = quote(
                            product, before,
                            np.full(rows, rival_buy, dtype=bool),
                        )
                    if own_active.any():
                        stocks[selling, column] += (own_prices[selling] > 1)
                        stocks[buying, column] -= 1
                    if rival_active:
                        if rival_sale:
                            stocks[:, column] += (rival_prices > 1)
                        elif rival_buy:
                            stocks[:, column] -= 1
        return (cash[1:] - cash[0]).tolist()

'''
text = text.replace(anchor, method + anchor)
forecast.write_text(text)

planner = Path('agents/planner.py')
text = planner.read_text()
start = text.index('def evaluate_targets(\n')
end = text.index('\n\ndef _choice_current_key', start)
old = text[start:end]
flows_scoped = 'flows_scoped' in old
new = '''def evaluate_targets(
    market, baseline, candidates, labor=None, position=(4, 4), *, flows_scoped=False
):
    \"\"\"Compare targets over the same min(16, remaining days) horizon.

    Candidate market values are settled as one exact NumPy batch instead of
    repeating the Python per-unit settlement loop for every candidate.
    \"\"\"
    del labor, position
    end = _cycle_end(market.day, market.end_day)
    scoped_market = copy(market)
    scoped_market.end_day = end

    def scoped(flow):
        result = Production()
        for field in (\"sales\", \"inputs\"):
            getattr(result, field).update(
                (d, value) for d, value in getattr(flow, field).items()
                if market.day <= d <= end)
        return result

    scoped_baseline = baseline if flows_scoped else scoped(baseline)
    eligible = []
    outputs = []
    for choice, output, cost in candidates:
        if market.day + _first_yield_age(choice[0]) > end:
            continue
        if not flows_scoped:
            output = scoped(output)
        eligible.append((choice, output, cost))
        outputs.append(output)
    values = scoped_market.marginal_values(scoped_baseline, outputs)
    return [
        TargetProfit(choice, output, market_cash, cost, 0.0)
        for (choice, output, cost), market_cash in zip(eligible, values)
    ]
'''
text = text[:start] + new + text[end:]
if flows_scoped:
    pass
elif 'audit=None, flows_scoped=False' in text:
    pass
planner.write_text(text)
