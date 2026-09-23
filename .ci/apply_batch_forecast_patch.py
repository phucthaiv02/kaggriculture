from pathlib import Path

forecast = Path('agents/forecast.py')
text = forecast.read_text()
init_anchor = '''        self.trade = lru_cache(maxsize=None)(self._trade)\n'''
if init_anchor not in text:
    raise SystemExit('forecast init anchor not found')
if 'self.settle_product = lru_cache(maxsize=None)(self._settle_product)' not in text:
    text = text.replace(
        init_anchor,
        init_anchor + '        self.settle_product = lru_cache(maxsize=None)(self._settle_product)\n',
    )

settle_anchor = '''    def _settle_day(self, stocks, flows, when, products=None, extra=None):\n'''
if settle_anchor not in text:
    raise SystemExit('settle_day anchor not found')
method = '''    def _settle_product(self, product, stock, own_amount, rival_amount):
        \"\"\"Exact one-product settlement transition, cached across candidates.\"\"\"
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

'''
if 'def _settle_product(' not in text:
    text = text.replace(settle_anchor, method + settle_anchor)

old = '''            own_units = abs(own_amount)\n            rival_units = abs(rival_amount)\n            for unit in range(max(own_units, rival_units)):\n                stock = stocks.get(product, 0)\n                quoted = []\n                if unit < own_units:\n                    price = self.price(product, stock if own_amount > 0 else stock - 1)\n                    quoted.append((0, own_amount, price))\n                if unit < rival_units:\n                    price = self.price(product, stock if rival_amount > 0 else stock - 1)\n                    quoted.append((1, rival_amount, price))\n\n                # Same-product units for both players are quoted against the\n                # same pre-round stock, then committed together.\n                for player, amount, price in quoted:\n                    stocks[product] = stocks.get(product, 0) + (\n                        int(price > 1) if amount > 0 else -1\n                    )\n                    if player == 0:\n                        cash += price if amount > 0 else -price\n'''
new = '''            product_cash, next_stock = self.settle_product(\n                product, stocks.get(product, 0), own_amount, rival_amount\n            )\n            stocks[product] = next_stock\n            cash += product_cash\n'''
if old not in text:
    raise SystemExit('legacy settle loop not found')
text = text.replace(old, new)
forecast.write_text(text)
