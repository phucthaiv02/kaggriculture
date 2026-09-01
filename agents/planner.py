"""Function 4: decide what to grow/raise where, and whether to buy more land.

Both decisions answer the same question -- "where does the next unit of
capital do the most good" -- so they live in one function rather than being
split across two: land is just another thing capital can buy, alongside a
seed or an animal.
"""

from __future__ import annotations

from collections import Counter

from kaggle_environments.envs.kaggriculture import kaggriculture as official_game
from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS as ENV_ANIMALS
from kaggle_environments.envs.kaggriculture.kaggriculture import SHOPS, market_price

from agents.schedules import CROP_FERTILIZE_DAYS, CROP_LAST_AGE, ONGOING_CROPS

CROPS = ("WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY")
ANIMALS = ("GOOSE", "COW", "SHEEP")
SEED_COST = {"WHEAT": 10, "CARROT": 20, "MELON": 80, "TOMATO": 50, "STRAWBERRY": 100}
CROP_YIELD = {"WHEAT": 4, "CARROT": 3, "MELON": 6, "TOMATO": 4, "STRAWBERRY": 4}
ANIMAL_COST = {name: data["cost"] for name, data in ENV_ANIMALS.items()}
ANIMAL_PRODUCT = {name: data["product"] for name, data in ENV_ANIMALS.items()}
ANIMAL_FIRST = {name: data["first_yield_day"] for name, data in ENV_ANIMALS.items()}
ANIMAL_INTERVAL = {name: data["interval"] for name, data in ENV_ANIMALS.items()}
ANIMAL_MAX_HELD = {name: data["max_held"] for name, data in ENV_ANIMALS.items()}
LAND_ORDER = official_game.LAND_ORDER
LAND_BUY_UTILIZATION = 0.75  # only buy once >=75% of owned land is already claimed

# Verified against the installed kaggriculture engine's own _town_consume
# (not docs/README.md's Configuration Defaults table, which states
# townCenterSellInterval=12 with a 2x/4x scale-up after day 10/20 -- neither
# is true of the actual installed engine: the real default is a flat 24,
# with no day-based scaling at all). turnsPerDay is also engine-verified.
TURNS_PER_DAY = 24
TOWN_SHOP_SELL_INTERVAL = 4
TOWN_CENTER_SELL_INTERVAL = 24
TOWN_CENTER_PRODUCTS = (*CROPS, *(ENV_ANIMALS[a]["product"] for a in ANIMALS))


def _expected_town_demand(remaining, unlocked_shops):
    """Units of each product the town center plus every *currently*
    unlocked shop will pull from the market for free over the next
    `remaining` days -- real demand that offsets how much a batch of new
    plantings will actually glut the market, which _expected_price ignored
    entirely before this. `unlocked_shops` (obs["town"]["unlocked_shops"])
    can list the same shop more than once (drawn with replacement each
    unlock); each listed instance consumes independently, same as the
    engine's own _town_consume.

    A shop that unlocks later (townShopUnlockInterval, a random draw from
    the remaining pool) is not counted -- which shop comes next is hidden
    information, so only what's already unlocked is a safe, known floor
    rather than a guess.
    """
    demand = Counter()
    turns = remaining * TURNS_PER_DAY
    shop_ticks = turns // TOWN_SHOP_SELL_INTERVAL
    for shop_name in unlocked_shops:
        products = SHOPS[shop_name]
        multiplier = 2 if len(products) == 1 else 1
        for product in products:
            demand[product] += shop_ticks * multiplier
    center_ticks = turns // TOWN_CENTER_SELL_INTERVAL
    for product in TOWN_CENTER_PRODUCTS:
        demand[product] += center_ticks
    return demand


def _expected_price(product, inventory, committed_units, town_demand=None):
    """The real market_price the engine would quote, priced as if
    `committed_units` extra units were already sitting in inventory, net of
    however much of that same batch the town center/shops will have already
    bought for free by the time it would actually sell (see
    _expected_town_demand) -- committing to a product the town is hungry for
    gluts the market less than this would otherwise assume.

    The market is a single shared per-unit lockstep queue: selling walks
    inventory up one unit at a time and *requotes after every single unit*
    (see kaggriculture's _process_market), so a tile's true expected revenue
    depends on how much of the same product every other tile already
    committed to producing this same planning pass -- not the live snapshot
    price alone. A flat "discount by portfolio share" cannot capture this:
    it treats a $250 MELON (T=300, i.e. price craters within a couple
    hundred extra units) the same as WHEAT (T=400 but far higher volume
    tolerance relative to its tiny unit price). Pricing the *marginal* unit
    with the engine's own curve gets this right for free.
    """
    committed = committed_units.get(product, 0)
    if town_demand:
        committed = max(0, committed - town_demand.get(product, 0))
    return market_price(product, inventory.get(product, 0) + committed)


def _fertilize_gain(name, remaining, crop_price, fertilizer_price):
    """Extra score from committing to fertilize this planting, at today's
    prices -- decided once, at planting time, because the WATER schedule for
    the whole planting depends on this commitment (see agents/schedules.py)."""
    if name not in CROP_FERTILIZE_DAYS:
        return 0
    extra_per_use = 2 if name != "CARROT" else 1
    applications = [age for age in CROP_FERTILIZE_DAYS[name] if age <= remaining]
    return len(applications) * (extra_per_use * crop_price - fertilizer_price)


def _score(name, end_day, day, inventory, wheat_price, committed_units, unlocked_shops=()):
    """Return profit per day the tile is occupied -- profit already nets out
    the seed/animal cost, so dividing by that cost again would count it
    twice; only the days-tied-up divisor stays, to make a fast, modest
    crop and a slow, lucrative one comparable on the same per-day basis.
    An animal's capital stays committed for the rest of the season once
    placed (no mid-game replant of the same tile), so its cycle length is
    what remains of the game, not a fixed maturation window.
    """
    remaining = end_day - day
    town_demand = _expected_town_demand(remaining, unlocked_shops)
    if name in CROPS:
        cycle = CROP_LAST_AGE[name]
        if cycle > remaining:
            return None, False
        price = _expected_price(name, inventory, committed_units, town_demand)
        fertilizer_price = _expected_price("FERTILIZER", inventory, {})
        base = price * CROP_YIELD[name] - SEED_COST[name]
        gain = _fertilize_gain(name, remaining, price, fertilizer_price)
        profit = base + max(0, gain)
        return profit / cycle, gain > 0
    first = ANIMAL_FIRST[name]
    if first > remaining:
        return None, False
    interval = ANIMAL_INTERVAL[name]
    production_days = 1 + (remaining - first) // interval
    if production_days == 1:
        # Placing this late means only the animal's very first production
        # tick will happen before end_day. That tick banks CARE bonus from
        # placement all the way to first_yield_day with nothing harvested
        # yet to reset it, so it saturates at max_held -- verified directly
        # against the real interpreter (fed+cared every day from placement
        # to the first possible tick): GOOSE=4, COW=6, SHEEP=6, not the
        # interval+1 steady-state figure below, which is for a tick
        # shortly after a *previous* harvest already emptied the bank.
        units = ANIMAL_MAX_HELD[name]
    else:
        units = production_days * (interval + 1)
    price = _expected_price(ANIMAL_PRODUCT[name], inventory, committed_units, town_demand)
    # A fed, living animal makes fertilizer_available every single day from
    # placement (see kaggriculture's _daily_refresh_animals) -- roughly one
    # collectible+sellable FERTILIZER unit/day, a steady side income on top
    # of its main product that a raw WOOL/MILK/EGG-only tally misses. Priced
    # the same marginal way as everything else: FERTILIZER is a single
    # shared market too, so committing many animals in the same pass must
    # discount each other's fertilizer income exactly like it discounts
    # their shared WOOL/MILK/EGG price. No shop or the town center buys
    # FERTILIZER (see TOWN_CENTER_PRODUCTS/SHOPS), so town_demand never
    # actually offsets it -- passed anyway for uniformity.
    fertilizer_price = _expected_price("FERTILIZER", inventory, committed_units, town_demand)
    fertilizer_income = remaining * fertilizer_price
    profit = units * price + fertilizer_income - ANIMAL_COST[name] - remaining * wheat_price
    return profit / remaining, False


def _commitment_deltas(name, fertilize, end_day, day, placed_day=None):
    """Products (and quantities) this planting will actually dump on the
    market -- used to charge the *next* candidate for the price impact of
    this one. An animal contributes to both its main product and to
    FERTILIZER (a side effect of being fed every day it's alive).

    `placed_day` distinguishes an *already-established* animal (pass its
    real placed_day, from the tile) from a not-yet-placed candidate (leave
    None, which treats today as the placement day -- the correct count for
    something that would only start its clock now). An established animal
    is already past its own first_yield_day and producing every `interval`
    days on schedule; counting `production_days` from *today's* remaining
    days as if it just started, as the None case does, silently erases its
    real remaining output once `remaining` alone drops below its
    first_yield_day -- confirmed directly (seed 1, day 25): committed WOOL
    from 17 already-producing SHEEP fell to 0 under that formula purely
    because 30-25=5 is less than SHEEP's first_yield_day of 6, letting new
    SHEEP keep scoring as if the wool market were untouched. Counting ticks
    from the animal's real placed_day instead fixes that regardless of how
    close to end_day the season is.
    """
    if name in CROPS:
        return {name: CROP_YIELD[name]}
    interval = ANIMAL_INTERVAL[name]
    first = ANIMAL_FIRST[name]
    if placed_day is None:
        remaining = end_day - day
        production_days = max(0, 1 + (remaining - first) // interval)
        occurred_ticks = 0
    else:
        age = day - placed_day
        end_age = end_day - placed_day
        total_ticks = (end_age - first) // interval + 1 if end_age >= first else 0
        occurred_ticks = (age - first) // interval + 1 if age >= first else 0
        production_days = max(0, total_ticks - occurred_ticks)
    if occurred_ticks == 0 and production_days == 1:
        # Same first-tick-only case _score handles: nothing harvested yet,
        # so the single remaining tick saturates at max_held rather than
        # the interval+1 steady-state figure below.
        units = ANIMAL_MAX_HELD[name]
    else:
        units = production_days * (interval + 1)
    return {
        ANIMAL_PRODUCT[name]: units,
        "FERTILIZER": end_day - day,
    }


CHEAPEST_COST = min(*SEED_COST.values(), *ANIMAL_COST.values())  # WHEAT, $10


def _category_discount(name, category_capital, total_capital, cost):
    """Discount by how much of the portfolio's *capital* (not tile count) is
    already committed to this asset's category (crop vs animal), scaled up
    for a costlier candidate.

    Even after pricing the marginal unit correctly (_expected_price), a
    fast-cycling crop can replant itself many times before end_day while an
    animal's capital stays committed for the rest of the season -- so crops
    still win almost every time on plain profit/day, and an all-crop
    portfolio never sees a WOOL/MILK/EGG market to fall back on if crop
    prices crash together. This is a second, coarser discount on top of the
    per-product one: it doesn't change which crop or which animal is best,
    only pushes some capital toward the category that has none yet.

    `_score` is plain profit/day now (no cost normalization -- profit
    already nets out cost), so a $500 SHEEP's raw score can be 10x a $10
    WHEAT's even though a single SHEEP already ties up as much capital as
    50 WHEAT plantings. A flat share-based cap can't tell those apart: it
    squeezes an expensive and a cheap candidate by the same amount at the
    same category share, letting the cheap one's edge just get scaled down
    proportionally right along with it -- confirmed directly (seed 1): SHEEP
    kept winning even at 73% existing ANIMAL share (a 30% floor still left
    it far above WHEAT's discounted score), spending the farm down to $4 and
    starving several SHEEP of FEED for lack of hands. Weighting the share by
    cost relative to the cheapest option (WHEAT) before applying it makes a
    single expensive placement count for as much of its category as the many
    WHEAT-equivalent plantings its capital could otherwise have funded, so
    it hits the squeeze much sooner than a cheap one at the same dollar
    share.
    """
    if not total_capital:
        return 1.0
    category = "CROP" if name in CROPS else "ANIMAL"
    share = category_capital.get(category, 0) / total_capital
    weighted_share = share * (cost / CHEAPEST_COST) ** 0.5
    return max(0.0, 1 - min(1.0, weighted_share))


def best_target(
    end_day, day, inventory, wheat_price, committed_units,
    category_capital=None, total_capital=0, unlocked_shops=(),
):
    """Best (name, fertilize) for one open tile, pricing every candidate's
    output as if `committed_units` (from other tiles already decided this
    same pass) were already sold, net of what the town/shops will buy for
    free over that time -- see _expected_price -- and discounting by
    category capital share -- see _category_discount."""
    best_choice, best_score = None, float("-inf")
    for name in (*CROPS, *ANIMALS):
        score, fertilize = _score(
            name, end_day, day, inventory, wheat_price, committed_units, unlocked_shops
        )
        if score is None:
            continue
        cost = SEED_COST[name] if name in CROPS else ANIMAL_COST[name]
        score *= _category_discount(name, category_capital or {}, total_capital, cost)
        if score > best_score:
            best_choice, best_score = (name, fertilize), score
    return best_choice


def should_buy_land(farm, active_positions):
    """Buy the next quadrant only once real production -- not just a target
    decision -- already fills most of the land currently owned. A target is
    assigned to (almost) every open tile the instant it unlocks (see
    plan_targets), so gating on "has a target" would trigger on day 0 before
    a single seed is planted; gating on what's actually growing/placed
    reflects real land pressure instead."""
    n_extra = len(farm["unlocked_quadrants"]) - 1
    if n_extra >= len(LAND_ORDER):
        return False
    if not active_positions:
        return False
    tiles = farm["tiles"]
    occupied = 0
    for x, y in active_positions:
        tile = tiles[y][x]
        if isinstance(tile, dict) and (tile.get("kind") == "PLANT" or "animal" in tile):
            occupied += 1
    return occupied / len(active_positions) >= LAND_BUY_UTILIZATION


def plan_targets(obs, targets, active_positions, end_day):
    """Update `targets` in place: claim newly-unlocked tiles and replant
    tiles whose crop just finished its cycle. Returns nothing; the caller
    (agents/expansion_agent.py) owns the dict across days.

    `committed_units` seeds from the *existing* portfolio (not just new picks
    this call), so a lone tile finishing its cycle on day 20 still sees the
    20-tile MELON block already established on day 0 -- otherwise each day's
    small batch of reassignments would start from a blank slate and could
    all pile onto the same already-saturated product. It then also
    accumulates every new choice made in this same call, so the 2nd, 3rd,
    ... tile assigned to the same product this pass is correctly priced as
    selling into a market the earlier picks already loaded up -- this is
    what makes per-product diversification emerge from the price math
    itself instead of an arbitrary portfolio cap. The market doesn't care
    which planning pass a unit of supply came from, so this stays scoped to
    the whole farm.

    `category_capital`/`total_capital`, by contrast, start empty every call
    and only accumulate picks made *within* this same pass -- they measure
    whether this batch of decisions is itself piling onto one category, not
    whether the farm has ever owned an animal before. Seeding them from the
    whole existing portfolio was tried and found wrong: a farm with a
    modest, deliberate handful of animals from days ago would already sit
    at a high ANIMAL capital share on its own, so _category_discount would
    floor every animal candidate to a hard 0 for the rest of the game (see
    _category_discount's cost weighting) regardless of whether *this*
    batch was actually overcommitting to animals at all -- confirmed
    directly (seed 1): a 25-tile quadrant opened with 6 pre-existing
    animals never got the chance to place a single new one even though
    none of those 25 tiles' own choices had anything to do with animals yet.
    """
    day = obs["day"]
    inventory = obs["market"]["inventory"]
    wheat_price = _expected_price("WHEAT", inventory, {})
    unlocked_shops = obs["town"]["unlocked_shops"]
    farm = obs["farms"][obs["player"]]
    committed_units = Counter()
    category_capital, total_capital = Counter(), 0
    for position, value in targets.items():
        if value is None:
            continue
        name, fertilize = value
        placed_day = None
        if name in ANIMALS:
            x, y = position
            tile = farm["tiles"][y][x]
            if isinstance(tile, dict) and tile.get("animal") == name:
                placed_day = tile["placed_day"]
        committed_units.update(_commitment_deltas(name, fertilize, end_day, day, placed_day))

    def choose():
        nonlocal total_capital
        choice = best_target(
            end_day, day, inventory, wheat_price, committed_units,
            category_capital, total_capital, unlocked_shops,
        )
        if choice:
            name, fertilize = choice
            committed_units.update(_commitment_deltas(name, fertilize, end_day, day))
            cost = SEED_COST[name] if name in CROPS else ANIMAL_COST[name]
            category_capital["CROP" if name in CROPS else "ANIMAL"] += cost
            total_capital += cost
        return choice

    # Tried and reverted: staggering a fresh batch's claims (only half per
    # call, so their planted_day -- and therefore recurring WATER schedule --
    # doesn't all land on the same days) to smooth hand demand. Measured
    # worse in practice (a 25-tile board's net profit dropped from $1,244 to
    # $53, seed 1): the delayed half loses real production days it never
    # gets back, and a 25-tile board's actual peak workload (~57 actions on
    # its busiest day, seed 1) was never close to saturating even a handful
    # of hands to begin with -- so the smoothing bought nothing there and
    # only cost the delay everywhere. May still be worth revisiting
    # specifically for a batch that *is* provably too big for current hand
    # capacity (e.g. right after buying land), rather than unconditionally.
    replanning = []
    new_positions = [position for position in active_positions if position not in targets]
    new_slots = len(new_positions)
    shed_access = ((4, 4), (5, 4), (4, 5), (5, 5))
    distance = lambda p: min(
        abs(p[0] - s[0]) + abs(p[1] - s[1]) for s in shed_access
    )
    selected_new = set(
        sorted(new_positions, key=lambda p: (distance(p), p[1], p[0]))[:new_slots]
    )
    for position in active_positions:
        x, y = position
        current = targets.get(position)
        if position not in targets:
            if position in selected_new:
                replanning.append(position)
            else:
                targets[position] = None
            continue
        if current is None:
            continue
        tile = farm["tiles"][y][x]
        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
            continue
        crop, age = tile["crop"], day - tile["planted_day"]
        finished = (
            (crop not in ONGOING_CROPS and age >= CROP_LAST_AGE[crop])
            or (crop in ONGOING_CROPS and age >= CROP_LAST_AGE[crop] and not tile.get("yield_units", 0))
        )
        if finished and current[0] == crop:
            replanning.append(position)

    # Portfolio choice is position-independent, but structure travel is not:
    # assign every animal selected in this batch to the candidate tiles
    # closest to a SHED access square. This keeps newly built PASTURE/COOP
    # beside the SHED instead of depending on row-major board order.
    choices = []
    for position in replanning:
        choice = choose()
        # Near season end no candidate may have time to return a profit.
        # A finished, already-claimed crop must still be harvested/replanted
        # instead of being abandoned until it decays into a WEED.
        if choice is None and position in targets and targets[position] is not None:
            choice = targets[position]
        choices.append(choice)
    nearest_positions = sorted(
        replanning,
        key=lambda p: (
            min(abs(p[0] - s[0]) + abs(p[1] - s[1]) for s in shed_access),
            p[1],
            p[0],
        ),
    )
    animal_choices = [choice for choice in choices if choice and choice[0] in ANIMALS]
    other_choices = [choice for choice in choices if not choice or choice[0] not in ANIMALS]
    for position, choice in zip(nearest_positions, animal_choices + other_choices):
        targets[position] = choice

    # A later replant pass may select an animal when only a far crop tile
    # happens to finish that day. Swap that pending structure target with the
    # nearest non-animal target instead of building a remote PASTURE/COOP.
    # Established animals/structures are never moved.
    tiles = farm["tiles"]
    pending_animals = [
        position
        for position in active_positions
        if targets.get(position)
        and targets[position][0] in ANIMALS
        and not (
            isinstance(tiles[position[1]][position[0]], dict)
            and tiles[position[1]][position[0]].get("kind") in ("PASTURE", "COOP")
        )
    ]
    crop_slots = [
        position
        for position in active_positions
        if targets.get(position) and targets[position][0] in CROPS
    ]
    for animal_position in sorted(pending_animals, key=distance):
        nearer = [p for p in crop_slots if distance(p) < distance(animal_position)]
        if not nearer:
            continue
        crop_position = min(nearer, key=lambda p: (distance(p), p[1], p[0]))
        targets[animal_position], targets[crop_position] = (
            targets[crop_position],
            targets[animal_position],
        )
        crop_slots.remove(crop_position)
        crop_slots.append(animal_position)
