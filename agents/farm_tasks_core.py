"""Function 1: read the farm and list what each tile needs done.

Pure function of (obs, targets) -> list[Task]. No side effects, no hidden
state -- everything a tile needs is derivable from the current observation
plus the standing "what should live here" decision (targets), which is
function 4's job to produce.

A target is `(name, fertilize) | None`: `name` is a crop or animal, and
`fertilize` is a commitment made once at planting time (see agents/
planner.py) because the correct WATER schedule for an ongoing crop depends
on whether it will be fertilized -- it is not a fertilize/day decision that
can be revisited daily (see agents/schedules.py's module docstring).
"""

from __future__ import annotations

from collections import Counter
from agents.horizon import SEASON_END_DAY, can_start, can_start_today
from dataclasses import dataclass, field

from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS as ENV_ANIMALS
from kaggle_environments.envs.kaggriculture.kaggriculture import market_price

from agents.schedules import (
    ONGOING_CROPS, animal_maintenance_can_still_pay, cycle_finished,
    is_maintenance_day, should_care_animal, should_feed_animal,
    should_fertilize_today,
)

CROPS = ("WHEAT", "CARROT", "MELON", "TOMATO", "STRAWBERRY")
ANIMALS = ("GOOSE", "COW", "SHEEP")
BUILD = {"GOOSE": "BUILD_COOP", "COW": "BUILD_PASTURE", "SHEEP": "BUILD_PASTURE"}
ANIMAL_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
SEED_COST = {"WHEAT": 10, "CARROT": 20, "MELON": 80, "TOMATO": 50, "STRAWBERRY": 100}
ANIMAL_COST = {name: data["cost"] for name, data in ENV_ANIMALS.items()}
SELLABLE_PRODUCTS = set(CROPS) | {
    ENV_ANIMALS[name]["product"] for name in ANIMALS
} | {"FERTILIZER"}


@dataclass
class Task:
    position: tuple[int, int]
    actions: list  # ordered engine ops for this tile, e.g. [["WATER"], ["HARVEST"]]
    needs: Counter = field(default_factory=Counter)  # items to PICKUP before arriving
    urgent: bool = False  # animal care -- a missed day is a lasting loss (escape)
    sells: Counter = field(default_factory=Counter)  # units this task will add to inventory
    ends_cycle: bool = False  # tile becomes free/replantable after this task
    immediate_drop: bool = False  # opening-only return to shed after this task
    refinance_feed: bool = False  # opening-only fertilizer sale -> wheat -> feed
    animal_harvest: bool = False  # ready animal output outranks all other tile work
    deadline: int | None = None  # absolute engine step before a crop decays
    must_liquidate: bool = False  # cash-critical work must end with a shed DROP
    pinned_worker: int | None = None  # final-day carried stock belongs to this worker
    immediate_transition: bool = False  # opening harvest/build stays in one visit


def _new_planting_actions(name, fertilize_commit, seeds_available, animals_available, wheat_available,
                           needs_dig, needs_build=True, wheat_needs_pickup=True,
                           day=0, end_day=SEASON_END_DAY):
    """Actions to start a fresh planting/placement on an empty (or weedy) tile.

    `needs_build` is False only for an animal moving onto a structure that's
    already the right kind and empty (e.g. a PASTURE an earlier animal
    escaped from) -- BUILD_PASTURE there would just no-op (the engine
    requires `tile is None`), silently wasting one of that worker's turns.
    """
    if not can_start(name, day, end_day):
        return None, Counter()
    if name in CROPS:
        if not seeds_available:
            return None, Counter()
        actions = ([["DIG"]] if needs_dig else []) + [["PLANT", name], ["WATER"]]
        return actions, Counter()
    build_step = [[BUILD[name]]] if needs_build else []
    actions = ([["DIG"]] if needs_dig else []) + build_step
    # Building the structure does not require the animal or its feed. Do it
    # immediately when a crop frees the target tile; PLACE can follow now if
    # inputs are ready, or on a later day without rebuilding the pasture.
    feed_on_placement = should_feed_animal(name, 0)
    if not (animals_available and (wheat_available or not feed_on_placement)):
        return (actions or None), Counter()
    actions += [["PLACE", name]]
    needs = Counter({name: 1})
    if feed_on_placement:
        actions.append(["FEED"])
        if should_care_animal(name, 0):
            actions.append(["CARE"])
        if wheat_needs_pickup:
            needs["WHEAT"] = 1
    return actions, needs


def build_tasks(
    obs,
    targets,
    assume_crop_seeds=False,
    assume_animal_inputs=False,
    prioritize_fertilizer_drop=False,
):
    """List every tile's required actions for today, given the standing targets.

    `targets` maps position -> (name, fertilize) | None. Positions with no
    target (planner decided nothing profitable fits there, or land not yet
    claimed) are skipped entirely -- an unclaimed tile generates no task.
    """
    day = obs["day"]
    end_day = obs.get("_planning_end_day", SEASON_END_DAY)
    final_day = day >= end_day
    farm = obs["farms"][obs["player"]]
    shed = obs["private"]["shed"]
    seeds_left = Counter(obs["private"]["seeds"])
    if assume_crop_seeds:
        # Hour-0 labor sizing happens before this turn's BUY_SEED orders land.
        # Model those planned seeds so HAND demand includes PLANT + WATER,
        # while the real hour-1 queue still uses only inventory actually
        # delivered by the market.
        for target in targets.values():
            if target and target[0] in CROPS:
                seeds_left[target[0]] += 1
    animals_left = Counter({a: shed.get(a, 0) for a in ANIMALS})
    wheat_left = shed.get("WHEAT", 0)
    if assume_animal_inputs:
        for position, target in targets.items():
            if not target or target[0] not in ANIMALS:
                continue
            x, y = position
            tile = farm["tiles"][y][x]
            if not (isinstance(tile, dict) and tile.get("animal") == target[0]):
                animals_left[target[0]] += 1
                wheat_left += int(should_feed_animal(target[0], 0))
    fertilizer_left = shed.get("FERTILIZER", 0)
    prices = obs["market"]["prices"]
    tasks = []

    # On the final actionable day, inventory already carried in the field is
    # itself liquidation work. Pin one shed-return task to each carrier so the
    # scheduler cannot accidentally give that DROP to a different worker.
    if final_day:
        worker_positions = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
        for index, (position, inventory) in enumerate(
            zip(worker_positions, obs["private"]["inventories"])
        ):
            carried = Counter({
                item: inventory.get(item, 0)
                for item in SELLABLE_PRODUCTS
                if inventory.get(item, 0) > 0
            })
            if carried:
                tasks.append(Task(
                    position,
                    [],
                    urgent=True,
                    sells=carried,
                    must_liquidate=True,
                    pinned_worker=index,
                ))

    for position, target in targets.items():
        x, y = position
        tile = farm["tiles"][y][x]

        # Final-day liquidation mode: no WATER/FERTILIZE/DIG/PLANT/PLACE/
        # FEED/CARE can create bankable money after the terminal boundary.
        # Harvest whatever is already ready, collect already-created animal
        # fertilizer, and make the route include a real shed DROP.
        if final_day:
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop = tile["crop"]
                quantity = tile.get("yield_units", 0)
                if quantity > 0:
                    tasks.append(Task(
                        position,
                        [["HARVEST"]],
                        urgent=True,
                        sells=Counter({crop: quantity}),
                        must_liquidate=True,
                    ))
            elif isinstance(tile, dict) and tile.get("animal"):
                actions, sells = [], Counter()
                harvest_ready = tile.get("yield_units", 0) > 0
                if harvest_ready:
                    product = ENV_ANIMALS[tile["animal"]]["product"]
                    actions.append(["HARVEST"])
                    sells[product] += tile["yield_units"]
                if tile.get("fertilizer_available"):
                    actions.append(["COLLECT_FERTILIZER"])
                    sells["FERTILIZER"] += 1
                if actions:
                    tasks.append(Task(
                        position,
                        actions,
                        urgent=True,
                        sells=sells,
                        animal_harvest=harvest_ready,
                        must_liquidate=True,
                    ))
            continue

        if target is None:
            # An idle target must not abandon already-grown output or buy a
            # fallback seed just to trigger a harvest.
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop = tile["crop"]
                if tile.get("yield_units", 0) > 0:
                    actions = [] if tile.get("watered_today") else [["WATER"]]
                    tasks.append(Task(position, actions + [["HARVEST"]], urgent=True,
                                      sells=Counter({crop: tile["yield_units"]}),
                                      immediate_drop=position in obs.get("_opening_early_harvest_positions", ()),
                                      must_liquidate=position in obs.get("_opening_early_harvest_positions", ())))
                elif cycle_finished(crop, day - tile["planted_day"], tile):
                    tasks.append(Task(position, [["DIG"]], ends_cycle=True))
            continue
        name, fertilize_commit = target
        actions, needs, sells = [], Counter(), Counter()
        urgent, ends_cycle = False, False
        immediate_drop = position in obs.get("_opening_early_harvest_positions", ())
        refinance_feed = False
        live_animal = tile.get("animal") if isinstance(tile, dict) else None

        if live_animal:
            age = day - tile["placed_day"]
            maintenance_can_pay = animal_maintenance_can_still_pay(
                live_animal, age, day, end_day
            )
            feed_today = maintenance_can_pay and should_feed_animal(live_animal, age)
            care_today = maintenance_can_pay and should_care_animal(live_animal, age)
            harvest_ready = tile.get("yield_units", 0) > 0
            # Animal output is capped on-tile. Harvesting as soon as it is
            # available prevents a later production tick from being clipped,
            # so it has the same scheduling priority as required maintenance.
            urgent = harvest_ready or feed_today or care_today
            if harvest_ready:
                product = ENV_ANIMALS[live_animal]["product"]
                tasks.append(
                    Task(
                        position,
                        [["HARVEST"]],
                        urgent=True,
                        sells=Counter({product: tile["yield_units"]}),
                        animal_harvest=True,
                    )
                )
            fed_from_refinance = False
            if tile.get("fed_today"):
                if care_today and not tile.get("cared_today"):
                    actions.append(["CARE"])
            elif feed_today and wheat_left > 0:
                wheat_left -= 1
                needs["WHEAT"] += 1
                actions.append(["FEED"])
                if care_today:
                    actions.append(["CARE"])
            elif feed_today and tile.get("fertilizer_available") and (
                prioritize_fertilizer_drop or farm.get("money", 0) < prices["WHEAT"]
            ):
                # No feed on hand: collect the fertilizer, but do not force
                # an immediate SHED trip. It will be dropped after the
                # worker's route if that fits, otherwise automatically at
                # the end of the day.
                actions.append(["COLLECT_FERTILIZER"])
                sells["FERTILIZER"] += 1
                fed_from_refinance = True
                immediate_drop = prioritize_fertilizer_drop
                refinance_feed = prioritize_fertilizer_drop
            elif feed_today:
                # Keep the survival task in the schedule even when hour-1
                # inventory is temporarily empty. feed_wheat_order can buy
                # WHEAT on any later market pass; omitting the task here
                # permanently left this animal unfed for the whole day.
                needs["WHEAT"] += 1
                actions.append(["FEED"])
                if care_today:
                    actions.append(["CARE"])
            if tile.get("fertilizer_available") and not fed_from_refinance:
                # Fed normally above, but the animal also has fertilizer
                # sitting on the tile from an earlier CARE bonus -- collect
                # that too, it is free money either way.
                actions.append(["COLLECT_FERTILIZER"])
                sells["FERTILIZER"] += 1
                immediate_drop = prioritize_fertilizer_drop

        elif isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop, age = tile["crop"], day - tile["planted_day"]
            if fertilizer_left and should_fertilize_today(crop, age, fertilize_commit):
                fertilizer_left -= 1
                needs["FERTILIZER"] += 1
                actions.append(["FERTILIZE"])
            if is_maintenance_day(crop, age, fertilize_commit) and not tile.get("watered_today"):
                actions.append(["WATER"])
                # Every scheduled WATER is protective work. Waiting until a
                # crop has already missed once leaves no scheduling margin:
                # one more miss turns it into a WEED at day end.
                urgent = True
            if crop in ONGOING_CROPS and tile.get("yield_units", 0):
                # Harvest only after today's irrigation.  A crop can carry
                # yield from an earlier production tick into a non-standard
                # maintenance day; harvesting that cached yield without
                # WATER is legal in the engine but violates the opening
                # policy and gives up today's growth opportunity.
                if not tile.get("watered_today") and ["WATER"] not in actions:
                    actions.append(["WATER"])
                actions.append(["HARVEST"])
                sells[crop] += tile["yield_units"]
            # A one-time crop whose target has since moved on to something
            # else may be cut short: harvest whatever yield has already
            # accumulated instead of waiting out its full water schedule,
            # trading the later bonus increments for freeing the tile
            # sooner (e.g. the opening book converting a WHEAT tile to an
            # animal as soon as there's anything to harvest, not on WHEAT's
            # own 4-day clock). Never pre-empt an ongoing crop this way --
            # that would forfeit entire future production cycles, not a few
            # bonus units.
            # Only explicitly designated opening tiles may exit early.
            # All other target changes wait for the verified max-yield age.
            early_exit = (
                position in obs.get("_opening_early_harvest_positions", ())
                and crop == "WHEAT"
                and tile.get("yield_units", 0) > 0
            )
            if cycle_finished(crop, age, tile) or early_exit:
                ends_cycle = True
                if crop not in ONGOING_CROPS:
                    if not tile.get("watered_today") and ["WATER"] not in actions:
                        actions.append(["WATER"])
                    actions.append(["HARVEST"])
                    sells[crop] += tile.get("yield_units", 0)
                    if tile.get("yield_units", 0) > 0:
                        # A one-time crop starts decaying (yield_units -1
                        # every other turn) the day after max_yield_day,
                        # with only that one day of grace -- marking it
                        # urgent starting the day it's *first* ready (not
                        # waiting for decay to actually begin) is what
                        # gives a capacity-pressured day any real chance to
                        # still reach it in time; waiting one more day
                        # measurably lost more crops to WEED than it saved
                        # (confirmed directly, seed 1: 19 tiles lost vs 4-8
                        # marking it urgent from the first ready day).
                        urgent = True
                else:
                    actions.append(["DIG"])
                harvested_feed = crop == "WHEAT" and tile.get("yield_units", 0) > 0 and name in ANIMALS
                new_actions, new_needs = _new_planting_actions(
                    name, fertilize_commit, seeds_left[name] if name in CROPS else 0,
                    animals_left[name] if name in ANIMALS else 0,
                    wheat_left + (1 if harvested_feed else 0),
                    False,
                    wheat_needs_pickup=not harvested_feed,
                    day=day, end_day=end_day,
                )
                if new_actions:
                    if name in CROPS:
                        seeds_left[name] -= 1
                    elif new_needs[name]:
                        animals_left[name] -= 1
                        wheat_left -= new_needs["WHEAT"]
                        if harvested_feed:
                            sells["WHEAT"] -= 1
                            if not sells["WHEAT"]:
                                del sells["WHEAT"]
                    actions += new_actions
                    needs += new_needs

        elif name in ANIMALS:
            if (isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE")
                    and tile["kind"] != ANIMAL_STRUCTURE[name]):
                continue  # A stale target cannot convert a permanent structure.
            available = animals_left[name] if name in ANIMALS else 0
            already_built = isinstance(tile, dict) and tile.get("kind") == ANIMAL_STRUCTURE[name]
            new_actions, new_needs = _new_planting_actions(
                name, fertilize_commit, 0, available, wheat_left,
                needs_dig=isinstance(tile, dict) and not already_built,
                needs_build=not already_built,
                day=day, end_day=end_day,
            )
            if new_actions:
                if new_needs[name]:
                    animals_left[name] -= 1
                    wheat_left -= new_needs["WHEAT"]
                actions += new_actions
                needs += new_needs

        elif name in CROPS and seeds_left[name] and not (
            isinstance(tile, dict) and tile.get("kind") in ("COOP", "PASTURE")
        ):
            new_actions, new_needs = _new_planting_actions(
                name, fertilize_commit, seeds_left[name], 0, 0, isinstance(tile, dict),
                day=day, end_day=end_day,
            )
            if new_actions:
                seeds_left[name] -= 1
                actions += new_actions
                needs += new_needs

        if actions:
            tasks.append(
                Task(
                    position,
                    actions,
                    needs,
                    urgent,
                    sells,
                    ends_cycle,
                    immediate_drop,
                    refinance_feed,
                    immediate_transition=(position in obs.get("_opening_early_harvest_positions", ())
                                          or any(a[0] == "PLANT" for a in actions)),
                    must_liquidate=bool(
                        (
                            obs.get("_opening_refinance_first")
                            and (immediate_drop or (name in ANIMALS and not live_animal))
                        )
                        or (
                            obs.get("_liquidate_fertilizer_first")
                            and sells.get("FERTILIZER", 0) > 0
                        )
                    ),
                    animal_harvest=False,
                    deadline=(tile.get("max_lifespan_step") if ends_cycle and isinstance(tile, dict) else None),
                )
            )
    return tasks


def reserved_items(tasks):
    """Items today's tasks will PICKUP and consume -- selling.py must hold
    these back rather than sell them out from under a FERTILIZE/FEED task."""
    return sum((task.needs for task in tasks), Counter())


def _animal_and_seed_demand(
    obs, targets, active_positions, replant_same_crop=False
):
    """Shared scan behind both purchase_orders and feed_wheat_order: how
    many live animals need feeding today, how much WHEAT today's harvests
    will already provide toward that, what's missing to fill every animal
    target, and what crop seeds are missing."""
    day = obs["day"]
    end_day = obs.get("_planning_end_day", SEASON_END_DAY)
    farm = obs["farms"][obs["player"]]
    shed = obs["private"]["shed"]
    seeds_left = Counter(obs["private"]["seeds"])
    carried = Counter()
    for inventory in obs["private"]["inventories"]:
        carried.update(inventory)

    seed_demand, animal_missing, live_animals = Counter(), Counter(), 0
    wheat_incoming = 0  # WHEAT today's tasks will harvest -- see purchase_orders
    for position in active_positions:
        target = targets.get(position)
        if target is None:
            continue
        name, _ = target
        x, y = position
        tile = farm["tiles"][y][x]
        if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "WHEAT":
            age = day - tile["planted_day"]
            exits_early = name != "WHEAT" and tile.get("yield_units", 0) > 0
            # Finished WHEAT harvested today is valid coverage for the
            # conservative next-day feed reserve. Without counting it, Day 4
            # spends the last $60 buying redundant feed and leaves one of the
            # seven replacement seeds unfunded until after its PLANT action.
            finished_today = cycle_finished("WHEAT", age, tile)
            if exits_early or finished_today:
                wheat_incoming += tile.get("yield_units", 0)
        if name in CROPS:
            if not can_start_today(name, obs):
                continue
            if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                seed_demand[name] += 1
            else:
                crop, age = tile["crop"], day - tile["planted_day"]
                if cycle_finished(crop, age, tile) and (
                    replant_same_crop or name != crop
                ):
                    seed_demand[name] += 1
        elif name in ANIMALS:
            if isinstance(tile, dict) and tile.get("animal") == name:
                age = day - tile["placed_day"]
                if (
                    not tile.get("fed_today")
                    and animal_maintenance_can_still_pay(name, age, day, end_day)
                ):
                    live_animals += 1
            elif can_start_today(name, obs):
                animal_missing[name] += 1
    animal_slots = sum(animal_missing.values())
    for name in ANIMALS:
        available = shed.get(name, 0) + carried[name]
        animal_missing[name] = max(0, animal_missing[name] - available)
    pending_feed = animal_slots - sum(animal_missing.values())
    seed_demand = Counter({name: max(0, n - seeds_left[name]) for name, n in seed_demand.items()})
    return (
        seed_demand,
        animal_missing,
        live_animals,
        pending_feed,
        wheat_incoming,
        carried["WHEAT"],
    )


def _tomorrow_feed_need(obs, farm, active_positions, animal_missing):
    """Top up a four-day feed window after the opening animal purchases."""
    day = obs["day"]
    end_day = obs.get("_planning_end_day", SEASON_END_DAY)
    from agents.maintenance import should_feed
    # Build the opening first; retain harvested reserves without buying them
    # ahead of the scheduled animal placements.
    if day < 3 or sum(animal_missing.values()):
        return 0
    total = 0
    for position in active_positions:
        tile = farm["tiles"][position[1]][position[0]]
        if not isinstance(tile, dict) or not tile.get("animal"):
            continue
        animal = tile["animal"]
        for when in range(day + 1, min(day + (2 if day == 3 else 4), end_day + 1)):
            age = when - tile["placed_day"]
            if (should_feed(animal, age, position)
                    and animal_maintenance_can_still_pay(animal, age, when, end_day)):
                total += 1
    return total


def feed_wheat_order(obs, targets, active_positions):
    """Just today's WHEAT-for-feed shortfall, safe (and needed) to call every
    turn of the day, not only hour 0/1: COLLECT_FERTILIZER -> return to shed
    -> DROP -> SELL only happens once a task queue actually runs it (hour
    2+), so the cash to fund this same day's feed purchase may not exist
    until partway through the day. purchase_orders' bigger seed/animal
    purchases stay hour-0/1-only (they are not this time-sensitive and
    shouldn't be resubmitted every turn), but wheat is cheap and this check
    is idempotent -- it only ever asks for the current shortfall."""
    (
        _seed_demand,
        animal_missing,
        live_animals,
        pending_feed,
        wheat_incoming,
        carried_wheat,
    ) = _animal_and_seed_demand(
        obs, targets, active_positions
    )
    shed = obs["private"]["shed"]
    farm = obs["farms"][obs["player"]]
    tomorrow_feed = _tomorrow_feed_need(obs, farm, active_positions, animal_missing)
    wheat_needed = max(
        0,
        live_animals + pending_feed + tomorrow_feed
        - shed.get("WHEAT", 0) - carried_wheat,
    )
    return [["BUY_PRODUCT", "WHEAT", wheat_needed]] if wheat_needed else []


def purchase_orders(
    obs, targets, active_positions, available_money=None, available_wheat=None,
    replant_same_crop=False,
):
    """What the shed is missing to cover every target's standing need: seeds
    for an empty/finished crop tile, an animal for an empty/escaped pen, and
    the wheat both feeding and any pending animal purchase will require.
    build_tasks only PICKUPs what's already in the shed -- this is what puts
    it there."""
    farm = obs["farms"][obs["player"]]
    end_day = obs.get("_planning_end_day", SEASON_END_DAY)
    (
        seed_demand,
        animal_missing,
        live_animals,
        pending_feed,
        wheat_incoming,
        carried_wheat,
    ) = _animal_and_seed_demand(
        obs, targets, active_positions, replant_same_crop=replant_same_crop
    )
    shed = obs["private"]["shed"]
    wheat_on_hand = (
        shed.get("WHEAT", 0) if available_wheat is None else available_wheat
    ) + carried_wheat

    # Feed reserve remains deliberately conservative (it may cover a verified
    # schedule gap too): spare wheat is cheap insurance against delayed
    # placement/refinance, while build_tasks decides the exact ages on which
    # FEED and CARE actions actually run.
    wheat_inventory = obs["market"]["inventory"].get("WHEAT", 0)
    tomorrow_feed = _tomorrow_feed_need(obs, farm, active_positions, animal_missing)
    wheat_needed_now = max(
        0,
        live_animals
        + pending_feed
        + max(0, tomorrow_feed - wheat_incoming)
        - wheat_on_hand,
    )

    def wheat_cost(quantity, offset=0):
        return sum(
            market_price("WHEAT", wheat_inventory - offset - unit - 1,
                         obs["market"].get("params"))
            for unit in range(quantity)
        )

    money = farm["money"] if available_money is None else available_money
    money -= wheat_cost(wheat_needed_now)

    # Orders are processed strictly in sequence -- one order's quantity is
    # fully exhausted (as far as cash allows) before the next order is even
    # attempted (see kaggriculture's _process_market). Requesting the full
    # target-based animal demand (dozens, on a cold start) would let the
    # first animal type alone eat the whole budget, leaving nothing for the
    # wheat that actually lets a newly-bought animal get PLACEd this same
    # cycle. So each animal request is pre-capped to what's affordable
    # including its own feed, and cash is debited as-if-spent between types
    # so a later type doesn't also assume the same money is still free.
    animal_orders = []
    # Cheapest-first prevents row-major target order from buying a $500
    # SHEEP while leaving an already-built $400 COW pasture empty when only
    # one of the two is currently affordable.
    for name, n in sorted(animal_missing.items(), key=lambda item: ANIMAL_COST[item[0]]):
        if not n:
            continue
        quantity = 0
        for _ in range(n):
            prior_new_feed = sum(int(order[2]) for order in animal_orders) + quantity
            total_new_feed = prior_new_feed + 1
            prior_wheat = max(
                0,
                live_animals + pending_feed + prior_new_feed
                - wheat_on_hand - wheat_incoming,
            )
            candidate_wheat = max(
                0,
                live_animals + pending_feed + total_new_feed
                - wheat_on_hand - wheat_incoming,
            )
            extra_wheat = candidate_wheat - prior_wheat
            feed_cost = wheat_cost(max(0, extra_wheat), prior_wheat)
            if money < ANIMAL_COST[name] + feed_cost:
                break
            money -= ANIMAL_COST[name] + feed_cost
            quantity += 1
        if quantity:
            animal_orders.append(["BUY_ANIMAL", name, quantity])

    new_animal_feed = sum(int(order[2]) for order in animal_orders)
    unfunded_animals = sum(animal_missing.values()) - new_animal_feed
    # Do not turn every last coin into seeds. Animals must be fed again
    # tomorrow before today's fertilizer can be collected and sold; without
    # this reserve an opening that is affordable on paper loses an animal
    # to two consecutive no-op FEED attempts.
    # Day 0 is the one exception: filling the fixed 25-tile opening requires
    # nearly all starting capital, and every newly placed animal produces
    # fertilizer that the now-safe refinance route can turn into day-1 feed.
    # From day 1 onward keep cash feed coverage instead of relying entirely
    # on that just-in-time loop. Near the season boundary, stop reserving
    # tomorrow's feed once tomorrow's maintenance cannot reach another sale.
    next_day_feed_reserve = (
        0
        if obs["day"] == 0 or obs["day"] >= end_day - 1
        else wheat_cost(live_animals + pending_feed + new_animal_feed)
    )

    # Same affordability-capping as animals, and for the same reason: with
    # both WHEAT and MELON seeds demanded and cash too short for both,
    # requesting the full count of whichever crop the position scan hit
    # first would let it eat the whole remaining budget, leaving nothing for
    # the other. Cheapest-first instead maximizes how many tiles actually
    # get planted with what's left, rather than funding whichever crop
    # happened to be listed first.
    seed_orders = []
    # Complete animal targets before spending their accumulating fertilizer
    # proceeds on replacement seeds. In the opening this is what preserves
    # enough cash to buy and PLACE the sixth animal on day index 3.
    for name, n in (() if unfunded_animals > 0 else sorted(
        seed_demand.items(), key=lambda item: SEED_COST[item[0]]
    )):
        if not n:
            continue
        spendable = max(0, money - next_day_feed_reserve)
        quantity = min(n, int(spendable // SEED_COST[name]))
        if quantity:
            seed_orders.append(["BUY_SEED", name, quantity])
            money -= quantity * SEED_COST[name]

    # A WHEAT tile finishing its cycle today will HARVEST before this day's
    # animal FEED tasks run (build_tasks queues HARVEST ahead of FEED/CARE),
    # so that yield already covers today's feed without buying anything --
    # counting it here is what stops the agent from spending cash on wheat
    # it's about to harvest for free.
    wheat_needed = max(
        0,
        live_animals + pending_feed + new_animal_feed
        + tomorrow_feed - wheat_on_hand - wheat_incoming,
    )

    orders = []
    if wheat_needed:
        orders.append(["BUY_PRODUCT", "WHEAT", wheat_needed])
    orders += animal_orders + seed_orders
    return orders
