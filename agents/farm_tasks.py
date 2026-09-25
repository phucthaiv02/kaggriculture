"""Read the farm and list what each tile needs done.

Pure function of (obs, targets) -> list[Task]. No side effects, no hidden
state -- everything a tile needs is derivable from the current observation
plus the standing "what should live here" decision (targets), produced by the planner.

A target is `(name, fertilizer_plan) | None`: `name` is a crop or animal.
Dynamic planner targets carry the exact fertilizer ages selected for the
current crop cycle (plus forecast-only later cycles); opening targets retain
the historical boolean flag.  agents/schedules.py normalizes both forms.
"""

from __future__ import annotations

from collections import Counter
from agents.horizon import SEASON_END_DAY, can_start, can_start_today
from dataclasses import dataclass, field

from kaggle_environments.envs.kaggriculture.kaggriculture import ANIMALS as ENV_ANIMALS
from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS as ENV_CROPS
from kaggle_environments.envs.kaggriculture.kaggriculture import market_price

from agents.schedules import (
    ONGOING_CROPS, cycle_turns_over_today, is_maintenance_day, should_care_animal,
    should_feed_animal, should_fertilize_today, animal_feed_end_age, should_harvest_animal,
)

from agents.products import ANIMALS, ANIMAL_COST, ANIMAL_STRUCTURE, BUILD, CROPS, SEED_COST


@dataclass
class Task:
    position: tuple[int, int]
    actions: list  # ordered engine ops for this tile, e.g. [["WATER"], ["HARVEST"]]
    needs: Counter = field(default_factory=Counter)  # items to PICKUP before arriving
    urgent: bool = False  # retained for compatibility with hand-built tasks
    sells: Counter = field(default_factory=Counter)  # units this task will add to inventory
    ends_cycle: bool = False  # tile becomes free/replantable after this task
    immediate_drop: bool = False  # opening-only return to shed after this task
    refinance_feed: bool = False  # opening-only fertilizer sale -> wheat -> feed
    animal_harvest: bool = False  # task includes a planned animal HARVEST
    cashout: bool = False  # reserve a return/DROP and a SELL before terminal
    terminal_day: bool = False  # terminal observation removes a worker turn
    mandatory: bool | None = None  # None preserves legacy hand-built tasks
    value: float = 0.0  # estimated incremental cash before today's extra labor


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
    # Building the structure does not require the animal or its feed. Once the
    # animal itself is available, however, keep the complete PLACE -> FEED ->
    # CARE dependency in the admitted task even if WHEAT is not in the shed
    # yet. The market/feed path can materialize that missing input later in the
    # same day; dropping PLACE here made a successfully bought SHEEP disappear
    # from the rolling task graph entirely.
    if not animals_available:
        return (actions or None), Counter()
    actions += [["PLACE", name]]
    needs = Counter({name: 1})
    feed_on_placement = should_feed_animal(name, 0)
    if feed_on_placement:
        actions.append(["FEED"])
        if should_care_animal(name, 0):
            actions.append(["CARE"])
        if wheat_needs_pickup:
            needs["WHEAT"] = 1
    return actions, needs


def executable_targets(obs, targets):
    """Keep pending reprices harvest-only after a cycle, including empty tiles."""
    result = dict(targets)
    tiles = obs["farms"][obs["player"]]["tiles"]
    for position in obs.get("_pending_targets", ()):
        if position not in result:
            continue
        x, y = position
        tile = tiles[y][x]
        if isinstance(tile, dict) and tile.get("animal"):
            continue
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop = tile["crop"]
            if not cycle_turns_over_today(crop, obs["day"] - tile["planted_day"], tile):
                target = targets[position]
                if not target or target[0] != crop:
                    result[position] = (crop, tile.get("fertilized_until_day", -1) >= tile["planted_day"])
                continue
        result[position] = None
    return result


def animal_output_at_risk(tile, day):
    """Held output is full or the next production refresh would clip it."""
    rules = ENV_ANIMALS[tile["animal"]]
    next_age = day + 1 - tile["placed_day"]
    produces = (next_age >= rules["first_yield_day"]
                and (next_age - rules["first_yield_day"]) % rules["interval"] == 0)
    held = tile.get("yield_units", 0)
    incoming = (1 + tile.get("pending_care_bonus", 0)) if produces else 0
    return held >= rules["max_held"] or held + incoming > rules["max_held"]


def build_tasks(
    obs,
    targets,
    assume_crop_seeds=False,
    assume_animal_inputs=False,
    prioritize_fertilizer_drop=False,
    include_physical=True,
):
    """List every tile's required actions for today, given the standing targets.

    `targets` maps position -> (name, fertilize) | None. Physical producers
    are maintained independently of replacement targets. Partial intraday
    calls set include_physical=False to limit work to the supplied positions.
    """
    targets = executable_targets(obs, targets)
    day = obs["day"]
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
    terminal = day == obs.get("_planning_end_day", SEASON_END_DAY)
    tasks = []

    task_targets = dict(targets)
    if include_physical:
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                if isinstance(tile, dict) and (tile.get("animal") or tile.get("kind") == "PLANT"):
                    task_targets.setdefault((x, y), None)

    for position, target in task_targets.items():
        x, y = position
        tile = farm["tiles"][y][x]
        if terminal and isinstance(tile, dict) and tile.get("kind") == "PLANT":
            if tile.get("yield_units", 0) > 0:
                crop = tile["crop"]
                rules = ENV_CROPS[crop]
                age = day - tile["planted_day"]
                amount = tile["yield_units"]
                actions = []
                # WATER can still add an immediate unit on a one-time crop;
                # retain that profitable last increment, not growth next day.
                if (not rules["ongoing"] and not tile.get("watered_today")
                        and (rules["max_yield_day"] + 1) // 2 <= age <= rules["max_yield_day"]
                        and amount < rules["max_yield"]):
                    actions.append(["WATER"])
                    amount = min(rules["max_yield"], amount + (
                        2 if tile.get("fertilized_until_day", -1) >= day else 1))
                tasks.append(Task(position, actions + [["HARVEST"]],
                                  sells=Counter({crop: amount}), mandatory=False))
            continue
        live_animal = tile.get("animal") if isinstance(tile, dict) else None
        if live_animal:
            target = (live_animal, False)
        if target is None:
            # An idle target must not abandon already-grown output or buy a
            # fallback seed just to trigger a harvest.
            x, y = position
            tile = farm["tiles"][y][x]
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                crop = tile["crop"]
                age = day - tile["planted_day"]
                turns_over = cycle_turns_over_today(crop, age, tile)
                if tile.get("yield_units", 0) > 0:
                    if crop in ONGOING_CROPS:
                        scheduled_water = is_maintenance_day(
                            crop, age,
                            tile.get("fertilized_until_day", -1) >= day,
                            None if prioritize_fertilizer_drop else position,
                            tile["planted_day"],
                        )
                        need_water = (
                            not tile.get("watered_today")
                            and (scheduled_water or tile.get("consecutive_unwatered", 0) >= 1)
                        )
                    else:
                        # Preserve the historical one-time-crop cashout bonus.
                        need_water = not tile.get("watered_today")
                    actions = [["WATER"]] if need_water else []
                    actions.append(["HARVEST"])
                    if turns_over:
                        actions.append(["DIG"])
                    tasks.append(Task(
                        position, actions, urgent=True, ends_cycle=turns_over,
                        sells=Counter({crop: tile["yield_units"]}), mandatory=True,
                    ))
                elif turns_over:
                    tasks.append(Task(position, [["DIG"]], ends_cycle=True))
                elif tile.get("consecutive_unwatered", 0) >= 1 and not tile.get("watered_today"):
                    tasks.append(Task(position, [["WATER"]], urgent=True))
            continue
        name, fertilize_commit = target
        x, y = position
        tile = farm["tiles"][y][x]
        actions, needs, sells = [], Counter(), Counter()
        urgent, ends_cycle = False, False
        immediate_drop, refinance_feed = False, False
        live_animal = tile.get("animal") if isinstance(tile, dict) else None

        if live_animal:
            age = day - tile["placed_day"]
            last_age = obs.get("_planning_end_day", SEASON_END_DAY) - tile["placed_day"]
            feed_today = should_feed_animal(live_animal, age, last_age) or (
                day < obs.get("_planning_end_day", SEASON_END_DAY)
                and tile.get("consecutive_unfed", 0) >= 1)
            care_today = should_care_animal(live_animal, age, last_age)
            harvest_ready = should_harvest_animal(
                live_animal, age, tile.get("yield_units", 0),
                force=age == last_age or animal_output_at_risk(tile, day),
            )
            urgent = harvest_ready or feed_today or care_today
            if harvest_ready:
                product = ENV_ANIMALS[live_animal]["product"]
                # Keep all work at one producer in one daily task.  Splitting
                # HARVEST from FEED/CARE allowed two workers to visit the same
                # pen and made the runtime opportunistic override look useful.
                actions.append(["HARVEST"])
                sells[product] += tile["yield_units"]
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
            if (should_fertilize_today(crop, age, fertilize_commit)
                    and tile.get("fertilized_until_day", -1) < day + 2):
                # A committed fertilizer event is a real scheduled input, not
                # an opportunistic action conditional on current shed stock.
                # purchase_orders funds any shortfall before this route runs.
                if fertilizer_left > 0:
                    fertilizer_left -= 1
                needs["FERTILIZER"] += 1
                actions.append(["FERTILIZE"])
            if (is_maintenance_day(
                    crop, age, fertilize_commit,
                    None if prioritize_fertilizer_drop else position,
                    tile["planted_day"],
                ) or tile.get("consecutive_unwatered", 0) >= 1) and not tile.get("watered_today"):
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
                # Opening keeps its historical water-before-harvest rule.
                # Post-opening, equivalent spatial WATER profiles decide the
                # protective day; harvesting cached output alone must not
                # synchronize every ongoing crop back onto the same day.
                if (prioritize_fertilizer_drop and not tile.get("watered_today")
                        and ["WATER"] not in actions):
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
            early_exit = (
                crop not in ONGOING_CROPS and name != crop and tile.get("yield_units", 0) > 0
            )
            if cycle_turns_over_today(crop, age, tile) or early_exit:
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
                    day=day, end_day=obs.get('_planning_end_day', SEASON_END_DAY),
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
                day=day, end_day=obs.get('_planning_end_day', SEASON_END_DAY),
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
                day=day, end_day=obs.get('_planning_end_day', SEASON_END_DAY),
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
                    animal_harvest=bool(live_animal and harvest_ready),
                )
            )
    investment_values = {}
    for task in tasks:
        tile = farm["tiles"][task.position[1]][task.position[0]]
        if task.mandatory is None:
            capped_animal_output = bool(task.animal_harvest and animal_output_at_risk(tile, day))
            starving_animal = bool(
                isinstance(tile, dict) and tile.get("animal")
                and tile.get("consecutive_unfed", 0) >= 1
                and ["FEED"] in task.actions
            )
            physical = isinstance(tile, dict) and bool(
                tile.get("animal") or tile.get("kind") == "PLANT"
            )
            required_ops = {"WATER", "FERTILIZE", "HARVEST", "FEED", "CARE", "DIG"}
            task.mandatory = bool(
                capped_animal_output or starving_animal
                or (physical and any(op[0] in required_ops for op in task.actions))
            )
        task.value = 0.0 if task.mandatory else task_cash_value(obs, task, investment_values)
    if terminal:
        for task in tasks:
            task.terminal_day = True
            # A sale has value only if the worker can bring it home. The
            # economic hire gate now also protects fertilizer-only returns.
            task.cashout = any(amount > 0 for amount in task.sells.values())
    return tasks


def task_cash_value(obs, task, investment_values=None):
    """Conservative current sale quote; investments use remaining net output.

    Route travel/returns are priced by the capacity and marginal hire search.
    New investments use remaining sales minus seeds/animals and future inputs.
    The target planner has already charged forecast labor when choosing them;
    dividing their cash again by visit count would reject profitable starts.
    """
    from agents.forecast import production

    market = obs["market"]
    if investment_values is None:
        investment_values = {}

    def quote(item, amount):
        stock = market["inventory"].get(item, 0)
        cash = 0
        for _ in range(max(0, amount)):
            price = market_price(item, stock, market.get("params"))
            cash += price
            stock += int(price > 1)
        return cash

    value = sum(quote(item, amount) for item, amount in task.sells.items())
    tile = obs["farms"][obs["player"]]["tiles"][task.position[1]][task.position[0]]
    operations = {op[0] for op in task.actions}
    if isinstance(tile, dict) and tile.get("kind") == "PLANT" and "WATER" in operations:
        # A scheduled irrigation protects at least one unit of the crop's
        # remaining yield. Terminal WATER+HARVEST already includes that unit
        # in task.sells, so do not count it twice.
        if "HARVEST" not in operations:
            value += quote(tile["crop"], 1)
    elif isinstance(tile, dict) and tile.get("animal"):
        product = ENV_ANIMALS[tile["animal"]]["product"]
        if "FEED" in operations:
            value += max(0, quote(product, 1) + quote("FERTILIZER", 1)
                         - market["prices"].get("WHEAT", 0))
        if "CARE" in operations:
            value += quote(product, 1)
    for op in task.actions:
        if op[0] not in ("PLANT", "PLACE"):
            continue
        name = op[1]
        if name in investment_values:
            value += investment_values[name]
            continue
        output = production(name, False, obs["day"], obs.get("_planning_end_day", SEASON_END_DAY))
        sales = sum(output.sales.values(), Counter())
        inputs = sum(output.inputs.values(), Counter())
        net = sum(quote(item, amount) for item, amount in sales.items())
        net -= sum(market["prices"].get(item, 0) * amount for item, amount in inputs.items())
        owned_animal = name in ANIMALS and (
            obs["private"]["shed"].get(name, 0)
            or any(inventory.get(name, 0) for inventory in obs["private"]["inventories"])
        )
        net -= SEED_COST[name] if name in CROPS else (0 if owned_animal else ANIMAL_COST[name])
        investment_values[name] = max(0, net)
        value += investment_values[name]
    return value


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
    targets = executable_targets(obs, targets)
    day = obs["day"]
    farm = obs["farms"][obs["player"]]
    shed = obs["private"]["shed"]
    seeds_left = Counter(obs["private"]["seeds"])
    carried = Counter()
    for inventory in obs["private"]["inventories"]:
        carried.update(inventory)

    seed_demand, animal_missing, live_animals = Counter(), Counter(), 0
    terminal = day >= obs.get("_planning_end_day", SEASON_END_DAY)
    wheat_incoming = 0  # WHEAT today's tasks will harvest -- see purchase_orders
    for position in active_positions:
        target = targets.get(position)
        x, y = position
        tile = farm["tiles"][y][x]
        if isinstance(tile, dict) and tile.get("animal"):
            name = tile["animal"]
            if not terminal and not tile.get("fed_today") and (
                day - tile["placed_day"] <= animal_feed_end_age(
                    name, obs.get("_planning_end_day", SEASON_END_DAY) - tile["placed_day"])
                or tile.get("consecutive_unfed", 0) >= 1
            ):
                live_animals += 1
            continue
        if target is None:
            continue
        name, _ = target
        if isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == "WHEAT":
            age = day - tile["planted_day"]
            exits_early = name != "WHEAT" and tile.get("yield_units", 0) > 0
            # Finished WHEAT harvested today is valid coverage for the
            # conservative next-day feed reserve. Without counting it, Day 4
            # spends the last $60 buying redundant feed and leaves one of the
            # seven replacement seeds unfunded until after its PLANT action.
            finished_today = cycle_turns_over_today("WHEAT", age, tile)
            if exits_early or finished_today:
                wheat_incoming += tile.get("yield_units", 0)
        if name in CROPS:
            if not can_start_today(name, obs):
                continue
            if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                seed_demand[name] += 1
            else:
                crop, age = tile["crop"], day - tile["planted_day"]
                if cycle_turns_over_today(crop, age, tile) and (
                    replant_same_crop or name != crop
                ):
                    seed_demand[name] += 1
        elif name in ANIMALS:
            if not terminal and can_start_today(name, obs):
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


def feed_wheat_reserve(obs, targets, active_positions):
    """WHEAT that must remain in the shed for committed feed demand."""
    (
        _seed_demand,
        animal_missing,
        live_animals,
        pending_feed,
        _wheat_incoming,
        carried_wheat,
    ) = _animal_and_seed_demand(obs, targets, active_positions)
    farm = obs["farms"][obs["player"]]
    tomorrow_feed = (
        3 <= obs["day"] < obs.get("_planning_end_day", SEASON_END_DAY)
        and not sum(animal_missing.values())
    ) * sum(
        1
        for position in active_positions
        if isinstance(farm["tiles"][position[1]][position[0]], dict)
        and farm["tiles"][position[1]][position[0]].get("animal")
        and should_feed_animal(
            farm["tiles"][position[1]][position[0]]["animal"],
            obs["day"] + 1 - farm["tiles"][position[1]][position[0]]["placed_day"],
            obs.get("_planning_end_day", SEASON_END_DAY)
            - farm["tiles"][position[1]][position[0]]["placed_day"],
        )
    )
    return max(0, live_animals + pending_feed + tomorrow_feed - carried_wheat)


def feed_wheat_order(obs, targets, active_positions):
    """Just today's WHEAT-for-feed shortfall, safe (and needed) to call every
    turn of the day, not only hour 0/1: COLLECT_FERTILIZER -> return to shed
    -> DROP -> SELL only happens once a task queue actually runs it (hour
    2+), so the cash to fund this same day's feed purchase may not exist
    until partway through the day. purchase_orders' bigger seed/animal
    purchases stay hour-0/1-only (they are not this time-sensitive and
    shouldn't be resubmitted every turn), but wheat is cheap and this check
    is idempotent -- it only ever asks for the current shortfall."""
    shed = obs["private"]["shed"]
    wheat_needed = max(
        0, feed_wheat_reserve(obs, targets, active_positions)
        - shed.get("WHEAT", 0),
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

    # Exact fertilizer demand for committed events that are due today.  This
    # mirrors build_tasks so the market plan can fund FERTILIZE even when the
    # shed starts the morning empty.  Opening targets use False and therefore
    # preserve their sell-fertilizer bootstrap behavior.
    fertilizer_demand = 0
    effective_targets = executable_targets(obs, targets)
    for position in active_positions:
        target = effective_targets.get(position)
        if not target or target[0] not in CROPS:
            continue
        x, y = position
        tile = farm["tiles"][y][x]
        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"
                and tile.get("crop") == target[0]):
            continue
        age = obs["day"] - tile["planted_day"]
        if (should_fertilize_today(tile["crop"], age, target[1])
                and tile.get("fertilized_until_day", -1) < obs["day"] + 2):
            fertilizer_demand += 1
    fertilizer_needed = max(0, fertilizer_demand - shed.get("FERTILIZER", 0))

    # Feed reserve remains deliberately conservative (it may cover a verified
    # schedule gap too): spare wheat is cheap insurance against delayed
    # placement/refinance, while build_tasks decides the exact ages on which
    # FEED and CARE actions actually run.
    wheat_inventory = obs["market"]["inventory"].get("WHEAT", 0)
    tomorrow_feed = (
        3 <= obs["day"] < obs.get("_planning_end_day", SEASON_END_DAY)
        and not sum(animal_missing.values())
    ) * sum(
        1
        for position in active_positions
        if isinstance(farm["tiles"][position[1]][position[0]], dict)
        and farm["tiles"][position[1]][position[0]].get("animal")
        and should_feed_animal(
            farm["tiles"][position[1]][position[0]]["animal"],
            obs["day"] + 1 - farm["tiles"][position[1]][position[0]]["placed_day"],
            obs.get("_planning_end_day", SEASON_END_DAY)
            - farm["tiles"][position[1]][position[0]]["placed_day"],
        )
    )
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

    fertilizer_inventory = obs["market"]["inventory"].get("FERTILIZER", 0)
    fertilizer_cost = sum(
        market_price(
            "FERTILIZER", fertilizer_inventory - unit - 1,
            obs["market"].get("params"),
        )
        for unit in range(fertilizer_needed)
    )
    money -= fertilizer_cost

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
    # From day 1 onward keep cash feed coverage, but credit WHEAT that is
    # guaranteed to arrive from today's admitted harvest before reserving
    # more market cash for that same coverage.
    next_day_feed_reserve = (
        0
        if obs["day"] == 0
        else wheat_cost(max(
            0,
            live_animals + pending_feed + new_animal_feed - wheat_incoming,
        ))
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
        - wheat_on_hand - wheat_incoming,
    )

    orders = []
    if wheat_needed:
        orders.append(["BUY_PRODUCT", "WHEAT", wheat_needed])
    if fertilizer_needed:
        orders.append(["BUY_PRODUCT", "FERTILIZER", fertilizer_needed])
    orders += animal_orders + seed_orders
    return orders
