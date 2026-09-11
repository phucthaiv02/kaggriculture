"""Function 5: wire functions 1-4 into one per-turn agent.

Morning planning establishes maintenance routes and the opening portfolio.
Each later observation also discovers newly unlocked land and newly vacant
tiles. Intraday scheduling preserves committed routes, replants in place
where possible, and uses remaining time and cash for additional work/hires.

Hour 0 is unavoidably orders-only: movement is applied *before* market orders
each turn (see kaggriculture's interpreter), so at hour 0 (a) a hand just
requested via HIRE does not exist yet to move, and (b) any BUY_SEED/
BUY_ANIMAL/BUY_PRODUCT submitted this same hour has not landed in the shed
yet either -- building today's task list from that stale, pre-purchase shed
would find nothing to do (an animal target with 0 animals in the shed is not
a task). So tasks/queues are built starting hour 1, once hour 0's orders
have actually taken effect; that is also the earliest a freshly hired hand
can move, so nothing is lost by waiting for it. hour 1 also gets its own
market submission as a free second attempt at anything hour 0's 10-order cap
truncated.
"""

from __future__ import annotations

from collections import Counter

from kaggle_environments.envs.kaggriculture.kaggriculture import LAND_PRICES, market_price

from agents.farm_tasks import (
    ANIMAL_COST,
    ANIMALS,
    CROPS,
    SEED_COST,
    build_tasks,
    feed_wheat_order,
    purchase_orders,
    reserved_items,
)
from agents.opening_book import make_opening_controller, should_buy_land_on_schedule
from agents.intraday import queue_commitments, schedule_open_tiles, schedule_idle_drops
from agents.planner import SEASON_END_DAY, plan_targets
from agents.horizon import can_start_today
from agents.scheduler import MAX_HANDS, build_queues, hands_needed
from agents.schedules import is_maintenance_day, should_care_animal, should_feed_animal
from agents.maintenance import should_feed, should_care, should_water
from agents.selling import sell_orders, update_selling_state


def _investment_sales(obs, reservations, selling_state, targets, hires):
    """Release held crops when cash cannot cover currently planned inputs."""
    sales = sell_orders(obs, reservations, selling_state=selling_state)
    liquidated = sell_orders(
        obs, reservations, selling_state=selling_state, needs_investment=True,
    )
    if sales == liquidated:
        return sales
    farm = obs["farms"][obs["player"]]
    # A finite, nonbinding budget asks the existing purchase planner for
    # demand before its affordability caps (it converts seed budgets to int).
    orders = purchase_orders(
        obs, targets, _active_positions(farm), available_money=10**12,
        replant_same_crop=True,
    )
    cost = _hire_costs(farm, hires)
    if should_buy_land_on_schedule(obs, farm):
        cost += LAND_PRICES[len(farm["unlocked_quadrants"]) - 1]
    for op, item, quantity in orders:
        if op == "BUY_SEED":
            cost += SEED_COST[item] * quantity
        elif op == "BUY_ANIMAL":
            cost += ANIMAL_COST[item] * quantity
        elif op == "BUY_PRODUCT":
            stock = obs["market"]["inventory"].get(item, 0)
            cost += sum(market_price(item, stock - unit - 1, obs["market"].get("params"))
                        for unit in range(quantity))
    if cost > farm["money"] + _sale_revenue(obs, sales):
        sales = liquidated
    return sales

def _protect_animal_structures(farm, operations):
    """Discard stale DIG/PLANT commands on permanent animal structures."""
    operations = list(operations)
    positions = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
    protected = {p for p in positions
                 if isinstance(farm["tiles"][p[1]][p[0]], dict)
                 and (farm["tiles"][p[1]][p[0]].get("animal")
                      or farm["tiles"][p[1]][p[0]].get("kind") in ("COOP", "PASTURE"))}
    for index, position in enumerate(positions[:len(operations)]):
        operation = operations[index]
        if operation and operation[0] in ("BUILD_COOP", "BUILD_PASTURE", "PLACE"):
            protected.add(position)
        if operation and operation[0] in ("DIG", "PLANT") and position in protected:
            operations[index] = ["PASS"]
    return operations


BOARD_SIZE = 10


def _active_positions(farm):
    tiles = farm["tiles"]
    return [(x, y) for y in range(BOARD_SIZE) for x in range(BOARD_SIZE) if tiles[y][x] != "LOCKED"]


def _open_shed_access(farm):
    half = BOARD_SIZE // 2
    positions = ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))
    return tuple((x, y) for x, y in positions if farm["tiles"][y][x] != "LOCKED")


def _hire_costs(farm, number):
    """Exact Fibonacci cost of `number` additional hires this day."""
    a, b = 1, 1
    costs = []
    for _ in range(farm.get("hires_today", 0) + number):
        costs.append(a)
        a, b = b, a + b
    return sum(costs[farm.get("hires_today", 0):])


def _affordable_hires(farm, desired, money):
    """How many of `desired` sequential HIRE orders can really execute."""
    affordable = 0
    for number in range(1, desired + 1):
        if _hire_costs(farm, number) > money:
            break
        affordable = number
    return affordable


def _is_protective_task(task):
    """Work that must survive capacity cuts before expansion/replacement."""
    if task.urgent or task.animal_harvest:
        return True
    return any(action and action[0] == "HARVEST" for action in task.actions)


def _capacity_safe_queues(
    tasks, farmer_start, hand_count, existing_hands, shed_access,
    pending_hand_budget=22, existing_hand_budget=23,
):
    """Pack against the workers we can actually afford.

    Try the complete workload first. If any protective work is unassigned,
    discard expansion/replacement work for this morning and repack only
    protective work. Only admitted tasks may reserve inventory or suppress
    intraday rescue.
    """
    kwargs = dict(
        pending_hand_budget=pending_hand_budget,
        existing_hand_budget=existing_hand_budget,
    )
    plans, unassigned = build_queues(
        tasks, farmer_start, hand_count, existing_hands, shed_access, **kwargs
    )
    protective_debt = [task for task in unassigned if _is_protective_task(task)]
    active_tasks = tasks
    if protective_debt:
        active_tasks = [task for task in tasks if _is_protective_task(task)]
        plans, unassigned = build_queues(
            active_tasks, farmer_start, hand_count, existing_hands, shed_access, **kwargs
        )
        protective_debt = list(unassigned)
    unassigned_ids = {id(task) for task in unassigned}
    admitted = [task for task in active_tasks if id(task) not in unassigned_ids]
    return plans, admitted, unassigned, protective_debt


def _maintenance_hires(obs, targets, plans, shed_access):
    """Fund missing protective routes once intraday proceeds become available."""
    farm = obs["farms"][obs["player"]]
    positions = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
    endpoints, committed, _, _ = queue_commitments(positions, plans, max_steps=24 - obs["hour"])
    pending = {position: target for position, target in targets.items()
               if position not in committed}
    tasks = [task for task in build_tasks(obs, pending, prioritize_fertilizer_drop=True)
             if _is_protective_task(task)]
    if not tasks:
        return 0
    remaining = 24 - obs["hour"]
    budgets = [max(0, remaining - len(plans[i].queue)) for i in range(len(positions))]
    # Reserve immediate feed before optional hires; the four-day stock top-up
    # must not consume money needed to prevent today's crops from dying.
    feed = sum(task.needs.get("WHEAT", 0) for task in tasks)
    shortfall = max(0, feed - obs["private"]["shed"].get("WHEAT", 0))
    cash = max(0, farm["money"] - shortfall * obs["market"]["prices"]["WHEAT"])
    limit = _affordable_hires(farm, MAX_HANDS - len(farm["hands"]), cash)
    best, debt = 0, None
    for extra in range(limit + 1):
        _, missing = build_queues(
            tasks, endpoints[0], len(farm["hands"]) + extra,
            tuple(endpoints[1:len(positions)]), shed_access,
            worker_budgets=budgets + [max(0, remaining - 1)] * extra,
        )
        if debt is None or len(missing) < debt:
            best, debt = extra, len(missing)
        if not missing:
            break
    return best


def _sale_revenue(obs, sales):
    """Exact same-turn proceeds before subsequent BUY orders execute."""
    inventory = dict(obs["market"]["inventory"])
    params = obs["market"].get("params")
    revenue = 0
    for _op, item, quantity in sales:
        for _ in range(int(quantity)):
            stock = inventory.get(item, 0)
            price = market_price(item, stock, params)
            revenue += price
            inventory[item] = stock + int(price > 1)
    return revenue


def _maximum_cash_after_sales(obs, farm, sales, hires):
    """Spendable cash after every preceding SELL and HIRE is processed."""
    return max(
        0,
        farm["money"] + _sale_revenue(obs, sales) - _hire_costs(farm, hires),
    )


def _reserve_feed_for_affordable_animals(obs, farm, targets, reservations, sales, hires):
    """Keep shed WHEAT for animals the same market pass can actually buy."""
    reserve = dict(reservations)
    wheat_free = max(0, obs["private"]["shed"].get("WHEAT", 0) - reserve.get("WHEAT", 0))
    if not wheat_free:
        return reserve

    carried = {name: 0 for name in ANIMALS}
    for inventory in obs["private"]["inventories"]:
        for name in ANIMALS:
            carried[name] += inventory.get(name, 0)
    missing = {name: 0 for name in ANIMALS}
    tiles = farm["tiles"]
    for (x, y), target in targets.items():
        if not target or target[0] not in ANIMALS:
            continue
        name = target[0]
        tile = tiles[y][x]
        if not (isinstance(tile, dict) and tile.get("animal") == name):
            missing[name] += 1
    for name in ANIMALS:
        missing[name] = max(
            0,
            missing[name] - obs["private"]["shed"].get(name, 0) - carried[name],
        )

    money = _maximum_cash_after_sales(obs, farm, sales, hires)
    extra_feed = 0
    for name in sorted(ANIMALS, key=ANIMAL_COST.get):
        for _ in range(missing[name]):
            if extra_feed >= wheat_free or money < ANIMAL_COST[name]:
                break
            money -= ANIMAL_COST[name]
            extra_feed += 1
    reserve["WHEAT"] = reserve.get("WHEAT", 0) + extra_feed
    return reserve


def _hire_and_buy_orders(
    obs,
    farm,
    targets,
    hand_target,
    pending_sales=(),
    buy_inputs=True,
    replant_same_crop=False,
    reserve_hire_budget=True,
):
    hires = max(0, hand_target - len(farm["hands"]))
    hire_orders = [["HIRE"] for _ in range(hires)]
    purchase = []
    if buy_inputs:
        available_money = _maximum_cash_after_sales(
            obs, farm, pending_sales, hires if reserve_hire_budget else 0
        )
        wheat_sold = sum(
            int(order[2]) for order in pending_sales
            if order[0] == "SELL" and order[1] == "WHEAT"
        )
        purchase = purchase_orders(
            obs,
            targets,
            _active_positions(farm),
            available_money=available_money,
            available_wheat=max(0, obs["private"]["shed"].get("WHEAT", 0) - wheat_sold),
            replant_same_crop=replant_same_crop,
        )
    # Inputs that make today's precomputed tile tasks executable must land
    # before optional labor spending.  Reserving the cost of every desired
    # HIRE here could reduce seed affordability to zero, leaving those hands
    # with no PLANT work to execute.
    feed = [order for order in purchase if order[0] == "BUY_PRODUCT"]
    animals = [order for order in purchase if order[0] == "BUY_ANIMAL"]
    seeds = [order for order in purchase if order[0] == "BUY_SEED"]
    if reserve_hire_budget:
        # Outside the opening, protect survival capacity before committing
        # capital to new production.
        orders = feed + hire_orders + animals + seeds
    else:
        # The opening's fixed portfolio depends on same-day seeds/animals;
        # buying these before optional labor prevents a queued PLANT no-op.
        orders = feed + animals + seeds + hire_orders
    if should_buy_land_on_schedule(obs, farm):
        # This debug purchase must claim its cash before optional hires and
        # inputs; putting it last can submit the order yet leave it unfunded.
        orders.insert(0, ["BUY_LAND"])
    return orders


def make_agent(
    end_day=SEASON_END_DAY, seed=0, opening_version="classic",
    planner_version="concentration",
):
    del seed  # every decision reacts to live prices/shed state; nothing to seed
    targets = {}
    state = {
        "day": -1, "hand_target": 0, "plans": [], "reserved": {},
        "opening_active": False, "scheduled_placements": set(),
        "unverified_hand_indices": set(),
        "deferred_expansion_positions": set(),
        "protective_debt_positions": set(),
    }
    opening_governs = make_opening_controller(opening_version)
    selling_state = {}

    def _validate_predicted_hands(farm, hour):
        """Discard a plan built for a hand hired mid-day once that hand
        actually spawns somewhere other than predicted.

        A hand hired at hour K exists starting hour K+1 (agents/scheduler.py
        predicts its spawn tile via the engine's NWSE/least-occupied rule),
        but that rule runs against occupancy *after* hour K's own movement
        -- which, for the hands already real at hour K, is whatever their
        just-built queue does that hour, not their pre-move hour-K position
        the prediction was necessarily computed from. A single top-up hand
        is close to always right; several at once (the case a big land
        purchase can trigger via plan_targets, see hour 0 above) raises the
        odds one guess is wrong. Every step downstream of a wrong start is
        silently wrong too -- not just the first one -- since the whole
        route was built as a fixed offset from that assumed tile, so an
        undetected miss can waste that worker's entire day and drop
        whatever survival work it was carrying (e.g. an animal's FEED).
        Clearing the queue here makes the mismatch immediately visible as
        an idle worker instead, which _schedule_late_placements can then
        route correctly from the hand's real position.
        """
        hands = tuple(map(tuple, farm["hands"]))
        still_unverified = set()
        for index in state["unverified_hand_indices"]:
            hand_index = index - 1
            if hand_index >= len(hands):
                if hour == 1:
                    still_unverified.add(index)
                continue
            plan = state["plans"][index] if index < len(state["plans"]) else None
            if plan is not None and plan.start != hands[hand_index]:
                plan.queue = []
        state["unverified_hand_indices"] = still_unverified
        if hour >= 2:
            # Morning HIRE orders have resolved. Failed/truncated purchases
            # must release their tasks and inputs for intraday scheduling.
            del state["plans"][len(hands) + 1:]

    def _animal_still_needs_attention(tile, name, day, position):
        """True if a *placed* animal still has real, unfulfilled survival
        work today (feed, care, an overdue harvest, or waiting fertilizer).

        Re-deriving this via a real (non-assumed) build_tasks call is always
        safe to repeat even if another worker's queue also happens to cover
        it -- FEED/CARE/HARVEST are each a same-day no-op once done (see
        kaggriculture's rules), unlike a fresh PLANT/PLACE, which would
        double-consume a seed or animal if two workers both attempted it.
        That asymmetry is why only the survival case is rescued here for
        every idle worker, not first-time plantings.
        """
        age = day - tile["placed_day"]
        harvest_ready = tile.get("yield_units", 0) > 0
        needs_feed = (should_feed(name, age, position) or tile.get("consecutive_unfed", 0)) and not tile.get("fed_today")
        needs_care = (
            should_care(name, age, position)
            and tile.get("fed_today")
            and not tile.get("cared_today")
        )
        return harvest_ready or needs_feed or needs_care

    def _crop_still_needs_water(tile, name, fertilize_commit, day, position):
        """True if a *planted* crop still needs today's WATER and hasn't
        gotten it yet.

        Same idempotent-action reasoning as _animal_still_needs_attention:
        WATER is a same-day no-op once done, so re-deriving and re-queuing
        it here is safe even if the tile's real hour-1 task also queued it
        (unlike a fresh PLANT, which would double-spend a seed -- tried
        rescuing a cycle-finished tile's harvest+replant transition here
        too, regardless of whether the tile's current crop still matched
        its target, and reverted it: without any way to tell "another
        worker's queue already has this, just hasn't reached it yet" apart
        from "already done", a second idle worker could re-attempt the same
        transition while the first was still correctly en route, and the
        duplicate PICKUP could take the last seed before the original
        worker got there -- confirmed directly, seed 1: exactly one of that
        day's routine WHEAT replantings came up short this way). Missing
        WATER isn't a delayed harvest like a missed animal FEED -- two
        consecutive misses turn the tile into an unrecoverable WEED (see
        kaggriculture's rules) -- so a crop whose main-queue WATER got
        dropped for lack of hands had no way back before this: confirmed
        directly (seed 1), a steady trickle of WHEAT/MELON tiles turned to
        WEED over the game specifically from repeated missed waterings,
        with no rescue path analogous to the animal one above.
        """
        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
            return False
        if tile.get("watered_today"):
            return False
        crop = tile["crop"]
        age = day - tile["planted_day"]
        current_fertilized = (
            fertilize_commit
            if crop == name
            else tile.get("fertilized_until_day", -1) >= tile["planted_day"]
        )
        return bool(tile.get("consecutive_unwatered", 0)) or should_water(crop, age, current_fertilized, position)

    def _replan_all_from_actual(obs, farm, day, hour):
        """Rebuild routes after morning hires or opening input deliveries.

        The Day-4 SHEEP arrives after fertilizer sales. Repack the remaining
        maintenance and its now-executable PLACE using existing workers;
        morning labor sizing never assumes this purchase has arrived.
        Repack same-day harvest work once replacement seeds arrive too, so
        harvesters can replant during the same visit when the planner retains
        a WHEAT target.
        """
        delivered = (
            opening_version == "melon_v2" and state["opening_active"] and day == 3
            and obs["private"]["shed"].get("SHEEP", 0) > 0
            and not state.get("opening_sheep_delivered")
        )
        replant_delivered = (
            opening_version == "melon_v2" and state["opening_active"]
            and day == 4 and obs["private"]["seeds"].get("WHEAT", 0) >= sum(
                target is not None and target[0] == "WHEAT"
                for target in targets.values()
            )
            and not state.get("opening_replant_delivered")
        )
        if hour != 2 and not delivered and not replant_delivered:
            return
        if replant_delivered:
            state["opening_replant_delivered"] = True
        if delivered:
            state["opening_sheep_delivered"] = True
        tasks = build_tasks(
            obs, targets, prioritize_fertilizer_drop=True
        )
        state["deferred_expansion_positions"] = {
            position for position in state["deferred_expansion_positions"]
            if not isinstance(farm["tiles"][position[1]][position[0]], dict)
        }
        tasks = [
            task for task in tasks
            if task.position not in state["deferred_expansion_positions"]
        ]
        actual_hands = tuple(map(tuple, farm["hands"]))
        remaining = 24 - hour
        plans, admitted, unassigned, protective_debt = _capacity_safe_queues(
            tasks, tuple(farm["farmer"]), len(actual_hands), actual_hands,
            _open_shed_access(farm), pending_hand_budget=remaining,
            existing_hand_budget=remaining,
        )
        state["plans"] = plans
        state["reserved"] = reserved_items(admitted)
        state["protective_debt_positions"] = {task.position for task in protective_debt}
        state["scheduled_placements"] = {
            task.position for task in admitted
            if any(action[0] in ("PLANT", "PLACE") for action in task.actions)
        }
        state["unverified_hand_indices"] = set()

    def _schedule_late_placements(obs, farm, day, hour):
        """Use newly bought animals, and rescue any live animal whose
        FEED/CARE/HARVEST fell through, once the morning queue has gone
        idle.

        A worker's whole day is planned in one shot at hour 1 from a
        *predicted* spawn tile for any hand hired that same hour (see
        agents/scheduler.py's predicted_hand_starts) -- an approximation
        that can miss once several hands are hired within the same hour (a
        big land purchase is exactly when agents/planner.py's plan_targets
        can suddenly need many more hands than a normal day), silently
        routing that worker's queued FEED/CARE to the wrong tile all day.
        Re-scanning every already-placed animal for unmet survival needs
        here -- not just not-yet-placed ones -- gives some other worker
        that goes idle later the same day a real chance to still catch it,
        rather than losing the animal to a missed feeding.
        """
        if hour < 2 or not state["plans"]:
            return
        worker_positions = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
        idle_indices = [
            index for index, plan in enumerate(state["plans"])
            if not plan.queue and index < len(worker_positions)
        ]
        pending_targets = {}
        state["scheduled_placements"] = {
            position for position in state["scheduled_placements"]
            if not (
                isinstance(farm["tiles"][position[1]][position[0]], dict)
                and farm["tiles"][position[1]][position[0]].get("animal")
                == (targets.get(position) or (None, False))[0]
            )
        }
        # Injected harvests and input retries can push a queued WATER past
        # dusk. Such work is no longer covered and must be rescued now.
        committed = queue_commitments(
            worker_positions, state["plans"], max_steps=24 - hour
        )[1]
        ordered_positions = sorted(
            targets,
            key=lambda position: (
                position not in state["protective_debt_positions"],
                position[1], position[0],
            ),
        )
        for position in ordered_positions:
            target = targets[position]
            if not target:
                continue
            if position in committed:
                continue
            x, y = position
            tile = farm["tiles"][y][x]
            name, fertilize_commit = target
            if name in ANIMALS:
                placed = isinstance(tile, dict) and tile.get("animal") == name
                if not placed:
                    if position not in state["scheduled_placements"]:
                        pending_targets[position] = target
                elif _animal_still_needs_attention(tile, name, day, position):
                    pending_targets[position] = target
            elif (
                position in state["deferred_expansion_positions"]
                and not isinstance(tile, dict)
                and position not in state["scheduled_placements"]
            ):
                pending_targets[position] = target
            elif _crop_still_needs_water(tile, name, fertilize_commit, day, position):
                pending_targets[position] = target
        if not pending_targets:
            return
        tasks = build_tasks(
            obs,
            pending_targets,
            prioritize_fertilizer_drop=True,
        )
        placement_tasks = [
            task for task in tasks
            if any(
                action[0] in ("PLANT", "PLACE", "FEED", "CARE", "HARVEST", "COLLECT_FERTILIZER", "WATER")
                for action in task.actions
            )
        ]
        if not placement_tasks:
            return
        shed_access = _open_shed_access(farm)
        remaining = 24 - hour
        if idle_indices:
            # Every idle worker here already has a real position (it's
            # mid-day, not the hour-0 hiring pass), so build_queues would
            # otherwise budget each one a *full* day (FARMER_BUDGET/
            # HAND_BUDGET) instead of the `remaining` turns actually left --
            # pending_hand_budget alone never reaches them, since none of
            # them fall past `hand_starts`' length. existing_hand_budget
            # overrides that. Confirmed directly (seed 1, day 23): without
            # it, two workers were routed 17-18 steps long against 16 real
            # turns left, for tasks including a SHEEP already a day overdue.
            plans, unassigned = build_queues(
                placement_tasks,
                worker_positions[idle_indices[0]],
                len(idle_indices) - 1,
                tuple(worker_positions[index] for index in idle_indices[1:]),
                shed_access,
                pending_hand_budget=remaining,
                existing_hand_budget=remaining,
            )
            # build_queues/_pack only ever inserts a task into a bucket once
            # its own budget check confirms it fits (now that
            # existing_hand_budget makes that check accurate -- see above),
            # so `plans` is always safe to apply as-is. Requiring the whole
            # batch to have zero `unassigned` before applying *any* of it
            # used to throw away every position that DID fit -- confirmed
            # directly (seed 1, day 23): with 11-12 animals pending and only
            # 10 idle workers, 1-2 positions genuinely had nowhere to fit
            # every single hour from 2 through 8, discarding the other 9-10
            # (a SHEEP a day overdue among them) purely to protect those 1-2
            # from being *skipped this hour* -- they simply retry on the
            # next idle check instead, same as any task that was never
            # pending_targets in the first place.
            unassigned_positions = {task.position for task in unassigned}
            for index, plan in zip(idle_indices, plans):
                state["plans"][index] = plan
            state["scheduled_placements"].update(
                task.position for task in placement_tasks
                if task.position not in unassigned_positions
                and any(action[0] in ("PLANT", "PLACE") for action in task.actions)
            )
            state["reserved"] = reserved_items(placement_tasks)
            return

        return

    def agent(obs, configuration=None):
        effective_end = end_day
        if configuration is not None:
            # The final observation is terminal; no action can run from it.
            last_action_day = (int(configuration['episodeSteps']) - 2) // int(configuration.get('turnsPerDay', 24))
            effective_end = min(effective_end, last_action_day)
        obs = dict(obs, _planning_end_day=effective_end)
        day, hour = obs["day"], obs["hour"]
        if opening_version == "melon_v2" and day == 4:
            # The day-5 planner needs animal fertilizer as working capital for
            # replacement seeds. Collect and bank it before harvest/transition
            # routes consume the remaining turns.
            obs["_liquidate_fertilizer_first"] = True
        farm = obs["farms"][obs["player"]]

        if state["opening_active"]:
            opening_governs(obs, targets, _active_positions(farm))

        update_selling_state(obs, selling_state)

        # React to every successful land purchase, including one submitted
        # mid-day or after the opening has already handed off.
        if hour > 0 and any(position not in targets for position in _active_positions(farm)):
            positions = _active_positions(farm)
            state["deferred_expansion_positions"].update(
                position for position in positions if position not in targets
            )
            state["opening_active"] = opening_governs(obs, targets, positions)
            if not state["opening_active"]:
                plan_targets(
                    obs, targets, positions, effective_end, planner_version
                )

        if hour == 0:
            positions = _active_positions(farm)
            # The hand-specified NW opening (agents/opening_book.py) governs
            # targets in place until the first BUY_LAND succeeds; from then
            # on the ROI planner takes over.
            state["opening_active"] = opening_governs(obs, targets, positions)
            if not state["opening_active"]:
                plan_targets(
                    obs, targets, positions, effective_end, planner_version
                )
            else:
                # The opening book only ever assigns the original NW tiles
                # (agents/opening_book.py's build_opening_targets). Without
                # this, a quadrant bought on LAND_BUY_DAYS would stay
                # target-less -- and therefore untouched by build_tasks --
                # until the first-land handoff. Route brand-new tiles through the
                # same ROI planner (profit/day, crop vs animal, priced off
                # today's market) the instant they unlock instead, without
                # touching the opening book's own NW allocation.
                new_positions = [p for p in positions if p not in targets]
                if new_positions:
                    plan_targets(
                        obs, targets, new_positions, effective_end,
                        planner_version,
                    )
            tasks = build_tasks(
                obs, targets,
                assume_crop_seeds=day > 0,
                assume_animal_inputs=state["opening_active"] and (opening_version != "melon_v2" or day == 0),
                prioritize_fertilizer_drop=True,
            )  # include the PLANT + WATER enabled by today's seed purchases
            tasks = [
                task for task in tasks
                if task.position not in state["deferred_expansion_positions"]
                or isinstance(farm["tiles"][task.position[1]][task.position[0]], dict)
            ]
            hand_target, _dropped = hands_needed(
                tasks,
                tuple(farm["farmer"]),
                tuple(map(tuple, farm["hands"])),
                _open_shed_access(farm),
            )
            state.update(
                day=day, hand_target=hand_target, plans=[], reserved={},
                scheduled_placements=set(), unverified_hand_indices=set(),
            )
            reservations = reserved_items(tasks)
            # First value every non-WHEAT sale, then decide how much WHEAT
            # must stay in the shed for animals that cash can fund today.
            non_wheat_reservations = dict(reservations)
            non_wheat_reservations["WHEAT"] = obs["private"]["shed"].get("WHEAT", 0)
            hires = max(0, hand_target - len(farm["hands"]))
            non_wheat_sales = _investment_sales(obs, non_wheat_reservations, selling_state, targets, hires)
            reservations = _reserve_feed_for_affordable_animals(
                obs, farm, targets, reservations, non_wheat_sales, hires
            )
            sales = _investment_sales(obs, reservations, selling_state, targets, hires)
            return {
                "farmer": ["PASS"], "hands": [["PASS"] for _ in farm["hands"]],
                "market": (
                    sales + _hire_and_buy_orders(
                        obs, farm, targets, hand_target, pending_sales=sales,
                        replant_same_crop=True,
                        reserve_hire_budget=not state["opening_active"],
                    )
                )[:10],
            }

        if state["day"] == day and not state["plans"]:
            # Hour-0 planner purchases have already resolved by hour 1.
            # Build execution work from inventory that actually reached the
            # shed; never invent a seed here and make a worker wait on PLANT.
            tasks = build_tasks(
                obs,
                targets,
                prioritize_fertilizer_drop=True,
            )
            state["deferred_expansion_positions"] = {
                position for position in state["deferred_expansion_positions"]
                if not isinstance(farm["tiles"][position[1]][position[0]], dict)
            }
            tasks = [
                task for task in tasks
                if task.position not in state["deferred_expansion_positions"]
            ]
            existing_hands = tuple(map(tuple, farm["hands"]))
            shed_access = _open_shed_access(farm)
            hand_count, _dropped = hands_needed(
                tasks,
                tuple(farm["farmer"]),
                existing_hands,
                shed_access,
                pending_hand_budget=22,
            )
            missing_hands = max(0, hand_count - len(existing_hands))
            affordable_hands = _affordable_hires(
                farm, missing_hands, farm["money"]
            )
            hand_count = len(existing_hands) + affordable_hands
            state["hand_target"] = hand_count
            plans, admitted_tasks, _unassigned, protective_debt = _capacity_safe_queues(
                tasks,
                tuple(farm["farmer"]),
                hand_count,
                existing_hands,
                shed_access,
                pending_hand_budget=22,
                existing_hand_budget=23,
            )
            state["plans"], state["reserved"] = plans, reserved_items(admitted_tasks)
            state["protective_debt_positions"] = {task.position for task in protective_debt}
            # Indices beyond today's already-real hands got a route built
            # from predicted_hand_starts' *guess* at where they'll spawn --
            # see _validate_predicted_hands below for why that guess can be
            # wrong, and what happens once it's checked against reality.
            state["unverified_hand_indices"] = set(
                range(len(existing_hands) + 1, hand_count + 1)
            )
            state["scheduled_placements"] = {
                task.position
                for task in admitted_tasks
                if any(action[0] in ("PLANT", "PLACE") for action in task.actions)
            }

        if state["unverified_hand_indices"]:
            _validate_predicted_hands(farm, hour)
        _replan_all_from_actual(obs, farm, day, hour)
        # Morning hires already have pending orders/plans. Extra hires here
        # are only for newly available work after that pass has settled.
        hire_costs = (
            [_hire_costs(farm, count) for count in range(1, MAX_HANDS - len(farm["hands"]) + 1)]
            if hour >= 2 and not state["unverified_hand_indices"]
            and not (state["opening_active"] and opening_version == "melon_v2" and day < 4) else []
        )
        purchase_targets, expansion_hires = schedule_open_tiles(
            obs, targets, state["plans"], _open_shed_access(farm), hire_costs,
        )
        if state["opening_active"] and opening_version == "melon_v2":
            # Scheduled animals remain purchase commitments while workers
            # finish real maintenance; do not drop demand just because their
            # pasture has not yet been admitted to a route.
            purchase_targets = targets
        _schedule_late_placements(obs, farm, day, hour)
        maintenance_hires = 0
        if hour >= 2 and not (state["opening_active"] and opening_version == "melon_v2" and day < 4):
            maintenance_hires = _maintenance_hires(
                obs, targets, state["plans"], _open_shed_access(farm)
            )
            expansion_hires = max(expansion_hires, maintenance_hires)
        schedule_idle_drops(obs, state["plans"], _open_shed_access(farm))
        # Remaining PICKUPs, including newly appended work, must be protected
        # from the same turn's sales.
        positions = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
        state["reserved"] = queue_commitments(positions, state["plans"])[3]

        plans = state["plans"]
        farmer_plan = plans[0] if plans else None
        farmer_op = farmer_plan.queue.pop(0) if farmer_plan and farmer_plan.queue else ["PASS"]
        hand_ops = []
        for index in range(len(farm["hands"])):
            plan = plans[index + 1] if index + 1 < len(plans) else None
            hand_ops.append(plan.queue.pop(0) if plan and plan.queue else ["PASS"])

        # A planned seed purchase may land later in the day after fertilizer
        # is sold. Never consume a queued PLANT as a no-op before its real
        # seed exists; hold it at the front of that worker's queue instead.
        worker_ops = [farmer_op, *hand_ops]
        seeds_available = dict(obs["private"]["seeds"])
        shed_available = Counter(obs["private"]["shed"])
        for index, operation in enumerate(worker_ops):
            if operation and operation[0] in ('PLANT', 'PLACE') and not can_start_today(operation[1], obs):
                worker_ops[index] = ['PASS']
                if index < len(plans):
                    queue = plans[index].queue
                    # Cancel care for the rejected new producer, preserving
                    # the worker's route and other tiles' harvest tasks.
                    while queue and queue[0][0] in ('WATER', 'FERTILIZE', 'FEED', 'CARE'):
                        queue.pop(0)
                continue
            if operation and operation[0] == "PICKUP":
                item = operation[1]
                requested = operation[2] if len(operation) > 2 else 1
                taken = min(requested, shed_available[item])
                shed_available[item] -= taken
                if taken < requested:
                    plan = plans[index] if index < len(plans) else None
                    if plan is not None:
                        plan.queue.insert(0, ["PICKUP", item, requested - taken])
                    worker_ops[index] = ["PICKUP", item, taken] if taken else ["PASS"]
                continue
            if not operation or operation[0] != "PLANT":
                continue
            crop = operation[1]
            if seeds_available.get(crop, 0) > 0:
                seeds_available[crop] -= 1
                continue
            # A missing seed means the market order did not actually land
            # (cash/order-cap race, etc.). Do not pin this worker behind a
            # retrying PLANT: discard the planting attempt and its immediate
            # crop-care tail so later WATER/HARVEST work can still execute.
            plan = plans[index] if index < len(plans) else None
            if plan is not None:
                while plan.queue and plan.queue[0][0] in ("WATER", "FERTILIZE"):
                    plan.queue.pop(0)
            worker_ops[index] = ["PASS"]
        farmer_op, hand_ops = worker_ops[0], worker_ops[1:]

        # Hard real-time rule: if a worker is already standing on ready
        # animal output, HARVEST now. Put its queued operation back instead
        # of discarding it, so FEED/CARE/WATER still executes next turn.
        worker_ops = [farmer_op, *hand_ops]
        worker_positions = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
        harvested_positions = set()
        for index, position in enumerate(worker_positions[:len(worker_ops)]):
            x, y = position
            tile = farm["tiles"][y][x]
            ready_animal = (
                isinstance(tile, dict)
                and tile.get("animal")
                and tile.get("yield_units", 0) > 0
            )
            if not ready_animal or position in harvested_positions:
                continue
            displaced = worker_ops[index]
            if displaced != ["HARVEST"]:
                plan = plans[index] if index < len(plans) else None
                if plan is not None and displaced != ["PASS"]:
                    plan.queue.insert(0, displaced)
                worker_ops[index] = ["HARVEST"]
            harvested_positions.add(position)

        worker_ops = _protect_animal_structures(farm, worker_ops)

        farmer_op, hand_ops = worker_ops[0], worker_ops[1:]

        # Sell and buy every turn so harvests and fertilizer can fund new
        # production immediately. Vacant-tile purchases are restricted to
        # the capacity admitted by the intraday scheduler.
        market = _investment_sales(
            obs, state["reserved"], selling_state, purchase_targets,
            expansion_hires + (max(0, state["hand_target"] - len(farm["hands"])) if hour == 1 else 0),
        )
        if hour == 1:
            # Morning top-up hires retain their existing sizing. Input
            # shopping below also covers feasible newly unlocked tiles.
            market += _hire_and_buy_orders(
                obs,
                farm,
                targets,
                state["hand_target"],
                buy_inputs=False,
            )
        else:
            market += feed_wheat_order(obs, targets, _active_positions(farm))

        # Fertilizer dropped on the previous turn is sold first. Its exact
        # proceeds are therefore available to BUY_ANIMAL later in this same
        # market pass; the animal is observed and scheduled for PLACE on a
        # subsequent turn once the morning work queues are idle.
        if hour >= 1:
            wheat_sold = sum(
                int(order[2]) for order in market
                if order[0] == "SELL" and order[1] == "WHEAT"
            )
            # Only SELL orders add cash.  ``market`` can also contain a
            # BUY_PRODUCT feed order; treating that as sale revenue would
            # make an animal look affordable before it really is.
            sales = [order for order in market if order[0] == "SELL"]
            available_money = _maximum_cash_after_sales(
                obs, farm, sales, expansion_hires + sum(order[0] == "HIRE" for order in market)
            )
            order_targets = purchase_targets
            if opening_version == "melon_v2" and day == 4:
                # Day-5 fertilizer is scarce working capital with only a few
                # turns left after delivery. Fund cheap crop targets first so
                # workers can PLANT today; animal targets remain for day 6.
                order_targets = {
                    position: target
                    for position, target in purchase_targets.items()
                    if target and target[0] in CROPS
                }
            late_orders = purchase_orders(
                obs,
                order_targets,
                _active_positions(farm),
                available_money=available_money,
                available_wheat=max(
                    0, obs["private"]["shed"].get("WHEAT", 0) - wheat_sold
                ),
                replant_same_crop=True,
            )
            existing_feed_buy = any(
                order[0] == "BUY_PRODUCT" and order[1] == "WHEAT"
                for order in market
            )
            for order in late_orders:
                if order[0] == "BUY_ANIMAL":
                    market.append(order)
                elif order[0] == "BUY_SEED":
                    market.append(order)
                elif order[0] == "BUY_PRODUCT" and not existing_feed_buy:
                    market.append(order)
                    existing_feed_buy = True

        # Inputs precede optional expansion hires. If the ten-order cap cuts
        # off some hires, the next observation schedules only real workers.
        if maintenance_hires:
            # Cash is already available and immediate feed was budgeted by
            # _maintenance_hires. Protect rescue labor before reserve top-ups.
            first_purchase = next((i for i, order in enumerate(market)
                                   if order[0] != "SELL"), len(market))
            market[first_purchase:first_purchase] = [["HIRE"]] * maintenance_hires
        market += [["HIRE"] for _ in range(expansion_hires - maintenance_hires)]
        return {"farmer": farmer_op, "hands": hand_ops, "market": market[:10]}

    return agent
