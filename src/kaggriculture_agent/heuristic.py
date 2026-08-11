"""Deterministic 5x5 heuristic for validating every crop and animal workflow."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")

# Fill the initial NW quadrant. Wheat and livestock stay near the shed because
# they need more frequent service; slow Melons occupy the farther rows.
WHEAT_POSITIONS = (
    (0, 4),
    (1, 4),
    (2, 4),
    (4, 4),
    (0, 3),
    (1, 3),
    (2, 3),
    (2, 2),
    (3, 2),
)
MELON_POSITIONS = (
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (0, 1),
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 1),
    (0, 2),
    (1, 2),
)
CROP_LAYOUT = {
    **dict.fromkeys(WHEAT_POSITIONS, "WHEAT"),
    **dict.fromkeys(MELON_POSITIONS, "MELON"),
}
NE_CROP_POSITIONS = tuple((x, y) for y in range(5) for x in range(5, 10))
ANIMAL_LAYOUT = {
    (3, 4): "COW",
    (3, 3): "COW",
    (4, 2): "SHEEP",
    (4, 3): "SHEEP",
}
CropLayout = dict[tuple[int, int], str | None]
ANIMAL_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
CROP_HARVEST_DAY = {"WHEAT": 4, "CARROT": 3, "TOMATO": 11, "STRAWBERRY": 16, "MELON": 10}
ONGOING_CROPS = frozenset(("TOMATO", "STRAWBERRY"))
CROP_SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
ONE_TIME_YIELD = {"WHEAT": 4, "CARROT": 3, "MELON": 6}
ONGOING_YIELD_DAYS = {"TOMATO": (8, 9, 10, 11), "STRAWBERRY": (10, 12, 14, 16)}

SHED_ACCESS_NW = (4, 4)
SHED_ACCESS_TILES = ((4, 4), (5, 4), (4, 5), (5, 5))
MAX_MARKET_ORDERS = 10
FEED_BUFFER_DAYS = 2
FAST_SELL_FINAL_DAYS = 3
NEXT_QUADRANT = "NE"
EXACT_PARTITION_LIMIT = 12
EXACT_TRAVEL_LIMIT = 8


@dataclass(frozen=True)
class Task:
    target: tuple[int, int]
    operation: str
    priority: int
    item: str | None = None
    deadline: int = 23


@dataclass(frozen=True)
class ActionChain:
    target: tuple[int, int]
    commands: tuple[tuple[str, ...], ...]
    priority: int
    family: str


@dataclass(frozen=True)
class CropProjection:
    crop: str
    total_profit: float
    profit_per_day: float
    harvests: int


def project_crop_profit(
    crop: str,
    plant_day: int,
    season_days: int,
    sale_price: float,
    slots: int = 1,
) -> CropProjection:
    """Project repeated monocrop profit over the remaining season window."""
    last_day = season_days - 1
    if slots <= 0 or plant_day > last_day:
        return CropProjection(crop, 0.0, 0.0, 0)

    cycle_day = plant_day
    profit_per_slot = 0.0
    harvests_per_slot = 0
    while cycle_day <= last_day:
        if crop in ONGOING_CROPS:
            harvest_days = [
                cycle_day + offset for offset in ONGOING_YIELD_DAYS[crop] if cycle_day + offset <= last_day
            ]
            if not harvest_days:
                break
            profit_per_slot += len(harvest_days) * sale_price - CROP_SEED_COST[crop]
            harvests_per_slot += len(harvest_days)
            cycle_day += CROP_HARVEST_DAY[crop] + 1
        else:
            harvest_day = cycle_day + CROP_HARVEST_DAY[crop]
            if harvest_day > last_day:
                break
            profit_per_slot += ONE_TIME_YIELD[crop] * sale_price - CROP_SEED_COST[crop]
            harvests_per_slot += 1
            cycle_day = harvest_day

    total_profit = profit_per_slot * slots
    window_days = max(1, season_days - plant_day)
    return CropProjection(
        crop=crop,
        total_profit=total_profit,
        profit_per_day=total_profit / window_days,
        harvests=harvests_per_slot * slots,
    )


def choose_replacement_crop(
    obs: dict[str, Any], plant_day: int, slots: int, season_days: int = 30
) -> tuple[str | None, dict[str, CropProjection]]:
    prices = (obs.get("market", {}) or {}).get("prices", {}) or {}
    projections = {
        crop: project_crop_profit(
            crop,
            plant_day,
            season_days,
            float(prices.get(crop, 0)),
            slots,
        )
        for crop in CROPS
    }
    candidate = max(
        CROPS,
        key=lambda crop: (
            projections[crop].profit_per_day,
            projections[crop].total_profit,
            -CROP_HARVEST_DAY[crop],
        ),
    )
    selected = candidate if projections[candidate].profit_per_day > 0 else None
    return selected, projections


def _hire_price(hire_index: int) -> int:
    if hire_index <= 1:
        return 1
    previous, current = 1, 1
    for _ in range(2, hire_index + 1):
        previous, current = current, previous + current
    return current


def _tile_at(farm: dict[str, Any], position: tuple[int, int]) -> Any:
    x, y = position
    return farm["tiles"][y][x]


def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _move_towards(position: tuple[int, int], target: tuple[int, int]) -> list[str]:
    x, y = position
    tx, ty = target
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return ["PASS"]


def _nearest_shed_access(position: tuple[int, int]) -> tuple[int, int]:
    return min(SHED_ACCESS_TILES, key=lambda access: (_distance(position, access), access))


def _crop_task(
    position: tuple[int, int],
    crop: str | None,
    tile: Any,
    day: int,
    seeds: dict[str, int],
    hour: int = 0,
) -> Task | None:
    if tile is None:
        # Empty managed tiles are continuations after HARVEST as well as initial
        # setup. Never let a lone unit PLANT on the final turn because the seed
        # would become a weed before it can WATER; cooperative chains may still
        # complete PLANT -> WATER together on hour 23.
        if hour >= 23:
            return None
        return Task(position, "PLANT", 0, crop) if crop and seeds.get(crop, 0) > 0 else None
    if tile == "LOCKED":
        return None
    if not isinstance(tile, dict):
        return None
    if tile.get("kind") == "WEED":
        return Task(position, "DIG", 0) if crop else None
    if tile.get("kind") != "PLANT":
        return Task(position, "DIG", 5) if "animal" not in tile else None
    active_crop = tile.get("crop")
    if active_crop not in CROPS:
        return Task(position, "DIG", 5)
    age = day - int(tile.get("planted_day", day))
    harvestable = int(tile.get("yield_units", 0)) > 0
    if not tile.get("watered_today", False):
        # A new plant already has consecutive_unwatered=1, so watering is the
        # highest-priority action on the turn immediately after planting. On
        # max-yield day it also starts the hard WATER -> HARVEST chain.
        priority = (
            0
            if int(tile.get("consecutive_unwatered", 0))
            or (active_crop in ONGOING_CROPS and harvestable)
            or age >= CROP_HARVEST_DAY[active_crop]
            else 1
        )
        return Task(position, "WATER", priority)
    if active_crop in ONGOING_CROPS and harvestable:
        # A ready ongoing plant still obeys the hard WATER -> HARVEST order.
        return Task(position, "HARVEST", 0)
    if harvestable and age >= CROP_HARVEST_DAY[active_crop]:
        # Harvest is a hard deadline on planted_day + max_yield_day.
        return Task(position, "HARVEST", 0)
    return None


def _animal_task(position: tuple[int, int], animal: str, tile: Any) -> Task | None:
    structure = ANIMAL_STRUCTURE[animal]
    if tile is None:
        # Fetch first, then BUILD and PLACE on consecutive turns at the tile.
        return Task(position, f"BUILD_{structure}", 2, animal)
    if tile == "LOCKED" or not isinstance(tile, dict):
        return None
    if tile.get("kind") == "WEED":
        return Task(position, "DIG", 5)
    if "animal" not in tile:
        if tile.get("kind") != structure:
            return Task(position, "DIG", 5)
        return Task(position, "PLACE", 2, animal)
    if tile.get("animal") != animal:
        return None  # Occupied animal structures cannot be dug up.
    if not tile.get("fed_today", False):
        priority = 0 if int(tile.get("consecutive_unfed", 0)) else 1
        return Task(position, "FEED", priority, "WHEAT")
    if tile.get("fertilizer_available", False):
        # Fertilizer does not accumulate, so every available day is a deadline.
        return Task(position, "COLLECT_FERTILIZER", 0)
    if int(tile.get("yield_units", 0)) > 0:
        return Task(position, "HARVEST", 2)
    if not tile.get("cared_today", False):
        return Task(position, "CARE", 3)
    return None


def build_tasks(obs: dict[str, Any], crop_layout: CropLayout | None = None) -> list[Task]:
    """Return at most one currently useful task for each managed tile."""
    farm = obs["farms"][int(obs["player"])]
    seeds = (obs.get("private", {}) or {}).get("seeds", {}) or {}
    day = int(obs.get("day", 0))
    tasks: list[Task] = []
    crop_layout = CROP_LAYOUT if crop_layout is None else crop_layout
    for position, crop in crop_layout.items():
        task = _crop_task(position, crop, _tile_at(farm, position), day, seeds, int(obs.get("hour", 0)))
        if task:
            tasks.append(task)
    for position, animal in ANIMAL_LAYOUT.items():
        task = _animal_task(position, animal, _tile_at(farm, position))
        if task:
            tasks.append(task)
    return tasks


def build_action_chains(obs: dict[str, Any], crop_layout: CropLayout | None = None) -> list[ActionChain]:
    """Create same-turn chains whose command order changes the resulting tile."""
    farm = obs["farms"][int(obs["player"])]
    private = obs.get("private", {}) or {}
    seeds = private.get("seeds", {}) or {}
    day = int(obs.get("day", 0))
    chains: list[ActionChain] = []
    crop_layout = CROP_LAYOUT if crop_layout is None else crop_layout
    for position, crop in crop_layout.items():
        tile = _tile_at(farm, position)
        if tile is None and crop and int(seeds.get(crop, 0)) > 0:
            chains.append(ActionChain(position, (("PLANT", crop), ("WATER",)), 3, "crop"))
            continue
        active_crop = tile.get("crop") if isinstance(tile, dict) else None
        if active_crop not in ONGOING_CROPS and _crop_is_harvestable(tile, active_crop, day):
            commands: list[tuple[str, ...]] = []
            if not tile.get("watered_today", False):
                commands.append(("WATER",))
            commands.append(("HARVEST",))
            if crop and int(seeds.get(crop, 0)) > 0:
                commands.extend((("PLANT", crop), ("WATER",)))
            chains.append(ActionChain(position, tuple(commands), 0, "crop"))

    for position, _animal in ANIMAL_LAYOUT.items():
        tile = _tile_at(farm, position)
        if not (isinstance(tile, dict) and "animal" in tile):
            continue
        commands = []
        if not tile.get("fed_today", False):
            commands.append(("FEED",))
        if tile.get("fertilizer_available", False):
            commands.append(("COLLECT_FERTILIZER",))
        if not tile.get("cared_today", False):
            commands.append(("CARE",))
        if len(commands) >= 2:
            chains.append(ActionChain(position, tuple(commands), 1, "animal"))
    return sorted(chains, key=lambda chain: (chain.priority, chain.target))


def required_unit_count(obs: dict[str, Any], clusters: list[list[Asset]]) -> int:
    """Return the minimum units required by the feasible daily routes.

    Same-tile chains are opportunistic: existing units may cooperate to finish
    them in one turn, but a shorter chain is not by itself a reason to hire an
    otherwise unnecessary hand.
    """
    return max(1, len(clusters))


def _inventory_total(private: dict[str, Any], item: str) -> int:
    total = int((private.get("shed", {}) or {}).get(item, 0))
    for inventory in private.get("inventories", []) or []:
        total += int((inventory or {}).get(item, 0))
    return total


def _matching_crop(tile: Any, crop: str) -> bool:
    return isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile.get("crop") == crop


def _matching_animal(tile: Any, animal: str) -> bool:
    return isinstance(tile, dict) and tile.get("animal") == animal


def required_seed_purchases(obs: dict[str, Any], crop_layout: CropLayout) -> dict[str, int]:
    """Return seed quantities needed for empty, transitioning, and near-due tiles."""
    farm = obs["farms"][int(obs["player"])]
    seeds = (obs.get("private", {}) or {}).get("seeds", {}) or {}
    day = int(obs.get("day", 0))
    requirements: dict[str, int] = {}
    for crop in Counter(crop for crop in crop_layout.values() if crop):
        target_tiles = [
            _tile_at(farm, position)
            for position, expected_crop in crop_layout.items()
            if expected_crop == crop
        ]
        unmatched = sum(not _matching_crop(tile, crop) for tile in target_tiles)
        replacement_soon = sum(
            crop not in ONGOING_CROPS
            and _matching_crop(tile, crop)
            and day + 1 - int(tile.get("planted_day", day)) >= CROP_HARVEST_DAY[crop]
            for tile in target_tiles
        )
        missing = max(0, unmatched + replacement_soon - int(seeds.get(crop, 0)))
        if missing:
            requirements[crop] = missing
    return requirements


Asset = tuple[str, tuple[int, int], str]


def _cluster_distance(left: list[Asset], right: list[Asset]) -> int:
    """Distance used by deterministic agglomerative geographic clustering."""
    return min(_distance(a[1], b[1]) for a in left for b in right)


def _crop_is_harvestable(tile: Any, crop: str, day: int) -> bool:
    if not (isinstance(tile, dict) and tile.get("crop") == crop):
        return False
    age = day - int(tile.get("planted_day", day))
    ready = int(tile.get("yield_units", 0)) > 0
    return ready and (crop in ONGOING_CROPS or age >= CROP_HARVEST_DAY[crop])


def estimate_asset_steps(obs: dict[str, Any], asset: Asset) -> int:
    """Estimate mandatory service actions for this asset during the current day."""
    kind, position, name = asset
    farm = obs["farms"][int(obs["player"])]
    tile = _tile_at(farm, position)
    day = int(obs.get("day", 0))
    if kind == "crop":
        if tile is None:
            return 2 if name else 0  # PLANT + WATER, unless this tile retires.
        if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
            return 3 if name else 0  # DIG + PLANT + WATER.
        active_crop = tile.get("crop")
        if active_crop not in CROPS:
            return 3
        steps = 0 if tile.get("watered_today", False) else 1
        if _crop_is_harvestable(tile, active_crop, day):
            # Ongoing crops stay in place. One-time crops need HARVEST followed
            # by PLANT + WATER for the replacement.
            steps += 1 if active_crop in ONGOING_CROPS or not name else 3
        return steps
    if tile is None:
        # PICKUP animal + BUILD + PLACE + FEED. The shared wheat PICKUP is
        # charged once by _route_steps.
        return 4
    if not isinstance(tile, dict) or "animal" not in tile:
        if isinstance(tile, dict) and tile.get("kind") == ANIMAL_STRUCTURE[name]:
            return 3  # PICKUP animal + PLACE + FEED.
        return 5  # DIG + PICKUP animal + BUILD + PLACE + FEED.
    steps = 0 if tile.get("fed_today", False) else 1
    if int(tile.get("yield_units", 0)) > 0:
        steps += 1
    if not tile.get("cared_today", False):
        steps += 1
    if tile.get("fertilizer_available", False):
        steps += 1
    return steps


def _travel_steps(cluster: list[Asset]) -> int:
    """Exact shortest open route from the shed for a small cluster."""
    targets = list(dict.fromkeys(asset[1] for asset in cluster))
    if not targets:
        return 0
    if len(targets) > EXACT_TRAVEL_LIMIT:
        remaining = targets[:]
        position = min(SHED_ACCESS_TILES, key=lambda access: sum(_distance(access, target) for target in targets))
        steps = 0
        while remaining:
            target = min(remaining, key=lambda candidate: (_distance(position, candidate), candidate))
            steps += _distance(position, target)
            position = target
            remaining.remove(target)
        return steps

    costs = {
        (1 << index, index): min(_distance(access, target) for access in SHED_ACCESS_TILES)
        for index, target in enumerate(targets)
    }
    for mask in range(1, 1 << len(targets)):
        for last in range(len(targets)):
            current = costs.get((mask, last))
            if current is None:
                continue
            for nxt in range(len(targets)):
                bit = 1 << nxt
                if mask & bit:
                    continue
                key = (mask | bit, nxt)
                candidate = current + _distance(targets[last], targets[nxt])
                costs[key] = min(costs.get(key, candidate), candidate)
    full_mask = (1 << len(targets)) - 1
    return min(costs[(full_mask, last)] for last in range(len(targets)))


def _route_steps(obs: dict[str, Any], cluster: list[Asset]) -> int:
    service_steps = sum(estimate_asset_steps(obs, asset) for asset in cluster)
    animal_tiles = [asset for asset in cluster if asset[0] == "animal"]
    farm = obs["farms"][int(obs["player"])]
    needs_feed_pickup = any(
        isinstance(_tile_at(farm, asset[1]), dict)
        and "animal" in _tile_at(farm, asset[1])
        and not _tile_at(farm, asset[1]).get("fed_today", False)
        for asset in animal_tiles
    )
    return service_steps + _travel_steps(cluster) + int(needs_feed_pickup)


def _merge_routes_greedily(obs: dict[str, Any], assets: list[Asset], turn_budget: int) -> list[list[Asset]]:
    """Merge nearby assets while the complete route still fits this day."""
    clusters = [[asset] for asset in assets]
    while True:
        candidates: list[tuple[int, int, int, int]] = []
        for left_index, left in enumerate(clusters):
            for right_index in range(left_index + 1, len(clusters)):
                right = clusters[right_index]
                merged = [*left, *right]
                route_steps = _route_steps(obs, merged)
                if route_steps <= turn_budget:
                    candidates.append((_cluster_distance(left, right), route_steps, left_index, right_index))
        if not candidates:
            return clusters
        _, _, left_index, right_index = min(candidates)
        clusters[left_index] = [*clusters[left_index], *clusters[right_index]]
        clusters.pop(right_index)


def _consolidate_routes(
    obs: dict[str, Any], clusters: list[list[Asset]], turn_budget: int
) -> list[list[Asset]]:
    """Eliminate a route when all its assets can fit into the remaining routes."""
    while len(clusters) > 1:
        improved = False
        for victim_index in sorted(range(len(clusters)), key=lambda index: len(clusters[index])):
            victim = sorted(
                clusters[victim_index],
                key=lambda asset: (
                    estimate_asset_steps(obs, asset),
                    min(_distance(access, asset[1]) for access in SHED_ACCESS_TILES),
                ),
                reverse=True,
            )
            hosts = [cluster[:] for index, cluster in enumerate(clusters) if index != victim_index]

            def distribute(asset_index: int) -> bool:
                if asset_index == len(victim):
                    return True
                asset = victim[asset_index]
                seen_hosts: set[tuple[Asset, ...]] = set()
                for host_index, host in enumerate(hosts):
                    signature = tuple(host)
                    if signature in seen_hosts:
                        continue
                    seen_hosts.add(signature)
                    if _route_steps(obs, [*host, asset]) > turn_budget:
                        continue
                    hosts[host_index].append(asset)
                    if distribute(asset_index + 1):
                        return True
                    hosts[host_index].pop()
                return False

            if distribute(0):
                clusters = hosts
                improved = True
                break
        if not improved:
            return clusters
    return clusters


def _minimum_route_partition(obs: dict[str, Any], assets: list[Asset], turn_budget: int) -> list[list[Asset]]:
    """Find the fewest workload-feasible routes, with deterministic tie breaks."""
    if len(assets) > EXACT_PARTITION_LIMIT:
        return _consolidate_routes(obs, _merge_routes_greedily(obs, assets, turn_budget), turn_budget)

    route_cost: dict[int, int] = {}

    def cost(mask: int) -> int:
        if mask not in route_cost:
            cluster = [asset for index, asset in enumerate(assets) if mask & (1 << index)]
            route_cost[mask] = _route_steps(obs, cluster)
        return route_cost[mask]

    @lru_cache(maxsize=None)
    def solve(remaining: int) -> tuple[int, int, tuple[int, ...]] | None:
        if remaining == 0:
            return 0, 0, ()
        first = remaining & -remaining
        best: tuple[int, int, tuple[int, ...]] | None = None
        subset = remaining
        while subset:
            if subset & first and cost(subset) <= turn_budget:
                tail = solve(remaining ^ subset)
                if tail is not None:
                    candidate = (tail[0] + 1, tail[1] + cost(subset), (subset, *tail[2]))
                    if best is None or candidate < best:
                        best = candidate
            subset = (subset - 1) & remaining
        return best

    full_mask = (1 << len(assets)) - 1
    solution = solve(full_mask)
    if solution is None:
        # Every normal asset fits in one day. Keep an explicit fallback for a
        # malformed/custom workload so the live agent can still make progress.
        return [[asset] for asset in assets]
    return [[asset for index, asset in enumerate(assets) if mask & (1 << index)] for mask in solution[2]]


def generate_unit_clusters(
    obs: dict[str, Any] | None = None,
    assets: list[Asset] | None = None,
    crop_layout: CropLayout | None = None,
) -> list[list[Asset]]:
    """Build compact routes from the current strategic layout.

    Crops and animals may share a route. Animal pickup overhead is included in
    the route cost, so the partition uses the fewest units that can finish all
    planned service and travel within the remaining turns of the day.
    """
    if obs is None:
        raise ValueError("observation is required for workload-aware clustering")
    crop_layout = CROP_LAYOUT if crop_layout is None else crop_layout
    if assets is None:
        assets = [
            *(("crop", position, crop or "") for position, crop in crop_layout.items()),
            *(("animal", position, animal) for position, animal in ANIMAL_LAYOUT.items()),
        ]
    # Assets with no work remaining today need no route visit.
    assets = [asset for asset in assets if estimate_asset_steps(obs, asset) > 0]
    if not assets:
        return [[]]

    # Keep livestock on a dedicated shed-adjacent route. Shared feed pickup and
    # completing FEED -> COLLECT at each tile make its daily deadlines reliable.
    # A newly hired crop hand is absent at hour 0, appears at hour 1, starts one
    # move away from NW access, and the live priority scheduler can differ by a
    # step from the shortest-path estimate. Reserve all three turns.
    hour = int(obs.get("hour", 0))
    hand_budget = max(1, 18 - hour)
    farmer_budget = max(1, 23 - hour)
    animals = [asset for asset in assets if asset[0] == "animal"]
    crops = [asset for asset in assets if asset[0] == "crop"]
    animal_routes = _minimum_route_partition(obs, animals, farmer_budget) if animals else []
    # Never build one combinatorial partition across quadrants. Each quadrant
    # has its own shed access and routes crossing the center are not useful;
    # separating them also keeps the >12-target heuristic fast at 46 crops.
    crop_groups = [
        [asset for asset in crops if asset[1][0] < 5 and asset[1][1] < 5],
        [asset for asset in crops if asset[1][0] >= 5 and asset[1][1] < 5],
        [asset for asset in crops if asset[1][0] < 5 and asset[1][1] >= 5],
        [asset for asset in crops if asset[1][0] >= 5 and asset[1][1] >= 5],
    ]
    crop_routes = [
        route
        for group in crop_groups
        if group
        for route in _minimum_route_partition(obs, group, hand_budget)
    ]
    return [*animal_routes, *crop_routes]


def market_actions(
    obs: dict[str, Any],
    clusters: list[list[Asset]] | None = None,
    crop_layout: CropLayout | None = None,
) -> list[list[Any]]:
    """Fund the managed assets, sell output, and unlock the next quadrant."""
    player = int(obs["player"])
    farm = obs["farms"][player]
    private = obs.get("private", {}) or {}
    shed = private.get("shed", {}) or {}
    crop_layout = CROP_LAYOUT if crop_layout is None else crop_layout
    orders: list[list[Any]] = []

    # Seeds precede hiring so a large expansion cannot let HIRE consume all ten
    # market slots while existing workers have nothing to plant. Strategic crop
    # selection has already reserved the projected labor and feed expense.
    for crop, missing in required_seed_purchases(obs, crop_layout).items():
        orders.append(["BUY_SEED", crop, missing])

    if int(obs.get("hour", 0)) < 18:
        active_clusters = clusters or generate_unit_clusters(obs, crop_layout=crop_layout)
        target_hands = max(0, required_unit_count(obs, active_clusters) - 1)
        missing_hands = max(0, target_hands - len(farm.get("hands", []) or []))
        orders.extend([["HIRE"] for _ in range(missing_hands)])

    placed_animals = sum(
        _matching_animal(_tile_at(farm, position), animal) for position, animal in ANIMAL_LAYOUT.items()
    )
    planned_animals = max(placed_animals, len(ANIMAL_LAYOUT))
    wheat_target = planned_animals * FEED_BUFFER_DAYS
    wheat_total = _inventory_total(private, "WHEAT")
    if wheat_total < wheat_target:
        # Keep two days of feed before committing capital to slow crops or
        # additional livestock. An existing animal escaping is more expensive.
        orders.append(["BUY_PRODUCT", "WHEAT", wheat_target - wheat_total])

    # Count shed and carried animals so BUY_ANIMAL is never repeated while the
    # placement state machine is in progress.
    for animal, target_count in Counter(ANIMAL_LAYOUT.values()).items():
        placed = sum(
            _matching_animal(_tile_at(farm, position), animal)
            for position, expected_animal in ANIMAL_LAYOUT.items()
            if expected_animal == animal
        )
        missing = max(0, target_count - placed - _inventory_total(private, animal))
        if missing:
            orders.append(["BUY_ANIMAL", animal, missing])

    # Keep the feed reserve.  All other finished products can be sold because
    # the goal of this version is lifecycle correctness, not market timing.
    for product in PRODUCTS:
        quantity = int(shed.get(product, 0))
        if product == "WHEAT":
            quantity = max(0, quantity - wheat_target)
        if quantity > 0:
            orders.append(["SELL", product, quantity])

    # BUY_LAND is deliberately after SELL: market orders are processed in list
    # order, so proceeds from this turn can immediately fund the NE quadrant.
    # Keep one order slot for it while locked; a failed attempt is a harmless
    # no-op and will be retried on the next turn.
    if NEXT_QUADRANT not in (farm.get("unlocked_quadrants", []) or []):
        return [*orders[: MAX_MARKET_ORDERS - 1], ["BUY_LAND"]]

    # Labor and setup correctness have precedence over selling.
    return orders[:MAX_MARKET_ORDERS]


def _execute_task(
    task: Task,
    position: tuple[int, int],
    inventory: dict[str, int],
    available_shed: dict[str, int],
) -> list[Any]:
    requires_inventory = task.operation in ("PLACE", "FEED") or task.operation.startswith("BUILD_")
    if requires_inventory and task.item and int(inventory.get(task.item, 0)) <= 0:
        shed_access = _nearest_shed_access(position)
        if position != shed_access:
            return _move_towards(position, shed_access)
        available = int(available_shed.get(task.item, 0))
        if available <= 0:
            return ["PASS"]
        quantity = min(3 if task.item == "WHEAT" else 1, available)
        available_shed[task.item] = available - quantity
        return ["PICKUP", task.item, quantity]
    if position != task.target:
        return _move_towards(position, task.target)
    if task.operation in ("PLANT", "PLACE"):
        return [task.operation, task.item]
    return [task.operation]


def _task_for_asset(
    obs: dict[str, Any],
    farm: dict[str, Any],
    private: dict[str, Any],
    asset: tuple[str, tuple[int, int], str],
    crop_layout: CropLayout | None = None,
) -> Task | None:
    kind, target, name = asset
    tile = _tile_at(farm, target)
    if kind == "crop":
        desired_crop = crop_layout.get(target, name) if crop_layout is not None else name
        return _crop_task(
            target,
            desired_crop,
            tile,
            int(obs.get("day", 0)),
            private.get("seeds", {}) or {},
            int(obs.get("hour", 0)),
        )
    return _animal_task(target, name, tile)


def _task_cost(
    task: Task, position: tuple[int, int], inventory: dict[str, int]
) -> tuple[int, int, int, int, int]:
    requires_inventory = task.operation in ("PLACE", "FEED") or task.operation.startswith("BUILD_")
    has_item = task.item is None or int(inventory.get(task.item, 0)) > 0
    if requires_inventory and not has_item:
        shed_access = _nearest_shed_access(position)
        distance = _distance(position, shed_access) + _distance(shed_access, task.target) + 2
    else:
        distance = _distance(position, task.target)
    # Survival work (priority 0/1) always precedes harvest/care/fertilizer.
    # Within the same tier, finish the current tile or a carried delivery before
    # travelling again. This prevents optional work from starving another tile.
    tier = task.priority if task.priority <= 1 else 2
    if requires_inventory and task.item and int(inventory.get(task.item, 0)) > 0:
        continuation = 0
    elif position == task.target:
        continuation = 0
    else:
        continuation = 1
    return tier, continuation, task.priority, task.deadline, distance


def _route_pickup_action(
    obs: dict[str, Any],
    cluster: list[Asset],
    position: tuple[int, int],
    inventory: dict[str, int],
    available_shed: dict[str, int],
) -> list[Any] | None:
    """Load the complete animal-route manifest before leaving the shed."""
    if position not in SHED_ACCESS_TILES or not any(asset[0] == "animal" for asset in cluster):
        return None
    farm = obs["farms"][int(obs["player"])]
    animal_manifest: Counter[str] = Counter()
    feed_units = 0
    for kind, target, animal in cluster:
        if kind != "animal":
            continue
        tile = _tile_at(farm, target)
        if not _matching_animal(tile, animal):
            animal_manifest[animal] += 1
            feed_units += 1  # Preload feed for the animal after placement.
        elif not tile.get("fed_today", False):
            feed_units += 1
    manifest = list(animal_manifest.items())
    if feed_units:
        manifest.append(("WHEAT", feed_units))

    for item, desired in manifest:
        carried = int(inventory.get(item, 0))
        missing = max(0, desired - carried)
        available = int(available_shed.get(item, 0))
        quantity = min(missing, available)
        if quantity > 0:
            available_shed[item] = available - quantity
            return ["PICKUP", item, quantity]
    return None


def _has_sale_inventory(inventory: dict[str, int], cluster: list[Asset]) -> bool:
    """Distinguish harvested products from animals/feed carried for service."""
    sellable = {item for item in PRODUCTS if int(inventory.get(item, 0)) > 0}
    if not sellable:
        return False
    if sellable == {"WHEAT"} and any(asset[0] == "animal" for asset in cluster):
        return False
    return True


def unit_actions(
    obs: dict[str, Any],
    clusters: list[list[Asset]] | None = None,
    crop_layout: CropLayout | None = None,
    prioritize_drop: bool = False,
) -> tuple[list[Any], list[list[Any]]]:
    player = int(obs["player"])
    farm = obs["farms"][player]
    private = obs.get("private", {}) or {}
    positions = [farm["farmer"], *(farm.get("hands", []) or [])]
    inventories = list(private.get("inventories", []) or [])
    available_shed = dict(private.get("shed", {}) or {})
    actions: list[list[Any]] = []
    clusters = clusters or generate_unit_clusters(obs, crop_layout=crop_layout)

    for unit_index, raw_position in enumerate(positions):
        position = tuple(map(int, raw_position))
        inventory = inventories[unit_index] if unit_index < len(inventories) else {}
        tasks = []
        if unit_index < len(clusters):
            pickup = _route_pickup_action(
                obs,
                clusters[unit_index],
                position,
                inventory,
                available_shed,
            )
            if pickup is not None:
                actions.append(pickup)
                continue
            tasks = [
                task
                for asset in clusters[unit_index]
                if (task := _task_for_asset(obs, farm, private, asset, crop_layout)) is not None
            ]
        # Finish survival/deadline work first. Once the route is safe, return
        # harvested goods immediately instead of carrying them until day end.
        urgent_tasks = [task for task in tasks if task.priority <= 1]
        cluster = clusters[unit_index] if unit_index < len(clusters) else []
        if prioritize_drop and _has_sale_inventory(inventory, cluster) and not urgent_tasks:
            shed_access = _nearest_shed_access(position)
            action = ["DROP"] if position == shed_access else _move_towards(position, shed_access)
            actions.append(action)
            continue
        task = min(tasks, key=lambda candidate: _task_cost(candidate, position, inventory)) if tasks else None
        if task is not None:
            action = _execute_task(task, position, inventory, available_shed)
        elif inventory:
            shed_access = _nearest_shed_access(position)
            action = ["DROP"] if position == shed_access else _move_towards(position, shed_access)
        else:
            action = ["PASS"]
        actions.append(action)

    return actions[0], actions[1:]


def _cooperative_overrides(
    obs: dict[str, Any], crop_layout: CropLayout | None = None, prioritize_drop: bool = False
) -> dict[int, list[Any]]:
    """Coordinate the highest-priority feasible chain across co-located units."""
    player = int(obs["player"])
    farm = obs["farms"][player]
    private = obs.get("private", {}) or {}
    positions = [tuple(farm["farmer"]), *(tuple(position) for position in farm.get("hands", []) or [])]
    inventories = list(private.get("inventories", []) or [])
    for chain in build_action_chains(obs, crop_layout):
        if chain.family == "animal" and prioritize_drop:
            pool = [
                index
                for index in range(1, len(positions))
                if index >= len(inventories)
                or not any(
                    item != "WHEAT" and int((inventories[index] or {}).get(item, 0)) > 0
                    for item in PRODUCTS
                )
            ]
        elif chain.family == "crop" and prioritize_drop:
            pool = [
                index
                for index in range(len(positions))
                if index >= len(inventories)
                or not any(int((inventories[index] or {}).get(item, 0)) > 0 for item in PRODUCTS)
            ]
        else:
            pool = list(range(1, len(positions))) if chain.family == "animal" else list(range(len(positions)))
        commands = chain.commands
        if len(pool) < len(commands):
            if chain.family == "animal" and len(pool) >= 2:
                # Execute the mandatory FEED+COLLECT pair; CARE can wait.
                commands = commands[:2]
            elif chain.family == "crop" and commands[:2] == (("WATER",), ("HARVEST",)):
                # Harvest now; the resulting empty tile produces a PLANT+WATER
                # chain from the next observation.
                commands = commands[:2]
            elif chain.family == "crop" and commands[0] == ("HARVEST",):
                commands = commands[:1]
            else:
                continue
        unit_count = len(commands)
        if len(pool) < unit_count:
            continue
        selected = sorted(pool, key=lambda index: (_distance(positions[index], chain.target), index))[
            :unit_count
        ]
        selected.sort()
        if not all(positions[index] == chain.target for index in selected):
            # Chains never pull units off their planned routes merely to save
            # turns at another unit's tile.
            continue

        feed_offset = next(
            (offset for offset, command in enumerate(commands) if command[0] == "FEED"),
            None,
        )
        if feed_offset is not None:
            carriers = [
                index
                for index in selected
                if index < len(inventories) and int((inventories[index] or {}).get("WHEAT", 0)) > 0
            ]
            if not carriers:
                continue
            carrier = carriers[0]
            carrier_offset = selected.index(carrier)
            selected[feed_offset], selected[carrier_offset] = selected[carrier_offset], selected[feed_offset]

        return {unit_index: list(commands[offset]) for offset, unit_index in enumerate(selected)}
    return {}


class HeuristicAgent:
    """Callable agent that keeps a stable route allocation for one game day."""

    def __init__(self) -> None:
        self._plan_key: tuple[int, int] | None = None
        self._clusters: list[list[Asset]] = []
        self._crop_layout: CropLayout = dict(CROP_LAYOUT)
        self.crop_choices: dict[int, tuple[str | None, dict[str, CropProjection]]] = {}
        self.expansion_choices: dict[int, tuple[str, dict[str, CropProjection]]] = {}
        self._last_expansion_attempt_day: int | None = None

    @staticmethod
    def _config_int(configuration: Any, key: str, default: int) -> int:
        if isinstance(configuration, dict):
            return int(configuration.get(key, default))
        return int(getattr(configuration, key, default) if configuration else default)

    def _available_seed_budget(
        self,
        obs: dict[str, Any],
        configuration: Any,
        crop_layout: CropLayout | None = None,
    ) -> float:
        player = int(obs["player"])
        farm = obs["farms"][player]
        private = obs.get("private", {}) or {}
        layout = self._crop_layout if crop_layout is None else crop_layout
        clusters = generate_unit_clusters(obs, crop_layout=layout)
        target_hands = max(0, required_unit_count(obs, clusters) - 1)
        current_hands = len(farm.get("hands", []) or [])
        missing_hands = max(0, target_hands - current_hands)
        hires_today = int(farm.get("hires_today", 0))
        labor_multiplier = self._config_int(configuration, "farmHandCostMult", 1)
        labor_cost = labor_multiplier * sum(
            _hire_price(hires_today + offset) for offset in range(missing_hands)
        )

        wheat_target = len(ANIMAL_LAYOUT) * FEED_BUFFER_DAYS
        feed_gap = max(0, wheat_target - _inventory_total(private, "WHEAT"))
        wheat_price = float(((obs.get("market", {}) or {}).get("prices", {}) or {}).get("WHEAT", 0))
        return max(0.0, float(farm.get("money", 0)) - labor_cost - feed_gap * wheat_price)

    def _activate_ne_layout(self, obs: dict[str, Any], configuration: Any) -> bool:
        """Add all 25 NE tiles as one affordable, profit-ranked crop cohort."""
        farm = obs["farms"][int(obs["player"])]
        if NEXT_QUADRANT not in (farm.get("unlocked_quadrants", []) or []):
            return False
        if any(position in self._crop_layout for position in NE_CROP_POSITIONS):
            return False

        day = int(obs.get("day", 0))
        if self._last_expansion_attempt_day == day:
            return False
        self._last_expansion_attempt_day = day
        episode_steps = self._config_int(configuration, "episodeSteps", 720)
        turns_per_day = self._config_int(configuration, "turnsPerDay", 24)
        season_days = max(1, (episode_steps + turns_per_day - 1) // turns_per_day)
        _unconstrained, projections = choose_replacement_crop(
            obs,
            plant_day=day,
            slots=len(NE_CROP_POSITIONS),
            season_days=season_days,
        )
        ranked_crops = sorted(
            CROPS,
            key=lambda crop: (
                projections[crop].profit_per_day,
                projections[crop].total_profit,
                -CROP_HARVEST_DAY[crop],
            ),
            reverse=True,
        )

        # Empty crop tiles all have the same PLANT+WATER workload regardless of
        # species, so one provisional layout gives the labor reserve shared by
        # every candidate.
        provisional_layout = {
            **self._crop_layout,
            **dict.fromkeys(NE_CROP_POSITIONS, ranked_crops[0]),
        }
        seed_budget = self._available_seed_budget(obs, configuration, provisional_layout)
        for crop in ranked_crops:
            if projections[crop].profit_per_day <= 0:
                break
            candidate_layout = {
                **self._crop_layout,
                **dict.fromkeys(NE_CROP_POSITIONS, crop),
            }
            seed_cost = sum(
                CROP_SEED_COST[item] * quantity
                for item, quantity in required_seed_purchases(obs, candidate_layout).items()
            )
            if seed_cost > seed_budget:
                continue
            self._crop_layout.update(dict.fromkeys(NE_CROP_POSITIONS, crop))
            self.expansion_choices[day] = crop, projections
            return True
        return False

    def _select_replacements(self, obs: dict[str, Any], configuration: Any) -> None:
        player = int(obs["player"])
        farm = obs["farms"][player]
        day = int(obs.get("day", 0))
        due_positions: list[tuple[int, int]] = []
        for position in self._crop_layout:
            tile = _tile_at(farm, position)
            if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
                continue
            active_crop = tile.get("crop")
            if active_crop in ONGOING_CROPS or active_crop not in CROP_HARVEST_DAY:
                continue
            age = day - int(tile.get("planted_day", day))
            if age >= CROP_HARVEST_DAY[active_crop]:
                due_positions.append(position)
        if not due_positions:
            return

        episode_steps = self._config_int(configuration, "episodeSteps", 720)
        turns_per_day = self._config_int(configuration, "turnsPerDay", 24)
        season_days = max(1, (episode_steps + turns_per_day - 1) // turns_per_day)
        _unconstrained, projections = choose_replacement_crop(
            obs,
            plant_day=day,
            slots=len(due_positions),
            season_days=season_days,
        )
        seed_budget = self._available_seed_budget(obs, configuration)
        selected: str | None = None
        ranked_crops = sorted(
            CROPS,
            key=lambda crop: (
                projections[crop].profit_per_day,
                projections[crop].total_profit,
                -CROP_HARVEST_DAY[crop],
            ),
            reverse=True,
        )
        for crop in ranked_crops:
            if projections[crop].profit_per_day <= 0:
                break
            candidate_layout = dict(self._crop_layout)
            for position in due_positions:
                candidate_layout[position] = crop
            seed_cost = sum(
                CROP_SEED_COST[item] * quantity
                for item, quantity in required_seed_purchases(obs, candidate_layout).items()
            )
            if seed_cost <= seed_budget:
                selected = crop
                break
        for position in due_positions:
            self._crop_layout[position] = selected
        self.crop_choices[day] = selected, projections

    def __call__(self, obs: dict[str, Any], configuration: Any = None) -> dict[str, Any]:
        plan_key = (int(obs["player"]), int(obs.get("day", 0)))
        new_game = int(obs.get("step", -1)) == 0
        new_day = self._plan_key != plan_key
        if new_day or new_game:
            if new_game:
                self._crop_layout = dict(CROP_LAYOUT)
                self.crop_choices = {}
                self.expansion_choices = {}
                self._last_expansion_attempt_day = None
            self._select_replacements(obs, configuration)
        layout_expanded = self._activate_ne_layout(obs, configuration)
        if new_day or new_game or layout_expanded:
            self._plan_key = plan_key
            self._clusters = generate_unit_clusters(obs, crop_layout=self._crop_layout)
        episode_steps = self._config_int(configuration, "episodeSteps", 720)
        turns_per_day = self._config_int(configuration, "turnsPerDay", 24)
        season_days = max(1, (episode_steps + turns_per_day - 1) // turns_per_day)
        prioritize_drop = int(obs.get("day", 0)) >= max(0, season_days - FAST_SELL_FINAL_DAYS)
        farmer, hands = unit_actions(
            obs,
            self._clusters,
            self._crop_layout,
            prioritize_drop=prioritize_drop,
        )
        actions = [farmer, *hands]
        for unit_index, action in _cooperative_overrides(
            obs,
            self._crop_layout,
            prioritize_drop=prioritize_drop,
        ).items():
            if unit_index < len(actions):
                actions[unit_index] = action
        return {
            "farmer": actions[0],
            "hands": actions[1:],
            "market": market_actions(obs, self._clusters, self._crop_layout),
        }
