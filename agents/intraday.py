"""Queue introspection and inventory reconciliation for rolling execution."""

from collections import Counter

from agents.horizon import can_start_today
from agents.products import ANIMALS, ANIMAL_STRUCTURE


MOVES = {"EAST": (1, 0), "WEST": (-1, 0), "SOUTH": (0, 1), "NORTH": (0, -1)}
TILE_ACTIONS = {
    "DIG", "PLANT", "WATER", "FERTILIZE", "HARVEST", "PLACE",
    "BUILD_COOP", "BUILD_PASTURE", "FEED", "CARE", "COLLECT_FERTILIZER",
}


def queue_commitments(positions, plans):
    """Recover destinations and input reservations from the current routes."""
    endpoints, occupied = [], set()
    seeds, supplies = Counter(), Counter()
    positions = [*positions, *(plan.start for plan in plans[len(positions):])]
    for index, position in enumerate(positions):
        x, y = position
        queue = plans[index].queue if index < len(plans) else []
        for operation in queue:
            op = operation[0]
            if op in MOVES:
                dx, dy = MOVES[op]
                x, y = x + dx, y + dy
            elif op in TILE_ACTIONS:
                occupied.add((x, y))
                if op == "PLANT":
                    seeds[operation[1]] += 1
            elif op == "PICKUP":
                supplies[operation[1]] += operation[2]
        endpoints.append((x, y))
    return endpoints, occupied, seeds, supplies


def reconcile_animals(obs, targets):
    """Pin owned animals to compatible vacant tiles until a daily plan places them."""
    farm = obs["farms"][obs["player"]]
    stock = Counter(obs["private"]["shed"])
    for inventory in obs["private"]["inventories"]:
        stock.update(inventory)
    pinned = set()
    for animal in ANIMALS:
        if not stock[animal] or not can_start_today(animal, obs):
            continue
        candidates = []
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                position = (x, y)
                if position in pinned:
                    continue
                compatible = tile is None or (
                    isinstance(tile, dict)
                    and tile.get("kind") == ANIMAL_STRUCTURE[animal]
                    and not tile.get("animal")
                )
                if compatible:
                    candidates.append(position)
        candidates.sort(
            key=lambda p: (
                targets.get(p) != (animal, False),
                farm["tiles"][p[1]][p[0]] is None,
                abs(p[0] - 4) + abs(p[1] - 4),
                p,
            )
        )
        for position in candidates[:stock[animal]]:
            targets[position] = (animal, False)
            pinned.add(position)
    return pinned
