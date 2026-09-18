"""Reserve a final return and sale before the episode's terminal observation."""
from agents.scheduler import nearest_shed, route
from agents.selling import SELLABLE

MOVES = {"EAST": (1, 0), "WEST": (-1, 0), "SOUTH": (0, 1), "NORTH": (0, -1)}


def liquidate_queues(obs, plans, last_action_step, shed_access):
    """Keep the longest queue prefix whose goods can still reach a sale.

    A DROP changes the next observation. SELL therefore needs a separate
    subsequent decision, even though worker operations run before the market.
    Inventory left on a hand at the terminal frame contributes no cash reward.
    """
    farm = obs['farms'][obs['player']]
    positions = [farm['farmer'], *farm['hands']]
    available = last_action_step - obs.get('step', obs['day'] * 24 + obs['hour'])  # reserve one decision for SELL
    inventories = obs['private']['inventories']
    for index, (plan, position) in enumerate(zip(plans, positions)):
        current = tuple(position)
        inventory = inventories[index] if index < len(inventories) else {}
        goods = any(inventory.get(item, 0) for item in SELLABLE)
        best = None
        for length in range(min(len(plan.queue), max(0, available)) + 1):
            shed = nearest_shed(current, shed_access)
            home = route(current, shed) + [['DROP']] if goods else []
            if length + len(home) <= available:
                best = plan.queue[:length] + home
            if length == len(plan.queue):
                break
            op = plan.queue[length][0]
            if op in MOVES:
                dx, dy = MOVES[op]
                current = current[0] + dx, current[1] + dy
            elif op in ('HARVEST', 'COLLECT_FERTILIZER') or (
                op == 'PICKUP' and plan.queue[length][1] in SELLABLE
            ):
                goods = True
            elif op == 'DROP':
                goods = False
        if best is not None:
            plan.queue = best
