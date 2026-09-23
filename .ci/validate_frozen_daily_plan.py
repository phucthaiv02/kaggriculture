from collections import Counter

import agents.expansion_agent as module
from agents.expansion_agent import make_agent
from agents.scheduler import WorkerPlan

agent = make_agent(29)
cells = dict(zip(agent.__code__.co_freevars, (cell.cell_contents for cell in agent.__closure__)))
state = cells["state"]
standing_targets = cells["targets"]
global_replan = cells["_global_replan"]

position = (0, 0)
standing_targets[position] = ("MELON", False)  # mutable future-morning target
state["daily_targets"] = {position: ("CARROT", False)}  # morning commitment
state["frozen_positions"] = {position}
state["opening_active"] = False

farm = {
    "farmer": [4, 4],
    "hands": [],
    "tiles": [[None for _ in range(10)] for _ in range(10)],
}
obs = {
    "player": 0,
    "day": 10,
    "hour": 5,
    "farms": [farm, farm],
    "private": {"inventories": [{}], "shed": {"WHEAT": 0}, "seeds": {}},
    "_pending_targets": {position},
}

captured = {}
original_build_tasks = module.build_tasks
original_build_queues = module.build_queues
original_reserved_items = module.reserved_items
try:
    def fake_build_tasks(replan_obs, target_map, **kwargs):
        captured["targets"] = dict(target_map)
        captured["pending"] = set(replan_obs.get("_pending_targets", ()))
        return []

    def fake_build_queues(*args, **kwargs):
        return [WorkerPlan((4, 4), [])], []

    module.build_tasks = fake_build_tasks
    module.build_queues = fake_build_queues
    module.reserved_items = lambda tasks: Counter()
    global_replan(obs, farm, 5)
finally:
    module.build_tasks = original_build_tasks
    module.build_queues = original_build_queues
    module.reserved_items = original_reserved_items

assert captured["targets"] == {position: ("CARROT", False)}, captured
assert position not in captured["pending"], captured
print("frozen daily target survives intraday reroute:", captured)
