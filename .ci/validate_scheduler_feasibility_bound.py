import base64
import gzip
import importlib.util
import json
import random
import sys
import time
from collections import Counter

from agents.farm_tasks import Task
import agents.scheduler as new


def load_old():
    spec = importlib.util.spec_from_file_location('scheduler_old', '/tmp/scheduler_old.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def task_index(tasks, values):
    mapping = {id(task): index for index, task in enumerate(tasks)}
    return [mapping[id(task)] for task in values]


def differential(old):
    rng = random.Random(20260923)
    item_names = ('WHEAT', 'FERTILIZER', 'COW', 'GOOSE')
    service_ops = ('FEED', 'CARE', 'COLLECT_FERTILIZER')
    normal_ops = ('WATER', 'HARVEST', 'FERTILIZE', 'DIG')

    for case in range(180):
        size = rng.randrange(0, 34)
        tasks = []
        for _ in range(size):
            position = (rng.randrange(10), rng.randrange(10))
            urgent = rng.random() < 0.25
            animal_harvest = urgent and rng.random() < 0.18
            action_count = rng.randrange(1, 5)
            if urgent and not animal_harvest and rng.random() < 0.65:
                first = rng.choice(service_ops)
            else:
                first = rng.choice(normal_ops)
            actions = [[first]] + [[rng.choice(normal_ops)] for _ in range(action_count - 1)]
            needs = Counter()
            for item in item_names:
                if rng.random() < 0.12:
                    needs[item] = rng.randrange(1, 4)
            immediate_drop = rng.random() < 0.06
            cashout = (not immediate_drop) and rng.random() < 0.05
            mandatory_roll = rng.random()
            mandatory = False if mandatory_roll < 0.22 else (True if mandatory_roll < 0.45 else None)
            tasks.append(Task(
                position, actions, needs=needs, urgent=urgent,
                sells=Counter({'WHEAT': rng.randrange(1, 4)}) if rng.random() < 0.12 else Counter(),
                immediate_drop=immediate_drop,
                refinance_feed=immediate_drop and rng.random() < 0.12,
                animal_harvest=animal_harvest,
                cashout=cashout,
                terminal_day=rng.random() < 0.04,
                mandatory=mandatory,
                value=float(rng.randrange(0, 250)),
            ))

        farmer = (rng.randrange(10), rng.randrange(10))
        existing_count = rng.randrange(0, 4)
        existing = tuple(rng.choice(old.SHED_ACCESS) for _ in range(existing_count))
        max_hands = rng.randrange(max(existing_count, 1), 9)
        pending_budget = rng.randrange(10, 24)
        existing_budget = rng.randrange(10, 24)
        economic = rng.random() < 0.55
        costs = ([float(1 << i) for i in range(max_hands - existing_count)] if economic else None)

        kwargs = dict(
            farmer_start=farmer,
            existing_hand_starts=existing,
            pending_hand_budget=pending_budget,
            existing_hand_budget=existing_budget,
            max_hands=max_hands,
            marginal_hire_costs=costs,
        )
        before = old.hands_needed(tasks, **kwargs)
        after = new.hands_needed(tasks, **kwargs)
        assert before[0] == after[0], ('hands', case, before[0], after[0])
        assert task_index(tasks, before[1]) == task_index(tasks, after[1]), ('missing', case)

        hand_count = rng.randrange(0, max_hands + 1)
        hand_starts = existing[:hand_count]
        queue_kwargs = dict(
            farmer_start=farmer,
            hand_count=hand_count,
            hand_starts=hand_starts,
            pending_hand_budget=pending_budget,
            existing_hand_budget=existing_budget,
            available_wheat=rng.randrange(0, 20),
        )
        old_plans, old_missing = old.build_queues(tasks, **queue_kwargs)
        new_plans, new_missing = new.build_queues(tasks, **queue_kwargs)
        assert [plan.queue for plan in old_plans] == [plan.queue for plan in new_plans], ('queues', case)
        assert task_index(tasks, old_missing) == task_index(tasks, new_missing), ('queue missing', case)
    print('random differential: 180/180 exact')


PAYLOAD = 'H4sIAANus2oC/+1ca3OiOhj+L/nM7oSbgN+oy1anVhx1xzntdBhW0y1TBA+ge7qO//0keKNUu4gXiOaLA0HyPnny5sntDTMQoJHteI73y5yiwP6Fes4IgaqsfpUFkVcFDoQRGuMEpcKBsWu/oQBUIQee7WAUgurjDIx8D72BqlTRhMpX/CRyXESePHoT1+Vm4NXxhqAK2k291QMcGAQ+zg7064ZObnGWXoSG1tDGeQjY3G87QgFOiPw46dl2Q4Tf8r0QDSaRM0XWxFv+B1R5Drw5yB3iNCcK4/uR/Z/lOs8oHNuetcBekVQMGAUYmfMHkT/jq4XFL/w8AVHv9n50DIzK9pyR7eKkmtlfgBysMPJqyiZMo3smyAhFH0thfyzZGldg2VPbce2fLuY/Cib42Rh5Q1wzFnnP+ul7E2xOmu/JKb8npzBVPnE7p4K0i1NBnnOf1n2319H7N0an808aLA8PBCtsBfuF/wxrDpj8gX4K94UpJGDWTLOd9NFb0+waaS/dQsw5vZTPRatyZlpJ6yee+sSlxSqTEijFKsGqmW1+9nIRlUYPoUQfKIFZRHsrf+f1UQ1K283SPhIoI5V0a+d2Rg/VI3k7oxDuZFSib1xFwwwlM5PiuZmUaNLNv472unXDaKcavVTscE/cd6YqFotXoFdTWRdQXBdQxNA5/xwwbb1SoBfQj3eb6qYRSwzxxQoZ093CdFc9iex+BlPM1T0I5+4eRDKupQNo7lX3IrAymMeECZpm7c74BrJc/G2exmqAwaTRny/BqUs0hojp3LMeKFhLy+slBSyoXaYzF1Lrx+wfaXFmuhewS9/rPC2CoEg81KPESfj2xfaGJPIJX0481x+8Ygv/TuxhgMtF0kGLLMe0yIy22wfkBSdA4ap4kNC0DKZSVEGW3gVT7WKwpnc65vFX1feezl2CNx6n8+ULolI8sMp3UKkIx6OSQbw4iEfczIWHb+buPQQtpHEzLadFyzNuNuerbmpZFE4jQMceQlKypH1+mNTEPOebiVHQhZ9I0ncHvVCnQ1k2s+XyBLhkwquVCO+FdEWlakhsdzXf7up7je8bxjeQM0SPBYucHrFS6jDIbYhVprtXPL4+v6jJp1+HOOnEdMc6xIncMmtfS1nIc65wTJGy8NGy41Uo8weFzWiufUZTBMR8EY5lOqx1fTDhUWFmOu/OU6ZOJQB8pPgp5t5MLBibmWCyFsdgXi9MCmK12WcBCoyqZG58ZScjKNg6v9xzGxREIhQSy1NwnDVOGQfOFBcJVGcgfCHlmi35qfIyt1oAJ4Xtmfd6z4wvE65I9j7ujabZWrihcXuLqSRpjeZdfKi+b5rN+KXvRqfXaDYejM7iDMniSDN5QuarcbbxjhUJ/wYhQqRgaywwCxRxDUXFWTjeFHmRHzhx2Phs/jQnFRS8oojku3r6ljCiyYq4saNpUNqY0tSK+t6apgmJskNeVpbF11QNrgjQNJLlggMeQllL8QBFWZvHlTBAyfJKlQ0QSdjAIN8eTYLgNX6NQVraJ0AW1gUor42/Nyyoc2w28n97xOjad8IXfxz7zY1+Z8RC0248POhWtx4fR7/p/GjV6la3He+JdO9Ns1dvGKun6fu20bNq+nfjw4vJO1Iry1M+2Dv9Cfmm6vx/ft1zuoxVAAA='


def replay_benchmark(old):
    import agents.expansion_agent as ea
    from agents import planner
    from agents.forecast import _cached_production_snapshot

    obs = json.loads(gzip.decompress(base64.b64decode(PAYLOAD)))
    farm = obs['farms'][obs['player']]
    active, targets = [], {}
    for y, row in enumerate(farm['tiles']):
        for x, tile in enumerate(row):
            if tile == 'LOCKED':
                continue
            position = (x, y)
            active.append(position)
            if isinstance(tile, dict):
                name = tile.get('animal') or (tile.get('crop') if tile.get('kind') == 'PLANT' else None)
                if name:
                    targets[position] = (name, False)
    _cached_production_snapshot.cache_clear()
    planner.plan_targets(obs, targets, active, 29, max_positions=10, replan_positions=set(active))
    tasks = ea.build_tasks(
        obs, {p: t for p, t in targets.items() if t is not None},
        assume_crop_seeds=True, assume_animal_inputs=True,
    )
    mandatory = [task for task in tasks if task.mandatory is not False]
    starts = tuple(map(tuple, farm['hands']))
    shed = ea._open_shed_access(farm)
    costs = [
        ea._hire_costs(farm, n + 1) - ea._hire_costs(farm, n)
        for n in range(new.MAX_HANDS - len(farm['hands']))
    ]

    def bench(module, work, marginal, rounds=3):
        original = module._pack_greedy
        calls = [0]
        def counted(*args, **kwargs):
            calls[0] += 1
            return original(*args, **kwargs)
        module._pack_greedy = counted
        try:
            elapsed = []
            result = None
            final_calls = 0
            for _ in range(rounds):
                calls[0] = 0
                start = time.perf_counter()
                result = module.hands_needed(
                    work, tuple(farm['farmer']), starts, shed,
                    marginal_hire_costs=marginal,
                )
                elapsed.append(time.perf_counter() - start)
                final_calls = calls[0]
            return min(elapsed), final_calls, result
        finally:
            module._pack_greedy = original

    old_all, old_all_calls, old_all_result = bench(old, tasks, costs)
    new_all, new_all_calls, new_all_result = bench(new, tasks, costs)
    old_mand, old_mand_calls, old_mand_result = bench(old, mandatory, None)
    new_mand, new_mand_calls, new_mand_result = bench(new, mandatory, None)
    assert old_all_result[0] == new_all_result[0]
    assert task_index(tasks, old_all_result[1]) == task_index(tasks, new_all_result[1])
    assert old_mand_result[0] == new_mand_result[0]
    assert task_index(mandatory, old_mand_result[1]) == task_index(mandatory, new_mand_result[1])
    old_total = old_all + old_mand
    new_total = new_all + new_mand
    print(f'day24 all old={old_all:.6f}s/{old_all_calls}greedy new={new_all:.6f}s/{new_all_calls}greedy')
    print(f'day24 mandatory old={old_mand:.6f}s/{old_mand_calls}greedy new={new_mand:.6f}s/{new_mand_calls}greedy')
    print(f'day24 sizing total old={old_total:.6f}s new={new_total:.6f}s speedup={old_total/max(new_total,1e-9):.2f}x')
    assert new_total < old_total * 0.90, (old_total, new_total)


old = load_old()
differential(old)
replay_benchmark(old)
