"""Opening regressions for rolling animal dependencies and harvest safety."""

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as official_game

from agents.expansion_agent import make_agent
from agents.farm_tasks import ANIMALS, animal_output_at_risk, build_tasks
from agents.rolling_scheduler import planning_observation
from experiments.crop_schedules import pass_agent
from tests.test_agents_integration import END_DAY, configuration

MOVES = {"EAST": (1, 0), "WEST": (-1, 0), "SOUTH": (0, 1), "NORTH": (0, -1)}


def _animal_targets(targets):
    return {
        position: target for position, target in targets.items()
        if target and target[0] in ANIMALS
    }


def _empty_animal_targets(obs, targets):
    return {
        position: target
        for position, target in _animal_targets(targets).items()
        if not (
            isinstance(obs.farms[0]["tiles"][position[1]][position[0]], dict)
            and obs.farms[0]["tiles"][position[1]][position[0]].get("animal")
            == target[0]
        )
    }


def _owned_animal_count(obs, species):
    total = obs.private["shed"].get(species, 0)
    total += sum(inventory.get(species, 0) for inventory in obs.private["inventories"])
    total += sum(
        1
        for row in obs.farms[0]["tiles"]
        for tile in row
        if isinstance(tile, dict) and tile.get("animal") == species
    )
    return total


def _planned_action_owners(starts, current_ops, plans, action_name):
    owners = {}
    for worker, start in enumerate(starts):
        x, y = start
        operations = []
        if worker < len(current_ops):
            operations.append(current_ops[worker])
        if worker < len(plans):
            operations.extend(plans[worker].queue)
        for operation in operations:
            if not operation:
                continue
            op = operation[0]
            if op in MOVES:
                dx, dy = MOVES[op]
                x, y = x + dx, y + dy
            elif op == action_name:
                owners.setdefault((x, y), []).append(worker)
    return {position: tuple(workers) for position, workers in owners.items()}


def test_intraday_animal_buy_materializes_admitted_place_dependency():
    env = make("kaggriculture", configuration=configuration(1), debug=False)
    agent = make_agent(END_DAY, seed=1)
    cells = {
        name: cell.cell_contents
        for name, cell in zip(agent.__code__.co_freevars, agent.__closure__)
    }
    planner_state = cells["state"]
    targets = cells["targets"]
    state = env.state
    pending_buy = None
    missing_position = None
    materialized = False

    for step in range(int(env.configuration.episodeSteps) - 1):
        obs = state[0].observation
        action = agent(obs)

        if obs.day == 3:
            if pending_buy is not None and obs.hour == pending_buy[0] + 1:
                _buy_hour, species, before_owned, candidate_position = pending_buy
                if _owned_animal_count(obs, species) > before_owned:
                    missing_position = candidate_position
                    x, y = missing_position
                    tile = obs.farms[0]["tiles"][y][x]
                    if isinstance(tile, dict) and tile.get("animal") == species:
                        materialized = True
                    else:
                        frozen = set(planner_state.get("frozen_positions", ()))
                        assert missing_position in frozen, (
                            missing_position, sorted(frozen), planner_state.get("purchase_positions")
                        )
                        frozen_targets = {
                            position: targets.get(position)
                            for position in frozen
                        }
                        replanned = build_tasks(
                            planning_observation(obs),
                            frozen_targets,
                            prioritize_fertilizer_drop=planner_state.get("opening_active", False),
                            include_physical=False,
                        )
                        target_tasks = [
                            task for task in replanned if task.position == missing_position
                        ]
                        assert target_tasks, (
                            missing_position,
                            targets.get(missing_position),
                            dict(obs["private"]["shed"]),
                            frozen_targets,
                        )
                        assert any(
                            ["PLACE", species] in task.actions
                            and task.needs.get(species, 0) >= 1
                            for task in target_tasks
                        ), [(task.actions, dict(task.needs)) for task in target_tasks]
                        materialized = True
                    pending_buy = None
                else:
                    pending_buy = None

            if pending_buy is None and not materialized:
                empty_targets = _empty_animal_targets(obs, targets)
                animal_orders = [
                    order for order in action.get("market", [])
                    if order[:1] == ["BUY_ANIMAL"]
                ]
                for order in animal_orders:
                    species = order[1]
                    candidates = [
                        position for position, target in empty_targets.items()
                        if target[0] == species
                        and position in planner_state.get("purchase_positions", set())
                    ]
                    if candidates:
                        pending_buy = (
                            obs.hour,
                            species,
                            _owned_animal_count(obs, species),
                            sorted(candidates)[0],
                        )
                        break

        if obs.day == 4 and obs.hour == 0:
            assert materialized, (
                "no successful Day-3 BUY_ANIMAL matched an admitted empty target",
                _empty_animal_targets(obs, targets),
                planner_state.get("purchase_positions"),
            )
            assert missing_position is not None
            x, y = missing_position
            tile = obs.farms[0]["tiles"][y][x]
            assert isinstance(tile, dict) and tile.get("animal") == targets[missing_position][0], (
                missing_position, tile, planner_state.get("plans")
            )
            animal_count = sum(
                1
                for row in obs.farms[0]["tiles"]
                for current in row
                if isinstance(current, dict) and current.get("animal")
            )
            assert animal_count == 6, animal_count
            return

        state[0].action = action
        state[1].action = pass_agent(state[1].observation)
        state = official_game.interpreter(state, env)
        state[0].observation.step = step + 1

    raise AssertionError("simulation ended before the opening animal regression completed")


def test_trace_first_at_risk_animal_harvest_route():
    """Expose where the first clipping-prevention HARVEST falls out of execution."""
    env = make("kaggriculture", configuration=configuration(1), debug=False)
    agent = make_agent(END_DAY, seed=1)
    cells = {
        name: cell.cell_contents
        for name, cell in zip(agent.__code__.co_freevars, agent.__closure__)
    }
    planner_state = cells["state"]
    targets = cells["targets"]
    state = env.state
    position = (1, 4)
    trace = []

    for step in range(int(env.configuration.episodeSteps) - 1):
        obs = state[0].observation
        action = agent(obs)
        farm = obs.farms[0]
        starts = [tuple(farm["farmer"]), *map(tuple, farm["hands"])]
        current_ops = [action.get("farmer", ["PASS"]), *action.get("hands", [])]

        if obs.day == 8:
            frozen = set(planner_state.get("frozen_positions", ()))
            frozen_targets = {p: targets.get(p) for p in frozen}
            generated = build_tasks(
                planning_observation(obs),
                frozen_targets,
                prioritize_fertilizer_drop=planner_state.get("opening_active", False),
                include_physical=False,
            )
            target_task = next((task for task in generated if task.position == position), None)
            owners = _planned_action_owners(
                starts, current_ops, planner_state.get("plans", ()), "HARVEST"
            )
            actual = next(
                (tuple(operation) for start, operation in zip(starts, current_ops)
                 if start == position and operation != ["PASS"]),
                None,
            )
            tile = farm["tiles"][position[1]][position[0]]
            trace.append((
                obs.hour,
                position in frozen,
                position in {task.position for task in planner_state.get("unassigned", ())},
                tuple(op[0] for op in target_task.actions) if target_task else None,
                owners.get(position),
                actual,
                tuple(len(plan.queue) for plan in planner_state.get("plans", ())),
                bool(planner_state.get("replan_needed")),
                bool(planner_state.get("route_invalidated")),
                len(farm["hands"]),
                tile.get("yield_units") if isinstance(tile, dict) else None,
            ))

        if obs.day == 8 and obs.hour == 23:
            tile = farm["tiles"][position[1]][position[0]]
            harvested = any(
                start == position and operation == ["HARVEST"]
                for start, operation in zip(starts, current_ops)
            )
            if isinstance(tile, dict) and animal_output_at_risk(tile, obs.day) and not harvested:
                raise AssertionError(
                    "risk_trace=" + repr(trace)
                    + f"; hand_target={planner_state.get('hand_target')}"
                    + f"; mandatory_hand_target={planner_state.get('mandatory_hand_target')}"
                )
            return

        state[0].action = action
        state[1].action = pass_agent(state[1].observation)
        state = official_game.interpreter(state, env)
        state[0].observation.step = step + 1

    raise AssertionError("simulation ended before day eight animal risk trace completed")
