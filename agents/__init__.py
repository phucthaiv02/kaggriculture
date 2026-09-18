"""Shared package hooks for the production agent.

The opening is deliberately left byte-for-byte equivalent.  Once the farm has
expanded, animal output and maintenance due on the same tile are kept in one
worker task so routing does not pay twice to visit the same pen.
"""


def _install_expanded_animal_coalescing():
    from . import farm_tasks as _farm_tasks

    original = _farm_tasks.build_tasks
    if getattr(original, "_coalesces_expanded_animal_work", False):
        return

    service_ops = {"FEED", "CARE", "COLLECT_FERTILIZER"}

    def build_tasks(obs, *args, **kwargs):
        tasks = original(obs, *args, **kwargs)
        farm = obs["farms"][obs["player"]]
        # The fixed NW opening has tightly-coupled refinancing and placement
        # timing.  Do not change its task boundaries at all.
        if len(farm.get("unlocked_quadrants", ())) <= 1 or len(tasks) < 2:
            return tasks

        merged = []
        index = 0
        while index < len(tasks):
            harvest = tasks[index]
            if harvest.animal_harvest and index + 1 < len(tasks):
                service = tasks[index + 1]
                same_pen_service = (
                    service.position == harvest.position
                    and not service.animal_harvest
                    and not service.ends_cycle
                    and service.actions
                    and all(action and action[0] in service_ops for action in service.actions)
                )
                if same_pen_service:
                    # build_tasks already emits HARVEST before maintenance.
                    # Preserve that order but make the visit indivisible so a
                    # second worker cannot be sent to this exact same tile.
                    harvest.actions.extend(service.actions)
                    harvest.needs.update(service.needs)
                    harvest.sells.update(service.sells)
                    harvest.urgent = harvest.urgent or service.urgent
                    harvest.immediate_drop = service.immediate_drop
                    harvest.refinance_feed = service.refinance_feed
                    harvest.deadline = harvest.deadline or service.deadline
                    harvest.cashout = harvest.cashout or service.cashout
                    merged.append(harvest)
                    index += 2
                    continue
            merged.append(harvest)
            index += 1
        return merged

    build_tasks.__name__ = original.__name__
    build_tasks.__doc__ = original.__doc__
    build_tasks._coalesces_expanded_animal_work = True
    _farm_tasks.build_tasks = build_tasks


_install_expanded_animal_coalescing()
del _install_expanded_animal_coalescing
