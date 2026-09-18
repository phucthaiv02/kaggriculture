"""Verify shortened-horizon care against engine actions and daily refresh."""
from collections import Counter
from kaggle_environments.envs.kaggriculture import kaggriculture as game
from agents.schedules import ANIMAL_PRODUCTION, should_feed_animal, should_care_animal


def simulate(animal, days, shortened):
    farm = {'tiles': [[game._new_animal(animal, 0)]], 'farmer': [0, 0], 'hands': []}
    private = {'inventories': [{}], 'shed': {}, 'seeds': {}}
    harvested, actions = Counter(), Counter()
    for age in range(days):
        tile = farm['tiles'][0][0]
        assert tile.get('animal') == animal, (animal, days, age, 'escaped')
        last_age = days - 1 if shortened else 29
        ops = []
        if tile.get('yield_units'):
            ops.append('HARVEST')
        if tile.get('fertilizer_available'):
            ops.append('COLLECT_FERTILIZER')
        if should_feed_animal(animal, age, last_age):
            private['inventories'][0]['WHEAT'] = 1
            ops.append('FEED')
        if should_care_animal(animal, age, last_age):
            ops.append('CARE')
        for op in ops:
            before = Counter(private['inventories'][0])
            game._apply_unit_action(farm, private, 0, [op], 1, age, 24)
            if op in ('HARVEST', 'COLLECT_FERTILIZER'):
                harvested.update(Counter(private['inventories'][0]) - before)
            actions[op] += 1
        private['inventories'][0].clear()
        if age < days - 1:
            game._daily_refresh_animals(farm, age)
    return harvested, actions


def verify():
    results = []
    for animal in ANIMAL_PRODUCTION:
        for days in range(1, 31):
            old_goods, old_actions = simulate(animal, days, False)
            goods, actions = simulate(animal, days, True)
            assert goods == old_goods, (animal, days, goods, old_goods)
            assert actions['FEED'] <= old_actions['FEED']
            assert actions['CARE'] <= old_actions['CARE']
            results.append((animal, days, dict(goods),
                            old_actions['FEED'] - actions['FEED'],
                            old_actions['CARE'] - actions['CARE']))
    return results


if __name__ == '__main__':
    results = verify()
    for animal, days, goods, feed, care in results:
        print(f'{animal} horizon={days}: goods={goods}, saved FEED={feed}, CARE={care}')
    print(f'{len(results)} horizons match engine harvest and fertilizer; no animal escapes.')
