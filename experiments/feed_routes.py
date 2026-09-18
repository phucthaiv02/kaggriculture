"""Count submitted routing/feed orders in replays (not successful engine effects)."""
from collections import Counter
import argparse
import json
from pathlib import Path


def analyze(replay, first_day=7, player=0):
    operations, pickups, feeds = Counter(), Counter(), Counter()
    for previous, current in zip(replay['steps'], replay['steps'][1:]):
        obs = previous[player]['observation']
        day = obs['day']
        if day < first_day:
            continue
        action = current[player].get('action') or {}
        for worker, op in enumerate([action.get('farmer', ['PASS']), *action.get('hands', [])]):
            if not op:
                continue
            operations[op[0]] += 1
            if op[0] == 'PICKUP' and op[1] == 'WHEAT':
                pickups[op[2]] += 1
            if op[0] == 'FEED':
                feeds[day, worker] += 1
    pickup_count = sum(pickups.values())
    pickup_units = sum(quantity * count for quantity, count in pickups.items())
    return {
        'first_day': first_day,
        'moves': sum(operations[op] for op in ('NORTH', 'SOUTH', 'EAST', 'WEST')),
        'wheat_pickups': pickup_count,
        'wheat_units_requested': pickup_units,
        'average_wheat_per_pickup': pickup_units / pickup_count if pickup_count else 0,
        'feed_worker_days': len(feeds),
        'feed_hand_days': sum(worker > 0 for day, worker in feeds),
        'average_feed_jobs_per_worker_day': sum(feeds.values()) / len(feeds) if feeds else 0,
        'wheat_pickups_by_quantity': dict(sorted(pickups.items())),
        'feeds_per_worker_day': dict(sorted(Counter(feeds.values()).items())),
        'operations': dict(operations),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replays', type=Path, nargs='+')
    parser.add_argument('--first-day', type=int, default=7)
    args = parser.parse_args()
    print(json.dumps({str(path): analyze(json.loads(path.read_text()), args.first_day)
                      for path in args.replays}, indent=2))
