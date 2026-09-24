from kaggle_environments.envs.kaggriculture import kaggriculture as game

from experiments.agent_v5_benchmark import replay_health


def _board(entries):
    board = [[None] * 3 for _ in range(3)]
    for (x, y), tile in entries.items():
        board[y][x] = tile
    return board


def _frame(entries):
    return [{"observation": {"farms": [{"tiles": _board(entries)}]}}]


def test_replay_health_counts_failures_and_turnover_gaps():
    replay = {
        "steps": [
            _frame({
                (0, 0): {"kind": "PLANT", "crop": "WHEAT"},
                (1, 0): {"kind": "PASTURE", "animal": "COW"},
                (2, 0): {"kind": "PLANT", "crop": "CARROT"},
            }),
            _frame({
                (0, 0): {"kind": "WEED"},
                (1, 0): {"kind": "PASTURE"},
            }),
            _frame({
                (0, 0): {"kind": "WEED"},
                (1, 0): {"kind": "PASTURE", "animal": "SHEEP"},
            }),
            _frame({
                (0, 0): {"kind": "WEED"},
                (1, 0): {"kind": "PASTURE", "animal": "SHEEP"},
                (2, 0): {"kind": "PLANT", "crop": "MELON"},
            }),
        ]
    }

    health = replay_health(replay)
    assert health["plant_to_weed"] == 1
    assert health["animal_escapes"] == 1
    assert health["turnover_gap_events"] == 2
    assert health["turnover_gap_mean_steps"] == 1.5
    assert health["turnover_gap_max_steps"] == 2
    assert health["open_turnover_gaps_at_end"] == 0


def test_engine_dig_cannot_remove_a_placed_animal():
    tile = {"kind": "PASTURE", "animal": "COW"}
    farm = {"tiles": [[tile]], "farmer": [0, 0], "hands": []}
    private = {"inventories": [{}], "shed": {}, "seeds": {}}

    game._apply_unit_action(
        farm, private, 0, ["DIG"],
        board_size=1, day=0, turns_per_day=24,
    )

    assert farm["tiles"][0][0] is tile
    assert farm["tiles"][0][0]["animal"] == "COW"
