from copy import deepcopy
import json
import re

from experiments.operations_report import Operations, render
from experiments.sales_report import analyze, operations_path, run
from test_sales_report import replay_fixture


def test_losses_exclude_random_weeds_and_track_each_player_and_used_land():
    operations = Operations(2, 24)
    before = [{"tiles": [[
        {"kind": "PLANT", "crop": "WHEAT", "consecutive_unwatered": 1, "watered_today": False},
        None,  # harvested/dug before refresh: random weed must not count
        {"kind": "PASTURE", "animal": "COW", "consecutive_unfed": 1, "fed_today": False},
        {"kind": "PLANT", "crop": "MELON", "max_lifespan_step": 23, "yield_units": 1},
    ]]}, {"tiles": [[{"kind": "PLANT", "crop": "CARROT"}, "LOCKED", None, None]]}]
    after = [{"tiles": [[{"kind": "WEED"}, {"kind": "WEED"}, {"kind": "PASTURE"}, {"kind": "WEED"}]]},
             {"tiles": [[{"kind": "PLANT", "crop": "CARROT"}, "LOCKED", {"kind": "COOP", "animal": "GOOSE"}, None]]}]
    operations.finish_turn(before, after, 23)
    operations.finish_turn(after, after, 24)  # persistent weeds aren't new deaths
    report = operations.report()
    assert report["totals"][0]["crop_deaths"] == 2
    assert report["totals"][0]["animal_losses"] == 1
    assert report["totals"][1]["crop_deaths"] == 0
    assert [event["cause"] for event in report["loss_events"]] == ["unwatered", "unfed", "decay"]
    assert {event["day"] for event in report["loss_events"]} == {0}
    p0_day0, p0_day1, p1_day0, _ = report["daily"]
    assert p0_day0["complete_day"] and not p0_day1["complete_day"]
    assert p0_day0["empty_structures"] == 1 and p0_day0["used_tiles"] == 0
    assert p1_day0["used_tiles"] == 2
    assert p1_day0["crop_tiles"] == p1_day0["animal_tiles"] == 1
    assert p1_day0["unlocked_tiles"] == 3


def test_action_counts_use_real_workers_group_moves_and_distinguish_noops():
    replay, _ = replay_fixture()
    initial = replay["steps"][0]
    final = deepcopy(initial)
    initial[0]["observation"]["farms"][0]["hands"] = [[4, 4]]
    initial[0]["observation"]["private"]["inventories"].append({})
    final[0]["action"] = {"farmer": ["NORTH"], "hands": [["WATER"], ["FEED"]]}
    final[1]["action"] = {}  # implicit PASS
    replay["steps"] = [initial, final]
    report = analyze(replay)["operations"]
    p0, p1 = report["totals"]
    assert p0["actions"] == {"MOVE": 1, "WATER": 1}
    assert p0["effective_actions"] == {"MOVE": 1}
    assert p1["actions"] == {"PASS": 1}
    assert p1["effective_actions"] == {}


def test_newly_planted_crop_dying_same_turn_is_counted():
    replay, _ = replay_fixture()
    initial = replay["steps"][0]
    initial[0]["observation"]["step"] = 23
    initial[0]["observation"]["private"]["seeds"]["WHEAT"] = 1
    final = deepcopy(initial)
    final[0]["action"] = {"farmer": ["PLANT", "WHEAT"]}
    final[0]["observation"]["farms"][0]["tiles"][4][4] = {"kind": "WEED"}
    replay["steps"] = [initial, final]
    report = analyze(replay)["operations"]
    assert report["totals"][0]["crop_deaths"] == 1
    assert report["loss_events"][0]["cause"] == "unwatered"
    assert report["loss_events"][0]["position"] == [4, 4]


def test_atomic_seed_shortage_counts_attempts_but_not_effective_plants():
    replay, _ = replay_fixture()
    initial = replay["steps"][0]
    initial[0]["observation"]["private"]["seeds"]["WHEAT"] = 1
    initial[0]["observation"]["private"]["inventories"].append({})
    initial[0]["observation"]["farms"][0]["hands"] = [[3, 4]]
    final = deepcopy(initial)
    final[0]["action"] = {"farmer": ["PLANT", "WHEAT"], "hands": [["PLANT", "WHEAT"]]}
    replay["steps"] = [initial, final]
    p0 = analyze(replay)["operations"]["totals"][0]
    assert p0["actions"] == {"PLANT": 2}
    assert p0["effective_actions"] == {}


def test_free_last_turn_hire_counts_even_when_daily_reset_removes_hands():
    replay, _ = replay_fixture()
    replay["configuration"]["farmHandCostMult"] = 0
    initial = replay["steps"][0]
    initial[0]["observation"]["step"] = 23
    final = deepcopy(initial)
    final[0]["action"] = {"market": [["HIRE"]]}
    replay["steps"] = [initial, final]
    report = analyze(replay)["operations"]
    assert report["totals"][0]["hires"] == 1
    assert report["daily"][0]["day"] == 0
    assert report["daily"][0]["complete_day"]


def test_unaffordable_hire_does_not_count():
    replay, _ = replay_fixture()
    initial = replay["steps"][0]
    initial[0]["observation"]["farms"][0]["money"] = 0
    final = deepcopy(initial)
    final[0]["action"] = {"market": [["HIRE"]]}
    replay["steps"] = [initial, final]
    assert analyze(replay)["operations"]["totals"][0]["hires"] == 0


def test_run_writes_two_offline_html_files_and_json(tmp_path):
    replay, _ = replay_fixture()
    source = tmp_path / "replay.json"
    source.write_text(json.dumps(replay))
    sales = run(source)
    operations = operations_path(sales)
    assert sales.name == "sales_analysis.html" and sales.is_file()
    assert operations.name == "operations_analysis.html" and operations.is_file()
    page = operations.read_text()
    assert "<svg" in page and "Player 0" in page and "Player 1" in page
    assert "Ô đang sản xuất" in page and "MOVE" in page and "FEED" in page
    assert not re.search(r'(?:src|href)=["\']https?://', page)
    report = json.loads((tmp_path / "sales_analysis.json").read_text())
    escaped = render(report["operations"], "<script>bad</script>")
    assert "&lt;script&gt;" in escaped and "<script>" not in escaped
    assert operations_path(tmp_path / "custom.html").name == "custom.operations.html"
