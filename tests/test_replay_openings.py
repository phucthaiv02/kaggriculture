from copy import deepcopy
import json

import pytest

from experiments.analyze_openings import analyze_inputs, extract_opening, mine_patterns, opening_families
from experiments.collect_replays import collect


def replay():
    farm = {"tiles": [[None]], "money": 3000, "hires_today": 0,
            "hands": [], "unlocked_quadrants": ["NW"]}
    frames = []
    for tick in range(5):
        farms = [deepcopy(farm), deepcopy(farm)]
        if tick >= 1:
            farms[1]["hires_today"] = 2
        if tick >= 2:
            farms[1]["tiles"][0][0] = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0}
        frames.append([{"observation": {"step": tick, "farms": farms},
                        "action": {"market": [["BUY_SEED", "MELON", 999]]},
                        "status": "DONE" if tick == 4 else "ACTIVE",
                        "reward": 10 if player == 1 else 5} for player in range(2)])
    return {"name": "kaggriculture", "configuration": {"turnsPerDay": 2}, "steps": frames}


def test_action_alignment_actual_effects_and_day_boundary():
    result = extract_opening(replay(), 1, days=1)
    assert result["opening_complete"]
    assert result["first_seen"] == {"HIRE": 0, "PLANT:WHEAT": 1}
    assert result["visible_changes"] == {"HIRE": 2, "PLANT:WHEAT": 1}
    assert "PLANT:MELON" not in result["visible_changes"]
    assert result["submitted_market_quantities"] == {"BUY_SEED:MELON": 1998}
    assert result["daily"][0]["complete_day"]
    assert result["outcome"] == "win"
    assert result["final_margin"] == 5


def test_partial_openings_and_failed_matches():
    data = replay()
    data["steps"] = data["steps"][:2]
    result = extract_opening(data, 1, days=2)
    assert not result["opening_complete"]
    assert result["outcome"] is None
    assert mine_patterns([result], min_support=1) == []
    data = replay()
    data["steps"][-1][0]["status"] = "ERROR"
    assert extract_opening(data, 1, 1)["outcome"] is None


def test_support_counts_openings_not_repetitions_and_separates_cohorts():
    row = {**extract_opening(replay(), 1, 1), "episode_id": "1", "team_id": 42, "team_name": "top"}
    row["sequence"] = ["HIRE", "PLANT:WHEAT"] * 5
    other = {**deepcopy(row), "episode_id": "2"}
    incompatible = {**deepcopy(row), "cohort": "other", "episode_id": "3"}
    patterns = mine_patterns([row, other, incompatible], min_support=2)
    assert patterns
    assert all(p["support"] == 2 and p["support_fraction"] == 1 for p in patterns)
    assert all(p["cohort"] == row["cohort"] for p in patterns)


class FakeSource:
    def __init__(self):
        self.downloads = 0

    def leaderboard(self, competition, count):
        return [{"teamId": 20, "teamName": "top", "score": "123"},
                {"teamId": 10, "teamName": "second", "score": "100"}][:count]

    def submissions(self, team_id):
        return [{"id": team_id, "publicScore": "100"},
                {"id": 999, "publicScore": "9"}]

    def episodes(self, submission_id):
        assert submission_id in (10, 20)  # numeric score sort, not lexicographic
        episode = {"id": 1, "state": "COMPLETED", "type": "EPISODE_TYPE_PUBLIC",
                   "endTime": "2026-09-09", "agents": [
                       {"index": 1, "teamId": 20, "submissionId": 20, "teamName": "top"},
                       {"index": 0, "teamId": 10, "submissionId": 10, "teamName": "second"}]}
        return [{**episode, "id": 2, "state": "RUNNING"},
                {**episode, "id": 3, "type": "EPISODE_TYPE_PRIVATE"}, episode]

    def download(self, episode_id, directory):
        assert episode_id == 1
        self.downloads += 1
        path = directory / "replay.json"
        path.write_text(json.dumps(replay()))
        return path


def test_collect_dedup_resume_and_correct_seat(tmp_path):
    source = FakeSource()
    manifest = collect(source, tmp_path, top=1, per_team=5)
    assert not manifest["errors"]
    assert manifest["episodes"]["1"]["target_players"] == [1]
    rows, skipped = analyze_inputs(tmp_path, days=1)
    assert not skipped
    assert len(rows) == 1 and rows[0]["player"] == 1 and rows[0]["team_id"] == 20
    manifest = collect(source, tmp_path, top=2)
    assert manifest["episodes"]["1"]["target_players"] == [0, 1]
    assert source.downloads == 1
    rows, skipped = analyze_inputs(tmp_path, days=1)
    assert not skipped and len(rows) == 2


def test_corrupt_cache_is_replaced_and_checksum_enforced(tmp_path):
    source = FakeSource()
    collect(source, tmp_path, top=1)
    path = tmp_path / "episode-1-replay.json"
    path.write_text("partial download")
    rows, skipped = analyze_inputs(tmp_path, days=1)
    assert not rows and "checksum" in skipped[0]["error"]
    collect(source, tmp_path, top=1)
    assert source.downloads == 2
    assert not analyze_inputs(tmp_path, days=1)[1]


def test_local_directory_and_wrapped_replay(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"replay": json.dumps(replay())}))
    (tmp_path / "a.result.json").write_text("{}")
    rows, skipped = analyze_inputs(tmp_path, days=1)
    assert not skipped and len(rows) == 2


def test_invalid_limits_and_noncontiguous_frames():
    with pytest.raises(ValueError):
        extract_opening(replay(), 1, days=0)
    data = replay()
    data["steps"][1][0]["observation"]["step"] = 0
    with pytest.raises(ValueError, match="contiguous"):
        extract_opening(data, 1, days=1)


def test_timed_families_distinguish_quantities():
    row = {**extract_opening(replay(), 1, 1), "episode_id": "1", "team_id": 1, "team_name": "one"}
    other = {**deepcopy(row), "episode_id": "2", "team_id": 2, "team_name": "two"}
    assert opening_families([row, other])[0]["teams"] == 2
    other["events"][0]["changes"]["HIRE"] = 3
    assert opening_families([row, other]) == []


def test_bad_download_is_not_cached(tmp_path):
    source = FakeSource()
    def download(episode_id, directory):
        path = directory / "replay.json"
        path.write_text('<html>not a replay</html>')
        return path
    source.download = download
    manifest = collect(source, tmp_path, top=1)
    assert manifest["errors"] and not manifest["episodes"]
    assert not (tmp_path / "episode-1-replay.json").exists()
    assert json.loads((tmp_path / "manifest.json").read_text())["errors"]


def test_draws_are_scored_separately():
    data = replay()
    data["steps"][-1][0]["reward"] = 10
    assert extract_opening(data, 1, 1)["outcome"] == "draw"
