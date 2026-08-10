from __future__ import annotations

from scripts.collect_replays import agent_index, replay_candidates


class Session:
    pass


def test_agent_index_defaults_to_agent_order() -> None:
    assert agent_index({"submissionId": 10}, 0) == 0
    assert agent_index({"index": 1}, 0) == 1


def test_candidates_only_label_matching_top_team() -> None:
    episode = {
        "id": 99,
        "state": "COMPLETED",
        "endTime": "2026-01-01",
        "agents": [
            {"submissionId": 10, "teamId": 100},
            {"submissionId": 20, "teamId": 200, "index": 1},
        ],
    }
    result = replay_candidates(
        Session(),
        [{"team_id": 100, "rank": 1}],
        {100: {10}},
        {10: [episode]},
        per_team=10,
        preferred_submissions={10},
    )
    assert result[99]["expert_indices"] == [0]
    assert result[99]["rank_by_player"] == {"0": 1}
    assert result[99]["involves_own_submission"]
