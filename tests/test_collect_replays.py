from __future__ import annotations

import io
import zipfile

import pytest

from scripts.collect_replays import (
    agent_index,
    authenticated_session,
    load_environment_file,
    own_submission_ids,
    replay_candidates,
    request_with_retry,
    top_teams,
)


class Session:
    pass


class Response:
    def __init__(
        self, payload=None, content: bytes = b"", status_code: int = 200, headers=None
    ) -> None:
        self.payload = payload
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return self.payload

    def raise_for_status(self) -> None:
        pass


class RequestSession:
    def __init__(self, payload) -> None:
        self.payload = payload

    def request(self, method, url, **kwargs):
        assert method == "GET"
        assert url.endswith("/submissions/list/kaggriculture")
        assert kwargs["params"] == {"pageSize": 100}
        return Response(self.payload)


def leaderboard_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "leaderboard.csv",
            "Rank,TeamId,TeamName,Score\n1,101,Alpha,123.5\n2,202,Beta,100.0\n",
        )
    return buffer.getvalue()


class LeaderboardSession:
    def request(self, method, url, **kwargs):
        assert method == "GET"
        assert url.endswith("/kaggriculture/leaderboard/download")
        return Response(content=leaderboard_archive())


class SequenceSession:
    def __init__(self, responses) -> None:
        self.responses = iter(responses)

    def request(self, method, url, **kwargs):
        return next(self.responses)


def test_authenticated_session_uses_bearer_environment_without_home(monkeypatch) -> None:
    monkeypatch.setattr("pathlib.Path.home", lambda: (_ for _ in ()).throw(AssertionError("home")))
    session = authenticated_session({"KAGGLE_API_TOKEN": "token-value"})
    assert session.headers["Authorization"] == "Bearer token-value"
    assert session.auth is None


def test_authenticated_session_ignores_legacy_environment_pair(monkeypatch) -> None:
    monkeypatch.setattr("pathlib.Path.home", lambda: (_ for _ in ()).throw(AssertionError("home")))
    with pytest.raises(RuntimeError, match="KAGGLE_API_TOKEN"):
        authenticated_session({"KAGGLE_USERNAME": "user", "KAGGLE_KEY": "key"})


def test_authenticated_session_rejects_missing_or_partial_environment() -> None:
    with pytest.raises(RuntimeError, match="KAGGLE_API_TOKEN"):
        authenticated_session({})
    with pytest.raises(RuntimeError, match="KAGGLE_API_TOKEN"):
        authenticated_session({"KAGGLE_USERNAME": "user"})


def test_load_environment_file_reads_dotenv_without_overriding_process_env(
    tmp_path, monkeypatch
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "KAGGLE_API_TOKEN=from-file\nIGNORED_SETTING=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("KAGGLE_API_TOKEN", "already-exported")
    assert load_environment_file(dotenv)
    assert authenticated_session().headers["Authorization"] == "Bearer already-exported"


def test_load_environment_file_allows_missing_default(tmp_path) -> None:
    assert not load_environment_file(tmp_path / ".env")


def test_own_submission_ids_accepts_ref_and_id_response_fields() -> None:
    session = RequestSession({"submissions": [{"ref": 10}, {"id": "20"}]})
    assert own_submission_ids(session, "kaggriculture") == [10, 20]


def test_top_teams_parses_rest_leaderboard_archive() -> None:
    assert top_teams(LeaderboardSession(), "kaggriculture", 1) == [
        {"rank": 1, "team_id": 101, "team_name": "Alpha", "score": 123.5}
    ]


def test_request_retry_honors_retry_after(monkeypatch) -> None:
    sleeps = []
    monkeypatch.setattr("scripts.collect_replays.time.sleep", sleeps.append)
    session = SequenceSession(
        [Response(status_code=429, headers={"Retry-After": "7"}), Response(payload={})]
    )
    assert request_with_retry(session, "GET", "https://example.test").status_code == 200
    assert sleeps == [7.0]


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
        per_team=1,
        preferred_submissions={10},
    )
    assert result[99]["expert_indices"] == [0]
    assert result[99]["rank_by_player"] == {"0": 1}
    assert result[99]["involves_own_submission"]
