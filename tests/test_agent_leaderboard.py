import json
import base64
import gzip
from pathlib import Path

from experiments.agent_leaderboard import (
    AgentSpec, _write_run_state, discover_public_agents, play_cached, prepare_run,
    render_html, run, standings,
)
from experiments.play_match import notebook_agent


def specs():
    return [AgentSpec("alpha", "a", lambda seed: None),
            AgentSpec("beta", "b", lambda seed: None)]


def test_standings_count_wins_draws_and_margins():
    table = standings(specs(), [
        {"players": ["alpha", "beta"], "rewards": [12, 10]},
        {"players": ["beta", "alpha"], "rewards": [7, 7]},
    ])
    assert table[0]["agent"] == "alpha"
    assert table[0]["score"] == .75
    assert table[0]["average_margin"] == 1
    assert table[1]["losses"] == 1


def test_play_cached_does_not_start_environment(monkeypatch, tmp_path):
    left, right = specs()
    monkeypatch.setattr("experiments.agent_leaderboard.version", lambda package: "test")
    monkeypatch.setattr("experiments.agent_leaderboard.configuration",
                        lambda seed: {"seed": seed})
    from experiments.agent_leaderboard import _match_key
    key = _match_key(left, right, 100, {"seed": 100})
    cache = tmp_path / "cache"
    cache.mkdir()
    expected = {"players": ["old alpha", "old beta"],
                "fingerprints": ["a", "b"], "rewards": [1, 0]}
    (cache / f"{key}.json").write_text(json.dumps(expected))
    monkeypatch.setattr("experiments.agent_leaderboard.make",
                        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    result, hit = play_cached(left, right, 100, cache)
    assert hit is True
    assert result["players"] == ["alpha", "beta"]
    assert result["rewards"] == [1, 0]


def test_html_escapes_agent_names():
    table = standings([AgentSpec("<agent>", "a", lambda seed: None),
                       AgentSpec("other", "b", lambda seed: None)], [])
    page = render_html(table, 100, 0, 1)
    assert "&lt;agent&gt;" in page
    assert "Seed 100" in page


def test_run_implicitly_includes_current(monkeypatch, tmp_path):
    played = []
    monkeypatch.setattr("experiments.agent_leaderboard.resolve_agent",
                        lambda value: AgentSpec(value, value, lambda seed: None))

    def fake_play(left, right, seed, cache_dir, force):
        played.append((left.label, right.label, seed))
        return {"players": [left.label, right.label],
                "fingerprints": [left.fingerprint, right.fingerprint],
                "rewards": [1, 0]}, False

    monkeypatch.setattr("experiments.agent_leaderboard.play_cached", fake_play)
    run(["pass"], tmp_path, 100, workers=1)
    assert played == [("current", "pass", 100)]


def test_discovers_supported_public_agents_in_filename_order(tmp_path):
    (tmp_path / "z.py").write_text("def agent(obs): pass")
    (tmp_path / "a.ipynb").write_text("{}")
    (tmp_path / "notes.txt").write_text("ignored")
    assert [Path(path).name for path in discover_public_agents(tmp_path)] == [
        "a.ipynb", "z.py",
    ]


def test_interrupted_seed_is_reused_then_next_run_gets_new_seed(monkeypatch, tmp_path):
    monkeypatch.setattr("experiments.agent_leaderboard.secrets.randbits",
                        lambda bits: 987654321)
    state, first = prepare_run(tmp_path, specs())
    assert first["seed"] == 987654321
    monkeypatch.setattr("experiments.agent_leaderboard.secrets.randbits",
                        lambda bits: 123)
    _, resumed = prepare_run(tmp_path, specs())
    assert resumed["seed"] == 987654321
    first["status"] = "completed"
    _write_run_state(tmp_path, state)
    _, second = prepare_run(tmp_path, specs())
    assert second["seed"] == 123


def test_rerun_current_only_forces_current_pairs(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("experiments.agent_leaderboard.resolve_agent",
                        lambda value: AgentSpec(value, value, lambda seed: None))

    def fake_play(left, right, seed, cache_dir, force):
        calls.append(({left.label, right.label}, force, seed))
        return {"players": [left.label, right.label],
                "fingerprints": [left.fingerprint, right.fingerprint],
                "rewards": [1, 0]}, not force

    monkeypatch.setattr("experiments.agent_leaderboard.play_cached", fake_play)
    run(["alpha", "beta"], tmp_path, 42, rerun_current=True, workers=1)
    assert [force for pair, force, seed in calls if "current" in pair] == [True, True]
    assert [force for pair, force, seed in calls if "current" not in pair] == [False]
    assert {seed for pair, force, seed in calls} == {42}


def test_notebook_agent_extracts_compressed_embedded_source(tmp_path):
    source = b"def agent(obs, config=None):\n    return {'farmer': ['PASS']}\n"
    payload = base64.b85encode(gzip.compress(source)).decode()
    notebook = tmp_path / "packed.ipynb"
    notebook.write_text(json.dumps({"cells": [{
        "cell_type": "code",
        "source": [
            "import base64, gzip\n",
            f"PAYLOAD = {payload!r}\n",
            "SOURCE = gzip.decompress(base64.b85decode(PAYLOAD))\n",
        ],
    }]}))
    agent, digest = notebook_agent(notebook)
    assert callable(agent)
    assert len(digest) == 64
