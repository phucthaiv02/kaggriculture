#!/usr/bin/env python3
"""Discover top leaderboard submissions and download their public replays."""

from __future__ import annotations

import argparse
import csv
import heapq
import io
import json
import os
import time
import zipfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


EPISODES_URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"
REPLAY_URL = "https://www.kaggle.com/api/v1/competitions/episodes/{episode_id}/replay"
LEADERBOARD_URL = "https://www.kaggle.com/api/v1/competitions/{competition}/leaderboard/download"
SUBMISSIONS_URL = "https://www.kaggle.com/api/v1/competitions/submissions/list/{competition}"


class RateLimitedSession(requests.Session):
    """Requests session with a minimum start-to-start interval."""

    def __init__(self, request_interval: float = 0.0) -> None:
        super().__init__()
        self.request_interval = max(0.0, float(request_interval))
        self._last_request_started: float | None = None

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        now = time.monotonic()
        if self._last_request_started is not None:
            remaining = self.request_interval - (now - self._last_request_started)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_started = time.monotonic()
        return super().request(method, url, **kwargs)


def load_environment_file(path: str | Path) -> bool:
    """Load a dotenv file without replacing variables already in the process."""
    dotenv_path = Path(path)
    if not dotenv_path.is_file():
        return False
    load_dotenv(dotenv_path=dotenv_path, override=False)
    return True


def authenticated_session(
    environment: Mapping[str, str] | None = None, request_interval: float = 0.0
) -> requests.Session:
    """Build a Kaggle session exclusively from environment variables.

    Bearer KAGGLE_API_TOKEN is preferred. The standard Kaggle
    KAGGLE_USERNAME/KAGGLE_KEY pair is used as a fallback. Values may already
    exist in the process or have been loaded from a dotenv file. This function
    deliberately never reads ~/.kaggle, ~/.config, or another credential file.
    """
    environment = os.environ if environment is None else environment
    session = RateLimitedSession(request_interval=request_interval)
    token = environment.get("KAGGLE_API_TOKEN", "").strip()
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
        return session
    username = environment.get("KAGGLE_USERNAME", "").strip()
    key = environment.get("KAGGLE_KEY", "").strip()
    if username and key:
        session.auth = (username, key)
        return session
    raise RuntimeError(
        "Kaggle credentials are missing. Set KAGGLE_API_TOKEN, or set both "
        "KAGGLE_USERNAME and KAGGLE_KEY in .env/process environment. "
        "No home-directory credential source is used."
    )


def request_with_retry(
    session: requests.Session, method: str, url: str, *, attempts: int = 8, **kwargs: Any
) -> requests.Response:
    for attempt in range(attempts):
        response = session.request(method, url, timeout=60, **kwargs)
        if response.status_code not in (429, 500, 502, 503, 504):
            response.raise_for_status()
            return response
        if attempt + 1 < attempts:
            retry_after = getattr(response, "headers", {}).get("Retry-After")
            try:
                delay = float(retry_after) if retry_after is not None else 2.0 ** (attempt + 1)
            except ValueError:
                delay = 2.0 ** (attempt + 1)
            delay = min(120.0, max(1.0, delay))
            print(
                f"HTTP {response.status_code} for {url}; retry "
                f"{attempt + 2}/{attempts} in {delay:.1f}s"
            )
            time.sleep(delay)
    response.raise_for_status()
    return response


def list_episodes(session: requests.Session, submission_id: int) -> list[dict[str, Any]]:
    response = request_with_retry(
        session,
        "POST",
        EPISODES_URL,
        json={"submissionId": int(submission_id)},
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    return list(response.json().get("episodes", []))


def http_error_status(error: requests.HTTPError) -> int | None:
    return error.response.status_code if error.response is not None else None


def top_teams(session: requests.Session, competition: str, count: int) -> list[dict[str, Any]]:
    response = request_with_retry(
        session, "GET", LEADERBOARD_URL.format(competition=competition)
    )
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
        with archive.open(csv_name) as raw:
            rows = list(csv.DictReader(line.decode("utf-8-sig") for line in raw))
    return [
        {
            "rank": int(row["Rank"]),
            "team_id": int(row["TeamId"]),
            "team_name": row["TeamName"],
            "score": float(row["Score"]),
        }
        for row in rows[:count]
    ]


def own_submission_ids(session: requests.Session, competition: str) -> list[int]:
    response = request_with_retry(
        session,
        "GET",
        SUBMISSIONS_URL.format(competition=competition),
        params={"pageSize": 100},
        headers={"Accept": "application/json"},
    )
    payload = response.json()
    rows = payload.get("submissions", []) if isinstance(payload, dict) else payload
    result = []
    for row in rows or []:
        reference = row.get("ref", row.get("id"))
        if reference is not None:
            result.append(int(reference))
    return result


def agent_index(agent: dict[str, Any], fallback: int) -> int:
    return int(agent.get("index", fallback))


def discover_submissions(
    session: requests.Session,
    roots: list[int],
    target_team_ids: set[int],
    max_queries: int,
) -> tuple[dict[int, set[int]], dict[int, list[dict[str, Any]]]]:
    team_submissions: dict[int, set[int]] = defaultdict(set)
    episode_cache: dict[int, list[dict[str, Any]]] = {}
    queued = set(roots)
    queue = [(-float("inf"), submission_id) for submission_id in roots]
    heapq.heapify(queue)
    while queue and len(episode_cache) < max_queries:
        _, submission_id = heapq.heappop(queue)
        if submission_id in episode_cache:
            continue
        try:
            episodes = list_episodes(session, submission_id)
        except requests.HTTPError as error:
            if http_error_status(error) == 429:
                print(
                    "Kaggle rate limit persisted after retries; stop discovery cleanly. "
                    "Run the same command again to continue."
                )
                break
            raise
        episode_cache[submission_id] = episodes
        for episode in episodes:
            for agent in episode.get("agents", []):
                if not agent.get("submissionId") or not agent.get("teamId"):
                    continue
                discovered = int(agent["submissionId"])
                team_submissions[int(agent["teamId"])].add(discovered)
                if discovered not in queued:
                    score = float(agent.get("updatedScore") or agent.get("initialScore") or 0.0)
                    heapq.heappush(queue, (-score, discovered))
                    queued.add(discovered)
        found = len(target_team_ids & team_submissions.keys())
        print(
            f"discovery queries={len(episode_cache)}/{max_queries} "
            f"top_teams_found={found}/{len(target_team_ids)} queue={len(queue)}"
        )
        if target_team_ids <= team_submissions.keys():
            break
    return team_submissions, episode_cache


def replay_candidates(
    session: requests.Session,
    targets: list[dict[str, Any]],
    team_submissions: dict[int, set[int]],
    episode_cache: dict[int, list[dict[str, Any]]],
    per_team: int,
    preferred_submissions: set[int] | None = None,
    max_submission_queries_per_team: int = 3,
) -> dict[int, dict[str, Any]]:
    preferred_submissions = preferred_submissions or set()
    selected: dict[int, dict[str, Any]] = {}
    for team in targets:
        team_id = int(team["team_id"])
        candidates: dict[int, dict[str, Any]] = {}
        episode_pool: dict[int, dict[str, Any]] = {
            int(episode["id"]): episode
            for episodes in episode_cache.values()
            for episode in episodes
            if episode.get("id")
        }
        def add_team_candidates(episodes: Any) -> None:
            for episode in episodes:
                matching = [
                    (agent_index(agent, index), int(agent.get("submissionId") or -1))
                    for index, agent in enumerate(episode.get("agents", []))
                    if int(agent.get("teamId") or -1) == team_id
                ]
                if matching and episode.get("state") == "COMPLETED":
                    candidates[int(episode["id"])] = {
                        "episode": episode,
                        "indices": [index for index, _ in matching],
                        "submission_id": matching[0][1],
                    }

        add_team_candidates(episode_pool.values())
        # Discovery already gives us a cross-team episode pool. Only query a
        # few uncached top-team submissions when that pool is insufficient;
        # the old exhaustive loop could burst through hundreds of requests.
        uncached = sorted(
            (
                submission_id
                for submission_id in team_submissions.get(team_id, set())
                if submission_id not in episode_cache
            ),
            reverse=True,
        )
        extra_queries = 0
        for submission_id in uncached:
            if len(candidates) >= per_team or extra_queries >= max_submission_queries_per_team:
                break
            extra_queries += 1
            try:
                episodes = list_episodes(session, submission_id)
            except requests.HTTPError as error:
                status = http_error_status(error)
                if status in (401, 403, 404, 429):
                    print(f"skip metadata submission={submission_id} status={status}")
                    if status == 429:
                        break
                    continue
                raise
            episode_cache[submission_id] = episodes
            add_team_candidates(episodes)
        ordered = sorted(
            candidates.values(),
            key=lambda row: (
                any(
                    int(agent.get("submissionId") or -1) in preferred_submissions
                    for agent in row["episode"].get("agents", [])
                ),
                row["episode"].get("endTime", ""),
            ),
            reverse=True,
        )
        # Recent top-vs-top matches can be listed before their replay becomes
        # visible. Keep a deep fallback pool and let the downloader skip 401/403.
        for row in ordered[: max(per_team * 20, 200)]:
            episode_id = int(row["episode"]["id"])
            entry = selected.setdefault(
                episode_id,
                {
                    "episode_id": episode_id,
                    "expert_indices": [],
                    "rank_by_player": {},
                    "source_team_ids": [],
                    "source_submission_ids": [],
                    "end_time": row["episode"].get("endTime", ""),
                    "involves_own_submission": any(
                        int(agent.get("submissionId") or -1) in preferred_submissions
                        for agent in row["episode"].get("agents", [])
                    ),
                },
            )
            for index in row["indices"]:
                if index not in entry["expert_indices"]:
                    entry["expert_indices"].append(index)
                entry["rank_by_player"][str(index)] = int(team["rank"])
            entry["source_team_ids"].append(team_id)
            entry["source_submission_ids"].append(int(row["submission_id"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition", default="kaggriculture")
    parser.add_argument("--top-teams", type=int, default=10)
    parser.add_argument("--episodes-per-team", type=int, default=50)
    parser.add_argument("--max-discovery-queries", type=int, default=500)
    parser.add_argument("--submission-id", type=int, action="append", default=[])
    parser.add_argument("--output", default="data/raw/top")
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Dotenv file containing Kaggle credentials (default: .env). Existing env wins.",
    )
    parser.add_argument("--request-delay", type=float, default=0.2)
    parser.add_argument(
        "--api-request-interval",
        type=float,
        default=1.0,
        help="Minimum seconds between Kaggle API requests (default: 1.0).",
    )
    parser.add_argument(
        "--max-submission-queries-per-team",
        type=int,
        default=3,
        help="Bound extra metadata lookups after graph discovery (default: 3).",
    )
    parser.add_argument(
        "--max-replay-attempts",
        type=int,
        default=0,
        help="Stop probing inaccessible replays after this many GETs (0 = automatic limit).",
    )
    args = parser.parse_args()
    if load_environment_file(args.env_file):
        print(f"loaded environment from {Path(args.env_file).resolve()}")
    session = authenticated_session(request_interval=args.api_request_interval)
    targets = top_teams(session, args.competition, args.top_teams)
    roots = list(args.submission_id)
    if not roots:
        roots = own_submission_ids(session, args.competition)
    if not roots:
        raise SystemExit("No root submissions. Pass at least one --submission-id.")
    print("targets:", ", ".join(f"#{row['rank']} {row['team_name']}" for row in targets))
    target_ids = {int(row["team_id"]) for row in targets}
    team_submissions, episode_cache = discover_submissions(
        session, roots, target_ids, args.max_discovery_queries
    )
    missing = [row for row in targets if int(row["team_id"]) not in team_submissions]
    if missing:
        names = ", ".join(f"#{row['rank']} {row['team_name']}" for row in missing)
        print(f"warning: submissions not discovered for {names}")
    selected = replay_candidates(
        session,
        targets,
        team_submissions,
        episode_cache,
        args.episodes_per_team,
        preferred_submissions=set(roots),
        max_submission_queries_per_team=args.max_submission_queries_per_team,
    )
    output = Path(args.output)
    replay_dir = output / "replays"
    replay_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    completed_by_team: dict[int, int] = defaultdict(int)
    ordered_entries = sorted(
        selected.values(),
        key=lambda row: (row["involves_own_submission"], row["end_time"]),
        reverse=True,
    )
    max_attempts = args.max_replay_attempts or max(200, args.top_teams * args.episodes_per_team * 20)
    attempts = 0
    rate_limited = False
    for number, entry in enumerate(ordered_entries, start=1):
        needed_teams = [
            team_id
            for team_id in entry["source_team_ids"]
            if completed_by_team[team_id] < args.episodes_per_team
        ]
        if not needed_teams:
            continue
        episode_id = int(entry["episode_id"])
        replay_path = replay_dir / f"episode-{episode_id}-replay.json"
        if not replay_path.exists():
            if attempts >= max_attempts:
                print(f"stopped after max_replay_attempts={max_attempts}")
                break
            attempts += 1
            try:
                response = request_with_retry(
                    session, "GET", REPLAY_URL.format(episode_id=episode_id)
                )
            except requests.HTTPError as error:
                status = http_error_status(error)
                if status in (401, 403, 404):
                    print(
                        f"skipped episode={episode_id} "
                        f"status={status} (replay is not accessible to this account)"
                    )
                    continue
                if status == 429:
                    print(
                        "Kaggle rate limit persisted after retries; stop replay downloads cleanly. "
                        "Downloaded files and manifest progress are retained."
                    )
                    rate_limited = True
                    break
                raise
            replay_path.write_bytes(response.content)
            time.sleep(max(0.0, args.request_delay))
        entry.pop("end_time", None)
        entry.pop("involves_own_submission", None)
        entry["replay_path"] = str(replay_path.relative_to(output))
        entries.append(entry)
        for team_id in needed_teams:
            completed_by_team[team_id] += 1
        progress = ", ".join(
            f"team={team_id}:{completed_by_team[team_id]}/{args.episodes_per_team}"
            for team_id in sorted(target_ids)
        )
        print(f"candidate={number}/{len(ordered_entries)} episode={episode_id} {progress}")
        if all(completed_by_team[team_id] >= args.episodes_per_team for team_id in target_ids):
            break
    manifest = {
        "competition": args.competition,
        "leaderboard_teams": targets,
        "replays": entries,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(entries)} expert replays to {output.resolve()}")
    incomplete = [
        team_id
        for team_id in sorted(target_ids)
        if completed_by_team[team_id] < args.episodes_per_team
    ]
    if incomplete:
        print(f"warning: replay quota incomplete for team IDs {incomplete} (Kaggle denied access)")
    if not entries:
        if rate_limited:
            raise SystemExit(
                "No replay completed before the Kaggle rate limit. Wait and rerun the same command."
            )
        raise SystemExit(
            "No top replay is accessible. Verify KAGGLE_API_TOKEN and that the competition "
            "exposes opponent replays."
        )


if __name__ == "__main__":
    main()
