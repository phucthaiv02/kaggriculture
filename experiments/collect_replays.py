"""Collect recent public Kaggriculture episodes from top teams via Kaggle API."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import sys
import tempfile
import time

from experiments.replay_data import read_replay, write_json


def score(value) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else -math.inf
    except (ValueError, TypeError):
        return -math.inf


class KaggleSource:
    """Official authenticated API; retry transient transport/server failures only."""

    def __init__(self, delay=1.0, retries=3):
        from kaggle.api.kaggle_api_extended import KaggleApi
        self.api = KaggleApi()
        if not hasattr(self.api, "competition_team_submissions"):
            raise RuntimeError("Install a recent API: python -m pip install -r requirements-replays.txt")
        self.api.authenticate()
        self.delay, self.retries = delay, retries
        self.last_request = 0.0

    def call(self, fn, *args, **kwargs):
        from requests.exceptions import ConnectionError, HTTPError, Timeout
        for attempt in range(self.retries + 1):
            time.sleep(max(0, self.delay - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                return fn(*args, **kwargs)
            except (ConnectionError, Timeout, HTTPError) as exc:
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
                if attempt == self.retries or (status is not None and
                                               status != 429 and status < 500):
                    raise
                time.sleep(min(30, 2 ** attempt))

    def leaderboard(self, competition, count):
        from kagglesdk.competitions.types.competition_api_service import ApiGetLeaderboardRequest
        rows, token, seen = [], None, set()
        while len(rows) < count:
            request = ApiGetLeaderboardRequest()
            request.competition_name = competition
            request.page_size = min(100, count - len(rows))
            if token:
                request.page_token = token
            with self.api.build_kaggle_client() as client:
                response = self.call(client.competitions.competition_api_client.get_leaderboard, request)
            rows.extend(row.to_dict(ignore_defaults=False) for row in response.submissions or [])
            token = response.next_page_token
            if not token:
                break
            if token in seen:
                raise ValueError("leaderboard returned a repeated page token")
            seen.add(token)
        return rows[:count]

    def submissions(self, team_id):
        return [x.to_dict(ignore_defaults=False) for x in
                self.call(self.api.competition_team_submissions, team_id) or []]

    def episodes(self, submission_id):
        return [x.to_dict(ignore_defaults=False) for x in
                self.call(self.api.competition_list_episodes, submission_id) or []]

    def download(self, episode_id, directory):
        self.call(self.api.competition_episode_replay, episode_id, path=str(directory), quiet=True)
        return directory / f"episode-{episode_id}-replay.json"


def collect(source, output: Path, top=10, per_team=10, submissions_per_team=1):
    if min(top, per_team, submissions_per_team) < 1:
        raise ValueError("collection limits must be positive")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "competition": "kaggriculture",
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "selection": {"top": top, "episodes_per_team": per_team,
                              "submissions_per_team": submissions_per_team},
                "leaderboard": source.leaderboard("kaggriculture", top),
                "teams": [], "episodes": {}, "errors": []}
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    for rank, team in enumerate(manifest["leaderboard"], 1):
        team_id = int(team["teamId"])
        selected = {**team, "rank": rank, "selected_submissions": [], "episode_ids": []}
        manifest["teams"].append(selected)
        try:
            submissions = sorted(source.submissions(team_id),
                                 key=lambda s: (score(s.get("publicScore")),
                                                s.get("dateSubmitted") or "", int(s["id"])),
                                 reverse=True)[:submissions_per_team]
            selected["selected_submissions"] = submissions
            candidates = {}
            for submission in submissions:
                for episode in source.episodes(int(submission["id"])):
                    if episode.get("state") == "COMPLETED" and episode.get("type") == "EPISODE_TYPE_PUBLIC":
                        candidates[int(episode["id"])] = episode
            allowed = {int(s["id"]) for s in submissions}
            for episode in sorted(candidates.values(),
                                  key=lambda e: (e.get("endTime") or "", int(e["id"])), reverse=True):
                if len(selected["episode_ids"]) >= per_team:
                    break
                episode_id = int(episode["id"])
                targets = [int(a["index"]) for a in episode.get("agents", [])
                           if int(a["submissionId"]) in allowed and int(a["teamId"]) == team_id]
                if not targets or any(index not in (0, 1) for index in targets):
                    manifest["errors"].append({"episode_id": episode_id, "error": "missing player mapping"})
                    continue
                try:
                    filename = f"episode-{episode_id}-replay.json"
                    destination = output / filename
                    try:
                        read_replay(destination)
                    except (OSError, ValueError):
                        with tempfile.TemporaryDirectory(dir=output, prefix="download-") as tmp:
                            downloaded = source.download(episode_id, Path(tmp))
                            read_replay(downloaded)
                            downloaded.replace(destination)
                    entry = manifest["episodes"].setdefault(str(episode_id), {
                        "file": filename, "metadata": episode, "target_players": [],
                        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()})
                    entry["target_players"] = sorted(set(entry["target_players"]) | set(targets))
                    selected["episode_ids"].append(episode_id)
                    print(f"#{rank} {team['teamName']}: episode {episode_id}", flush=True)
                except Exception as exc:
                    manifest["errors"].append({"episode_id": episode_id, "error": str(exc)})
                write_json(manifest_path, manifest)
            if not selected["episode_ids"]:
                manifest["errors"].append({"team_id": team_id, "error": "no usable public episodes"})
        except Exception as exc:
            manifest["errors"].append({"team_id": team_id, "error": str(exc)})
        write_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--episodes-per-team", type=int, default=10)
    parser.add_argument("--submissions-per-team", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("replays/top_players"))
    parser.add_argument("--delay", type=float, default=1.0, help="minimum seconds between API calls")
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if min(args.top, args.episodes_per_team, args.submissions_per_team) < 1 or args.delay < 0 or args.retries < 0:
        parser.error("limits must be positive; delay and retries must be nonnegative")
    try:
        manifest = collect(KaggleSource(args.delay, args.retries), args.output_dir,
                           args.top, args.episodes_per_team, args.submissions_per_team)
    except Exception as exc:
        parser.exit(1, f"Kaggle collection failed: {exc}\n")
    print(f"{len(manifest['episodes'])} unique replays; {len(manifest['errors'])} errors. "
          f"Manifest: {args.output_dir / 'manifest.json'}")
    if manifest["errors"] or not manifest["episodes"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
