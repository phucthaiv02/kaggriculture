#!/usr/bin/env python3
"""Run the packaged-agent smoke test in Kaggle's pinned CPU container."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


IMAGE = "gcr.io/kaggle-images/python:v163"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    archive = Path(args.archive).resolve()
    try:
        relative_archive = archive.relative_to(root)
    except ValueError as error:
        raise SystemExit("Archive must be inside the repository") from error
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--volume",
            f"{root}:/workspace:ro",
            "--workdir",
            "/workspace",
            IMAGE,
            "python",
            "scripts/runtime_smoke.py",
            str(relative_archive),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
