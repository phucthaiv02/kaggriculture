#!/usr/bin/env python3
"""Reject submission imports unavailable from the pinned Kaggle base image."""

from __future__ import annotations

import argparse
import ast
import sys
import tarfile
from pathlib import Path


ALLOWED_THIRD_PARTY = {"numpy", "torch"}
LOCAL_MODULES = {"kaggriculture_agent", "main"}
REQUIRED_FILES = {"main.py", "model.pt", "runtime_image.txt"}
MAX_SUBMISSION_BYTES = 100 * 1024 * 1024


def imported_roots(source: str, filename: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    args = parser.parse_args()
    archive_path = Path(args.archive)
    if archive_path.stat().st_size > MAX_SUBMISSION_BYTES:
        size_mib = archive_path.stat().st_size / 1024 / 1024
        raise SystemExit(f"Submission is {size_mib:.1f} MiB, exceeding the 100 MiB limit")
    violations: dict[str, list[str]] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        files = {member.name for member in archive.getmembers() if member.isfile()}
        missing = REQUIRED_FILES - files
        if missing:
            raise SystemExit(f"Missing runtime files: {sorted(missing)}")
        image_file = archive.extractfile("runtime_image.txt")
        if image_file is None:
            raise SystemExit("Cannot read runtime_image.txt")
        if image_file.read().decode("utf-8").strip() != "gcr.io/kaggle-images/python:v163":
            raise SystemExit("Runtime image is not pinned to v163")
        allowed = set(sys.stdlib_module_names) | ALLOWED_THIRD_PARTY | LOCAL_MODULES
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            source_file = archive.extractfile(name)
            if source_file is None:
                continue
            roots = imported_roots(source_file.read().decode("utf-8"), name)
            unexpected = sorted(roots - allowed)
            if unexpected:
                violations[name] = unexpected
    if violations:
        raise SystemExit(f"Forbidden runtime imports: {violations}")
    print(f"PASS: {archive_path} uses only stdlib + numpy + torch on Kaggle Python v163")


if __name__ == "__main__":
    main()
