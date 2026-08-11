#!/usr/bin/env python3
"""Create a dependency-free Kaggle submission archive."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="submissions/heuristic_agent.tar.gz")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="kaggriculture-heuristic-") as temporary:
        root = Path(temporary)
        shutil.copy2("main.py", root / "main.py")
        package = root / "kaggriculture_agent"
        package.mkdir()
        shutil.copy2("src/kaggriculture_agent/__init__.py", package / "__init__.py")
        shutil.copy2("src/kaggriculture_agent/heuristic.py", package / "heuristic.py")
        with tarfile.open(output, "w:gz") as archive:
            for path in sorted(root.rglob("*.py")):
                archive.add(path, arcname=path.relative_to(root))

    print(f"Saved {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
