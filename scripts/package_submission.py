#!/usr/bin/env python3
"""Create a minimal inference-only Kaggle archive."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path

import torch


INFERENCE_MODULES = ("__init__.py", "constants.py", "codec.py", "model.py", "inference.py")
MAX_SUBMISSION_BYTES = 100 * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="submissions/dynamic_agent.tar.gz")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    minimal = {
        "architecture": checkpoint["architecture"],
        "model_config": checkpoint["model_config"],
        "model": checkpoint["model"],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kaggriculture-submission-") as temporary:
        root = Path(temporary)
        shutil.copy2("main.py", root / "main.py")
        package = root / "kaggriculture_agent"
        package.mkdir()
        source = Path("src/kaggriculture_agent")
        for name in INFERENCE_MODULES:
            shutil.copy2(source / name, package / name)
        torch.save(minimal, root / "model.pt")
        (root / "runtime_image.txt").write_text(
            "gcr.io/kaggle-images/python:v163\n", encoding="utf-8"
        )
        with tarfile.open(output, "w:gz") as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=path.relative_to(root))
    if output.stat().st_size > MAX_SUBMISSION_BYTES:
        size_mib = output.stat().st_size / 1024 / 1024
        output.unlink()
        raise SystemExit(
            f"Submission is {size_mib:.1f} MiB, exceeding the 100 MiB limit; removed {output}"
        )
    print(f"Saved {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
