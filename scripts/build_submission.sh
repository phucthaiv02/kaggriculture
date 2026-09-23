#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
build_dir="$repo_dir/build"
submission="$build_dir/submission.tar.gz"
python_bin="${PYTHON_BIN:-python}"

mkdir -p "$build_dir"
archive_tmp="$(mktemp "$build_dir/submission.tar.gz.XXXXXX")"
smoke_dir="$(mktemp -d /tmp/kagriculture-submission.XXXXXX)"

cleanup() {
    rm -f -- "$archive_tmp"
    rm -rf -- "$smoke_dir"
}
trap cleanup EXIT

cd "$repo_dir"
tar \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -czf "$archive_tmp" \
    main.py agents

tar -xzf "$archive_tmp" -C "$smoke_dir"
cd "$smoke_dir"
"$python_bin" - <<'PY'
from kaggle_environments import make

import main

assert callable(main.agent), "main.agent must be callable"

env = make(
    "kaggriculture",
    configuration={"episodeSteps": 48, "seed": 1},
    debug=True,
)
env.run([main.agent, "pass"])
state = env.steps[-1][0]
assert state.status == "DONE", f"smoke test ended with status {state.status!r}"
PY

mv -f -- "$archive_tmp" "$submission"
printf 'Created %s\n' "$submission"
sha256sum "$submission"
