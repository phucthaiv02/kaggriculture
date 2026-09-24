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
stage_dir="$(mktemp -d /tmp/kagriculture-build.XXXXXX)"

cleanup() {
    rm -f -- "$archive_tmp"
    rm -rf -- "$smoke_dir"
    rm -rf -- "$stage_dir"
}
trap cleanup EXIT

# ``agents`` collides with kaggle_environments.envs.lux_ai_s3.agents when
# Kaggle executes main.py as raw source rather than importing it as a module.
# Give the submitted package a competition-specific name while leaving the
# development package (and all local imports/tests) unchanged.
cp -R "$repo_dir/agents" "$stage_dir/kagriculture_agent"
sed 's/from agents\./from kagriculture_agent./g; s/import agents\./import kagriculture_agent./g' \
    "$repo_dir/main.py" > "$stage_dir/main.py"
find "$stage_dir/kagriculture_agent" -type f -name '*.py' -exec \
    sed -i \
        -e 's/from agents\./from kagriculture_agent./g' \
        -e 's/import agents\./import kagriculture_agent./g' {} +

cd "$stage_dir"
tar \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -czf "$archive_tmp" \
    main.py kagriculture_agent

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
