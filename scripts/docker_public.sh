#!/usr/bin/env bash
# Public-data container check. Never echoes keys or reads .env into the shell.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$root"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="$root/runs/docker-$stamp"
if [[ -e "$run_dir" ]]; then
  echo "Run directory already exists; retry in a second." >&2
  exit 2
fi
mkdir -p "$run_dir/input" "$run_dir/artifacts" "$run_dir/output"
python src/prepare_inputs.py --train data/public_train.csv --test data/public_test.csv --out-dir "$run_dir/input"

# Docker Desktop needs Windows host paths when invoked from Git Bash. Stop MSYS
# rewriting container paths (/input, /artifacts, /output) into host paths.
host_run="$run_dir"
host_root="$root"
if command -v cygpath >/dev/null 2>&1; then
  host_run="$(cygpath -m "$run_dir")"
  host_root="$(cygpath -m "$root")"
  export MSYS_NO_PATHCONV=1
fi
env_args=(-e GEMINI_API_KEY)
if [[ -f "$root/.env" ]]; then
  env_args=(--env-file "$host_root/.env")
fi
docker build -t rulejudge .
docker run --rm "${env_args[@]}" \
  --mount "type=bind,source=$host_run/input,target=/input,readonly" \
  --mount "type=bind,source=$host_run/artifacts,target=/artifacts" rulejudge adapt
docker run --rm "${env_args[@]}" \
  --mount "type=bind,source=$host_run/input,target=/input,readonly" \
  --mount "type=bind,source=$host_run/artifacts,target=/artifacts,readonly" \
  --mount "type=bind,source=$host_run/output,target=/output" rulejudge eval
python src/evaluate.py --test data/public_test.csv --predictions "$run_dir/output/predictions.csv" --out "$run_dir/score.json"
printf 'Public Docker check complete: %s\n' "$run_dir"
