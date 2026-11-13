#!/usr/bin/env bash
# Copies the stdlib client next to loadgen.py (git-ignored). workload.json is versioned
# (regenerate with scripts/make_load_workload.py).
set -euo pipefail
cd "$(dirname "$0")"
cp ../../../../inference/client.py client.py
test -f workload.json || { echo "workload.json missing: run scripts/make_load_workload.py" >&2; exit 2; }
echo "staged load-test"
