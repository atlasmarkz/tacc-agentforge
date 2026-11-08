#!/usr/bin/env bash
# Usage: stage.sh smoke|serve   - copies the stdlib client next to serve.py and records the mode.
set -euo pipefail
cd "$(dirname "$0")"
mode=${1:?smoke or serve}
cp ../../../../inference/client.py client.py
printf '%s\n' "$mode" > mode.txt
echo "staged mode=$mode"
