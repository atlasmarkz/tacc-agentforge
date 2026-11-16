#!/usr/bin/env bash
# Usage: stage.sh smoke|serve|validate [--harness]
#   Copies the stdlib client next to serve.py and records the mode.
#   --harness (serve only): also stage the agent harness service (Task 5) into pkg/, so this one job runs vLLM
#   and, once vLLM is ready, the harness service next to it. Without it, pkg/ is removed and the job is vLLM only.
set -euo pipefail
cd "$(dirname "$0")"
mode=${1:?smoke, serve or validate}
harness=${2:-}
cp ../../../../inference/client.py client.py
printf '%s\n' "$mode" > mode.txt
rm -rf pkg harness-lib.sh
if [ "$harness" = --harness ]; then
    test "$mode" = serve || { echo "--harness only with serve" >&2; exit 2; }
    bash ../app-service/stage.sh "$PWD/pkg"
    cp ../app-service/harness-lib.sh harness-lib.sh
elif [ -n "$harness" ]; then
    echo "unknown option $harness" >&2
    exit 2
fi
echo "staged mode=$mode${harness:+ with harness}"
