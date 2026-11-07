#!/usr/bin/env bash
# Usage: stage.sh <group> [variant ...]   (2 GPUs). Copies the runner here and writes tuxiv.conf/group.txt.
set -euo pipefail
cd "$(dirname "$0")"
group=${1:?group name from configs/experiments.json}; shift
cp ../../../../inference/benchmark.py ../../../../inference/experiments.py .
cp ../../../../configs/experiments.json .
jobname="pexp-$group"; for v in "$@"; do jobname="$jobname-$v"; done
jobname=${jobname:0:40}
sed -e "s/__GPUS__/2/" -e "s/__JOBNAME__/$jobname/g" ../../vllm-experiments/tuxiv.conf.template > tuxiv.conf
printf '%s\n' "$group" > group.txt
if [ "$#" -gt 0 ]; then printf '%s\n' "$*" > only.txt; else rm -f only.txt; fi
echo "job-name=$jobname group=$group only=${*:-all}"
