#!/usr/bin/env bash
# CPU benchmark worker; reuses an existing GPU service. Checkpoints are in-memory.
set -euo pipefail
umask 077
: "${TACC_WORKDIR:?}"
: "${TACC_USERDIR:?}"
source /mnt/home/public/course-envs/CSIT6000V/project/activate.sh
trap 'rm -rf -- "$QWEN_RUN_DIR"' EXIT
cd "$TACC_WORKDIR"
export PYTHONPATH="$TACC_WORKDIR/pkg:$TACC_USERDIR/agent-pylib${PYTHONPATH:+:$PYTHONPATH}"
export BENCHMARK_ADDRESS_FILE="$TACC_USERDIR/tacc-agentforge-service/service-url.txt"
export BENCHMARK_MODEL=qwen3.8-27b
qwen-python -c 'import pydantic, langgraph; from budget.validator import CatalogValidator'
qwen-python -u benchmark.py
