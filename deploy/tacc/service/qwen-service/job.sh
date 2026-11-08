#!/usr/bin/env bash
# Reliable inference service for the agent harness: one vLLM server per job, address published when ready.
set -euo pipefail
umask 077
: "${TACC_WORKDIR:?Submit with tcloud submit}"
: "${TACC_USERDIR:?Submit with tcloud submit}"
: "${SLURM_JOB_ID:?Submit with tcloud submit}"
: "${CONDA_PREFIX:?SDK must activate the course management environment}"
test "$CONDA_PREFIX" = /mnt/home/public/course-envs/CSIT6000V/assignment1/mpi-gemm-v1
source /mnt/home/public/course-envs/CSIT6000V/project/activate.sh
trap 'rm -rf -- "$QWEN_RUN_DIR"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export QWEN_CACHE_HOME="$TACC_USERDIR/CSIT6000V-project-cache"
mkdir -p -m 700 "$QWEN_CACHE_HOME"
cd "$TACC_WORKDIR"
MODE="$(tr -d '[:space:]' < mode.txt 2>/dev/null || echo serve)"
BASE="$TACC_USERDIR/tacc-agentforge-service"
OUT="$BASE/service-$SLURM_JOB_ID"
mkdir -p "$OUT"
hostname
ARGS=(--out "$OUT" --address-file "$BASE/service-url.txt")
if [ "$MODE" = smoke ]; then
    # Validation run: publish to a separate file so a running real service is never overwritten.
    ARGS=(--out "$OUT" --address-file "$BASE/service-url-smoke.txt" --smoke --exit-after-smoke)
fi
qwen-python -u serve.py "${ARGS[@]}"
echo "RESULTS_DIR=$OUT"
