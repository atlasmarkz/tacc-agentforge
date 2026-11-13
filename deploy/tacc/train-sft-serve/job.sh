#!/usr/bin/env bash
# Serve an SFT adapter and the base model in one vLLM (public course runtime) and compare first actions.
# args.txt (git-ignored) must contain: --adapter <adapter directory in USERDIR>
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
test -s args.txt || { echo "args.txt needs: --adapter <directory>" >&2; exit 2; }
read -r -a ARGS <<< "$(cat args.txt)"
OUT="$TACC_USERDIR/tacc-agentforge-results/sft-serve-$SLURM_JOB_ID"
hostname
qwen-python -u serve_check.py --out "$OUT" "${ARGS[@]}"
echo "RESULTS_DIR=$OUT"
