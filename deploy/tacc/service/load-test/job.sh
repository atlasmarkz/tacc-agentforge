#!/usr/bin/env bash
# CPU job: open-loop load test and fault injection against the service job (start it in `validate` mode).
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
cd "$TACC_WORKDIR"
hostname
test -f client.py || { echo "Run stage.sh before submitting." >&2; exit 2; }
ARGS="$(cat args.txt 2>/dev/null || true)"
echo "args=$ARGS"
qwen-python -u loadgen.py --out "$TACC_USERDIR/tacc-agentforge-service/load-$SLURM_JOB_ID" $ARGS
