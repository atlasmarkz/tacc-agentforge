#!/usr/bin/env bash
# CPU job (Task 5): the agent harness as a long-running HTTP service (app/harness_service.py) on a compute node.
# It talks to the vLLM service job through its address file and publishes its own address and token in
# $TACC_USERDIR/tacc-agentforge-service/harness-url.txt / harness-token.txt for the local backend (app/backend.py),
# which reaches it through an SSH tunnel via the gateway. Run stage.sh before submitting.
# The usual way is now to run the harness inside the vLLM job (qwen-service/stage.sh serve --harness), which uses
# one job slot instead of two; this job remains for running the harness on its own.
# Checkpoints are kept in memory: USERDIR is NFS (findmnt: nfs4), where the SQLite store of PR #13 refuses to
# run, so this job does not use it and has no AGENT_SQLITE_STORAGE_VERIFIED gate.
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
test -d pkg/app || { echo "Run stage.sh before submitting." >&2; exit 2; }
hostname
source harness-lib.sh
harness_install_deps "$TACC_WORKDIR/pkg"
# Optional extra options (git-ignored args.txt), e.g. --explain or --profile tacc-thinking.
read -r -a EXTRA <<< "$(cat args.txt 2>/dev/null || true)"
# Stop accepting a few minutes before the 1-hour limit so running requests finish and the address is withdrawn.
harness_run "$TACC_WORKDIR/pkg" --deadline-s 3240 ${EXTRA[@]+"${EXTRA[@]}"}
