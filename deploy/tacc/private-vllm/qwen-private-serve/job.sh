#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${TACC_WORKDIR:?Submit with tcloud submit}"
: "${TACC_USERDIR:?Submit with tcloud submit}"
: "${SLURM_JOB_ID:?Submit with tcloud submit}"
: "${CONDA_PREFIX:?SDK must activate the course management environment}"
test "$CONDA_PREFIX" = /mnt/home/public/course-envs/CSIT6000V/assignment1/mpi-gemm-v1 || {
    printf 'Unexpected SDK environment: %s\nRun setup-sdk-account.sh from the same package as this job.\n' "$CONDA_PREFIX" >&2
    exit 1
}
printf 'SDK_BOOTSTRAP=%s\n' "$CONDA_PREFIX"
project="$TACC_USERDIR/CSIT6000V-private-vllm"
export QWEN_PRIVATE_ENV="$project/.venv"
if [[ -d "$project/.build-in-progress" || -e "$project/.build-failed" || ! -x "$QWEN_PRIVATE_ENV/bin/python" ]]; then
    echo 'Private vLLM is not ready. Complete or repair the build first; wait for PRIVATE VLLM BUILD PASSED. See README for interrupted-build recovery.' >&2
    exit 1
fi
cd "$project"
source /mnt/home/public/course-envs/CSIT6000V/project-private-vllm/activate.sh
trap 'rm -rf -- "$QWEN_RUN_DIR"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export QWEN_CACHE_HOME="$TACC_USERDIR/CSIT6000V-private-vllm/cache"
mkdir -p -m 700 "$QWEN_CACHE_HOME"
hostname
printf 'SERVICE_BASE_URL=http://%s:8000/v1\n' "$(hostname)"
qwen-python "$TACC_WORKDIR/check-private-env.py"
bash "$QWEN_PROJECT_ROOT/serve.sh"
