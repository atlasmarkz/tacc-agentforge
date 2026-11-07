#!/usr/bin/env bash
set -euo pipefail
: "${TACC_WORKDIR:?Submit with tcloud submit}"
: "${TACC_USERDIR:?Submit with tcloud submit}"
: "${SLURM_JOB_ID:?Submit with tcloud submit}"
: "${CONDA_PREFIX:?SDK must activate the course management environment}"
test "$CONDA_PREFIX" = /mnt/home/public/course-envs/CSIT6000V/assignment1/mpi-gemm-v1 || {
    printf 'Unexpected SDK environment: %s\nRun setup-sdk-account.sh from the same package as this job.\n' "$CONDA_PREFIX" >&2
    exit 1
}
printf 'SDK_BOOTSTRAP=%s\n' "$CONDA_PREFIX"
export QWEN_PRIVATE_ENV="$TACC_USERDIR/CSIT6000V-private-vllm/.venv"
source /mnt/home/public/course-envs/CSIT6000V/project-private-vllm/activate.sh
trap 'rm -rf -- "$QWEN_RUN_DIR"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
cd "$TACC_WORKDIR"
# Keep assignment separate from export so validation failures stop the job.
QWEN_BASE_URL=$(qwen-python qwen_client.py --check-url --base-url "$(cat server-url.txt)")
export QWEN_BASE_URL
# Course service traffic is internal; bypass inherited HTTP proxy settings.
service_host=${QWEN_BASE_URL#*://}
if [[ "$service_host" == \[* ]]; then
    service_host=${service_host#\[}
    service_host=${service_host%%\]*}
else
    service_host=${service_host%%[:/]*}
fi
export no_proxy="${no_proxy:+$no_proxy,}${service_host},127.0.0.1,localhost"
export NO_PROXY="$no_proxy"
qwen-python -u main.py
