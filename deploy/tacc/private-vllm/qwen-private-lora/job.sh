#!/usr/bin/env bash
# Run one experiment group (group.txt) against the PRIVATE vLLM build, in a single TACC job.
set -euo pipefail
umask 077
: "${TACC_WORKDIR:?Submit with tcloud submit}"
: "${TACC_USERDIR:?Submit with tcloud submit}"
: "${SLURM_JOB_ID:?Submit with tcloud submit}"
: "${CONDA_PREFIX:?SDK must activate the course management environment}"
test "$CONDA_PREFIX" = /mnt/home/public/course-envs/CSIT6000V/assignment1/mpi-gemm-v1
project="$TACC_USERDIR/CSIT6000V-private-vllm"
export QWEN_PRIVATE_ENV="$project/.venv"
if [[ -d "$project/.build-in-progress" || -e "$project/.build-failed" || ! -x "$QWEN_PRIVATE_ENV/bin/python" ]]; then
    echo 'Private vLLM is not ready; wait for PRIVATE VLLM BUILD PASSED.' >&2
    exit 1
fi
cd "$project"
source /mnt/home/public/course-envs/CSIT6000V/project-private-vllm/activate.sh
trap 'rm -rf -- "$QWEN_RUN_DIR"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export QWEN_CACHE_HOME="$project/cache"
mkdir -p -m 700 "$QWEN_CACHE_HOME"
cd "$TACC_WORKDIR"
OUT="$TACC_USERDIR/tacc-agentforge-results/private-lora-ovh-$SLURM_JOB_ID"
hostname
(cd "$project" && qwen-python -c "import vllm;print('VLLM_FILE',vllm.__file__,vllm.__version__)")
qwen-python -u lora_overhead.py --out "$OUT" --work "$QWEN_RUN_DIR/adapters"
echo "RESULTS_DIR=$OUT"
