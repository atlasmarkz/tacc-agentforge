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
GROUP="$(tr -d '[:space:]' < group.txt)"
ONLY="$(cat only.txt 2>/dev/null || true)"
OUT="$TACC_USERDIR/tacc-agentforge-results/private-$GROUP-$SLURM_JOB_ID"
hostname
# GPU preflight, the same as deploy/tacc/service/qwen-service/job.sh.
# On the 10-0-*-11 nodes Slurm sets CUDA_VISIBLE_DEVICES=2,3, but the job sees only its own two GPUs, numbered
# 0 and 1, so vLLM fails with NVMLError_InvalidArgument (jobs 28936, 28955; 29909 and 29910 during the README
# check). Renumber to what the job can see when the allocated IDs are out of range.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi > /dev/null; then
    visible=$(nvidia-smi -L | grep -c '^GPU ')
    IFS=, read -r -a allocated <<< "$CUDA_VISIBLE_DEVICES"
    for id in "${allocated[@]}"; do
        if [[ "$id" =~ ^[0-9]+$ ]] && [ "$id" -ge "$visible" ] && [ "${#allocated[@]}" -le "$visible" ]; then
            CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((${#allocated[@]} - 1)))
            echo "CUDA_VISIBLE_DEVICES ${allocated[*]} is outside the $visible visible GPUs; using $CUDA_VISIBLE_DEVICES"
            export CUDA_VISIBLE_DEVICES
            break
        fi
    done
fi
# Processes outside Slurm sometimes hold the allocated GPUs; vLLM would then fail with out-of-memory errors for
# every variant. Stop at once with the reason instead (resubmit; Slurm usually picks another node).
if command -v nvidia-smi > /dev/null; then
    busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2 > 2048 {printf "%s:%sMiB ", $1, $2}')
    if [ -n "$busy" ]; then
        echo "GPUs already in use by processes outside this job: $busy- not starting vLLM on $(hostname); resubmit"
        exit 3
    fi
fi
# Record which vLLM is imported, so results are tied to the private source tree.
(cd "$project" && qwen-python -c "import vllm,sys;print('VLLM_FILE',vllm.__file__,vllm.__version__)")
(cd "$project" && git -C vllm-src rev-parse HEAD 2>/dev/null || true)
qwen-python -u experiments.py --matrix experiments.json --group "$GROUP" \
    --out "$OUT" --deadline-s 3300 ${ONLY:+--only $ONLY}
echo "RESULTS_DIR=$OUT"
