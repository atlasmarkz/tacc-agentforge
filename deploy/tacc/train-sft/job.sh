#!/usr/bin/env bash
# SFT pilot (Task 7): LoRA on the course 27B INT4 checkpoint, 2 GPUs, one hour. Run stage.sh first.
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
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$TACC_WORKDIR"
test -f pkg/data/train.jsonl || { echo "Run stage.sh (with the training data) before submitting." >&2; exit 2; }
# peft/accelerate/flash-linear-attention from the train-env-setup job; then the staged repository code.
export PYTHONPATH="$TACC_WORKDIR/pkg:$TACC_USERDIR/pylib${PYTHONPATH:+:$PYTHONPATH}"
hostname
# Same as the feasibility test and the serving script: Slurm's default hides the GPUs from this interpreter.
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
read -r -a EXTRA <<< "$(cat args.txt 2>/dev/null || true)"
qwen-python -u -m training.sft --config pkg/configs/sft.yaml \
    --set "train_file=$TACC_WORKDIR/pkg/data/train.jsonl" \
    --set "output_dir=$TACC_USERDIR/adapters/sft-pilot-$SLURM_JOB_ID" ${EXTRA[@]+"${EXTRA[@]}"}
echo "ADAPTER_DIR=$TACC_USERDIR/adapters/sft-pilot-$SLURM_JOB_ID"
