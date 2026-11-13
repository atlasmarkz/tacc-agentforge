#!/usr/bin/env bash
# Usage: stage.sh <train.jsonl>  - copies the training code, config and data into pkg/ (git-ignored).
set -euo pipefail
data=$(realpath "${1:?training JSONL from training/prepare_data.py}")
cd "$(dirname "$0")"
root=../../..
rm -rf pkg
mkdir -p pkg/training pkg/configs pkg/data
cp "$root"/training/*.py pkg/training/
cp "$root/configs/sft.yaml" pkg/configs/
cp "$data" pkg/data/train.jsonl
manifest="${data%.jsonl}.manifest.json"
[ -f "$manifest" ] && cp "$manifest" pkg/data/train.manifest.json
echo "staged: $(wc -l < pkg/data/train.jsonl) examples, $(find pkg -type f | wc -l) files"
