#!/usr/bin/env bash
# Copies the harness (read-only, unchanged) and the model adapter into pkg/ for upload. pkg/ is git-ignored.
set -euo pipefail
cd "$(dirname "$0")"
root=../../../..
rm -rf pkg
mkdir -p pkg/inference
tar -C "$root" -cf - --exclude='.venv' --exclude='__pycache__' --exclude='tests' agent | tar -xf - -C pkg
cp "$root/inference/__init__.py" "$root/inference/client.py" "$root/inference/model_port.py" pkg/inference/
echo "staged: $(find pkg -type f | wc -l) files"
