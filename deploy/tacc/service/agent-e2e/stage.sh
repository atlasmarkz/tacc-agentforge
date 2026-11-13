#!/usr/bin/env bash
# Stage the harness, model adapter and KB package/configs. pkg/ is git-ignored.
set -euo pipefail
cd "$(dirname "$0")"
root=../../../..
rm -rf pkg
mkdir -p pkg/inference pkg/configs pkg/assets/knowledge
tar -C "$root" -cf - --exclude='.venv' --exclude='__pycache__' --exclude='tests' agent | tar -xf - -C pkg
tar -C "$root" -cf - --exclude='__pycache__' knowledge | tar -xf - -C pkg
tar -C "$root" -cf - --exclude="__pycache__" evaluations | tar -xf - -C pkg
cp "$root/configs/knowledge.json" "$root/configs/category_schemas.json" "$root/configs/shopping_queries.json" pkg/configs/
cp "$root/assets/knowledge/pilot-snapshot.zip" pkg/assets/knowledge/
cp "$root/inference/__init__.py" "$root/inference/client.py" "$root/inference/model_port.py" "$root/inference/harness_policy.py" pkg/inference/
echo "staged: $(find pkg -type f | wc -l) files"
