#!/usr/bin/env bash
# Stage the service, harness, knowledge package, configs and seed snapshot into pkg/ (git-ignored).
set -euo pipefail
cd "$(dirname "$0")"
root=../../../..
rm -rf pkg
mkdir -p pkg/inference pkg/configs pkg/assets/knowledge pkg/app/static
tar -C "$root" -cf - --exclude='.venv' --exclude='__pycache__' --exclude='tests' agent | tar -xf - -C pkg
tar -C "$root" -cf - --exclude='__pycache__' knowledge | tar -xf - -C pkg
cp "$root/app/__init__.py" "$root/app/harness_service.py" "$root/app/selftest.py" "$root/app/backend.py" pkg/app/
cp "$root/app/static/index.html" pkg/app/static/
cp "$root/configs/knowledge.json" "$root/configs/category_schemas.json" "$root/configs/shopping_queries.json"    "$root/configs/agent.yaml" pkg/configs/
cp "$root/assets/knowledge/pilot-snapshot.zip" pkg/assets/knowledge/
cp "$root/inference/__init__.py" "$root/inference/client.py" "$root/inference/model_port.py" pkg/inference/
echo "staged: $(find pkg -type f | wc -l) files"
