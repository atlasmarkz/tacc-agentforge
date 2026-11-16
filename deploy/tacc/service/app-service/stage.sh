#!/usr/bin/env bash
# Stage the service, harness, knowledge package, configs and seed snapshot (git-ignored).
#   stage.sh          -> app-service/pkg/ (harness as its own job)
#   stage.sh DEST     -> DEST (used by qwen-service/stage.sh serve --harness)
set -euo pipefail
cd "$(dirname "$0")"
root=$(cd ../../../.. && pwd)
dest=${1:-pkg}
rm -rf "$dest"
mkdir -p "$dest/inference" "$dest/configs" "$dest/assets/knowledge" "$dest/app/static"
tar -C "$root" -cf - --exclude='.venv' --exclude='__pycache__' --exclude='tests' agent | tar -xf - -C "$dest"
tar -C "$root" -cf - --exclude='__pycache__' knowledge | tar -xf - -C "$dest"
cp "$root/app/__init__.py" "$root/app/harness_service.py" "$root/app/selftest.py" "$root/app/backend.py" "$dest/app/"
cp "$root/app/static/index.html" "$dest/app/static/"
cp "$root/configs/knowledge.json" "$root/configs/category_schemas.json" "$root/configs/shopping_queries.json" \
   "$root/configs/agent.yaml" "$dest/configs/"
cp "$root/assets/knowledge/pilot-snapshot.zip" "$dest/assets/knowledge/"
cp "$root/inference/__init__.py" "$root/inference/client.py" "$root/inference/model_port.py" "$dest/inference/"
echo "staged harness: $(find "$dest" -type f | wc -l) files in $dest"
