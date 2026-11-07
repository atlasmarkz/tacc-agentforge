#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test -f "$root/vllm-src/pyproject.toml" || {
    echo 'Missing vllm-src/pyproject.toml. Run bash prepare-source.sh first.' >&2
    exit 1
}
# Upload source only; private environments and compiled binaries stay on TACC.
tar --exclude=.git --exclude=__pycache__ --exclude='*.so' \
    --exclude=./.venv --exclude=./venv --exclude=./build --exclude=./dist \
    -czf "$root/qwen-private-build/vllm-source.tar.gz" -C "$root/vllm-src" .
cd "$root/qwen-private-build"
"${TCLOUD_BIN:-tcloud}" submit
