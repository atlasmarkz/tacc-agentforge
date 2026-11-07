#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test ! -e "$root/vllm-src" || { echo 'vllm-src exists; keep your edited source.' >&2; exit 1; }
test -f "$root/vllm-v0.26.0.tar.gz" || {
    echo 'Missing vllm-v0.26.0.tar.gz. Re-extract the complete private package.' >&2
    exit 1
}
mkdir "$root/vllm-src"
tar -xzf "$root/vllm-v0.26.0.tar.gz" --strip-components=1 -C "$root/vllm-src"
echo 'Source ready locally. Edit vllm-src, then run bash submit-build.sh.'
