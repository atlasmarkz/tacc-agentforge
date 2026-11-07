#!/usr/bin/env bash
# Run the native tcloud SDK on Windows with a path-translating rsync in front of PATH.
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MSYS_NO_PATHCONV=1
shim=$(cygpath -w "$here/shim")
export PATH="$shim;$(cygpath -w /e/Code/.cache/tcloud/bin);$PATH"
exec /e/Code/.cache/tcloud/bin/tcloud-native.exe "$@"
