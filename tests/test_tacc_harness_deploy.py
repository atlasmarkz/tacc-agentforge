import os
from pathlib import Path
import subprocess
import sys


HARNESS_LIB = Path(__file__).resolve().parents[1] / "deploy/tacc/service/app-service/harness-lib.sh"


def run_cache_checks(tmp_path, script):
    return subprocess.run(
        ["bash", "-eu", "-o", "pipefail", "-c", script],
        env={**os.environ, "TACC_USERDIR": str(tmp_path), "LIB_SCRIPT": str(HARNESS_LIB), "PYTHON": sys.executable,
             "PYTHONPATH": ""},
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_incomplete_cache_is_replaced_and_then_reused(tmp_path):
    result = run_cache_checks(tmp_path, r"""
source "$LIB_SCRIPT"
pkg="$PWD/pkg"
cache="$TACC_USERDIR/agent-harness-deps"
mkdir -p "$pkg/agent" "$cache/build.incomplete"
printf 'langsmith==0.14.17\n' > "$pkg/agent/requirements.lock"
lock_sha=$(sha256sum "$pkg/agent/requirements.lock" | cut -d ' ' -f 1)
printf 'build.incomplete\n' > "$cache/$lock_sha"

harness_check_deps() {
    test "$1" = "$pkg" && test -s "$2/langsmith/__init__.py"
}
harness_build_deps() {
    mkdir -p "$2/langsmith"
    printf 'class RunTree: pass\n' > "$2/langsmith/__init__.py"
    harness_check_deps "$1" "$2" || return
    printf '%s\n' "$(basename "$2")" > "$3"
}

harness_install_deps "$pkg"
repaired="$HARNESS_LIB"
test "$repaired" != "$cache/build.incomplete"
test -d "$cache/build.incomplete"
harness_install_deps "$pkg"
test "$HARNESS_LIB" = "$repaired"
""")
    assert result.stdout.count("installing harness dependencies") == 1
    assert "harness dependencies already validated" in result.stdout


def test_failed_install_does_not_publish_cache(tmp_path):
    run_cache_checks(tmp_path, r"""
source "$LIB_SCRIPT"
pkg="$PWD/pkg"
cache="$TACC_USERDIR/agent-harness-deps"
mkdir -p "$pkg/agent"
printf 'langsmith==0.14.17\n' > "$pkg/agent/requirements.lock"

harness_build_deps() {
    touch "$2.ready"
    return 1
}

if harness_install_deps "$pkg"; then exit 1; fi
test -z "$(ls -A "$cache")"
""")


def test_full_import_probe_rejects_missing_runtree(tmp_path):
    run_cache_checks(tmp_path, r"""
source "$LIB_SCRIPT"
pkg="$PWD/pkg"
library="$PWD/library"
mkdir -p "$pkg/app" "$library/langsmith" "$library/langgraph/checkpoint/sqlite"
touch "$pkg/app/__init__.py" "$pkg/app/harness_service.py" "$pkg/app/formspec.py" "$pkg/app/narrative.py"
printf 'class AsyncSqliteSaver: pass\n' > "$library/langgraph/checkpoint/sqlite/aio.py"

qwen-python() {
    "$PYTHON" -S "$@"
}

if harness_check_deps "$pkg" "$library" 2> error.log; then exit 1; fi
grep -q 'cannot import name.*RunTree' error.log
printf 'class RunTree: pass\n' > "$library/langsmith/__init__.py"
harness_check_deps "$pkg" "$library"
""")
