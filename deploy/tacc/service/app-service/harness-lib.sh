# Sourced by app-service/job.sh (harness as its own CPU job) and qwen-service/job.sh (harness inside the vLLM job).
# Needs: the course runtime activated (qwen-python), TACC_USERDIR, and the staged package dir (stage.sh).

harness_python() (
    local pkg=$1 library=$2
    shift 2
    cd "$pkg" || return
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    PYTHONPATH="$pkg:$library${PYTHONPATH:+:$PYTHONPATH}" qwen-python "$@"
)

harness_check_deps() {
    harness_python "$1" "$2" -c 'from langsmith import RunTree; from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver; from app import harness_service, formspec, narrative'
}

harness_build_deps() (
    local pkg=$1 library=$2 marker=$3 reqs
    reqs=$(mktemp) || return
    trap 'rm -f -- "$reqs"' EXIT
    local pip_py="$TACC_USERDIR/../.Miniconda3/bin/python" newpip="$TACC_USERDIR/pip-new"
    if [ ! -d "$newpip/pip" ]; then
        "$pip_py" -m pip install --quiet --target "$newpip" "pip<25.1" || return
    fi
    local -a pip_cmd=(env PYTHONPATH="$newpip" "$pip_py" -m pip)
    "${pip_cmd[@]}" --version || return
    # Evaluate the universal lock with the course interpreter rather than Miniconda's Python 3.8.
    LOCK="$pkg/agent/requirements.lock" OUT="$reqs" qwen-python - <<'PY' || return
import os
from pathlib import Path
from packaging.requirements import Requirement

requirements = []
for line in Path(os.environ["LOCK"]).read_text().splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    requirement = Requirement(line)
    if requirement.marker is None or requirement.marker.evaluate():
        requirements.append(f"{requirement.name}{requirement.specifier}")
Path(os.environ["OUT"]).write_text("\n".join(requirements) + "\n")
PY
    local -a install_args=(--quiet --only-binary=:all --ignore-requires-python
        --python-version 3.12 --implementation cp --abi cp312
        --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64
        --platform manylinux_2_34_x86_64 --platform linux_x86_64
        --target "$library" --no-deps -r "$reqs")
    if [ -d "$pkg/wheels" ]; then
        install_args+=(--no-index --find-links "$pkg/wheels")
    fi
    PIP_DISABLE_PIP_VERSION_CHECK=1 "${pip_cmd[@]}" install "${install_args[@]}" || return
    harness_check_deps "$pkg" "$library" || return
    printf '%s\n' "${library##*/}" > "$library.ready" || return
    mv -f -- "$library.ready" "$marker"
)

harness_install_deps() {
    local pkg=$1 lock_sha marker generation
    local cache="$TACC_USERDIR/agent-harness-deps"
    lock_sha=$(sha256sum "$pkg/agent/requirements.lock" | cut -d ' ' -f 1) || return
    mkdir -p -m 700 "$cache" || return
    marker="$cache/$lock_sha"
    generation=$(cat "$marker" 2>/dev/null || true)
    if [[ "$generation" == build.* && "$generation" != */* && -d "$cache/$generation" ]] &&
        harness_check_deps "$pkg" "$cache/$generation"; then
        HARNESS_LIB="$cache/$generation"
        echo "harness dependencies already validated in $HARNESS_LIB"
        return 0
    fi
    HARNESS_LIB=$(mktemp -d "$cache/build.XXXXXXXX") || return
    echo "installing harness dependencies into $HARNESS_LIB"
    if ! harness_build_deps "$pkg" "$HARNESS_LIB" "$marker"; then
        rm -rf -- "$HARNESS_LIB" "$HARNESS_LIB.ready"
        echo 'harness dependency installation or import check failed; cache not published' >&2
        return 1
    fi
    echo 'harness dependencies installed and validated'
}

harness_run() {
    local pkg=$1
    shift
    harness_python "$pkg" "$HARNESS_LIB" -u -m app.harness_service "$@"
}
