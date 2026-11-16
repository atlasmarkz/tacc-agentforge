# Sourced by app-service/job.sh (harness as its own CPU job) and qwen-service/job.sh (harness inside the vLLM job).
# Needs: the course runtime activated (qwen-python), TACC_USERDIR, and the staged package dir (stage.sh).
#
#   harness_install_deps PKG     install the harness dependencies into $TACC_USERDIR/agent-pylib once
#   harness_run PKG ARGS...      run app/harness_service.py from PKG with ARGS (in the foreground)

HARNESS_LIB="$TACC_USERDIR/agent-pylib"

harness_install_deps() {
    local pkg=$1 lock_sha installed_sha
    # Same install as deploy/tacc/service/agent-e2e/job.sh (exact versions from agent/requirements.lock).
    lock_sha=$(sha256sum "$pkg/agent/requirements.lock" | cut -d ' ' -f 1)
    installed_sha=$(cat "$HARNESS_LIB/.agent-requirements-sha256" 2>/dev/null || true)
    if [ "$lock_sha" = "$installed_sha" ] && PYTHONPATH="$HARNESS_LIB" qwen-python -c "import langgraph, pydantic; from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver; assert pydantic.VERSION == '2.14.2'" 2>/dev/null; then
        echo "harness dependencies already installed in $HARNESS_LIB"
        return 0
    fi
    echo "installing harness dependencies into $HARNESS_LIB"
    # The course vLLM env has no pip, so use the user's own Miniconda pip (Python 3.8) and ask it for
    # Python 3.12 wheels (what qwen-python runs), as the train-env-setup job did.
    local pip_py=/mnt/home/markzhang/.Miniconda3/bin/python
    # Miniconda's pip is 20.2.4 (2020): too old for modern wheel tags (manylinux_2_17+). Bootstrap a
    # recent pip (pure Python, still supports Python 3.8) into USERDIR and use that one.
    local newpip="$TACC_USERDIR/pip-new"
    if [ ! -d "$newpip/pip" ]; then
        "$pip_py" -m pip install --quiet --target "$newpip" "pip<25.1"
    fi
    local pip_cmd=(env PYTHONPATH="$newpip" "$pip_py" -m pip)
    "${pip_cmd[@]}" --version
    # pip itself runs on Python 3.8, so evaluate the universal lock's markers with the actual runtime first;
    # --python-version alone does not change marker evaluation.
    local reqs
    reqs=$(mktemp)
    LOCK="$pkg/agent/requirements.lock" OUT="$reqs" qwen-python - <<'PY'
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
    # Binary dependencies (pydantic-core, xxhash, ormsgpack) need explicit Linux platform tags when
    # targeting a different Python version, otherwise pip only accepts pure-Python wheels.
    PIP_DISABLE_PIP_VERSION_CHECK=1 "${pip_cmd[@]}" install --quiet --only-binary=:all: --ignore-requires-python \
        --python-version 3.12 --implementation cp --abi cp312 \
        --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 \
        --platform manylinux_2_34_x86_64 --platform linux_x86_64 \
        --target "$HARNESS_LIB" --upgrade --no-deps -r "$reqs"
    rm -f "$reqs"
    PYTHONPATH="$HARNESS_LIB" qwen-python -c "import langgraph, pydantic; print('harness deps ok, pydantic', pydantic.VERSION)"
    printf '%s\n' "$lock_sha" > "$HARNESS_LIB/.agent-requirements-sha256"
}

harness_run() {
    local pkg=$1
    shift
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    (cd "$pkg" && PYTHONPATH="$pkg:$HARNESS_LIB${PYTHONPATH:+:$PYTHONPATH}" qwen-python -u -m app.harness_service "$@")
}
