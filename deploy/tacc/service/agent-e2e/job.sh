#!/usr/bin/env bash
# CPU job: run the agent harness against the vLLM service job, from a different node.
set -euo pipefail
umask 077
: "${TACC_WORKDIR:?Submit with tcloud submit}"
: "${TACC_USERDIR:?Submit with tcloud submit}"
: "${SLURM_JOB_ID:?Submit with tcloud submit}"
: "${CONDA_PREFIX:?SDK must activate the course management environment}"
test "$CONDA_PREFIX" = /mnt/home/public/course-envs/CSIT6000V/assignment1/mpi-gemm-v1
source /mnt/home/public/course-envs/CSIT6000V/project/activate.sh
trap 'rm -rf -- "$QWEN_RUN_DIR"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
cd "$TACC_WORKDIR"
hostname
LIB="$TACC_USERDIR/agent-pylib"
export AGENT_CHECKPOINT_DB="${AGENT_CHECKPOINT_DB:-$TACC_USERDIR/tacc-agentforge/checkpoints/agent.sqlite}"
# This acknowledgement must follow a storage review, not just a successful SQLite open.
# Known network filesystems are also rejected by agent.checkpoints.
test "${AGENT_SQLITE_STORAGE_VERIFIED:-0}" = 1 || {
    echo 'Verify durable storage supports SQLite WAL, then set AGENT_SQLITE_STORAGE_VERIFIED=1 in this job.' >&2
    exit 1
}
LOCK_SHA=$(sha256sum pkg/agent/requirements.lock | cut -d ' ' -f 1)
INSTALLED_SHA=$(cat "$LIB/.agent-requirements-sha256" 2>/dev/null || true)
# Harness dependencies (langgraph, pydantic 2) are installed once into USERDIR and reused by later jobs.
if [ "$LOCK_SHA" != "$INSTALLED_SHA" ] || ! PYTHONPATH="$LIB" qwen-python -c "import langgraph, pydantic; from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver; assert pydantic.VERSION == '2.14.2'" 2>/dev/null; then
    echo "installing harness dependencies into $LIB"
    # The course vLLM env has no pip, so use the user's own Miniconda pip (Python 3.8) and ask it for
    # Python 3.12 wheels (what qwen-python runs), as the train-env-setup job did.
    PIP_PY=/mnt/home/markzhang/.Miniconda3/bin/python
    # Miniconda's pip is 20.2.4 (2020): too old for modern wheel tags (manylinux_2_17+). Bootstrap a
    # recent pip (pure Python, still supports Python 3.8) into USERDIR and use that one.
    NEWPIP="$TACC_USERDIR/pip-new"
    if [ ! -d "$NEWPIP/pip" ]; then
        "$PIP_PY" -m pip install --quiet --target "$NEWPIP" "pip<25.1"
    fi
    PIP_PY_CMD=(env PYTHONPATH="$NEWPIP" "$PIP_PY" -m pip)
    "${PIP_PY_CMD[@]}" --version
    # Binary dependencies (pydantic-core, xxhash, ormsgpack) need explicit Linux platform tags when
    # targeting a different Python version, otherwise pip only accepts pure-Python wheels.
    # pip itself runs on Python 3.8, so evaluate the universal lock's markers with the
    # actual runtime first; --python-version alone does not change marker evaluation.
    qwen-python - <<'PY'
from pathlib import Path
from packaging.requirements import Requirement

requirements = []
for line in Path("pkg/agent/requirements.lock").read_text().splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    requirement = Requirement(line)
    if requirement.marker is None or requirement.marker.evaluate():
        requirements.append(f"{requirement.name}{requirement.specifier}")
Path("agent-requirements-runtime.txt").write_text("\n".join(requirements) + "\n")
PY
    PIP_DISABLE_PIP_VERSION_CHECK=1 "${PIP_PY_CMD[@]}" install --quiet --only-binary=:all: --ignore-requires-python \
        --python-version 3.12 --implementation cp --abi cp312 \
        --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 \
        --platform manylinux_2_34_x86_64 --platform linux_x86_64 \
        --target "$LIB" --upgrade --no-deps -r agent-requirements-runtime.txt
    PYTHONPATH="$LIB" qwen-python -c "import langgraph, pydantic; from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver; print('harness deps ok, pydantic', pydantic.VERSION)"
    printf '%s\n' "$LOCK_SHA" > "$LIB/.agent-requirements-sha256"
fi
export PYTHONPATH="$TACC_WORKDIR/pkg:$LIB${PYTHONPATH:+:$PYTHONPATH}"
qwen-python -u e2e.py
