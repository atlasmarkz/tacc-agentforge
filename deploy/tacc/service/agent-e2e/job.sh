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
# Harness dependencies (langgraph, pydantic 2) are installed once into USERDIR and reused by later jobs.
if ! PYTHONPATH="$LIB" qwen-python -c "import langgraph, pydantic; assert pydantic.VERSION.startswith('2.14')" 2>/dev/null; then
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
    PIP_DISABLE_PIP_VERSION_CHECK=1 "${PIP_PY_CMD[@]}" install --quiet --only-binary=:all: --ignore-requires-python \
        --python-version 3.12 --implementation cp --abi cp312 \
        --platform manylinux2014_x86_64 --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 \
        --platform manylinux_2_34_x86_64 --platform linux_x86_64 \
        --target "$LIB" --upgrade -r pkg/agent/requirements.txt
    PYTHONPATH="$LIB" qwen-python -c "import langgraph, pydantic; print('harness deps ok, pydantic', pydantic.VERSION)"
fi
export PYTHONPATH="$TACC_WORKDIR/pkg:$LIB${PYTHONPATH:+:$PYTHONPATH}"
SCRIPT="$(tr -d '[:space:]' < script.txt 2>/dev/null || echo e2e.py)"
ARGS="$(cat args.txt 2>/dev/null || true)"
echo "script=$SCRIPT args=$ARGS"
qwen-python -u "$SCRIPT" $ARGS
