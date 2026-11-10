#!/usr/bin/env bash
# CPU-only snapshot/query worker. No credential or public API access is needed.
set -euo pipefail
umask 077
: "${TACC_WORKDIR:?Submit with tcloud submit}"
: "${TACC_USERDIR:?Submit with tcloud submit}"
: "${SLURM_JOB_ID:?Submit with tcloud submit}"
cd "$TACC_WORKDIR"
test -f _stage/pkg/configs/knowledge.json || { echo 'Run stage.py before submitting.' >&2; exit 2; }

knowledge_run_dir=''
cleanup() {
    if [[ -n "$knowledge_run_dir" ]]; then rm -rf -- "$knowledge_run_dir"; fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

if [[ -n "${KNOWLEDGE_PYTHON:-}" ]]; then
    runner="$KNOWLEDGE_PYTHON"
else
    activation="${KNOWLEDGE_RUNTIME_ACTIVATE:-/mnt/home/public/course-envs/CSIT6000V/project/activate.sh}"
    test -f "$activation" || { echo 'Course runtime unavailable; configure KNOWLEDGE_RUNTIME_ACTIVATE or KNOWLEDGE_PYTHON.' >&2; exit 2; }
    source "$activation"
    knowledge_run_dir="${QWEN_RUN_DIR:-}"
    runner=qwen-python
fi

export PYTHONPATH="$TACC_WORKDIR/_stage/pkg${PYTHONPATH:+:$PYTHONPATH}"
"$runner" -c "import sys,sqlite3; assert sys.version_info >= (3,12), 'Python 3.12+ required'; c=sqlite3.connect(':memory:'); c.execute('CREATE VIRTUAL TABLE t USING fts5(text)'); print('KB_RUNTIME',sys.version.split()[0],sqlite3.sqlite_version)"
"$runner" -u worker.py
