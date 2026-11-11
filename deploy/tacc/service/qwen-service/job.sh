#!/usr/bin/env bash
# Reliable inference service for the agent harness: one vLLM server per job, address published when ready.
set -euo pipefail
umask 077
: "${TACC_WORKDIR:?Submit with tcloud submit}"
: "${TACC_USERDIR:?Submit with tcloud submit}"
: "${SLURM_JOB_ID:?Submit with tcloud submit}"
: "${CONDA_PREFIX:?SDK must activate the course management environment}"
test "$CONDA_PREFIX" = /mnt/home/public/course-envs/CSIT6000V/assignment1/mpi-gemm-v1
source /mnt/home/public/course-envs/CSIT6000V/project/activate.sh
HELPER=""
trap 'if [ -n "$HELPER" ]; then kill "$HELPER" 2>/dev/null || true; fi; rm -rf -- "$QWEN_RUN_DIR"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
export QWEN_CACHE_HOME="$TACC_USERDIR/CSIT6000V-project-cache"
mkdir -p -m 700 "$QWEN_CACHE_HOME"
cd "$TACC_WORKDIR"
MODE="$(tr -d '[:space:]' < mode.txt 2>/dev/null || echo serve)"
BASE="$TACC_USERDIR/tacc-agentforge-service"
OUT="$BASE/service-$SLURM_JOB_ID"
mkdir -p "$OUT"
hostname
ARGS=(--out "$OUT" --address-file "$BASE/service-url.txt")
if [ "$MODE" = smoke ]; then
    # Validation run: publish to a separate file so a running real service is never overwritten.
    ARGS=(--out "$OUT" --address-file "$BASE/service-url-smoke.txt" --smoke --exit-after-smoke)
fi
if [ "$MODE" = validate ]; then
    # Validation run (load and fault tests): the load job may ask this service to kill vLLM.
    ARGS+=(--allow-fault-injection)
fi
# serve.py runs inside the course runtime's proot root filesystem, where sbatch and the Slurm config do not
# exist (job 28581). For a handover it writes $OUT/handover-request; this helper, running here outside proot,
# submits the successor and writes the outcome to $OUT/handover-result.
handover_helper() {
    while true; do
        if [ -f "$OUT/handover-request" ]; then
            if result=$(/usr/bin/sbatch --parsable "$TACC_WORKDIR/configurations/run.slurm" 2>&1); then
                printf 'ok %s\n' "$result" > "$OUT/handover-result.tmp"
            else
                printf 'error %s\n' "$result" > "$OUT/handover-result.tmp"
            fi
            mv "$OUT/handover-result.tmp" "$OUT/handover-result"
            rm -f "$OUT/handover-request"
        fi
        sleep 2
    done
}
handover_helper &
HELPER=$!
# Optional extra serve.py options (git-ignored args.txt), e.g. --deadline-s 1500 --handover-lead-s 600
read -r -a EXTRA <<< "$(cat args.txt 2>/dev/null || true)"
qwen-python -u serve.py "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
echo "RESULTS_DIR=$OUT"
