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
HARNESS=""
HSTOP=""
HRUN=""
cleanup() {
    if [ -n "$HSTOP" ]; then
        touch "$HSTOP" 2>/dev/null || true
        # Kill the harness process itself, not only its wrapper subshell: once the wrapper is gone the proot
        # and Python processes are re-parented to init, Slurm's proctrack/linuxproc no longer sees them, and
        # they keep running on the node after the job (jobs 28925 and 28943 left two behind on 10-0-3-10).
        stop_harness_processes
    fi
    for pid in $HELPER $HARNESS; do kill "$pid" 2>/dev/null || true; done
    rm -rf -- "$QWEN_RUN_DIR" ${HRUN:+"$HRUN"}
}
stop_harness_processes() {
    local pattern="app.harness_service --stop-file $HSTOP"
    pkill -TERM -u "$(id -u)" -f "$pattern" 2>/dev/null || return 0
    for _ in 1 2 3 4 5; do
        pgrep -u "$(id -u)" -f "$pattern" > /dev/null || return 0
        sleep 1
    done
    pkill -KILL -u "$(id -u)" -f "$pattern" 2>/dev/null || true
}
trap cleanup EXIT
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
# On the 10-0-*-11 nodes Slurm sets CUDA_VISIBLE_DEVICES=2,3, but the job sees only its own two GPUs, numbered
# 0 and 1, so CUDA finds no device and vLLM fails with NVMLError_InvalidArgument (jobs 28936, 28955; probes
# 28956/28957). Renumber to what the job can see when the allocated IDs are out of range.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && command -v nvidia-smi > /dev/null; then
    visible=$(nvidia-smi -L | grep -c '^GPU ')
    IFS=, read -r -a allocated <<< "$CUDA_VISIBLE_DEVICES"
    for id in "${allocated[@]}"; do
        if [[ "$id" =~ ^[0-9]+$ ]] && [ "$id" -ge "$visible" ] && [ "${#allocated[@]}" -le "$visible" ]; then
            CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((${#allocated[@]} - 1)))
            echo "CUDA_VISIBLE_DEVICES ${allocated[*]} is outside the $visible visible GPUs; using $CUDA_VISIBLE_DEVICES"
            export CUDA_VISIBLE_DEVICES
            break
        fi
    done
fi
# Processes outside Slurm sometimes hold the allocated GPUs (10-0-1-10 and 10-0-1-11, probes 28961/28962: 15-23 GiB
# used per GPU with no job running). vLLM then spends ~20 minutes failing with out-of-memory errors; stop now instead.
if command -v nvidia-smi > /dev/null; then
    busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' '$2 > 2048 {printf "%s:%sMiB ", $1, $2}')
    if [ -n "$busy" ]; then
        echo "GPUs already in use by processes outside this job: $busy- not starting vLLM on $(hostname)"
        exit 3
    fi
fi
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
            # Run sbatch from this job's own results directory: the workdir can be deleted and re-created by a
            # later tcloud submit of the same directory, and sbatch then fails with "getcwd failed" (job 28999).
            if result=$(cd "$OUT" && /usr/bin/sbatch --parsable "$TACC_WORKDIR/configurations/run.slurm" 2>&1); then
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

# Task 5: with the harness staged (stage.sh serve --harness), this job also runs the agent harness service
# (app/harness_service.py), so vLLM and the harness need one job slot and the second slot stays free for the
# handover successor. Its dependency install overlaps with vLLM loading; the service starts once this job's vLLM
# is ready, publishes harness-url.txt / harness-token.txt like the app-service job, and calls vLLM through the
# address file, so after a handover it follows the successor's vLLM. A harness failure never stops vLLM.
# The harness gets its own runtime directory (its own proot rootfs). Sharing serve.py's rootfs, the harness
# stopped seeing files under /mnt/home (the stop file, the vLLM address file) as soon as serve.py's proot
# exited (job 28943: every request after that failed with model_unavailable).
if [ -d pkg/app ] && [ "$MODE" = serve ] && HRUN=$(bash "$QWEN_PROJECT_ROOT/prepare-runtime.sh"); then
    HSTOP="$OUT/harness-stop"
    source harness-lib.sh
    (
        export QWEN_RUN_DIR="$HRUN"
        harness_install_deps "$TACC_WORKDIR/pkg"
        # If vLLM never becomes ready (job 28955), the stop file ends the wait instead of the 120 s timeout.
        until grep -q '"event": "ready"' "$OUT/events.jsonl" 2>/dev/null; do
            if [ -e "$HSTOP" ]; then echo "harness: vLLM never became ready, not starting"; exit 0; fi
            sleep 5
        done
        # Optional harness options (git-ignored harness-args.txt), e.g. --explain or --profile tacc-thinking.
        read -r -a HARGS <<< "$(cat harness-args.txt 2>/dev/null || true)"
        harness_run "$TACC_WORKDIR/pkg" --stop-file "$HSTOP" ${HARGS[@]+"${HARGS[@]}"}
    ) > "$OUT/harness.log" 2>&1 &
    HARNESS=$!
    echo "harness service: starts when vLLM is ready, log $OUT/harness.log"
fi

stop_harness() {
    local mine waited=0
    mine=$(sed -n 's/.*READY harness service \(http[^ ]*\).*/\1/p' "$OUT/harness.log" 2>/dev/null | tail -n 1)
    # After a handover the successor publishes its own harness shortly after its vLLM: keep this one serving
    # until then (at most 5 minutes, never past the job limit), so a new request always finds a harness.
    if [ -n "$mine" ] && grep -q '"event": "superseded"' "$OUT/events.jsonl" 2>/dev/null; then
        while [ "$(cat "$BASE/harness-url.txt" 2>/dev/null)" = "$mine" ] && [ "$waited" -lt 300 ] && [ "$SECONDS" -lt 3450 ]; do
            sleep 5
            waited=$((waited + 5))
        done
        echo "harness: successor published after ${waited}s wait"
    fi
    # The service stops accepting, lets running requests finish and withdraws its own address.
    if kill -0 "$HARNESS" 2>/dev/null; then echo "harness: running, asking it to stop ($(date +%T))"; else echo "harness: already exited"; fi
    touch "$HSTOP"
    for _ in $(seq 1 60); do
        kill -0 "$HARNESS" 2>/dev/null || break
        sleep 2
    done
    if kill -0 "$HARNESS" 2>/dev/null; then
        echo "harness: still running after 120 s, terminating it ($(date +%T))"
        stop_harness_processes
    else
        echo "harness: exited ($(date +%T))"
    fi
    # Never leave a dead address behind: withdraw it here if it is still this job's own.
    if [ -n "$mine" ] && [ "$(cat "$BASE/harness-url.txt" 2>/dev/null)" = "$mine" ]; then
        rm -f "$BASE/harness-url.txt"
        echo "harness: address $mine withdrawn by job.sh"
    fi
}

# Optional extra serve.py options (git-ignored args.txt), e.g. --deadline-s 1500 --handover-lead-s 600
read -r -a EXTRA <<< "$(cat args.txt 2>/dev/null || true)"
set +e
qwen-python -u serve.py "${ARGS[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
CODE=$?
set -e
if [ -n "$HARNESS" ]; then
    stop_harness
fi
echo "RESULTS_DIR=$OUT"
exit "$CODE"
