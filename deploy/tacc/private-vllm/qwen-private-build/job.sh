#!/usr/bin/env bash
set -euo pipefail
: "${TACC_WORKDIR:?Submit with tcloud submit}"
: "${TACC_USERDIR:?Submit with tcloud submit}"
: "${SLURM_JOB_ID:?Submit with tcloud submit}"
: "${CONDA_PREFIX:?SDK must activate the course management environment}"
test "$CONDA_PREFIX" = /mnt/home/public/course-envs/CSIT6000V/assignment1/mpi-gemm-v1 || {
    printf 'Unexpected SDK environment: %s\nRun setup-sdk-account.sh from the same package as this job.\n' "$CONDA_PREFIX" >&2
    exit 1
}
printf 'SDK_BOOTSTRAP=%s\n' "$CONDA_PREFIX"
umask 077
project="$TACC_USERDIR/CSIT6000V-private-vllm"
export QWEN_PRIVATE_ENV="$project/.venv"
source /mnt/home/public/course-envs/CSIT6000V/project-private-vllm/activate.sh
source_tmp=$(mktemp -d /tmp/csit6000v-vllm-source.XXXXXX)
build_started=false
build_passed=false
cleanup() {
    status=$?
    trap - EXIT
    if "$build_started"; then
        if ! "$build_passed"; then
            touch "$project/.build-failed"
            echo 'PRIVATE VLLM BUILD FAILED: fix the error above and resubmit the build. Start the service after PRIVATE VLLM BUILD PASSED.' >&2
        fi
        rmdir "$project/.build-in-progress"
    fi
    rm -rf -- "$source_tmp" "$QWEN_RUN_DIR"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
tar -xzf "$TACC_WORKDIR/vllm-source.tar.gz" -C "$source_tmp"
test -f "$source_tmp/pyproject.toml" || {
    echo 'Invalid source archive: pyproject.toml is missing. Run prepare-source.sh and submit-build.sh again.' >&2
    exit 1
}
mkdir -p "$project"
mkdir "$project/.build-in-progress" || {
    echo 'A private build is already running, or a previous job left .build-in-progress. Check the previous build job before retrying; see README.' >&2
    exit 1
}
build_started=true
# Keep service startup gated on a successful build.
touch "$project/.build-failed"
# This build replaces only the submitted source mirror in this personal project.
rm -rf -- "$project/vllm-src"
python3 "$QWEN_PROJECT_ROOT/tooling/parallel-copy.py" "$source_tmp" "$project/vllm-src"
source_dir="$project/vllm-src"
cd "$project"
export UV_LINK_MODE=copy
export UV_CACHE_DIR="$QWEN_RUN_DIR/uv-cache"
export UV_SYSTEM_CERTS=true
uv="$QWEN_PROJECT_ROOT/tooling/uv"
qwen-run "$uv" venv --clear --python "$QWEN_PROJECT_ROOT/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12" "$QWEN_PRIVATE_ENV"
printf '%s\n' "$QWEN_PROJECT_ROOT/vllm-env/lib/python3.12/site-packages" \
    > "$QWEN_PRIVATE_ENV/lib/python3.12/site-packages/course-public-dependencies.pth"
qwen-run "$uv" pip install --python "$QWEN_PRIVATE_ENV/bin/python" \
    --target "$QWEN_PRIVATE_ENV/lib/python3.12/site-packages" \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    'setuptools==80.10.2' 'setuptools-scm>=8,<10' 'setuptools-rust>=1.9,<2' wheel
export VLLM_USE_PRECOMPILED=1
export VLLM_PRECOMPILED_WHEEL_LOCATION="$QWEN_PROJECT_ROOT/tooling/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.26.0
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
cd "$source_dir"
qwen-run "$uv" pip install --python "$QWEN_PRIVATE_ENV/bin/python" \
    --no-deps --no-build-isolation --editable .
cd "$project"
qwen-python "$TACC_WORKDIR/check-private-env.py"
rm -f "$project/.build-failed"
build_passed=true
echo 'PRIVATE VLLM BUILD PASSED'
