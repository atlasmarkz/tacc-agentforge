#!/usr/bin/env bash
# Submit a TACC job from Windows with the native tcloud SDK 0.4.1.
#
# Usage: bash deploy/tacc/tools/submit-win.sh <job-dir>      (job-dir holds tuxiv.conf)
#
# Why this exists: on Windows the SDK (a) hands rsync "E:\..." paths, (b) writes
# run.sh / run.slurm with backslash remote paths and appends the local job path to
# TACC_WORKDIR, and (c) fails its post-upload environment check. So we
#   1. run `tcloud submit` through tcloud-win.sh (rsync shim: tar-over-ssh upload),
#      which uploads the directory and generates run.sh / configurations/run.slurm,
#   2. repair those two generated files on the cluster, and
#   3. call sbatch ourselves.
# Nothing is submitted by step 1 (the SDK stops at the environment check).
set -euo pipefail
export MSYS_NO_PATHCONV=1   # keep /mnt/home/... arguments intact for the remote shell
job_dir=$(cd -- "${1:?job directory}" && pwd)
name=$(basename "$job_dir")
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ssh_exe='C:/Windows/System32/OpenSSH/ssh.exe'
key='C:/Users/Mark/.ssh/id_rsa_tacc'
target='markzhang@gw.turing.ust.hk'
home=/mnt/home/markzhang

( cd "$job_dir" && bash "$here/tcloud-win.sh" submit >/dev/null 2>&1 ) || true

"$ssh_exe" -p 30041 -i "$key" -o BatchMode=yes "$target" bash -s -- "$home" "$name" <<'REMOTE'
set -euo pipefail
home=$1; name=$2
dir=$home/WORKDIR/$name
cd "$dir"
test -f configurations/run.slurm -a -f run.sh || { echo "SDK did not generate run files in $dir" >&2; exit 1; }
# A backslash at end of line is a shell line continuation, not a path separator: keep it.
expr='s#\\$#@@EOL@@#; s#\\WORKDIR\\E:\\[^ ]*\\@@N@@#\\WORKDIR\\@@N@@#g; s#\\#/#g; s#@@EOL@@$#\\#'
expr=${expr//@@N@@/$name}
for f in run.sh configurations/run.slurm; do
  # drop the local Windows path the SDK appended after WORKDIR, then flip slashes
  sed -i -E "$expr" "$f"
done
chmod +x run.sh
mkdir -p "$home/USERDIR/slurm_log"
if cat run.sh configurations/run.slurm | grep -v '\\$' | grep -q '\\'; then echo "backslashes remain" >&2; exit 1; fi
sbatch configurations/run.slurm
REMOTE
