#!/usr/bin/env bash
# One-off probe: can a running job submit (and schedule) its own successor? Prints facts, changes nothing else.
set -uo pipefail
: "${TACC_WORKDIR:?}" "${TACC_USERDIR:?}" "${SLURM_JOB_ID:?}"
echo "node=$(hostname) job=$SLURM_JOB_ID user=$USER"
for c in sbatch squeue scancel scontrol; do printf '%s=%s\n' "$c" "$(command -v $c || echo MISSING)"; done
echo "--- run.slurm of this job exists?"; ls -l "$TACC_WORKDIR/configurations/run.slurm" 2>&1
echo "--- my jobs"; squeue -u "$USER" -h -o '%i %j %T %M %l' 2>&1
echo "--- time limit of this job"; scontrol show job "$SLURM_JOB_ID" 2>&1 | grep -oE 'TimeLimit=[^ ]+|StartTime=[^ ]+|EndTime=[^ ]+'
echo "--- test-only submission"; sbatch --test-only -p tacc -t 00:01:00 --wrap "true" 2>&1
echo "--- real submission of a 1-minute child that starts in 60 s"
child=$(sbatch --parsable -p tacc -t 00:01:00 -J chain-child --begin=now+60 \
        -o "$TACC_USERDIR/slurm_log/chain-child-%j.log" --wrap "hostname; date" 2>&1); rc=$?
echo "child_submit_rc=$rc child=$child"
sleep 5; echo "--- after submit"; squeue -u "$USER" -h -o '%i %j %T %r %S' 2>&1
echo "--- submission with dependency on this job"
dep=$(sbatch --parsable -p tacc -t 00:01:00 -J chain-dep --dependency=afterany:$SLURM_JOB_ID \
      -o "$TACC_USERDIR/slurm_log/chain-dep-%j.log" --wrap "hostname; date" 2>&1); echo "dep_submit_rc=$? dep=$dep"
squeue -u "$USER" -h -o '%i %j %T %r %S' 2>&1
echo PROBE_DONE
