"""Keeps the TACC vLLM service running while the cloud service is up (issue #25, cloud host).

The GPUs are only used while `agentforge.target` is started on the VM:
- start: `--start` clears the stop switch and submits the vLLM job (agentforge-cluster.service, ExecStart);
- while up: a timer runs the plain check every few minutes (below);
- stop: `--release` sets the stop switch and cancels the vLLM jobs (agentforge-cluster.service, ExecStop).
A crash restart of the backend or harness does not touch the cluster; only stopping the target does.

The check, over SSH, looks at the cluster and

- tops `service-chain.json` up to `--target-remaining` handovers, so the hourly handover chain never runs out
  (serve.py decrements it on every handover, RUNBOOK section 5);
- submits `WORKDIR/qwen-service/configurations/run.slurm` when no `qwen-service` job is running or queued, e.g.
  after a successor could not be placed, a job failed its GPU check, or the chain was broken by hand.

Safety:
- `USERDIR/tacc-agentforge-service/service-stop` (the RUNBOOK's stop switch) turns both off: nothing is
  submitted and the chain is not topped up while it exists.
- At most `--max-submits-per-hour` submissions per rolling hour (state in $AGENTFORGE_HOME/keeper.json): a job
  that fails right away (a node whose GPUs are taken outside Slurm) is not resubmitted in a tight loop.
- It submits whatever is staged in WORKDIR/qwen-service. Stage the vLLM-only service there first
  (stage.sh serve, no --harness; deploy/gcp/README.md).

  python -m app.service_keeper            # one check (what the timer runs)
  python -m app.service_keeper --dry-run  # show what it would do

Environment: TACC_SSH, TACC_SSH_PORT, TACC_SSH_KEY (as app/model_tunnel.py), AGENTFORGE_HOME.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

JOB_NAME = "qwen-service"

# Prints one key=value per line. Runs on the gateway as the account user.
STATUS_SCRIPT = r"""
base=$HOME/USERDIR/tacc-agentforge-service
echo "jobs=$(squeue -h -u "$USER" -n %(job)s -o '%%i:%%T' 2>/dev/null | tr '\n' ' ')"
echo "url=$(cat "$base/service-url.txt" 2>/dev/null)"
if [ -e "$base/service-stop" ]; then echo stop=1; else echo stop=0; fi
echo "remaining=$(sed -n 's/.*"remaining"[^0-9-]*\(-\{0,1\}[0-9]\{1,\}\).*/\1/p' "$base/service-chain.json" 2>/dev/null | head -1)"
test -f "$HOME/WORKDIR/%(job)s/configurations/run.slurm" && echo staged=1 || echo staged=0
test -d "$HOME/WORKDIR/%(job)s/pkg/app" && echo harness_staged=1 || echo harness_staged=0
""" % {"job": JOB_NAME}

TOPUP_SCRIPT = r"""
base=$HOME/USERDIR/tacc-agentforge-service
mkdir -p "$base"
printf '{"remaining": %d}\n' "$1" > "$base/service-chain.json.tmp" && mv "$base/service-chain.json.tmp" "$base/service-chain.json"
"""

START_SCRIPT = r"""
rm -f "$HOME/USERDIR/tacc-agentforge-service/service-stop"
"""

# Stop handing over and free the GPUs now: the stop switch keeps a successor from being submitted, scancel ends
# the running and queued service jobs (each withdraws its own address on SIGTERM, RUNBOOK section 6).
RELEASE_SCRIPT = r"""
base=$HOME/USERDIR/tacc-agentforge-service
mkdir -p "$base" && touch "$base/service-stop"
ids=$(squeue -h -u "$USER" -n %(job)s -o '%%i' 2>/dev/null | tr '\n' ' ')
if [ -n "$ids" ]; then scancel $ids && echo "cancelled $ids"; else echo "no service job"; fi
""" % {"job": JOB_NAME}

# sbatch from USERDIR, not the workdir (job.sh: a re-uploaded workdir broke sbatch's getcwd, job 28999).
SUBMIT_SCRIPT = r"""
mkdir -p "$HOME/USERDIR/slurm_log" && cd "$HOME/USERDIR" && sbatch --parsable "$HOME/WORKDIR/%(job)s/configurations/run.slurm"
""" % {"job": JOB_NAME}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


@dataclass
class ClusterState:
    jobs: list[str] = field(default_factory=list)   # "<id>:<STATE>" of qwen-service jobs (running or queued)
    url: str = ""
    stop: bool = False
    remaining: int = 0
    staged: bool = True
    harness_staged: bool = False


def parse_status(text: str) -> ClusterState:
    values = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    remaining = values.get("remaining", "").strip()
    return ClusterState(jobs=values.get("jobs", "").split(), url=values.get("url", "").strip(),
                        stop=values.get("stop") == "1",
                        remaining=int(remaining) if remaining.lstrip("-").isdigit() else 0,
                        staged=values.get("staged", "1") == "1", harness_staged=values.get("harness_staged") == "1")


def decide(state: ClusterState, submits: list[float], now: float, *, target_remaining: int = 3,
           max_submits_per_hour: int = 3) -> tuple[list[tuple], str]:
    """Actions for one check: ("topup", n) and/or ("submit",), and a one-line reason for the log."""
    if state.stop:
        return [], "service-stop is set: not submitting, not topping up the chain"
    actions: list[tuple] = []
    if state.remaining < target_remaining:
        actions.append(("topup", target_remaining))
    if state.jobs:
        return actions, f"running: {' '.join(state.jobs)}" + (f", url {state.url}" if state.url else ", not ready")
    if not state.staged:
        return actions, "no job, and WORKDIR/qwen-service has no run.slurm: stage and submit it once by hand"
    if state.harness_staged:
        return actions, "no job, but WORKDIR/qwen-service still has the in-cluster harness (pkg/): restage with stage.sh serve"
    recent = [t for t in submits if now - t < 3600]
    if len(recent) >= max_submits_per_hour:
        wait = int(3600 - (now - min(recent)))
        return actions, f"no job, but {len(recent)} submissions in the last hour: waiting {wait} s before the next"
    actions.append(("submit",))
    return actions, "no qwen-service job running or queued: submitting"


class Keeper:
    def __init__(self, ssh_base: list[str], target: str, state_file: Path):
        self.ssh_base, self.target, self.state_file = ssh_base, target, state_file

    def remote(self, script: str, *args: str) -> str:
        done = subprocess.run(self.ssh_base + [self.target, "bash", "-s", "--", *args], input=script,
                              capture_output=True, text=True, timeout=60)
        if done.returncode != 0:
            raise RuntimeError(f"ssh exit {done.returncode}: {done.stderr.strip()[-300:]}")
        return done.stdout

    def load(self) -> list[float]:
        try:
            return [float(t) for t in json.loads(self.state_file.read_text())["submits"]]
        except (OSError, ValueError, KeyError, TypeError):
            return []

    def save(self, submits: list[float]) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_name(self.state_file.name + ".tmp")
        tmp.write_text(json.dumps({"submits": submits[-20:]}))
        os.replace(tmp, self.state_file)

    def check(self, *, target_remaining: int, max_submits_per_hour: int, dry_run: bool = False) -> int:
        state = parse_status(self.remote(STATUS_SCRIPT))
        submits, now = self.load(), time.time()
        actions, reason = decide(state, submits, now, target_remaining=target_remaining,
                                 max_submits_per_hour=max_submits_per_hour)
        log(reason + (f" (chain {state.remaining})" if not state.stop else ""))
        for action in actions:
            if dry_run:
                log(f"dry run: would {action}")
                continue
            if action[0] == "topup":
                self.remote(TOPUP_SCRIPT, str(action[1]))
                log(f"chain topped up from {state.remaining} to {action[1]}")
            elif action[0] == "submit":
                submits.append(now)
                self.save(submits)                       # count it even if sbatch fails, so failures back off too
                job = self.remote(SUBMIT_SCRIPT).strip()
                log(f"submitted job {job}")
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-remaining", type=int, default=3, help="keep this many handovers allowed")
    ap.add_argument("--max-submits-per-hour", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--start", action="store_true", help="the service is starting: clear service-stop, then check "
                                                          "(submits right away instead of waiting for the timer)")
    mode.add_argument("--release", action="store_true", help="the service is stopping: set service-stop and cancel "
                                                            "the vLLM jobs so the GPUs are free")
    a = ap.parse_args()
    if not os.environ.get("TACC_SSH"):
        raise SystemExit("set TACC_SSH (user@gateway)")
    ssh = ["ssh", "-p", os.environ.get("TACC_SSH_PORT", "30041"),
           "-i", os.path.expanduser(os.environ.get("TACC_SSH_KEY", "~/.ssh/id_rsa_tacc")),
           "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
    home = Path(os.environ.get("AGENTFORGE_HOME", "/var/lib/agentforge"))
    keeper = Keeper(ssh, os.environ["TACC_SSH"], home / "keeper.json")
    try:
        if a.release:
            if a.dry_run:
                log("dry run: would set service-stop and cancel the qwen-service jobs")
                return 0
            log("releasing: " + keeper.remote(RELEASE_SCRIPT).strip())
            return 0
        if a.start and not a.dry_run:
            keeper.remote(START_SCRIPT)
            log("starting: service-stop cleared")
        return keeper.check(target_remaining=a.target_remaining, max_submits_per_hour=a.max_submits_per_hour,
                            dry_run=a.dry_run)
    except Exception as exc:                              # the next timer run tries again
        log(f"check failed: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
