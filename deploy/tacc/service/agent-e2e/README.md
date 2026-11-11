# Persistent agent runs on TACC

`e2e.py` opens one durable checkpointer for the job, cleans retained runs before
accepting work, and injects it into every Runtime. New requests get a session
prefix, so repeated jobs do not overwrite old runs. Reports include `run_id` and
the full `pending_question` needed to resume. The synthetic tools and validator
are unchanged; the suite still uses the real vLLM service.

Set `AGENT_EXPLAIN=1` in the job to opt into the post-validation explanation node
(off by default). Reports include the final explanation and metrics relative to
`explain_started`: first content event delay, duration, delta event count and final
characters. The raw adapter monitoring deltas remain separate measurements.
Evidence facts in this harness are explicitly mocked, like its catalog/validator.
This is instrumentation for a future TACC run; explanation latency and output
quality have not been measured on the cluster by the local tests.

## Storage prerequisite

Before submitting a job, confirm the checkpoint volume is persistent across jobs
and nodes and supports SQLite WAL, file locking and sync. Inspect the mount on
the compute node, for example with `findmnt -T "$TACC_USERDIR"`. A successful
SQLite open or local restart test alone does not establish support.

The default database is
`$TACC_USERDIR/tacc-agentforge/checkpoints/agent.sqlite`. Override
`AGENT_CHECKPOINT_DB` in `job.sh` if necessary; it must resolve to the same
supported storage in every job. Do not use a path containing a job ID or a
node-local temporary path for cross-job recovery.

Only after storage support is established, add
`export AGENT_SQLITE_STORAGE_VERIFIED=1` in `job.sh` before its verification gate.
This acknowledgement does not override the helper's rejection of known network
filesystems. SQLite WAL is unsupported on NFS and other network filesystems;
if USERDIR is such a volume, this SQLite deployment must not be enabled there.
Provision a supported persistent volume or integrate a reachable durable
database saver through `AgentRuntime(checkpointer=...)` first. A PostgreSQL
deployment is not provisioned by these scripts.

Only one host may own this database at a time. Stop the previous job before
starting its replacement or running manual cleanup. The helper holds a POSIX
lease for the connection lifetime; a process kill releases it automatically.
Never remove the `.lock`, `-wal` or `-shm` files to bypass another owner.

`job.sh` fingerprints the universal agent lockfile, evaluates its markers using
the actual qwen Python runtime, and installs those exact versions into USERDIR.
Run `bash stage.sh` before uploading a new package/lockfile through the SDK.

## Cross-job acceptance

This check uses explicit mock capabilities to isolate persistence from LLM
availability. It needs no model service. Run two non-overlapping CPU jobs using
the same verified store and the same source/lock versions. For each job, replace
the final `qwen-python -u e2e.py` line in `job.sh` as follows, stage and submit
through the normal TACC SDK workflow.

For job A:

```bash
qwen-python -u -m agent.examples.persistent_resume \
  --db "$AGENT_CHECKPOINT_DB" --events start \
  --run-id restart-UNIQUE_ID --question-timeout-s 7200 --hold-after-waiting
```

Wait until `run_waiting` appears in the log. Record its `run_id`,
`pending_question.question_id`, sequence and usage, then cancel/terminate job A.
Do not rely on an EXIT trap to save checkpoints; the example keeps the database
open after emitting its result. The local automated test explicitly uses SIGKILL.

For job B, retain the original IDs:

```bash
qwen-python -u -m agent.examples.persistent_resume \
  --db "$AGENT_CHECKPOINT_DB" --events resume \
  --run-id restart-UNIQUE_ID --question-id QUESTION_ID \
  --answer '{"purchase_cap":"100.00"}'
```

Expect `run_resumed`, then `run_completed` with the same run/snapshot, cumulative
three model calls and one tool call, and continuous event sequences. Repeat on a
different node when scheduling permits. Answer before the saved expiry; the
7200-second TTL can be increased up to seven days when queue delays require it.
An expired retained question returns `incomplete / question_expired`.

For real-model resumes, recreate `VllmModelPort`, the same tools, validator and
available knowledge snapshot, and use the saved IDs with `astream_resume`.
The mock restart example does not resume a real-model run with its original
capabilities. Restarting the vLLM service also requires its address file to point
at the new service before continuing model calls.

## Retention and errors

Before accepting requests, startup cleanup keeps unexpired questions, retains
expired questions for another day, and retains terminal runs for seven days
from their latest checkpoint. Unknown versions and unfinished runs are kept
for manual review. A whole thread and its writes are removed together.

While the host is stopped, preview or customize cleanup:

```bash
qwen-python -m agent.checkpoint_cleanup --db "$AGENT_CHECKPOINT_DB" --dry-run
qwen-python -m agent.checkpoint_cleanup --db "$AGENT_CHECKPOINT_DB" \
  --terminal-retention-s 604800 --expired-question-grace-s 86400
```

Missing or cleaned runs raise `run_not_found` without distinguishing why the
checkpoint is absent. Wrong question IDs raise `question_mismatch`; duplicate
answers to a finished run raise `not_waiting`. Database failures remain storage
failures. Cleanup does not renew question TTL or reset execution budgets.

TACC filesystem support and cross-job acceptance have not been verified by the
local test suite. Record node names, job IDs, versions and event logs when doing
the cluster acceptance run.
