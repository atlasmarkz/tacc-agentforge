"""agent-e2e job entry for the Task 6 baseline: runs evaluations/run.py from the staged package.
Without --out, results go to $TACC_USERDIR/tacc-agentforge-service/eval-<job id>/."""
import os
import sys

from evaluations.run import main

if __name__ == "__main__":
    if "--out" not in sys.argv:
        sys.argv += ["--out", os.path.join(os.environ["TACC_USERDIR"], "tacc-agentforge-service",
                                           f"eval-{os.environ.get('SLURM_JOB_ID', 'local')}")]
    main()
