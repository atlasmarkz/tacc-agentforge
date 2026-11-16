"""SDK entrypoint; the platform's bootstrap Python launches the job script."""
import os
import subprocess

raise SystemExit(subprocess.call(["bash", os.path.join(os.environ["TACC_WORKDIR"], "job.sh")]))
