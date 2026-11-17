import os
import subprocess
raise SystemExit(subprocess.call(['bash', os.path.join(os.environ['TACC_WORKDIR'], 'job.sh')]))
