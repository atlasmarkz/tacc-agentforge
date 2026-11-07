# Quick Command Reference: Fully Public Environment

Install and configure the SDK first by following [README.md](README.md). This page lists routine commands; replace `YOUR_*` and `NODE` with your own job IDs and service address.
Start in the package root, using the same Bash terminal throughout:

```bash
PROJECT_DIR=$(pwd -P)
bash setup-sdk-account.sh
```

Submit the service:

```bash
cd "$PROJECT_DIR/qwen-public-serve"
tcloud submit
```

Record and set the service job ID:

```bash
SERVICE_JOB_ID=YOUR_SERVICE_JOB_ID
tcloud ps -j "$SERVICE_JOB_ID"
tcloud cat "slurm_log/qwen-public-serve-${SERVICE_JOB_ID}.log"
```

Wait for `Application startup complete`, then use the current service address from the log to replace NODE before running:

```bash
cd "$PROJECT_DIR"
printf '%s\n' 'http://NODE:8000/v1' > qwen-public-client/server-url.txt
cd "$PROJECT_DIR/qwen-public-client"
tcloud submit
```

Record and set the client job ID. After completion, view and download the results:

```bash
CLIENT_JOB_ID=YOUR_CLIENT_JOB_ID
tcloud ps -j "$CLIENT_JOB_ID"
tcloud cat "slurm_log/qwen-public-client-${CLIENT_JOB_ID}.log"
cd "$PROJECT_DIR"
tcloud download "slurm_log/qwen-public-client-${CLIENT_JOB_ID}.log" "./client-${CLIENT_JOB_ID}.log"
tcloud download "CSIT6000V-project-results/job-${CLIENT_JOB_ID}.json" "./result-${CLIENT_JOB_ID}.json"
```

Cancel the service when finished:

```bash
tcloud cancel -j "$SERVICE_JOB_ID"
tcloud ps -j "$SERVICE_JOB_ID"
```

After restarting the service, record the new job ID and address. While waiting for readiness, repeatedly check the log for the same service job.
