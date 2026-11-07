# CSIT6000V Project: Public Dependencies + Private vLLM

User Guide | 2026-11-05

This package provides instructions for environment setup, service startup, and client calls. Refer to the official project brief for the project topic, grading criteria, deadline, and final deliverables. After running the tests, complete your submission as required by that brief.

Extract this package into a new local directory to preserve any existing client code and private source modifications.

The private version is intended for modifying the Python source code of vLLM. For writing model client programs, the public version is recommended.
This package uses `mpi-gemm-v1` as its SDK-managed environment entry point; the job scripts then activate the actual Project environment. Use the `setup-sdk-account.sh`, `job.sh`, and `tuxiv.conf` files supplied with this package together.

## 1. Install the SDK and Configure Your Account (First Use)

If you installed the official tcloud SDK for Assignment 1, you can reuse it. First run `tcloud --help` to confirm that the command is available.
Official tcloud SDK v0.4.1 downloads:

- [Linux x86_64](https://github.com/turingaicloud/quickstart/releases/download/v0.4.1/linux-amd64-v0.4.1.zip)
- [macOS Intel](https://github.com/turingaicloud/quickstart/releases/download/v0.4.1/macos-amd64-v0.4.1.zip)
- [Official Quickstart](https://github.com/turingaicloud/quickstart)

Download and extract the SDK that matches your system, then enter the directory containing the `tcloud` executable:

```bash
chmod +x ./tcloud
export PATH="$PWD:$PATH"
tcloud --help
```

The PATH setting above applies only to the current terminal. Set it again in a new terminal, or add the SDK's actual absolute directory to your shell configuration file.
For Windows, Apple Silicon, or unavailable download links, ask the teaching assistant for a supported course client. These two downloads support only the system architectures listed above.
The commands and scripts in this guide use Bash. Run `bash` in your terminal before starting.

Configure your account using your own TACC username and the private key corresponding to your registered public key. Replace the placeholders below:

```bash
tcloud config -u YOUR_USERNAME -f /path/to/your/private_key
tcloud init
tcloud env ls
```

The scripts use `~/.Miniconda3` in your platform account. For account initialization issues such as `Failed to get env list`, send the error message to the teaching assistant.
Your computer needs the SDK, Bash, and standard archive extraction/creation tools; the platform provides the course runtime environment.
Keep your private key in your personal SSH directory on the client computer, and specify that file's path when configuring your account.

## 2. Initialize This Package's Environment Entry Point

Enter the root directory of the extracted package (containing `setup-sdk-account.sh`) and run the following in your current Bash terminal:

```bash
PROJECT_DIR=$(pwd -P)
bash setup-sdk-account.sh
```

This package uses `/mnt/home/public/course-envs/CSIT6000V/assignment1/mpi-gemm-v1` as its SDK-managed environment entry point. Runtime dependencies are provided separately by the public Project activation script.
Run this package's setup script to confirm that your account can access the `mpi-gemm-v1` environment. If a configuration conflict occurs, send the message to the teaching assistant.
The platform provides the course runtime dependencies. Job resources and entry points are configured in `tuxiv.conf`; edit your programs in the client directory.

Run the remaining steps in the same Bash terminal. `PROJECT_DIR` must always point to this package's root directory; set it again after opening a new terminal.

## 3. Prepare the Source and Build on the Platform (First Use or After Source Changes)

```bash
cd "$PROJECT_DIR"
bash prepare-source.sh
```

The script extracts the bundled `vllm-v0.26.0.tar.gz` into `vllm-src`. It stops if the directory already exists to avoid overwriting your changes. Edit `vllm-src` on your own computer, then run:

```bash
bash submit-build.sh
```

The script packages your local source and submits it to the platform through the SDK for building. Record the returned build job ID and replace `YOUR_BUILD_JOB_ID` below:

```bash
BUILD_JOB_ID=YOUR_BUILD_JOB_ID
tcloud ps -j "$BUILD_JOB_ID"
tcloud cat "slurm_log/qwen-private-build-${BUILD_JOB_ID}.log"
```

Wait until the log shows `PRIVATE VLLM BUILD PASSED`, then proceed to start the service. While waiting, repeatedly check the log for the same build job.
The build requests 8 CPUs and 45 minutes. It uses the precompiled extensions matching vLLM v0.26.0 and supports Python-layer source modifications.
The build installation tools need access to the configured PyPI mirror. If networking or installation fails, fix the issue based on the log and resubmit the build. Start the service only after the build succeeds.

Before rebuilding, stop your private service and save your local source changes. Rebuilding replaces your personal `TACC_USERDIR/CSIT6000V-private-vllm/vllm-src` and `.venv`. If a build fails, fix the issue and rebuild successfully before starting the service.
The build and service scripts check the current build status. You can restart the service after a successful rebuild.
If an interrupted build keeps reporting that a build already exists, use `tcloud ps` to check the old build's status. Once the old job has ended, contact the teaching assistant to verify and remove the leftover `.build-in-progress` directory in your personal project directory, then submit the build again.

## 4. Start the Service

```bash
cd "$PROJECT_DIR/qwen-private-serve"
tcloud submit
```

Record the service job ID returned by the submission and replace `YOUR_SERVICE_JOB_ID`:

```bash
SERVICE_JOB_ID=YOUR_SERVICE_JOB_ID
tcloud ps -j "$SERVICE_JOB_ID"
tcloud cat "slurm_log/qwen-private-serve-${SERVICE_JOB_ID}.log"
```

Wait until the service log shows `Application startup complete`, then copy the complete address after `SERVICE_BASE_URL=` in the same log.
The first startup includes model loading, compilation, and warm-up. While waiting, repeatedly run the same `tcloud cat` command to check progress. Submit the client after the service is ready.

## 5. Set the Current Service Address and Submit the Client

Return to the package root and copy the complete address after `SERVICE_BASE_URL=` from the current service log. Use it to replace `http://NODE:8000/v1` below, keeping `/v1`:

```bash
cd "$PROJECT_DIR"
printf '%s\n' 'http://NODE:8000/v1' > qwen-private-client/server-url.txt
cat qwen-private-client/server-url.txt
cd "$PROJECT_DIR/qwen-private-client"
tcloud submit
```

Before submitting, replace `NODE` with the node address from the current service log and confirm that the service is ready. The script trims leading/trailing whitespace and Windows line endings, then validates the address format.
The client is submitted as a separate CPU job. Keep the service running while the client runs. Use `tcloud ps` to check queued jobs; `Resources` means the job is waiting for scheduling resources.
Record the returned client job ID and replace `YOUR_CLIENT_JOB_ID`:

```bash
CLIENT_JOB_ID=YOUR_CLIENT_JOB_ID
tcloud ps -j "$CLIENT_JOB_ID"
tcloud cat "slurm_log/qwen-private-client-${CLIENT_JOB_ID}.log"
```

The example should return `95` and the Chinese name for Beijing, or equivalent correct answers. It also prints token usage and the elapsed time for each complete non-streaming HTTP request. On completion, the example prints `RESULT_FILE=CSIT6000V-project-results/job-<actual-job-id>.json`, indicating that the results of both calls have been saved.

## 6. Save and Download Results

The example writes results to your personal `TACC_USERDIR/CSIT6000V-project-results/job-<client-job-id>.json`. `tcloud download` uses remote paths relative to your personal USERDIR, as follows:

```bash
cd "$PROJECT_DIR"
tcloud download "slurm_log/qwen-private-client-${CLIENT_JOB_ID}.log" "./client-${CLIENT_JOB_ID}.log"
tcloud download "CSIT6000V-project-results/job-${CLIENT_JOB_ID}.json" "./result-${CLIENT_JOB_ID}.json"
```

Write your own result files under `os.environ["TACC_USERDIR"]` as well, so they persist across jobs and can be downloaded. WORKDIR is used for code synchronization and execution for the current job. If a call fails, address the error using the client log and resubmit the client. After successful completion, download the JSON results for the corresponding job ID.

## 7. Stop the Service and Run Again

When finished, stop the service using the `SERVICE_JOB_ID` recorded earlier:

```bash
tcloud cancel -j "$SERVICE_JOB_ID"
tcloud ps -j "$SERVICE_JOB_ID"
```

The old address becomes invalid when the service ends or reaches its 1-hour time limit. For the next run, submit again with `tcloud submit` in the `qwen-private-serve` directory, wait until it is ready, and update `server-url.txt` using the new log.
If you are reusing the source from the last successful build, simply resubmit the service. After changing the source, stop the old service, submit a build, and start the service only after the build passes.
The private version shares one source tree and `.venv` per account. Build and switch versions sequentially, and stop your private service before every rebuild. Persistent results are saved separately by client job ID.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| `tcloud: command not found` | Check the SDK executable permissions and PATH in the current terminal. |
| `Failed to get env list` / account not initialized | Run config/init for your own account. If it still fails, send the error message to the teaching assistant. |
| `Unexpected SDK environment` / existing configuration needs review | Run setup using the matching files from this package. Send any environment conflict message to the teaching assistant. |
| `Set server-url.txt` / invalid address format | Copy the current service address, remove `SERVICE_BASE_URL=`, replace NODE with the actual node, and keep `/v1`. |
| Connection refused or timeout | Confirm that the service is still running, its log has reached startup complete, and the address is from the current service. Client calls must run inside a platform job. |
| HTTP 400/404 or similar errors | Check the service log, `/v1` path, request length, and model name `qwen3.8-27b`. |
| Log file not found | Use the actual numeric job ID. Check that you are using the correct log filename for the service or client, and wait for the job to start generating logs. |
| JSON result missing | Check that the client completed successfully. Relative paths are based on USERDIR, and the ID must be the client job ID. |
| `PRIVATE VLLM BUILD FAILED` / `Private vLLM is not ready` | Fix the issue using the build log and resubmit. Start the service after the build passes. See the source-build section for recovery after an interrupted build. |

## Editable Files, Resources, and Environment

- Put your client logic in `qwen-private-client/main.py` and supporting source files in the same client job directory. Use `from qwen_client import chat` to call the service.
- `qwen_client.py` uses the Python standard library and a 300-second timeout for each HTTP request. If a request fails, troubleshoot as instructed, then rerun the client.
- `job.general` in `tuxiv.conf` defines resources. The service uses a single node with 2 GPUs, 8 CPUs, and a 1-hour time limit; the client uses 1 CPU and 10 minutes. Before making batch calls, set time limits and resources according to your experiment size and course quota.
- Model name: `qwen3.8-27b`; port: 8000; context: 32K; concurrency: 4; text only. Public dependencies and the model are maintained centrally for the course.
- The actual Project activation entry point is `/mnt/home/public/course-envs/CSIT6000V/project-private-vllm/activate.sh`. The vLLM source and .venv are in your personal persistent directory; PyTorch / Transformers use public dependencies.
- The service reuses compilation caches in your personal persistent directory. Every startup still includes model loading, process initialization, and warm-up.
- Use the SDK for job submission, status checks, log viewing, result downloads, and job cancellation.

## File List

| Path | Purpose |
| --- | --- |
| `README.md` | Complete user guide and troubleshooting instructions. |
| `STUDENT-COMMANDS.md` | Short command index; refer to README for detailed instructions. |
| `setup-sdk-account.sh` | Initialize the SDK-managed environment entry point required by this package. |
| `qwen-private-serve/tuxiv.conf`, `job.sh` | Service resources and startup entry point. |
| `qwen-private-client/tuxiv.conf`, `job.sh` | CPU client resources and execution entry point. |
| `qwen-private-client/main.py`, `qwen_client.py` | Editable client example and HTTP functions. |
| `qwen-private-client/server-url.txt` | Current service address; contains the NODE placeholder by default. |
| `vllm-v0.26.0.tar.gz`, `prepare-source.sh`, `submit-build.sh` | Pinned source archive, local extraction, and build submission. |
| `qwen-private-build/` | Platform build configuration, startup script, Python launch bridge, and provenance checks. |
| `qwen-private-serve/main.py`, `check-private-env.py` | SDK launch bridge and checks for the origins of private/public dependencies. |

Reference: [Official SDK Job Configuration](https://github.com/turingaicloud/tcloud-sdk#job-configuration). Contact the teaching assistant for SDK installation issues.
