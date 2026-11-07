# Private vLLM feasibility plan (2026-11-06)

Status (2026-11-06): E0, E1 (two runs), E2, E3 and E4 (MLP, attention, linear attention, CUDA graphs) done; E5 and E6 not run.
The private build currently contains two logging markers (`AGENTFORGE-E2` in the scheduler, `AGENTFORGE-E4` in the LoRA manager). Results at the end.
Jobs are submitted with `deploy/tacc/tools/submit-win.sh <job-dir>` (see "Blocker" for why).

The official package `deploy/tacc/private-vllm/` lets us edit the Python layer of vLLM 0.26.0 and
rebuild it on TACC (precompiled extensions, so no C++/CUDA changes). Each experiment answers one
question, in dependency order. Stop at the first failure and record it.

| # | Question | How | Pass criterion | Needs |
|---|----------|-----|----------------|-------|
| E0 | Does the private build pass with the course environment? | `submit-build.sh` unchanged | log shows `PRIVATE VLLM BUILD PASSED` | 8 CPU, 45 min |
| E1 | Does the unmodified private server match the public one? | Serve with the same flags as `configs/experiments.json` `graphs` baseline; run `inference/benchmark.py` | TTFT/throughput within run-to-run noise of `inference/results/graphs-27795` | 2 GPU, 1 h |
| E2 | Can we change vLLM behaviour and observe it? | Add a one-line log/metric in the scheduler, rebuild, confirm it appears | the line is in the serve log and `check-private-env.py` still passes | rebuild |
| E3 | Does the 32K compiled configuration (official default) start? | Official launch settings, no `--enforce-eager` | `Application startup complete` within 900 s | 2 GPU |
| E4 | Can vLLM apply a LoRA adapter to this hybrid INT4 model? | Dummy PEFT-format adapter (zero B matrices) from the model config, `--enable-lora` | server starts and a request with the adapter returns text equal to the base model | adapter file |
| E5 | Does n-gram speculative decoding help on evidence-heavy prompts? | `--speculative-config` n-gram, shared-prefix workload | tokens/s vs baseline, acceptance rate recorded | 2 GPU |
| E6 | Per-request timing breakdown (queue, prefill, decode) | Python-level instrumentation from E2 | TTFT split reported separately from blocking-client duration, as `docs/evaluation-metrics.md` requires | E2 |

Notes
- E4 is the most important unknown for the post-training line; if it fails, the fallback is merging the
  adapter into dequantized weights and re-quantizing, which belongs to Task 8.
- One private build per account at a time; stop the private server before rebuilding.
- Change one factor per run and repeat at least twice, as in `inference/RESULTS.md`.
- E3 revisits the earlier finding that the 32K compiled launch did not become ready.

## Blocker

On this Windows machine the native `tcloud` client builds remote paths with backslashes and passes local
paths as `E:\...` to rsync:

- `tcloud ls` / `tcloud env ls` run `ls  \mnt\home\markzhang\USERDIR` and fail.
- `tcloud submit` fails with `The source and destination cannot both be remote` (rsync reads `E:` as a host).

`tcloud ps` works. The same error occurs from Git Bash and PowerShell, so it is not a shell issue.

Workaround (in `deploy/tacc/tools/`): `shim/rsync-wrap.sh` replaces the SDK's rsync upload with `tar | ssh`
(MSYS rsync cannot use Windows OpenSSH or Git's ssh here), and `submit-win.sh` repairs the backslash
paths the SDK writes into `run.sh` / `run.slurm` (including a local `E:\...` path appended to
`TACC_WORKDIR`) and calls `sbatch` itself. `tcloud ps/cat/cancel` still work as before.
Earlier jobs have forward-slash paths, so they were submitted some other way; the cause was not found.

## Results

### E0 build — passed (job 28310)
`PRIVATE VLLM BUILD PASSED`; `vllm` imports from `USERDIR/CSIT6000V-private-vllm/vllm-src`, `torch` and
`transformers` from the public environment. Build took roughly 10 minutes of the 45 requested.

### E1 private vs public baseline — within about 2% (job 28317, one run each)
Same flags (`graphs` baseline: compiled, CUDA graphs, 4K context, 2 GPUs). Output tokens/s, public
(job 27795) vs private (job 28317), 0 failed requests in both:

| workload | public | private | diff |
|---|---|---|---|
| distinct, c1 | 58.4 | 57.4 | -1.7% |
| shared prefix, c1 | 38.3 | 37.8 | -1.3% |
| distinct, c4 | 148.3 | 145.9 | -1.6% |
| shared prefix, c4 | 64.3 | 63.6 | -1.1% |
| distinct, c8 | 203.5 | 199.1 | -2.2% |
| shared prefix, c8 | 74.4 | 72.8 | -2.2% |

- Private is slightly lower in every cell. With one run per side the run-to-run noise is unknown, so this
  shows "no large difference", not "identical".
- Startup 896 s vs 300 s: `torch.compile` took 225 s on the private build's empty cache (public run: 48 s,
  warm). A second private start should be faster; not yet measured.
- KV cache 8.47 GiB / 161,109 tokens (private) vs 10.19 GiB / 193,877 tokens (public). Cause not
  investigated (the public run may have used a different memory setting or had a different free-memory
  reading at profiling time).
- Files: `inference/results/private-graphs-28317/`.

#### E1 repeat (job 28335, same config, warm compile cache)
Output tokens/s, public (27795) / private run 1 (28317) / private run 2 (28335); 0 failed requests:

| workload | public | private 1 | private 2 | run 2 vs run 1 | run 2 vs public |
|---|---|---|---|---|---|
| distinct, c1 | 58.4 | 57.4 | 57.7 | +0.5% | -1.3% |
| shared prefix, c1 | 38.3 | 37.8 | 37.9 | +0.5% | -0.8% |
| distinct, c4 | 148.3 | 145.9 | 146.6 | +0.5% | -1.1% |
| shared prefix, c4 | 64.3 | 63.6 | 63.7 | +0.2% | -0.9% |
| distinct, c8 | 203.5 | 199.1 | 201.7 | +1.3% | -0.9% |
| shared prefix, c8 | 74.4 | 72.8 | 72.7 | -0.2% | -2.3% |

- Run-to-run spread between the two private runs is at most 1.3%. Private is below public in all six cells in
  both runs (-0.8% to -2.3%), so a small private deficit of about 1% is plausible, but with two private runs and
  one public run it is not established. It is far smaller than any effect we would try to measure.
- Startup 275 s (was 896 s): the warm `torch.compile` cache removes the cold-start cost.
- KV cache 9.49 GiB / 180,224 tokens (run 1: 8.47 GiB / 161,109). Same weights (9.27 GiB), so the KV size
  varies between runs of the same config; the public/private KV difference is therefore not evidence of a
  private-build difference (public: 10.19 GiB / 193,877). Cause of the variation not investigated.
- Files: `inference/results/private-graphs-28335/`.

### E2 modify vLLM and observe it — passed (build job 28340, serve job 28342)
Edit: one `logger.info` in `Scheduler.add_request` (`vllm-src/vllm/v1/core/sched/scheduler.py`) logging request id,
prompt token count, running and waiting counts, tagged `AGENTFORGE-E2`. Source repacked with the same `tar`
command as `submit-build.sh` and rebuilt with `submit-win.sh`: `PRIVATE VLLM BUILD PASSED`, `check-private-env.py`
passed. The rebuild took about 10 minutes (dependency resolve 5m50s, build 3m37s).
A `graphs` baseline serve job then logged lines such as
`[scheduler.py:2139] AGENTFORGE-E2 add_request id=chatcmpl-... prompt_tokens=376 running=0 waiting=1`
(15 lines seen when checked), so a Python-level change reaches the running engine.
- Job 28342 was cancelled after the marker appeared; its benchmark numbers are not used (instrumented build,
  not complete).
- The marker is still in the private build. Remove it (or keep it for E6) before any timing comparison.
- Files: none saved; evidence is the serve log `USERDIR/slurm_log/agentforge-exp-28342.log` on the cluster.

### E4 vLLM applies a PEFT-format LoRA to the INT4 model — passed (job 28345)
Three dummy adapters in PEFT layout (`adapter_config.json` + `adapter_model.safetensors`, names
`base_model.model.model.language_model.layers.N.mlp.{gate,up,down}_proj.lora_{A,B}.weight`), rank 8, alpha 16,
all 64 layers, MLP projections only, 138 MB each. `zero`: B = 0; `small`: B ~ N(0, 0.01); `large`: B ~ N(0, 0.1).
One private-build server with `--enable-lora --max-lora-rank 8 --max-loras 3 --lora-modules ...`, `--enforce-eager`,
4K context, 2 GPUs; ready in 245 s. Greedy (temperature 0), 40 tokens, 3 prompts, same server for all four models.

| model | output equals base model (3 prompts) |
|---|---|
| zero | yes, all 3 |
| small | no: 2 of 3 differ, 1 identical ("The capital of France is ...") |
| large | no: all 3 are garbage text |

- The server loaded all three adapters and listed them in `/v1/models`; no request errors, no LoRA warnings in the log.
- `zero` matching the base exactly shows the adapter path adds nothing when B = 0; `small` and `large` changing the
  output shows the adapter weights are really applied on the quantized layers. This is a loading/application check,
  not a quality result.
- Consequence: the training side must save adapters in PEFT layout (the 2026-10-28 test saved a custom format),
  with these module names, to be servable. The merge-and-requantize fallback is not needed for MLP targets.
- Files: `inference/results/private-lora-e4-28345/results.json`; job in `deploy/tacc/private-vllm/qwen-private-lora/`.

#### E4b attention and linear-attention targets, CUDA graphs on — passed (job 28355)
Script `lora_e4b.py` reads each Linear's shape from the checkpoint (q_proj 5120->12288 incl. gate, k/v_proj 5120->1024,
o_proj 6144->5120; in_proj_qkv 5120->10240, in_proj_z 5120->6144, out_proj 6144->5120). Five adapters, rank 8:
`all_zero` (B=0 on MLP + attention + linear attention), `full_b`/`full_big` (q/k/v/o_proj on the 16 full-attention
layers, B std 0.05 / 0.5), `lin_b`/`lin_big` (in_proj_qkv, in_proj_z, out_proj on the 48 linear-attention layers, same
stds). Compared token logprobs on 3 prompts, 40 greedy tokens, against the base model on the same server.

| adapter | texts differing from base (of 3) | max abs logprob diff, eager | max abs logprob diff, compiled + CUDA graphs |
|---|---|---|---|
| all_zero | 0 | 0.0 | 0.0 |
| full_b | 3 | 1.07 | 1.52 |
| lin_b | 3 | 0.82 | 1.52 |
| full_big | 3 | 5.77 | 5.92 |
| lin_big | 3 | 5.73 | 6.10 |

- The activation log (my `AGENTFORGE-E4` line in `LoRAModelManager.activate_adapter`) shows what was applied:
  all_zero loaded 48 `in_proj_qkvz`, 48 `out_proj`, 64 `gate_up_proj`, 64 `down_proj`, 16 `qkv_proj`, 16 `o_proj`;
  `full_*` loaded 16 `qkv_proj` + 16 `o_proj`; `lin_*` loaded 48 `in_proj_qkvz` + 48 `out_proj`.
  `conv1d` and `in_proj_ba` have no adapter weights (not targeted). q/k/v and in_proj_qkv/in_proj_z are packed into
  vLLM's fused modules without extra work.
- Startup with LoRA: 230 s eager, 450 s compiled with CUDA graphs (compile cache partly warm). All adapters work in both modes.
- Single-stream decode, compiled + CUDA graphs, 128 tokens, one measurement pair each: base model 64.1-64.2 tok/s,
  `all_zero` adapter 49.0-49.1 tok/s (-23%), on the same LoRA-enabled server. The same 49 tok/s was measured
  in an earlier run in which no adapter weights were loaded (see the correction below), so the cost looks like the
  LoRA code path itself, not the weights. Not isolated further; concurrent-load throughput not measured.

Correction (what went wrong first): jobs 28347, 28349 and 28352 used a version of `lora_e4b.py` with a key prefix
`base_model.model.language_model...` (one `model.` short; E4 had it right). vLLM then loaded no weights at all
(`loaded={}` in the diagnostic log), so those jobs only showed that nothing changed, and an interim reading
"vLLM ignores attention LoRA" was wrong. Fixed in the script (`base_model.model.model.language_model...`).
Job 28354 died at start-up on node 10-0-1-11 (`NVMLError_InvalidArgument` while reading GPU properties);
28355 is the same code on 10-0-1-10.

- Not covered: an adapter produced by real training, switching between several adapters per request, and
  embedding/lm_head targets.
- Files: `inference/results/private-lora-e4e-28355/results.json`.

#### E4c where the ~24% LoRA slowdown comes from (job 28357) — located, not explained
Three servers with the `graphs` baseline flags (compiled + CUDA graphs); zero-B adapters (outputs equal base) covering
different module sets; decode of 256 tokens, 3 interleaved repeats (median); then 4 concurrent requests (aggregate tok/s).

| server | target | single stream tok/s | 4 concurrent tok/s |
|---|---|---|---|
| no LoRA | base | 65.2 | 195.4 |
| `--max-loras 5` | base | 65.1 | 194.8 |
| | all_zero (MLP + attn + linear attn) | 49.5 | 158.2 |
| | mlp_zero (64 layers, 2 modules each) | 49.5 | 151.6 |
| | full_zero (16 layers, q/k/v/o) | 49.5 | 158.3 |
| | lin_zero (48 layers) | 49.4 | 158.4 |
| `--max-loras 1` | base | 64.9 | 196.1 |
| | all_zero | 48.6 | 152.3 |
| | mlp_zero | 49.1 | 155.2 |
| | full_zero | 49.7 | 153.9 |
| | lin_zero | 49.4 | 153.9 |

- Turning LoRA on does not slow requests that do not use an adapter (65.1 and 64.9 vs 65.2 tok/s).
- A request that uses any adapter is about 24% slower per token single-stream (15.3 -> 20.2 ms/token) and about
  20-22% slower at 4 concurrent (194-196 -> 152-158 tok/s).
- The cost is the same whichever modules the adapter covers (16 layers of attention only costs as much as all
  400 modules) and independent of `--max-loras` (1 vs 5). So it is not proportional to the number of LoRA kernels or to
  slot count; it looks like a fixed per-step cost of running a batch that contains a LoRA request.
- Ruled out: CUDA graphs are not disabled for LoRA. The dispatcher captures separate graphs with and without LoRA
  (`cudagraph_dispatcher.py`, `_get_lora_cases`), and graph capture appears in the log.
- Not found: the actual source of the fixed ~4.7 ms/token. Candidates not yet tested: per-step LoRA mapping/metadata
  update on the CPU, the LoRA path touching `lm_head`/logits, or the "default LoRA kernel configs" Triton path
  (the log warns that tuned configs are not used). A profile of one decode step with and without an adapter would settle it.
- Practical consequence: serving a fine-tuned model through a LoRA adapter costs roughly a quarter of decode speed.
  Merging the adapter into the weights (not possible directly on INT4, would need dequantize + merge + re-quantize,
  Task 8) would remove it. Decide after measuring how much the final answer quality depends on the adapter.
- Files: `inference/results/private-lora-ovh-28357/results.json`.

### E6 per-request timing breakdown — works with vLLM's own metrics, no source change (job 28361)
vLLM already exports per-phase request times as histograms on `/metrics` (`request_queue_time_seconds`,
`request_prefill_time_seconds`, `request_decode_time_seconds`, `time_to_first_token_seconds`,
`e2e_request_latency_seconds`). `e6_timing.py` scrapes `_sum/_count` before and after each workload cell and reports the
mean per phase next to the streaming client's p50 (so a TTFT and the blocking duration are reported separately, as
`docs/evaluation-metrics.md` asks). `graphs` baseline flags, 8 prompts x 2 repeats per cell, 128 output tokens.
Server numbers are means (ms), client numbers are p50 (ms); they are not the same statistic.

Without prefix caching:

| cell | client TTFT | client e2e | server queue | server prefill | server decode | server e2e |
|---|---|---|---|---|---|---|
| distinct c1 | 295 | 2185 | 0.0 | 313.5 | 1787.0 | 2114.6 |
| shared c1 | 654 | 1491 | 0.0 | 633.7 | 801.3 | 1448.4 |
| distinct c4 | 1088 | 3431 | 0.0 | 657.5 | 2404.6 | 3260.6 |
| shared c4 | 1295 | 3554 | 0.0 | 1172.8 | 1940.0 | 3386.8 |
| distinct c8 | 1607 | 4978 | 0.9 | 1022.3 | 3360.8 | 4707.1 |
| shared c8 | 2685 | 6202 | 170.1 | 2117.8 | 3140.2 | 5921.2 |

With `--enable-prefix-caching`:

| cell | client TTFT | client e2e | server queue | server prefill | server decode | server e2e |
|---|---|---|---|---|---|---|
| distinct c1 | 294 | 2186 | 0.0 | 288.1 | 1781.1 | 2083.4 |
| shared c1 | 120 | 958 | 0.0 | 133.0 | 803.8 | 950.0 |
| distinct c4 | 842 | 3446 | 0.0 | 623.8 | 2508.4 | 3240.8 |
| shared c4 | 225 | 1407 | 0.0 | 173.8 | 1111.1 | 1340.3 |
| distinct c8 | 1594 | 4913 | 0.9 | 1011.6 | 3319.6 | 4655.0 |
| shared c8 | 565 | 1924 | 0.1 | 353.5 | 1347.3 | 1820.2 |

- Queueing is negligible up to concurrency 8 (max-num-seqs 8); the one exception is shared-prefix c8 without caching
  (170 ms mean). So time before the first token is almost all prefill, and prefill grows with concurrency because the
  prefills compete for the same 2048-token batch budget.
- Prefix caching removes most of the prefill for shared prefixes: c1 634 -> 133 ms, c4 1173 -> 174 ms, c8 2118 -> 354 ms
  server mean, and shortens client e2e by 36%-69%. Distinct prompts are unchanged, as expected.
- Decode at c1 is about 14 ms per token (1.79 s / ~128); it does not depend on caching.
- Client TTFT runs 10-20 ms above server TTFT at c1 and more at higher concurrency (HTTP, tokenization, streaming
  and thread scheduling in the client); report both.
- Caveat: the vLLM numbers are means of all requests including the one warm-up request the client sends per cell,
  and 16 requests per cell is small; treat differences under ~10% as noise.
- Files: `inference/results/private-e6-28361/results.json`; job in `deploy/tacc/private-vllm/qwen-private-e6/`.

### E5 n-gram speculative decoding — fails to start with compile + CUDA graphs (job 28360)
Group `spec-ngram`: `graphs` baseline vs `--speculative-config {"method":"ngram","num_speculative_tokens":3 or 5,
"prompt_lookup_max":4,"prompt_lookup_min":2}`. The baseline ran (startup 465 s; distinct c1 57.7 tok/s, shared c1 37.9,
distinct c4 147.0, shared c4 63.6, in line with E1). Both ngram variants failed before serving: during
`determine_available_memory` the worker raised `CUDA error: operation failed due to a previous error during capture`
(`cudaErrorStreamCaptureInvalidated`). So speculative decoding together with CUDA graph capture does not work on this
model and build.

#### E5 retry without CUDA graphs (group `spec-ngram-eager`, job 28364)
Eager baseline vs eager + ngram (`prompt_lookup_max` 4, `prompt_lookup_min` 2), 16 prompts x 2 repeats, 128 output
tokens, 0 failed requests everywhere. Speculative decoding starts and serves in eager mode (startup 380-400 s).
Output tokens/s (change vs the eager baseline), single run each:

| workload | eager baseline | ngram-k3 | ngram-k5 |
|---|---|---|---|
| distinct, c1 | 11.5 | 12.0 (+4.4%) | 12.0 (+4.9%) |
| shared prefix, c1 | 10.4 | 11.5 (+11.2%) | 11.4 (+10.2%) |
| distinct, c4 | 37.4 | 30.6 (-18.1%) | 30.4 (-18.7%) |
| shared prefix, c4 | 30.2 | 29.6 (-2.0%) | 29.1 (-3.7%) |

- Acceptance (server log, sampled 10 s windows, so only indicative): mean accepted length ranged 1.5-2.4 for k=3
  and 3.1-4.9 for k=5 (per-position acceptance 0.09-0.81), i.e. strongly dependent on the text; random-token prompts
  accept little, repeated evidence text accepts a lot.
- Result: a small gain for single requests (+4% to +11%) and a loss with 4 concurrent requests (-2% to -19%). The
  extra draft/verify work is not free at batch 4. Raising k from 3 to 5 changed nothing material.
- The decisive fact is that this only works in eager mode, where the baseline is already ~5x slower than the
  compiled + CUDA graphs baseline (11.5 vs 57.7 tok/s distinct c1). Eager + ngram (12.0) is far below graphs without
  speculation (57.7). Conclusion: n-gram speculative decoding is not worth pursuing on this model/build.
- Caveats: one run per cell, eager-mode noise not measured, prompts are synthetic (not the agent's real
  evidence-heavy prompts), only the n-gram method was tried (no draft model, no MTP head: the checkpoint has `mtp.*`
  weights, which would be the next thing to try if speculation is revisited).
- Files: `inference/results/private-spec-ngram-eager-28364/results.json`.

### E5b MTP-head speculative decoding — works in eager mode, fails with CUDA graphs (jobs 28389, 28390)

The checkpoint carries `mtp.*` weights; vLLM 0.26.0 supports them for this architecture as
`--speculative-config '{"method":"mtp","num_speculative_tokens":k}'` (`Qwen3_5MTP`). Same workload as E5
(512 in / 128 out, 16 prompts, 2 repeats), one measurement per cell.

**With compile + CUDA graphs (28389): both variants fail to start.** The baseline ran (57.6 tok/s, distinct c1).
Both `mtp-k1` and `mtp-k3` crash during CUDA graph capture with `RuntimeError: Triton Error [CUDA]: operation not
permitted when stream is capturing`, raised from `causal_conv1d_update` (linear-attention layer,
`vllm/model_executor/layers/mamba/ops/causal_conv1d.py`) — a Triton kernel is JIT-compiled for the first time
inside the capture. The variant labelled `mtp-k5` in that job's `results.json` actually used k=3
(a label typo in the first submission; fixed in `configs/experiments.json`).

**Eager (28390), output tokens/s (change vs eager baseline):**

| workload | eager baseline | mtp-k1 | mtp-k3 |
|---|---|---|---|
| distinct, c1 | 11.3 | 17.2 (+52%) | 24.4 (+116%) |
| shared prefix, c1 | 10.2 | 15.4 (+51%) | 20.7 (+103%) |
| distinct, c4 | 37.8 | 38.6 (+2%) | 63.0 (+67%) |
| shared prefix, c4 | 29.8 | 36.3 (+22%) | 48.0 (+61%) |

- Mean acceptance length from the server's periodic log lines (unweighted mean of 10 s windows):
  k1 about 1.84, k3 about 2.86. Far better than n-gram (1.5-2.4 at k=3).
- TTFT gets worse at c4 (distinct: 780 ms baseline, 1735-2954 ms with MTP).

**Conclusion.** MTP is a real speedup, unlike n-gram, but only in eager mode, where even the best cell
(24.4 tok/s) is far below the graphs baseline without speculation (57.6 tok/s). So MTP is not useful with this
build until it can run under CUDA graphs. A possible fix is pre-compiling the Triton kernel for the
speculative-decode shape before capture; not tried. Caveats: one measurement per cell, synthetic prompts,
acceptance lengths are log samples.

Files: `inference/results/private-spec-mtp-28389/`, `inference/results/private-spec-mtp-eager-28390/`.

### Node 10-0-1-11 cannot start vLLM (jobs 28354, 28358, 28359)
Every vLLM server started on 10-0-1-11 died during import with `NVMLError_InvalidArgument` in
`pynvml.nvmlDeviceGetHandleByIndex` (E5 baseline and variants, E6, and the first corrected E4 attempt). The same
code ran on 10-0-1-10. The three jobs produced no results. Workaround: `exclude=10-0-1-11` under `general:` in
`tuxiv.conf` (written to `#SBATCH --exclude`); E5 and E6 were resubmitted as 28360 and 28361. `stage.sh` regenerates
`tuxiv.conf` from the template and drops this line, so add it again after staging. Not reported to the cluster staff yet.

### E3 official 32K configuration — starts and serves (job 28322)
Course `serve.sh` settings (32768 context, `--max-num-seqs 4`, `--max-num-batched-tokens 4096`,
GPU utilization 0.90, prefix caching on, compiled, CUDA graphs), private build, 2 GPUs.
Ready after 340 s (`torch.compile` 103 s; 225 s in E1's cold start, so the cache is only partly warm),
peak GPU memory 21.8 GiB per GPU, 0 failed requests. The earlier failure to start the 32K compiled
configuration (public environment, 2026-10-16) did not reproduce here.

| workload | output tok/s | TTFT p50 (ms) | e2e p50 (ms) |
|---|---|---|---|
| distinct, c1 | 57.2 | 300 | 2228 |
| shared prefix, c1 | 57.4 | 125 | 1002 |
| distinct, c4 | 145.1 | 851 | 3497 |
| shared prefix, c4 | 155.3 | 158 | 1428 |
| distinct, c8 | 146.4 | 4289 | 6864 |
| shared prefix, c8 | 156.8 | 1530 | 2810 |

- KV cache 9.78 GiB = 278,528 tokens (8.5x concurrency at 32K).
- Prefix caching is on here, so shared-prefix throughput is not comparable with E1 (caching off).
  Do not read the shared-prefix c4 gain (63.6 -> 155.3 tok/s) as an effect of 32K; the 32K and caching
  changes are confounded in this run.
- c8 is capped by `--max-num-seqs 4`, so the extra requests queue (TTFT p50 4.3 s for distinct prompts).
- Not tested: prompts longer than the 512-token workload, so this shows the 32K server starts and
  serves, not that long contexts perform well.
- Files: `inference/results/private-official-32k-28322/`.
