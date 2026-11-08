# How to serve the course model more efficiently — analysis from measurements and engine principles

Date: 2026-11-07. Model: course `Qwen3.8-27B-AWQ-INT4`, vLLM 0.26.0, TP=2 on one TACC node (2 GPUs, about
23.7 GiB each; the GPU model is not recorded in the logs, RTX 3090 is only inferred from the memory size).

Evidence labels: **measured** = a number from a TACC job in this repo; **source** = read from vLLM code or logs;
**reasoned** = derived by arithmetic or from general knowledge, not tested here; **hypothesis** = needs an experiment.

## 1. Where the time goes (measured)

All numbers: compiled + CUDA graphs, 4K context, 512-token prompts, 128 output tokens, 16 requests per cell.

| Quantity | Value | Source |
|---|---|---|
| Decode time per token, one request | about 15 ms (eager: about 86 ms) | `inference/RESULTS.md`, job 28403 |
| Throughput, 1 / 4 / 8 concurrent | 58 / 147 / 204 tok/s | `inference/RESULTS.md` |
| Time to first token, 512-token prompt | about 300 ms, almost all prefill | E6, job 28361 |
| Queueing before prefill, up to 8 concurrent | about 0 ms (170 ms for shared-prefix c8 without caching) | E6 |
| Start-up of one server | 210-530 s, typically 265-280 s with graphs | `results.json` of 28335, 28403 |

Reading: decoding is the bulk of a request (about 1.8 s of 2.1 s at one request); prefill is the bulk of the
waiting before the first token; queueing is not the problem at this load.

## 2. Roofline check: how close is decoding to its limit? (reasoned)

At batch 1, decoding reads every weight once per token, so it is memory-bandwidth bound.

- Weights measured when loaded for the LoRA test: 7.2 + 11.6 = 18.8 GiB (INT4 packed; this number includes
  whatever the embeddings and output layer cost and was split unevenly by that script). With TP=2, about
  9.4 GiB per GPU per token.
- An RTX 3090 has about 936 GB/s (specification; the GPU model is not confirmed). 9.4 GiB / 936 GB/s is about 11 ms.
- Measured 15 ms per token, so decoding is already within roughly 1.4x of the bandwidth bound.

Consequence: there is at most about 30% left to gain from making the existing kernels faster. Bigger wins
must change how many bytes are read per generated token (speculative decoding, batching) or remove
non-weight time (communication, launch overhead). This matches the measurements: CUDA graphs (launch overhead,
+202%) and MTP (more tokens per weight read, +27% to +42%) helped a lot; the attention backend did not.

Uncertainty: the bound depends on the unconfirmed GPU model and on the 18.8 GiB figure; treat 11 ms as an
order-of-magnitude estimate.

## 3. What FlashAttention does and why it matters little for this model

FlashAttention computes attention in tiles held in on-chip memory with an online softmax, so the N x N score
matrix is never written to GPU memory. It saves memory traffic, which matters when attention over a long context
dominates. Related ideas: PagedAttention (KV cache in fixed blocks, no fragmentation), FlashDecoding (split the
key dimension so one long sequence uses all SMs).

For this model (source: vLLM log and earlier model inspection): 16 of 64 layers are full attention, 48 are
linear-attention (gated delta net) layers whose state is a fixed size and does not grow with context.

- vLLM picks `FLASH_ATTN` (FlashAttention-2) here (log). FlashAttention-3 targets Hopper GPUs, so it does not
  apply to this hardware (reasoned).
- Measured: swapping to the Triton attention backend changed nothing outside noise (66.1 vs 67.4 tok/s eager).
- At 512-token context the attention layers are a small part of each step. They matter more for long prompts
  (agent evidence prompts, the 32K configuration), where only 16 layers pay the quadratic cost.
- The 48 linear-attention layers use Triton kernels (`causal_conv1d`, the delta-rule kernels) instead of
  FlashAttention; their speed is a separate lever (and the E5c fix touched exactly that path).

Conclusion: tuning attention kernels is low return at short context. Re-check with long prompts before spending
time on it.

## 4. Techniques used by modern engines, mapped to this deployment

| Technique | Origin | Status here | Evidence |
|---|---|---|---|
| CUDA graphs | vLLM, TensorRT-LLM | on in the course server | measured +202% throughput |
| Continuous batching, chunked prefill | vLLM, Orca | on | chunked prefill off: about the same |
| Prefix caching (radix/hash KV reuse) | SGLang RadixAttention, vLLM | on in the official 32K script, off in our default | measured: shared prefix, one request, first token 654 -> 120 ms; end-to-end -36% to -69% |
| Speculative decoding with the model's MTP head | DeepSeek-style MTP, vLLM | not in the official script | measured +27% (k=1), +42% (k=3) single request, about 0% at 4 concurrent |
| N-gram speculative decoding | vLLM | not worth it | measured -19% to +11% in eager mode |
| Asynchronous scheduling (CPU scheduling overlapped with GPU) | vLLM, DeepInfer lists it | already enabled | source: log line "Asynchronous scheduling is enabled" |
| Structured output (JSON schema / grammar) | vLLM, SGLang, DeepInfer | available, not measured | for valid tool calls |
| Tensor parallel vs replicas | all | TP=2 | not compared (section 6) |
| Quantized KV cache (FP8) | vLLM | failed to start with FlashInfer (no `nvcc` on nodes) | Triton backend variant untested |
| Prefill/decode disaggregation, expert parallelism, multi-node | DeepInfer, Dynamo | not applicable | one node, dense (non-MoE) model |

### DeepInfer specifically

Read from its GitHub README (via a fetch; the page does not state everything): DeepSeek's framework, built mainly
on vLLM with ideas from SGLang and TensorRT-LLM; it lists only DeepSeek-V4.1-Flash as supported, NVIDIA SM100
and Huawei Ascend 950 as hardware, offline inference in the quick start, and names no quantization format. So it
cannot run this Qwen model on this hardware, and I would not try. What is transferable is the ideas, and the
useful ones are already present in vLLM 0.26.0 (asynchronous scheduling is on; MTP speculative decoding works).
Its "confidence-scheduled verification" for speculative decoding is the one idea not available to us, and I have
not checked what vLLM offers equivalent to it.

Other engines (SGLang, TensorRT-LLM, LMDeploy) were not evaluated. A likely obstacle is that the nodes have no
`nvcc` (that is what broke FlashInfer), and SGLang depends on FlashInfer; this is a guess, not tested.

## 5. Serving an API on TACC: constraints that matter more than kernel speed

- A job lasts at most 1 hour and server start-up takes 4-9 minutes (measured), so 7-15% of each job is lost
  to start-up. Reusing the torch.compile cache across jobs (a persistent cache directory in USERDIR) shortens it;
  how much was not measured separately.
- The project's design keeps model calls inside TACC (`tasks/todo.md`, Task 5), so the API is a request bridge,
  not a public endpoint; latency includes that transport.
- vLLM already provides the OpenAI-compatible server, the Qwen reasoning parser (used in all our runs) and a Qwen
  tool parser (`qwen3_engine_tool_parser.py` is in the source tree; not exercised in any measured run).
- A shopping agent repeats the same system prompt and tool schemas on every call, which is the ideal case for
  prefix caching (measured gain above).

## 6. Proposed experiments, in priority order (all untested)

1. **Combined configuration.** CUDA graphs + prefix caching + MTP k=3 on agent-shaped prompts (long shared system
   prompt, 1-3K tokens of evidence, short structured output). The three gains were measured separately; their
   interaction was not.
2. **Longer prompts.** 2K, 8K, 24K input tokens: shows where prefill and the 16 attention layers start to
   dominate, and whether FlashAttention-level work becomes relevant.
3. **Concurrency limits.** `--max-num-seqs` 16 and `--max-num-batched-tokens` 4096 with MTP; MTP's gain
   disappears at 4 concurrent, so find the load where it should be switched off.
4. **TP=2 vs two TP=1 replicas.** Weights (about 19 GiB incl. embeddings) may just fit one 24 GiB GPU with a small KV
   cache. If they do, two independent servers avoid all-reduce time and double throughput under load.
   Hypothesis: the course script sets `NCCL_P2P_DISABLE=1` and disables the custom all-reduce, so TP=2 pays for
   all-reduce through a slow path; `nvidia-smi topo -m` and an all-reduce timing would confirm.
5. **FP8 KV cache via the Triton backend** (`--attention-backend TRITON_ATTN --kv-cache-dtype fp8`) for memory
   headroom at 32K.
6. **Structured output overhead** for tool calls: latency and validity rate with and without a JSON schema.

Each runs with the existing `inference/experiments.py` runner (new groups in `configs/experiments.json`);
experiments 4 and 5 need small new job scripts.
