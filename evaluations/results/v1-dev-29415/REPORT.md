# Benchmark report: project-v1-synthetic-fixtures

- split: `dev`；configs: `B0, B1, B2, B3, A1, A2`；repeats: 1
- attempts recorded: 60
- dataset status: `awaiting_post_freeze_category`
- human review merged: no（Faithfulness 为 N/A，需要 `python -m evaluations.review`）

## 主结果（proposal 表）

| Method | Configuration | TSR % ↑ | Faithfulness ↑ | BVR % ↓ | p95 s ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| B0 Single call | No extra text; no budget tool | 30.0 | N/A | 0.0 | 5.34 |
| B1 Keyword RAG | FTS5 evidence; single call; no budget tool | 30.0 | N/A | 0.0 | 6.12 |
| B2 Structured RAG | Filtered FTS5; single call; no budget tool | 30.0 | N/A | 0.0 | 5.94 |
| **B3 Full advisor** | B2 + bounded loop + budget tool | 90.0 | N/A | 0.0 | 10.07 |
| A1 Without loop | B3; one-pass plan with the same tools | 90.0 | N/A | 0.0 | 6.87 |
| A2 Without validator | B3; budget tool disabled | 10.0 | N/A | 0.0 | 11.11 |

口径：

- **TSR** = 成功完成次数 / 全部计划尝试次数；超时、崩溃、缺失记录都计入分母。
- **Faithfulness** = 两位评审对可评判事实声明的支持率；没有标注时为 N/A，不能用 0 或 1 代替。
- **BVR** = 已交付且费用可核实的计划中超出真实可用预算的比例；没有已交付计划时为 N/A。
- **p95** 是 adapter 子进程完整往返耗时，不含浏览器、桥接和用户输入等待；不是 proposal 的完整请求 p95。

## 自动指标与计数

| Config | runs | plans | TSR n/m (95% CI) | BVR n/m | amount errors | unverifiable plans | constraint violations | ungrounded IDs | invalid actions | model calls | tokens in/out |
| --- | ---: | ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: | --- |
| B0 | 10 | 7 | 3/10 ([10.0, 60.0]) | 0/6 | 83.3 (5/6) | 1 | 0.0 (0/7) | 0.0 (0/7) | 1 | 1.1 | 812.9/164.9 |
| B1 | 10 | 7 | 3/10 ([10.0, 60.0]) | 0/6 | 83.3 (5/6) | 1 | 0.0 (0/7) | 0.0 (0/7) | 1 | 1.1 | 1359.9/165.7 |
| B2 | 10 | 7 | 3/10 ([10.0, 60.0]) | 0/6 | 83.3 (5/6) | 1 | 0.0 (0/7) | 0.0 (0/7) | 1 | 1.1 | 1359.9/165.7 |
| B3 | 10 | 5 | 9/10 ([70.0, 100.0]) | 0/5 | 0.0 (0/5) | 0 | 0.0 (0/5) | 0.0 (0/5) | 1 | 3.3 | 5628.7/114.9 |
| A1 | 10 | 5 | 9/10 ([70.0, 100.0]) | 0/5 | 0.0 (0/5) | 0 | 0.0 (0/5) | 0.0 (0/5) | 1 | 2.2 | 2718.5/124.3 |
| A2 | 10 | 6 | 1/10 ([0.0, 30.0]) | 0/5 | 80.0 (4/5) | 1 | 0.0 (0/6) | 0.0 (0/6) | 4 | 2.8 | 3547.2/263.2 |

## 按任务类型 TSR %

| Config | basket | dialogue | ledger | single | uncertainty |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 50.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| B1 | 50.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| B2 | 50.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| B3 | 100.0 | 50.0 | 100.0 | 100.0 | 100.0 |
| A1 | 100.0 | 50.0 | 100.0 | 100.0 | 100.0 |
| A2 | 50.0 | 0.0 | 0.0 | 0.0 | 0.0 |

单元格是 TSR %；完整的分母、区间和 `by_category` 在 `report.json` 的 `strata` 中。

## 按语言 TSR %

| Config | en | zh |
| --- | ---: | ---: |
| B0 | 20.0 | 40.0 |
| B1 | 20.0 | 40.0 |
| B2 | 20.0 | 40.0 |
| B3 | 100.0 | 80.0 |
| A1 | 100.0 | 80.0 |
| A2 | 0.0 | 20.0 |

单元格是 TSR %；完整的分母、区间和 `by_category` 在 `report.json` 的 `strata` 中。

## 按预算口径 TSR %

| Config | all_in | pre_tax |
| --- | ---: | ---: |
| B0 | 25.0 | 33.3 |
| B1 | 25.0 | 33.3 |
| B2 | 25.0 | 33.3 |
| B3 | 100.0 | 83.3 |
| A1 | 100.0 | 83.3 |
| A2 | 25.0 | 0.0 |

单元格是 TSR %；完整的分母、区间和 `by_category` 在 `report.json` 的 `strata` 中。

## 配对差值（按 case 的 95% bootstrap 区间）

| Compare | Δ TSR (pp) | 95% CI (pp) | cases |
| --- | ---: | --- | ---: |
| B0 → B1 | +0.0 | [0.0, 0.0] | 10 |
| B1 → B2 | +0.0 | [0.0, 0.0] | 10 |
| A1 → B3 | +0.0 | [0.0, 0.0] | 10 |
| A2 → B3 | +80.0 | [50.0, 100.0] | 10 |

区间包含 0 时，不要声称该配置优于对照。

## 运行溯源

- split: `dev`；configs: `B0, B1, B2, B3, A1, A2`；repeats: 1
- dataset: status=`awaiting_post_freeze_category`, dev=10, heldout=32, pending=8
- dataset hashes: `cases.jsonl`=cf137191182d, `catalogs.json`=4bb5be6d765c
- adapter: model=`qwen3.8-27b`, runtime=`agent.AgentRuntime`, adapter=`evaluations.v1.agent_adapter:respond`, validator=`budget.validator.AgentPlanValidator (integer cents; independent of evaluation oracle)`, temperature=`0`, thinking=`False`, max_output_tokens=`1024`
- model env: BENCHMARK_BASE_URL=`http://127.0.0.1:18967/v1`
- service: job_id=`29415`, node=`10-0-1-10`, address=`http://10.0.1.10:18966/v1`
- code fingerprint: `e3254efdc25c` (72 files)
- timing boundary: `adapter subprocess round-trip; excludes browser`

> 本报告由 `python -m evaluations.bench report` 从 `report.json` 生成；模型、服务参数和原始记录以同目录的 `manifest.json` / `records.jsonl` 为准。
