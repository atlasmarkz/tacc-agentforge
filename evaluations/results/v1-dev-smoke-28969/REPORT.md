# Benchmark report: project-v1-synthetic-fixtures

- split: `unknown`；configs: `B0, B1, B2, B3, A1, A2`；repeats: 1
- attempts recorded: 18
- dataset status: `awaiting_post_freeze_category`
- human review merged: no（Faithfulness 为 N/A，需要 `python -m evaluations.review`）

## 主结果（proposal 表）

| Method | Configuration | TSR % ↑ | Faithfulness ↑ | BVR % ↓ | p95 s ↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| B0 Single call | No extra text; no budget tool | 0.0 | N/A | 0.0 | 12.60 |
| B1 Keyword RAG | FTS5 evidence; single call; no budget tool | 0.0 | N/A | 0.0 | 13.65 |
| B2 Structured RAG | Filtered FTS5; single call; no budget tool | 0.0 | N/A | 0.0 | 13.62 |
| **B3 Full advisor** | B2 + bounded loop + budget tool | 100.0 | N/A | 0.0 | 11.65 |
| A1 Without loop | B3; one-pass plan with the same tools | 100.0 | N/A | 0.0 | 11.66 |
| A2 Without validator | B3; budget tool disabled | 33.3 | N/A | 0.0 | 14.99 |

口径：

- **TSR** = 成功完成次数 / 全部计划尝试次数；超时、崩溃、缺失记录都计入分母。
- **Faithfulness** = 两位评审对可评判事实声明的支持率；没有标注时为 N/A，不能用 0 或 1 代替。
- **BVR** = 已交付且费用可核实的计划中超出真实可用预算的比例；没有已交付计划时为 N/A。
- **p95** 是 adapter 子进程完整往返耗时，不含浏览器、桥接和用户输入等待；不是 proposal 的完整请求 p95。

## 自动指标与计数

| Config | runs | plans | TSR n/m (95% CI) | BVR n/m | amount errors | unverifiable plans | constraint violations | ungrounded IDs | invalid actions | model calls | tokens in/out |
| --- | ---: | ---: | --- | --- | --- | ---: | --- | --- | ---: | ---: | --- |
| B0 | 3 | 2 | 0/3 ([0.0, 0.0]) | 0/2 | 100.0 (2/2) | 0 | 0.0 (0/2) | 0.0 (0/2) | 0 | 1.3 | 1007.3/184.3 |
| B1 | 3 | 2 | 0/3 ([0.0, 0.0]) | 0/2 | 100.0 (2/2) | 0 | 0.0 (0/2) | 0.0 (0/2) | 0 | 1.3 | 1703.0/186.7 |
| B2 | 3 | 2 | 0/3 ([0.0, 0.0]) | 0/2 | 100.0 (2/2) | 0 | 0.0 (0/2) | 0.0 (0/2) | 0 | 1.3 | 1703.0/186.7 |
| B3 | 3 | 2 | 3/3 ([100.0, 100.0]) | 0/2 | 0.0 (0/2) | 0 | 0.0 (0/2) | 0.0 (0/2) | 0 | 3.0 | 5246.3/108.7 |
| A1 | 3 | 2 | 3/3 ([100.0, 100.0]) | 0/2 | 0.0 (0/2) | 0 | 0.0 (0/2) | 0.0 (0/2) | 0 | 2.0 | 2577.7/132.3 |
| A2 | 3 | 2 | 1/3 ([0.0, 100.0]) | 0/2 | 100.0 (2/2) | 0 | 0.0 (0/2) | 0.0 (0/2) | 0 | 3.7 | 6262.3/262.0 |

## 按任务类型 TSR %

| Config | basket | dialogue | uncertainty |
| --- | ---: | ---: | ---: |
| B0 | 0.0 | 0.0 | 0.0 |
| B1 | 0.0 | 0.0 | 0.0 |
| B2 | 0.0 | 0.0 | 0.0 |
| B3 | 100.0 | 100.0 | 100.0 |
| A1 | 100.0 | 100.0 | 100.0 |
| A2 | 0.0 | 0.0 | 100.0 |

单元格是 TSR %；完整的分母、区间和 `by_category` 在 `report.json` 的 `strata` 中。

## 按语言 TSR %

| Config | en | zh |
| --- | ---: | ---: |
| B0 | 0.0 | 0.0 |
| B1 | 0.0 | 0.0 |
| B2 | 0.0 | 0.0 |
| B3 | 100.0 | 100.0 |
| A1 | 100.0 | 100.0 |
| A2 | 0.0 | 100.0 |

单元格是 TSR %；完整的分母、区间和 `by_category` 在 `report.json` 的 `strata` 中。

## 按预算口径 TSR %

| Config | all_in | pre_tax |
| --- | ---: | ---: |
| B0 | 0.0 | 0.0 |
| B1 | 0.0 | 0.0 |
| B2 | 0.0 | 0.0 |
| B3 | 100.0 | 100.0 |
| A1 | 100.0 | 100.0 |
| A2 | 0.0 | 50.0 |

单元格是 TSR %；完整的分母、区间和 `by_category` 在 `report.json` 的 `strata` 中。

## 配对差值（按 case 的 95% bootstrap 区间）

| Compare | Δ TSR (pp) | 95% CI (pp) | cases |
| --- | ---: | --- | ---: |
| B0 → B1 | +0.0 | [0.0, 0.0] | 3 |
| B1 → B2 | +0.0 | [0.0, 0.0] | 3 |
| A1 → B3 | +0.0 | [0.0, 0.0] | 3 |
| A2 → B3 | +66.7 | [0.0, 100.0] | 3 |

区间包含 0 时，不要声称该配置优于对照。

## 运行溯源

- split: `unknown`；configs: `B0, B1, B2, B3, A1, A2`；repeats: 1
- dataset: status=`awaiting_post_freeze_category`, dev=10, heldout=32, pending=8
- dataset hashes: `cases.jsonl`=ee22a161b6fa, `catalogs.json`=4bb5be6d765c
- adapter: model=`qwen3.8-27b`, runtime=`agent.AgentRuntime`, adapter=`evaluations.v1.agent_adapter:respond`, validator=`budget.validator.AgentPlanValidator (integer cents; independent of evaluation oracle)`, temperature=`0`, thinking=`False`, max_output_tokens=`1024`
- code fingerprint: `b2484375df47` (57 files)
- timing boundary: `adapter subprocess round-trip; excludes browser`

> 本报告由 `python -m evaluations.bench report` 从 `report.json` 生成；模型、服务参数和原始记录以同目录的 `manifest.json` / `records.jsonl` 为准。
