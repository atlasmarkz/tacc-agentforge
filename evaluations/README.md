# Benchmark 运行手册（v1）

本目录是项目 benchmark 的唯一入口。设计背景、数据契约和未完成项见
[benchmark-design](../docs/benchmark-design.md)，指标定义见
[evaluation-metrics](../docs/evaluation-metrics.md)。旧 v0 用例和结果说明已归档到
[README-v0.md](README-v0.md)，不要用 v0 的 TSR 口径解释 v1。

当前范围：**10 个开发用例 + 32 个保留用例 + 8 个待冻结释放的第五类用例**，6 个配置
`B0/B1/B2/B3/A1/A2`，统一用独立 oracle 和确定性规则评分。正式计划是 40 个 heldout
用例 × 6 配置 × 3 次 = **720 次尝试**。数据是显式标注的 synthetic fixture，不是
SerpApi 真实商品，也不是浏览器端到端结果。

---

## 1. 30 秒检查（不需要模型）

```bash
cd tacc-agentforge
python3.12 -m venv .venv
.venv/bin/python -m pip install -r agent/requirements.lock

PYTHON=.venv/bin/python
$PYTHON -m evaluations.bench check

# 不需要模型端点：从已有运行目录生成 proposal 风格 REPORT.md
$PYTHON -m evaluations.bench report \
  --input evaluations/results/v1-replay-smoke-2026-11-15
# 输出 evaluations/results/v1-replay-smoke-2026-11-15/REPORT.md
```

`check` 会检查数据哈希、冻结状态、adapter 声明、Python 依赖和模型端点；没有配置端点
时只给 warning，仍然可以查看/渲染已有结果。

样例报告：

- **2026-11-17 修复 prompt 后新跑 dev**：[10 个 dev 用例 × 1 次 = 60 attempts，job 29437](results/v1-dev-29437/REPORT.md)
- 修复前对照：[job 29415](results/v1-dev-29415/REPORT.md)（旧 prompt，仅供对比）
- 历史 3 用例冒烟：[job 28969](results/v1-dev-smoke-28969/REPORT.md)（旧夹具）
- 当前数据契约回放：[replay](results/v1-replay-smoke-2026-11-15/REPORT.md)（不是新模型运行）

job 29437 的 proposal 四列如下（TACC 2-GPU `qwen-service`，`qwen3.8-27b`，
数据哈希 `cf137191...`；**dev、1 次 repeat、无人工 Faithfulness**，不是正式 heldout 结果）：

| Method | TSR % | Faithfulness | BVR % | p95 s |
| --- | ---: | ---: | ---: | ---: |
| B0 Single call | 40.0 | N/A | 0.0 | 7.49 |
| B1 Keyword RAG | 40.0 | N/A | 0.0 | 7.72 |
| B2 Structured RAG | 40.0 | N/A | 0.0 | 7.81 |
| B3 Full advisor | 90.0 | N/A | 0.0 | 10.29 |
| A1 Without loop | 90.0 | N/A | 0.0 | 6.92 |
| A2 Without validator | 20.0 | N/A | 0.0 | 14.78 |

这次 dev 的自动诊断：B0/B1/B2 的金额错误率都是 4/6；B3/A1 为 0/5；A2 为 3/5。
B3/A1 的 TSR 90%，B0/B1/B2 都是 40%，A2 是 20%。由于只有 10 个 dev case，区间很宽，
且 B0/B1/B2 在这批 fixture 上结果相同；不能把差异当成正式模型结论。完整计算见
[REPORT.md](results/v1-dev-29437/REPORT.md)。

job 29415 是修复 prompt 前的旧结果（B0/B1/B2 TSR 30%、金额错误 5/6）；只用于说明
`budget_basis` 歧义确实会压低单次调用配置，不能和 29437 混算。

正式 40 个 heldout 用例 × 3 次 = **720 次尚未运行**：需要先 freeze/release 第五类，
并配置稳定的模型端点（约 2 小时）。本报告运行完已释放 TACC 服务作业，当前没有常驻服务。

---

## 2. 接模型跑 dev smoke

以下命令中的 `$PYTHON` 指第 1 节创建的 `.venv/bin/python`；新开 shell 请重新 `export PYTHON=.venv/bin/python`。

模型需要是 **OpenAI 兼容、免鉴权**的 `/v1` 端点（TACC 的 `qwen-service` 就是这样）。
也可以直接给 `--base-url` / `--address-file`，命令会自动设置对应环境变量。

```bash
export BENCHMARK_BASE_URL=http://<host>:<port>/v1
export BENCHMARK_MODEL=qwen3.8-27b

# 可选：TACC 服务地址文件（ModelClient 每次调用前重新读取）
# export BENCHMARK_ADDRESS_FILE=$TACC_USERDIR/tacc-agentforge-service/service-url.txt

$PYTHON -m evaluations.bench check --out /tmp/v1-smoke
$PYTHON -m evaluations.bench run \
  --split dev --only v1-dev-01,v1-dev-03,v1-dev-04 --repeats 1 \
  --out /tmp/v1-smoke
```

第一次只跑 3 个 dev 用例 × 6 配置 = 18 次，确认 6 个配置都能完成、金额/引用检查正常。
成功后会打印 `Report: /tmp/v1-smoke/REPORT.md`。

完整 dev（10 用例 × 6 配置 × 1 次 = 60 次，约 10 分钟）：

```bash
$PYTHON -m evaluations.bench run --split dev --repeats 1 --out /tmp/v1-dev
```

`check` 会请求 `<endpoint>/models` 验证地址，并确认端点确实提供 `BENCHMARK_MODEL`
（默认 `qwen3.8-27b`）这个 alias；不会触发生成。如果端点需要 API key，当前 client
不会发送认证头，需要加一个本地反代或在冻结前扩展 client。

---

## 3. 正式 heldout（40 用例 × 6 配置 × 3 次 = 720）

正式 heldout 是 sealed 的，必须先冻结代码、释放第五类，再显式 `--unseal-heldout`：

```bash
$PYTHON -m evaluations.v1.suite freeze
$PYTHON -m evaluations.v1.suite release-fifth
$PYTHON -m evaluations.v1.suite verify

$PYTHON -m evaluations.bench check --split heldout --unseal-heldout
$PYTHON -m evaluations.bench run \
  --split heldout --unseal-heldout --repeats 3 \
  --out /tmp/v1-heldout
```

注意：

- **freeze 是单向门**。`suite freeze` 会给 `agent/`、`budget/`、`inference/`、
  `knowledge/`、`configs/`、`evaluations/` 下所有 `.py/.yaml/.sql` 生成指纹；冻结后
  任何代码改动都会让 release 失效。代码定稿后再 freeze。
- 720 次约 **2 小时**（按当前冒烟每次约 10 秒估算）。端点需要在这段时间稳定可用；
  TACC 的 qwen-service 需要开启 handover，并留一个作业名额给继任服务。
- 单次 episode 默认 `--timeout 240`（intent 提取 + agent 110s deadline 的最坏情况留余量）；
  如果模型明显更慢，可显式调大。
- 中断后继续，不丢已完成的 attempts：

  ```bash
  $PYTHON -m evaluations.bench run --split heldout --unseal-heldout --repeats 3 \
    --out /tmp/v1-heldout --resume
  # 其余参数必须和第一次一致：data/configs/adapter/timeout/adapter manifest
  ```

  `--resume` 只补缺失的 roster 条目；如果数据、代码、adapter 或 roster 改变，会拒绝继续。
- 如果只是 `records.jsonl` 损坏或想按完整 roster 重新出分，使用：

  ```bash
  $PYTHON -m evaluations.v1.run --rescore /tmp/v1-heldout
  ```

  `--rescore` 会把缺失条目记为失败，**不会**重新调用模型；不要把它当续跑。

---

## 4. 指标与 proposal 的对应关系

`REPORT.md` 的主表就是 proposal 的四列：TSR、Faithfulness、BVR、p95。

| Proposal 指标 | 报告字段 / 来源 | 自动？ | 说明 |
| --- | --- | --- | --- |
| TSR % ↑ | `summary.*.tsr` | 是 | 成功 / 全部尝试；缺失、超时、崩溃计入分母 |
| Faithfulness ↑ | `review.py score` 的 `faithfulness_micro` | 否 | 两位评审 claim-level 标注；没有标注时 N/A |
| BVR % ↓ | `summary.*.bvr` | 是 | 已交付且费用可核实的计划中，超出真实可用预算的比例 |
| p95 s ↓ | `summary.*.wall_s.p95` | 是 | **adapter 子进程完整往返**，不含浏览器/桥接/用户等待 |
| exact amount error | `summary.*.amount_error_rate` | 是 | 报告金额与独立 Decimal 枚举不一致的比例 |
| hard constraint violation | `summary.*.constraint_violation` | 是 | 已交付计划违反规格/数量/约束的比例 |
| citation / grounding | `ungrounded_ids`、`evidence_of_another_product`、`ids_not_in_context` | 是 | ID 存在且对应商品、实际可见 |
| preference fit 1–5 | `review.py score` 的 `preference_mean` | 否 | 两位评审独立打分，报告加权 kappa |
| Recall@5 / MRR@5 | `evaluations/retrieval.py` + `relevance_pool.jsonl` | 否 | 需要先人工标注证据相关性，不在 v1 fixture 轨道 |
| TTFT / TPOT / 吞吐 | `inference/` 服务基准 | 否 | 不是本 benchmark 的 fixture 轨道 |

### 加入人工评审

```bash
# 1) 导出盲评包（会生成 reviewer-a/b 和 private-key；private-key 不要外发）
$PYTHON -m evaluations.review export --records /tmp/v1-dev/records.jsonl \
  --out /tmp/v1-dev/review

# 2) 两位不同评审分别填写 reviewer-a.jsonl / reviewer-b.jsonl（rubric 在目录内）

# 3) 汇总结论
$PYTHON -m evaluations.review score \
  --a /tmp/v1-dev/review/reviewer-a.jsonl \
  --b /tmp/v1-dev/review/reviewer-b.jsonl \
  --key /tmp/v1-dev/review/private-key.jsonl \
  --out /tmp/v1-dev/review-report.json

# 4) 把 Faithfulness / preference 合进主报告
$PYTHON -m evaluations.bench report \
  --input /tmp/v1-dev --review /tmp/v1-dev/review-report.json
```

没有标签时 Faithfulness 保持 N/A；不要用 0 或 1 代替，也不要用单一“标准答案”做字符串匹配。

---

## 5. 输出文件

每次运行目录包含：

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 数据哈希、roster、configs/repeats、adapter 声明、代码指纹、git commit/dirty、模型/服务信息、resume 记录 |
| `records.jsonl` | 每次尝试的 outcome、耗时、tokens、调用数；heldout 会剥离 turns/traces |
| `scores.jsonl` | 每次尝试的确定性评分 |
| `report.json` | summary/strata/paired 的机器可读指标 |
| `REPORT.md` | `evaluations.bench` 自动生成的 proposal 风格报告 |

运行目录必须不存在（或使用 `--resume`）；不要覆盖已有正式结果。

---

## 6. 当前限制

- 数据是 synthetic fixture：不能声称真实 SerpApi 目录、现实价格正确性或完整应用正确率。
- `B1/B2` 的检索差异只在 fixture 候选上验证；真实知识库检索收益需要 KB 快照轨道。
- 10 / 32 个用例规模很小，case-level bootstrap 区间会很宽；3 次 repeat 是同一 case 的重复，不是独立样本。
- `p95` 目前只是 adapter 往返；浏览器到结果的完整请求 p95、TTFT/TPOT 需要应用/推理观测。
- 正式 720 次需要稳定模型端点和约 2 小时连续运行；没有服务时只能跑 `check`/`report` 或已有历史结果。
- 公开 ShoppingBench 250 题是另一条轨道，见 [恢复状态](results/shoppingbench-recovery-2026-11-14/README.md) 和 [公开评分冒烟](results/shoppingbench-public-smoke-2026-11-15/README.md)。

---

## 7. 常见问题

| 现象 | 处理 |
| --- | --- |
| `heldout sealed` | 先 `suite freeze`、`release-fifth`、`verify`，再加 `--unseal-heldout` |
| `release or code changed after freeze` | 代码在 freeze 后改过；恢复 pre-release 数据后重新 freeze/release，不能继续用旧 release |
| `check` 报端点不可用 | 确认 URL 是 `.../v1`，`GET .../v1/models` 可访问；端点不能用旧地址文件 |
| 端点返回 401/403 | 当前 `ModelClient` 不带 API key；加反代或在冻结前扩展认证 |
| `out` 已存在 | 换一个新目录；确认是同一 run 时用 `--resume` |
| `--resume` 报 manifest/adapter/code 改变 | 不能混跑；重新开始一次 run，或先重新 freeze |
| 运行被 kill | 用同样的参数加 `--resume`；不要手改 roster |
| `report.json` 缺失 | `python -m evaluations.v1.run --rescore DIR` 重建评分（缺失记为失败） |
| dev 想保存 prompt/reply | `--record-traces`；heldout 不允许导出原始轨迹 |

---

## 8. TACC 上跑

`deploy/tacc/service/benchmark/` 是 **dev-only smoke worker**：它只打包 dev split、固定 3 个
用例、30 分钟上限，不能直接跑 720。正式 720 建议在不受 1 小时作业限制的主机/云主机上跑，
模型服务由 TACC 提供；或者把正式 runner 分片并实现 resume（当前 `evaluations.bench` 已支持
单进程 `--resume`，但 TACC 分片调度仍需额外脚本）。不要在共享 TACC USERDIR 落 heldout 数据。
