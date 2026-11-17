# Benchmark 调研与实施说明

可直接运行的命令、指标表与排错见 [evaluations/README.md](../evaluations/README.md)。

调研日期：2026-11-13。课程规模。**现成 benchmark 可复用；采用公开环境外部对照＋项目冻结任务主评测。** 模型实验状态与限制见文末；单元测试不作为模型分数。

## 公开 benchmark 选择

| Benchmark | 公开性与能力 | 采用方式 |
|---|---|---|
| [ShoppingBench](https://github.com/yjwjy/ShoppingBench) / [AAAI论文](https://arxiv.org/html/2508.04266v2) | 900测试任务，Coupon & Budget 250题；代码Apache-2.0，公开商品沙箱、工具与评分。 | 本轮唯一外部对照。预算只覆盖商品价格与优惠券，不代替周期预算、税运费、引用评测。 |
| [EComAgentBench](https://github.com/Morizeyao/EComAgentBench_) | 662英文任务，SQLite+FTS5、画像、脚本澄清、评论；代码MIT，商品数据须遵守独立数据条款，数据库约25GB。 | 后续多轮需求获取评测候选；最终仅单商品，LLM裁判需要额外资源。 |
| [ShoppingComp](https://github.com/ByteDance-BandAI/ShoppingComp) | 97常规＋48陷阱，中英文、原子rubric、参考证据，公开任务与judge。 | 借鉴中文报告、引用和约束评测；开放世界任务不能直接给小型目录评分。 |
| [Shopping Reasoning Bench](https://huggingface.co/datasets/amazon/ShoppingReasoningBench) | 525任务、10,863专家rubric；CC-BY-NC-4.0，需接受HF访问条件。 | 回答质量补充；没有工具环境和独立预算账本。 |
| [WebShop](https://github.com/princeton-nlp/WebShop) | 开放购物环境，商品搜索、页面操作、单商品购买。 | 接口与项目工具型顾问差异较大，本轮不运行。 |

[Ragas](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) 是指标与裁判框架，[vLLM bench](https://docs.vllm.ai/en/latest/benchmarking/cli/) 是性能工具，都不是完整购物任务数据集。

## 两条评测轨道

### ShoppingBench 外部对照

保持官方商品库、全部250道voucher测试题、工具与评分函数。单独环境安装官方依赖（Java21、Pyserini、约1.4GB压缩商品文档/4.8GB解压及索引），通过官方 `OPENAI_BASE_URL` 接部署模型。先验证少量任务的API接入，再运行完整250题；保存官方生成配置、prompt、模型和服务参数。预算子集不需要将知识类在线搜索实验混入结果。

固定完整git提交ID，禁止修改任务/工具/reward后声称是原版。按官方README从 `src/agent` 运行voucher rollout；完成后用官方环境的Python运行本项目审计器：

```bash
python -m evaluations.shoppingbench --repo /path/to/ShoppingBench \
  --revision FULL_COMMIT_SHA --rollouts /absolute/path/to/voucher-rollout.jsonl \
  --model MODEL_ID --out /absolute/path/to/shoppingbench-report.json
```

审计器调用上游 `eval_voucher`，保留规则与预算判定，分母固定250；额外记录缺失输出、完成率、工具次数和输入哈希/依赖。上游[原评分代码](https://raw.githubusercontent.com/yjwjy/ShoppingBench/main/src/agent/run_evaluate.py)遍历已有rollout，缺失任务可能未计入；本入口补齐该分母，报告时注明。token数在上游轨迹未提供时为null，不猜测。Base/SFT须同一基座、同一环境，官方harness结果不代表生产B3。

### 项目v1

旧42例及历史结果保留为v0，不更改旧分数含义。新增独立v1契约/枚举器/评分器与adapter运行入口。目前交付 **10dev＋32heldout，另外8题受冻结门槛保护**。费用夹具全部标注synthetic，不是新增采集的真实商品；不能声称实现了真实目录评测或现实价格正确性。

完整v1目标40测试题：单商品12、购物篮8、账本6、多轮8、证据不足6；五品类各8，中文/英文各20；当前中英文由同一模板直译生成并共享英文品类词，只能作语言外壳分层，不能当作独立中文任务。夹具总量与报价年龄不代表真实采集质量。单币种USD，不隐式换汇。

```bash
python -m evaluations.v1.suite verify
# 完成agent/prompt/tool冻结后，由评测负责人执行：
python -m evaluations.v1.suite freeze
python -m evaluations.v1.suite release-fifth
python -m evaluations.v1.suite verify
```

`freeze`记录agent、knowledge、inference、configs、evaluation代码哈希；之后改变实现须重新版本化评测。第五类desk_lamps的schema通过数据加入，不修改agent代码。生成器公开，所以该门槛是过程约束，不是访问控制或预训练污染证明；第五类及所有测试衍生内容必须从训练、校准和提示调优中排除。只有release之后才允许正式heldout运行。

## 契约与确定性评分

- 用例：schema_version、case_id/split、品类/语言/能力标签、budget_basis、用户原话、初始及最终约束、购物篮requirements、脚本交互、catalog_id。
- catalog：明确synthetic，稳定offer/evidence ID、币种、十进制价格、每件税费、每订单组运费、规格、冲突/过期标志。相同运费组只收一次；不同组分别计费；冲突组无法核实。
- `pre_tax`不含税，`all_in`包含每件税费。unknown不补零；所有成本必须能核实。金额必须是非负、有限、精确到分的十进制字符串。
- 可用资金为 `min(cap, periodic-spent-reserve)`；仅有一种来源用该来源，周期账本缺支出/预留需追问，不默认零。负余额不会变成额外可用资金。
- oracle独立穷举有限目录组合，不导入生产工具/validator；接受所有可行SKU组合。购物篮每需求类别选择一个报价，支持数量；同类别多报价拆单不在本版定义内。
- plan：items、每行subtotal、shipping、total、remaining，引用须存在、对应商品、实际可见且包含该报价的事实证据。
- no_feasible_plan：只有全部组合已被排除才成立；必须指定catalog_id和有效原因。insufficient_evidence：仍有未排除未知组合，不能声称全目录无解。ask_user必须询问恰当必要字段。无关追问失败。
- 不做字符串匹配唯一“标准答案”。金额、数量、预算、未知和引用ID确定性检查；引用内容是否支持自然语言声明仍需人工/校准裁判，ID通过不等于Faithfulness。

## 运行与对照

```bash
export BENCHMARK_BASE_URL=http://localhost:8000/v1
export BENCHMARK_MODEL=qwen3.8-27b
python -m evaluations.v1.run --configs B0,B1,B2 --split dev --repeats 3 \
  --adapter evaluations.v1.model_adapter:respond \
  --adapter-manifest evaluations/v1/adapter.example.json --out /tmp/benchmark-v1-dev
```

自带model adapter实现B0/B1/B2的夹具单次调用：B0使用共同结构化候选、无证据检索；B1用SQLite FTS5对用户消息取最多5证据；B2在此基础上按消息中出现的品类过滤候选（未命中则回退全目录），与B1产生真实的信息集差异。fixture候选仍按请求品类限定，真实过滤检索收益须在KB快照轨道测量，不能用此处差值作证。

生产adapter契约为 `module:function(payload)->record`；每轮隔离子进程，有总episode deadline。payload只含用户可见输入、历史、目录和输出契约，不含oracle/最终约束/结构化参考要求。adapter自行维护配置差异：B0结构化单次；B1关键词检索；B2过滤检索（品类过滤候选）；B3有界循环＋真实校验器；A1检索→一次规划→校验、不修订；A2关闭提案校验与 `assess_catalog`，由模型自行报告金额。B1/B2的候选范围因此不同；B3/A1/A2共用全目录候选，不额外提供参考category/retrieval_query。

**B3/A1/A2 已接入现有 AgentRuntime，业务校验由 `budget.validator` 独立实现**；这是 normalized fixture 轨道，不等于完整 SerpApi/浏览器系统。校验器以整数分核算，oracle以Decimal独立枚举。模型先从用户原话和历史提取需求，错误提取仍由oracle判失败，不注入参考requirements。B3允许有界工具循环与校验反馈；A1固定流程且只允许一次规划；A2不提供提案校验或目录评估工具，保留未经校验的提案，并让模型自行报告金额；报告阶段只允许重排/保留商品与数量，不得改变商品或数量。提取和金额报告的调用均计入成本。

示例声明 `agent-adapter.example.json` 只对真实校验配置B3/A1登记。旧 `adapter.example.json` 的单次调用入口仍拒绝冒充完整advisor。能力声明是审计记录，不是自动证明。禁止把独立oracle注入业务validator。v0的RecordingValidator仍为历史诊断。

脚本澄清仅在正确字段出现后返回固定答案；新增最终 `expected=clarify` 用例（最终约束缺 purchase_cap），由模型询问，adapter不做预算预检查。修订场景先独立评分初始方案，再要求保留未变需求。最终输出对最终约束核验。dev原始turns需显式 `--record-traces` 才保存；heldout记录会剥离 turns/model_trace/events/parsed_requirements 等训练衍生内容。

完整正式运行：release后使用生产adapter，6配置×40题×3=720；用循环平衡配置顺序；`deploy/tacc/service/qwen-service/service.json` 已移除前缀缓存，manifest记录实际服务参数/依赖。同case三次重复作同一统计簇，paired bootstrap按case，不把120视为独立样本。B3/A1/A2用于迭代/validator比较，不用B2→B3单独解释loop因果。

中断后按预写roster补齐缺失失败，不丢弃部分repeat：

```bash
python -m evaluations.v1.run --rescore /tmp/benchmark-v1-dev
```

计时目前是adapter子进程完整往返，**不是浏览器到结果**；浏览器/桥接耗时须接生产应用观测。流式model adapter保存TTFT；TPOT/吞吐/显存复用现有inference性能轨道，不能用模型耗时代替用户延迟。

## 人工标注、指标与交付状态

主表TSR、Faithfulness、BVR、用户完整请求p95，公开/项目两表分开。报告金额错误、各能力/语言/预算轨道TSR、Recall@5/MRR@5、引用支持/覆盖、失败原因、tokens/调用/成功任务成本。没有计划时BVR=null；未知费用的可负担声明失败。裁判不可用时Faithfulness=null。

人工工作不能由生成器伪造：现有602行relevance_pool仍待标注；两人独立判断证据是否支持必要事实，分歧按归档来源解决，并从池外抽样查漏。只标池内时报告pooled recall，不能声称完整Recall。偏好1–5分：1不满足用途，2明显缺陷，3满足主要偏好但解释欠缺，4充分满足且权衡合理，5充分满足、权衡和费用完整；预算/硬约束失败不能靠偏好补分。两人盲评并报告加权kappa及分歧。LLM裁判固定模型/版本/prompt/随机设置，在双人开发标签上校准，失败单列。

验收：等额/差一分、数量、合并运费、多解、unknown/冲突/过期、无解与缺证据区分、错误/虚构引用、修订保留要求、非法动作、超时及缺失输出分母。当前42条夹具已加入 pre_tax 17.00、all_in 18.00 的等额边界、11.99 的差一分 no_feasible、以及 dev/heldout 各1条最终 clarify。保留至少一条真实失败轨迹；不将单元测试造出的失败当模型实验。

已交付：调研、独立v1夹具/枚举器/评分器、全部六配置的真实AgentRuntime接入、独立业务校验器、720次调度、冻结/第五类门槛、带调度终态确认的恢复监督器、外部上游评分审计器、外部环境预检与隔离运行、随机盲评ID及加权kappa。恢复批次实际停在183/190条（其中121条API-complete），没有最终report/ASR；不得把183条当作API有效。真实KB快照到normalized contract转换、人工相关性/偏好/裁判校准、浏览器指标仍需独立验收。不得将fixture跑分写成真实价格或完整应用正确率。

延迟降低20%或token减少30%且TSR损失≤5个百分点是工程目标；40题不足以作强非劣性统计结论。


## 2026-11-14 执行入口与人工交接

六配置开发集：

```bash
python -m evaluations.v1.run --configs B0,B1,B2,B3,A1,A2 --split dev --repeats 3 \
  --adapter evaluations.v1.agent_adapter:respond \
  --adapter-manifest evaluations/v1/agent-adapter.example.json --out /tmp/v1-six-config-dev
```

模型地址可通过 `BENCHMARK_BASE_URL` 或 `BENCHMARK_ADDRESS_FILE` 提供。TACC CPU worker在 `deploy/tacc/service/benchmark/`，使用内存检查点，不对SQLite持久卷作未经验证的声明。`stage.py`只打包公共源代码和 dev split（heldout不进入 `pkg`）。worker等待18966独立端口，拒绝8000上的陈旧地址，并在设置 `BENCHMARK_EXPECTED_SERVICE_JOB` 时校验该作业的 `service-info.json`/address；服务实际端口必须与worker配置一致。

模型服务 `serve.py --port` 支持独立端口，每次启动前检查占用，并要求本次启动日志出现Application startup complete后才作生成探测。修复前的28966/28967触及已有端口，**不得用于模型比较**。28968的新端口不再误判就绪，但10-0-1-10的两GPU均已占用23500/24576MiB，CUDA初始化OOM；这是基础设施失败，不是模型任务失败。

人审包（每case/config只取最早一次repeat，去掉配置名；blind ID使用随机token，private-key仅由评测负责人保管并已被 `.gitignore` 排除）：

```bash
python -m evaluations.review export --records /path/to/records.jsonl --out /tmp/blind-review
# 两位不同评审独立填写 reviewer-a.jsonl / reviewer-b.jsonl 后：
python -m evaluations.review score --a /tmp/blind-review/reviewer-a.jsonl \
  --b /tmp/blind-review/reviewer-b.jsonl --key /tmp/blind-review/private-key.jsonl --out /tmp/review-report.json
```

空白标签、未检查声明完整性、同一评审冒充两人、重复/缺失ID均拒绝评分。无可评判事实声明时Faithfulness=N/A，不把沉默评为满分。人审输出与TSR报告分表；原始分歧保留，不自动替两人裁决。

公开轨道固定提交 `204dfba767cf849fe33535e42293e8973fb017ea`。官方LFS语料压缩1471795629字节，SHA256为 `a0ba9a8618ce72a7b383a6de31acf6e5bf138873da0d13ad8a1a6b0271e8e9d1`，解压5112146514字节。2026-11-15已按该commit重建索引（19 files、3271147867 bytes、SHA256 `2012a83fcbf08e301ce2981413bf2bcb10fb2371316d89d6971aa78fc4cfc080`），`preflight.ready=true`；用官方`eval_voucher`对历史中断批次冒烟：250/250、ASR 4.4%、API-incomplete 204、scoring errors 0，产物在 `evaluations/results/shoppingbench-public-smoke-2026-11-15/`。下载/校验/审计冒烟不等于干净模型跑分。

```bash
python -m evaluations.shoppingbench_run --repo /path/to/ShoppingBench --check-corpus --out /tmp/public-preflight
```

`--run`只在隔离环境内运行：上游python_execute会执行模型生成代码，因此使用 `deploy/benchmark/ShoppingBench.Dockerfile`，只挂公共代码、语料、索引和该次输出目录，不挂用户home/密钥。Docker构建上下文须含官方requirements.txt；使用CPU PyTorch 2.9.1及OpenAI SDK 1.109.1，其余实际依赖通过pip-freeze归档。Java21与Python3.12在容器内提供。

```bash
docker build -f /path/to/tacc-agentforge/deploy/benchmark/ShoppingBench.Dockerfile \
  -t agentforge-shoppingbench:204dfba /path/to/ShoppingBench
# 在同一镜像下建立官方Lucene索引（resources中只放已验证解压文档）：
python deploy/benchmark/build_shoppingbench_index.py --repo /path/to/ShoppingBench
# 有可用的模型网关后，挂本项目公共代码至/project并令PYTHONPATH=/project：
# python -m evaluations.shoppingbench_run --repo /bench --out /outputs/run --model qwen3.8-27b --check-corpus --run
```

索引构建入口拒绝覆盖已有 `indexes`，在临时目录运行官方脚本，只暴露解压JSONL，避免gzip与JSONL重复索引；`index-build.json` 记录 image、解压语料SHA256、index文件数/字节数和合并SHA256，`preflight` 要求该清单存在，审计报告会带上 `index_build`/`preflight`。正式运行只读挂载公共源和索引，输出目录单独可写。模型配置为官方30步harness、4并发、temperature0、max_tokens2048、关闭thinking，作为本项目明确记录的设置，不冒称复现论文所有设置。官方轨迹未记录token usage时保持null。

## 中断后的恢复（2026-11-14）

首批公开轨道写出250条记录，但204条含API重试耗尽后的空响应，只有46条每一步均有模型响应。该批官方规则ASR为11/250=4.4%，**这是受基础设施中断影响的审计数字，不作模型能力结论**。原始轨迹及报告在 `evaluations/results/shoppingbench-interrupted-2026-11-14/`，恢复结果另存，不覆盖首批。

`shoppingbench_recovery.prepare` 的筛选条件仅为空API响应；完整但错误的推荐不重试。新轨迹文件先保留46条API完整记录，未修改的官方producer仍读取全部250题，并跳过这46题后执行剩余204题。源码、商品库、rubric、模型生成参数均不因得分调整。

`deploy/benchmark/recover_public_run.py` 将SSH控制连接、容器及调度监督与交互会话分离：

- 只控制自身提交并通过账户检查的作业；每轮默认两次一小时分配，可用 `--max-allocations` 调整（1–10），总等待/执行上限默认四小时（`--max-hours`）。
- `squeue` 空输出不再当作FINISHED：必须由 `sacct` 确认终态，或保持 `UNKNOWN/waiting_for_allocation`，避免SSH抖动/提交注册竞态导致误取消或重复提交。
- 服务切换前暂停容器；每次目标IP/端口变化都重建SSH转发（不复用旧control socket），确认新作业自己的service-info与模型别名后才恢复。
- `scancel` 检查返回码并以 `squeue` 确认消失后才记录 `allocation_released`；失败写入 `cleanup_error`，发布器不得继续自动评分。
- 完成、错误、停止请求或达到限额时释放自己提交的作业，不结束其他用户作业；`stop.request` 是显式停止入口；中间轨迹和原始失败均保留。所有终态记录 `completed/api_complete/inherited_api_complete/retried_new_api_complete`；发生新的API失败时状态标为 `completed_with_api_failures`，不冒充干净评测。

`publish_public_run.py` 在监督结束且每个作业都有确认的 `allocation_released` 后，才以相同官方评分器作CPU评分，自动压缩轨迹、复制模型/依赖/恢复策略及服务元数据，并写出结果说明。出现 `cleanup_error` 时状态标为 `publish_blocked_cleanup`，不评分/不发布。人类不用再发送“继续”才能触发评分；队列和GPU资源仍可能决定实际完成时间。

本次恢复的提交快照和后续结果位于 `evaluations/results/shoppingbench-recovery-2026-11-14/`。当前归档实际停在 `paused_allocation_limit`：rollouts有190条distinct，其中121条API-complete、69条API-incomplete，`supervisor.json`记录183条（历史产物，未改写）；没有最终 `report.json`/恢复ASR。调度状态不等于已完成评测，恢复ASR必须等实际运行和CPU评分完成；在终态确认修复前不得发布。2026-11-15只重建了索引并对历史中断轨迹做了官方评分冒烟，没有启动新的模型/GPU作业，也没有新的恢复ASR。
