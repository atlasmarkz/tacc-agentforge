# Shopping knowledge service：运行与部署说明

本模块提供 SerpApi 增量补采、SQLite/FTS5 知识库、不可变快照、一个 schema 发现工具和四个查询工具。下列命令从 **GitHub 仓库根目录**执行，无需创建作者机器上的目录。查询部分使用 Python 3.12+ 标准库；Agent 联调和完整测试另外安装项目已锁定的依赖。

## 1. 从 GitHub 克隆后直接运行

仓库包含 `assets/knowledge/pilot-snapshot.zip`（约7MB）和其 SHA256，首次运行从本地种子包初始化，无需 API key、模型服务或重新爬取数据。

Linux/macOS/WSL：

```bash
git clone <仓库地址>
cd tacc-agentforge
python3.12 -m venv .venv
PYTHON="$PWD/.venv/bin/python"

# 可选：将运行数据放在仓库中已被Git忽略的目录。
# 不设置时默认为仓库旁边的 ../tacc-agentforge-data。
export KNOWLEDGE_DATA_ROOT="$PWD/.local/knowledge"

"$PYTHON" -m knowledge bootstrap
"$PYTHON" -m knowledge doctor
"$PYTHON" -m knowledge products --category earbuds --limit 3
"$PYTHON" -m knowledge search '降噪 耳机' --category earbuds --k 5
"$PYTHON" -m knowledge audit
```

`bootstrap` 校验ZIP的SHA256和快照内所有文件/证据，成功后原子安装；重复执行只验证既有快照，不会覆盖它。损坏的已安装快照会报错。`doctor` 应返回 `ready: true`、`fts5: true`、`verification.valid: true`。

Windows PowerShell查询示例：

```powershell
py -3.12 -m venv .venv
$env:KNOWLEDGE_DATA_ROOT = Join-Path (Get-Location) '.local\knowledge'
.\.venv\Scripts\python.exe -m knowledge bootstrap
.\.venv\Scripts\python.exe -m knowledge doctor
.\.venv\Scripts\python.exe -m knowledge search 'noise cancelling earbuds' --category earbuds
```

采集器使用POSIX任务锁，采集和TACC提交脚本请在Linux/macOS/WSL运行。本地自动化验证环境为macOS；TACC Linux CPU任务已通过快照校验和查询smoke验证。

## 2. 配置与路径优先级

版本化默认配置位于 `configs/knowledge.json`：

| 配置 | 用途 |
|---|---|
| `data_root` | 原始批次、运行数据库、快照的根目录；相对路径以项目根目录为基准 |
| `snapshot_id` | 固定、可复现实验快照，当前为 `snapshot_5bcd00723fb0b7589cbe50c8` |
| `seed.archive / seed.sha256` | 仓库内种子ZIP路径与校验值 |
| `service.cache_size` | 结果LRU项数，默认512；0可关闭结果缓存 |
| `service.max_result_bytes` | 工具返回上限，默认8192 bytes |
| `service.price_ttl_hours` | 价格年龄阈值，默认24小时 |

优先级为 **显式命令行/函数参数 > 环境变量 > JSON默认值**。环境变量为 `KNOWLEDGE_CONFIG`、`KNOWLEDGE_DATA_ROOT`、`KNOWLEDGE_SNAPSHOTS_ROOT`、`KNOWLEDGE_SNAPSHOT_ID`。相对环境变量/命令行路径以当前工作目录为基准。

自定义配置文件放在项目 `configs/` 下，通过 `python -m knowledge --settings configs/knowledge.local.json ...` 选择；其中的相对路径以该 `configs/` 的上级目录为基准。个人配置请自行加入Git忽略或使用环境变量。

```bash
"$PYTHON" -m knowledge --data-root /your/data/root bootstrap
"$PYTHON" -m knowledge --data-root /your/data/root products --category earbuds
"$PYTHON" -m knowledge products --snapshot-id snapshot_5bcd00723fb0b7589cbe50c8 --limit 1
```

正常查询显式固定配置中的快照，不会自动切换到后来采集的数据。服务运行中切换快照需要宿主为下一次run指定新ID。

## 3. 手工查询与分页

```bash
# 筛选基础价格≤USD100，排除价格未知的候选。
"$PYTHON" -m knowledge products --category earbuds --currency USD \
  --max-price-minor 10000 --filters '{"include_unknown":false}' --limit 3

# 规格三态：满足 / 不满足 / 未知；明确不满足者排除。
"$PYTHON" -m knowledge products --category chargers \
  --filters '{"hard_constraints":{"power":{"op":"gte","value":"65","unit":"W"}},"include_unknown":true}'

# 将ID替换为查询实际返回的值。
"$PYTHON" -m knowledge offers OFFER_ID
"$PYTHON" -m knowledge evidence EVIDENCE_ID
"$PYTHON" -m knowledge evidence EVIDENCE_ID --cursor CURSOR
"$PYTHON" -m knowledge products --category earbuds --limit 1 --cursor CURSOR
```

金额采用最小货币单位，10000美分=USD100。`include_unknown=true`默认保留待澄清候选；false排除未知约束。翻页时保持相同筛选条件/ID列表。返回中包含 `bytes`、`truncated`、`next_cursor`；报价批量截断使用 `remaining_offer_ids`，只包含完全未返回的offer组。

检索保留Top-K证据ID，并在字节不足时缩短片段，通过 `resolve_evidence` 展开全文。无法容纳元数据或完整单offer上下文组时返回明确错误。终端JSON带缩进，字节限制针对工具使用的紧凑UTF-8序列化，不是终端打印长度。

## 4. Harness / Agent 接入

需要现有Agent框架时安装锁定依赖：

```bash
"$PYTHON" -m pip install -r agent/requirements.lock
```

用同一配置构造服务，无需写任何人的本机路径：

```python
from knowledge.settings import load_settings
from knowledge.service import KnowledgeService
from knowledge.agent_tools import build_knowledge_tools

settings = load_settings()
kb = KnowledgeService(settings.snapshots_root, **settings.service)
tools = build_knowledge_tools(kb)

categories = kb.describe_filters(settings.snapshot_id)
schema = kb.describe_filters(settings.snapshot_id, category="earbuds")
page = kb.search_products(settings.snapshot_id, {"category": "earbuds"}, limit=5)
evidence = kb.retrieve_evidence(
    settings.snapshot_id, "noise cancelling comfortable", page["candidate_set_id"], k=5
)

# 注入 AgentDependencies(model=..., tools=tools, validator=...)。
# AgentRequest.snapshot_id = settings.snapshot_id。
# 所有活跃run结束后，由宿主关闭共享工作线程：
tools.get("search_products").executor.close()
```

每个harness进程创建一次service和registry，在同一个事件循环内复用。snapshot ID来自可信 `CallContext`，模型参数不包含数据库路径。

| Python接口 | 行为 |
|---|---|
| `describe_filters(snapshot_id, category=None, cursor=None)` | 不传品类时分页列出合法品类；传品类时分页列出 canonical 属性、值类型、操作符和单位 |
| `search_products(snapshot_id, filters, limit=5, cursor=None)` | 完整匹配范围的candidate_set_id，加分页紧凑商品卡片 |
| `retrieve_evidence(snapshot_id, query, candidate_set_id, k=5, mode='filtered')` | 固定候选范围内的Top-K证据 |
| `get_offer_facts(snapshot_id, offer_ids)` | 当前报价、精确金额、费用状态、observation和evidence IDs |
| `resolve_evidence(snapshot_id, evidence_ids, cursor=None)` | 原文、URL、时间、raw hash和JSON pointer |

`candidate_set_id`在服务内存中活跃。服务重启或集合淘汰后，重新执行商品查询；不同快照的集合不能互用。B1通过 `build_knowledge_tools(kb, retrieval_mode='keyword')` 选择关键词证据，B2/B3用filtered，保持候选/价格输入一致。

模型可先调用 `describe_filters({})` 获取当前快照的品类，再调用 `describe_filters({"category":"earbuds"})` 获取该类属性，不能由模型指定 `snapshot_id`。发现结果中的 `categories` 和 `attributes` 分别用于品类页和属性页；若有 `next_cursor`，保持同一 `category` 继续查询。只返回查询用的 canonical 名称，不返回导入别名；属性出现在 schema 中不表示每个商品都有已知值。

非法品类、属性、值类型等查询参数返回可恢复的 `invalid_arguments`，短错误提示包含合法值或修正要求，Agent 可据此调用发现工具并更正下一次查询。指定品类时只允许该类的属性；未指定品类时可使用各类属性的并集。顶层 `type` 与 `hard_constraints` 的属性不是同一概念，例如机械轴应根据键盘 schema 使用 `switch_type`，不要写成 `hard_constraints.type`。配置、可信快照选择和快照完整性错误仍不可恢复。发现和失败查询都计入既有调用预算，不会自动重试或扩充预算。

issue #4 的本地验收使用真实快照、真实适配器和脚本模型，覆盖错误反馈后更正查询。本地回归不能替代真实模型验收：后续应固定代码、快照和 issue #6 的部署策略，重跑原五个场景，不传 `--hint` 或 `available_categories`，确认不再因可修正的品类/属性错误直接失败；本次修复不自动提交 TACC 作业。

CLI演示需要传入显式快照位置，可使用环境变量或配置输出填写：

```bash
SNAPSHOT_ID=snapshot_5bcd00723fb0b7589cbe50c8
"$PYTHON" -m knowledge.examples.agent_demo \
  --snapshots-root "$KNOWLEDGE_DATA_ROOT/data/knowledge/snapshots" \
  --snapshot-id "$SNAPSHOT_ID"
```

该示例使用真实归档和脚本模型，不调用LLM。由于费用与SKU尚未确认，预期 `waiting_user`；预算验证器未接入时不能产出已验证购买方案。字节限制不能替代ModelPort对实际tokenizer及4K上下文的检查。

## 5. 数据版本、重新采集和评测

当前种子来源于88个成功响应：315个发现商品组，其中80个有详情（四类各20），251条报价、380条评论、2507条属性、3525条证据。商品组身份仍待审核，属性/评论可为系列范围；税费未知不会填零，显示总价不代表到手预算。

种子不包含可增量写入的工作数据库或原始采集任务目录。首次团队查询使用已冻结快照，`audit`会在没有工作数据库时审计默认快照。要重新采集/重建，先取得或采集原始run，将其放在 `$KNOWLEDGE_DATA_ROOT/data/serpapi/`，再执行：

```bash
"$PYTHON" -m knowledge plan --target-per-category 10
# collect是明确联网操作，运行前另配SERPAPI_API_KEY或--key-file，重试计入50次上限。
# 同一任务恢复时使用原--session-dir，不能新建任务来扩大本轮预算。
"$PYTHON" -m knowledge collect --session-dir "$KNOWLEDGE_DATA_ROOT/data/serpapi/run-team-001" \
  --max-attempts 50 --target-per-category 10
"$PYTHON" -m knowledge ingest
"$PYTHON" -m knowledge snapshot
```

种子本身的provenance保留历史导入来源名称，仅用于溯源，不参与路径解析，也不要求组员建立同名目录。后续发布新种子时重新打包完整快照，并同时更新 `configs/knowledge.json` 的快照ID和ZIP SHA256。

```bash
"$PYTHON" -m knowledge label-development --output "$KNOWLEDGE_DATA_ROOT/reports/development-labels.json"
"$PYTHON" -m knowledge evaluate --labels "$KNOWLEDGE_DATA_ROOT/reports/development-labels.json" \
  --output "$KNOWLEDGE_DATA_ROOT/reports/retrieval.json"
"$PYTHON" -m knowledge benchmark --iterations 100 --scale 10 \
  --output "$KNOWLEDGE_DATA_ROOT/reports/benchmark.json"
"$PYTHON" -m pytest tests agent/tests -q
```

程序化开发标签不等于人工或held-out相关性；不可将其分数外推为回答Faithfulness或完整推荐成功率。10倍基准扩大证据量，商品和价格不变，记录缓存开关、冷/暖与并发边界。CPU查询时延不包括LLM、采集或SDK桥接。

第五类按proposal保留到agent/prompt/tools冻结后接入。Top-K证据不能证明完整目录无解。

## 6. TACC与GitHub交付

[TACC CPU worker部署说明](../deploy/tacc/knowledge-worker/README.md)包括上传准备、SDK提交、运行时覆盖、USERDIR持久化、节点本地查询和日志位置。

其他组员应使用自己的Tcloud账户执行初始化，将固定快照安装到 `$TACC_USERDIR/tacc-agentforge-data/data/knowledge/snapshots/`。不要依赖其他账户的作业路径或节点临时目录。

本次远端验证覆盖 `search_products`、`retrieve_evidence` 和 `resolve_evidence`；`get_offer_facts` 已有本地测试，但未在该次smoke中调用。集群p95性能基准、真实LLM/harness调用和完整预算验证仍待执行；`smoke`成功不代表这些项目已通过。

提交GitHub时应包含 `knowledge/`、`configs/knowledge.json`、`configs/category_schemas.json`、`assets/knowledge/`、测试和部署脚本。`assets/knowledge/pilot-snapshot.zip`是可版本化的离线复现种子，不要漏传。

运行数据、个人配置、密钥、`.venv`、`_stage`和日志不应提交。项目忽略规则已覆盖默认运行路径与TACC生成目录。此模块不会自动push或发布Release。
