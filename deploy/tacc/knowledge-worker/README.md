# TACC知识库CPU任务

这是知识库的离线验证/查询任务：**1个节点、2个CPU、无GPU、最长20分钟**。它沿用仓库现有Tcloud SDK入口和课程公共Python包装器，不需要个人目录路径、SerpApi key、模型服务或额外pip安装。

配置与脚本已通过本机模拟及真实TACC CPU smoke验证。集群性能基准和真实LLM/harness联调仍待运行；详见[验证范围](#7-验证范围)。

## 1. 准备上传目录

先按 `deploy/tacc/README.md` 配置自己的Tcloud账户。在仓库根目录执行：

```bash
python3.12 deploy/tacc/knowledge-worker/stage.py --dry-run
python3.12 deploy/tacc/knowledge-worker/stage.py
```

`stage.py`校验种子ZIP的SHA256，仅复制知识库代码、三个公共配置和种子到 `_stage/pkg/`。它不复制 `.env`、密钥、个人数据根目录或虚拟环境。`_stage/`已被Git忽略，提交代码前不需要运行stage；每次上传前重新stage。

首次stage会从 `runtime.example.json` 创建本地 `runtime.json`。之后保留你的配置，不会覆盖。生成的runtime.json也已被Git忽略。

## 2. 选择模式

编辑 `deploy/tacc/knowledge-worker/runtime.json`：

```json
{
  "mode": "smoke",
  "data_subdir": "tacc-agentforge-data",
  "copy_snapshot_to_local": true,
  "benchmark_iterations": 100,
  "benchmark_scale": 10,
  "query": "noise cancelling earbuds",
  "category": "earbuds"
}
```

| 模式 | 行为 |
|---|---|
| `verify` | 初始化并完整校验快照 |
| `smoke` | 校验后运行商品筛选、Top-5检索和证据展开，默认模式 |
| `benchmark` | 校验后执行本节点的真实/10倍证据性能基准 |

默认快照ID读取 `_stage/pkg/configs/knowledge.json`。可在runtime.json额外指定 `snapshot_id` 选择USERDIR内已有的另一快照；如果选择的快照未安装，种子ZIP也必须与其匹配。

`data_subdir`只能是 `TACC_USERDIR` 内的相对目录。`copy_snapshot_to_local=true`会将快照复制到 `SLURM_TMPDIR` 或节点系统临时目录，再验证并只读查询；任务结束自动清理这份临时副本。原始持久快照保持在USERDIR，避免每次查询在共享存储上读写工作库。

## 3. SDK提交与查看结果

```bash
cd deploy/tacc/knowledge-worker
TCLOUD_BIN=${TCLOUD_BIN:-tcloud}
"$TCLOUD_BIN" submit

# 用提交返回的作业ID替换JOB_ID。
"$TCLOUD_BIN" ps -j JOB_ID
"$TCLOUD_BIN" cat slurm_log/knowledge-worker-JOB_ID.log

# 相对路径以当前账户USERDIR为基准；默认smoke模式。
"$TCLOUD_BIN" cat tacc-agentforge-results/knowledge-JOB_ID/smoke.json
"$TCLOUD_BIN" download tacc-agentforge-results/knowledge-JOB_ID/smoke.json ./smoke-JOB_ID.json
```

成功日志包含：

```text
KB_RUNTIME <Python版本> <SQLite版本>
{"status":"ready", ...}
KB_REPORT=<TACC_USERDIR>/tacc-agentforge-results/knowledge-JOB_ID/smoke.json
```

持久数据：

```text
$TACC_USERDIR/tacc-agentforge-data/data/knowledge/snapshots/<snapshot_id>/
$TACC_USERDIR/tacc-agentforge-results/knowledge-<job_id>/<mode>.json
```

首次安装从上传种子解压并校验；后续作业验证后复用。已有同ID快照损坏时任务会失败，不会静默覆盖。SDK需要下载结果时使用课程SDK自己的文件管理命令和日志中显示的路径。

`ps`用于查看排队或运行状态；作业结束后可能不再列出。以对应节点日志和结果JSON核对 `status=ready`、`execution_context=tacc_job` 及 `verification.valid=true`，不能仅凭SDK显示“submitted”判断验证通过。每次使用自己提交得到的作业ID；知识库worker不是需要一直运行的服务。

## 4. 公共运行时与替代Python

默认SDK管理环境为 `mpi-gemm-v1`；`job.sh`随后激活现有课程公共环境：

```text
/mnt/home/public/course-envs/CSIT6000V/project/activate.sh
```

通过 `qwen-python` 使用实际Python 3.12运行。脚本先检查Python版本和SQLite FTS5能力，管理bootstrap的Python不会用于知识库查询。

若课程路径变化，可在账户/job环境设置 `KNOWLEDGE_RUNTIME_ACTIVATE` 为新的公共activate脚本。若已准备自己的Python 3.12+和FTS5，则设置 `KNOWLEDGE_PYTHON` 为**单个Python可执行文件路径**，跳过公共激活。不要把带参数的整条shell命令放进这个变量。

`KNOWLEDGE_RUNTIME_CONFIG`可以指定另一个runtime JSON文件，文件必须存在。配置代码不包含任何组员的用户名或私有Conda路径。

## 5. Harness与模型任务的关系

本任务用于准备和验证知识库，不提供HTTP监听端口，也不需要一直占用一个独立服务作业。Harness运行时在其CPU进程内构造 `KnowledgeService`，读取USERDIR中的同一已校验快照。

```python
import os
from pathlib import Path
from knowledge.settings import load_settings
from knowledge.service import KnowledgeService
from knowledge.agent_tools import build_knowledge_tools

data_root = Path(os.environ["TACC_USERDIR"]) / "tacc-agentforge-data"
settings = load_settings(data_root=data_root,
    snapshots_root=data_root / "data/knowledge/snapshots")
kb = KnowledgeService(settings.snapshots_root, **settings.service)
tools = build_knowledge_tools(kb)
# 注入现有AgentDependencies，AgentRequest.snapshot_id=settings.snapshot_id。
```

现有 `deploy/tacc/service/agent-e2e/stage.sh` 已将knowledge代码和配置一起打包，供harness后续接入；它不重复上传种子，先用本worker完成USERDIR初始化。实际harness使用Agent适配器时仍需要其现有Pydantic/LangGraph环境。模型请求继续由TACC内的独立GPU服务处理，知识库任务不启动模型。

如需相同节点的稳定查询性能，harness也可把整个已验证快照复制到节点临时目录后构造服务。不要在已发布快照中增量写入数据；更新数据应在工作库导入并导出新ID。

## 6. 本机模拟验证

无需Tcloud或GPU，也能验证上传包、USERDIR持久化和节点临时副本清理：

```bash
python3.12 deploy/tacc/knowledge-worker/stage.py
TASK_DIR="$PWD/deploy/tacc/knowledge-worker"
mkdir -p .local/tacc-user
export TACC_WORKDIR="$TASK_DIR"
export TACC_USERDIR="$PWD/.local/tacc-user"
export SLURM_JOB_ID=local-smoke
export KNOWLEDGE_PYTHON="$PWD/.venv/bin/python"
export KNOWLEDGE_LOCAL_TEST=1
bash "$TASK_DIR/job.sh"
```

本机报告明确标记 `execution_context=local_simulation`，不代表已在TACC部署或测得集群时延。

## 7. 验证范围

已完成的真实TACC smoke验证覆盖快照安装和完整校验、节点临时副本查询，以及 `search_products`、`retrieve_evidence` 和 `resolve_evidence`。

`get_offer_facts` 已有本地测试，但未在该次远端smoke中调用；集群查询p95、真实LLM/harness和完整预算算法仍待验证。运行本文的 `benchmark` 模式后，才能报告相应节点的性能结果。节点临时查询路径不应保存为harness配置。
