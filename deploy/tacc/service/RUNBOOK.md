# 推理服务运行手册

本手册说明如何在 TACC 上启动、检查、调用、恢复和停止 Qwen3.8-27B 推理服务,以及常见问题怎么处理。读者不需要看代码。所有数字都来自实测作业,作业号写在括号里,原始数据在 `inference/results/` 下。

## 0. 速查

| 要做的事 | 命令或位置 |
|---|---|
| 启动服务 | `bash deploy/tacc/service/qwen-service/stage.sh serve`,然后提交该目录(见第 2 节) |
| 连同 harness 服务一起启动(Task 5) | `stage.sh serve --harness`,一个作业里跑 vLLM 和 harness,第二个名额留给交接(见第 2、5 节) |
| 服务是否可用 | `$TACC_USERDIR/tacc-agentforge-service/service-url.txt` 存在,且内容是一个地址 |
| 服务事件 | `$TACC_USERDIR/tacc-agentforge-service/service-<作业号>/events.jsonl` |
| 调用 | 集群内用 `inference/client.py` 的 `ModelClient(address_file=...)`(见第 4 节) |
| 停止 | 先写 `service-stop`,再 `scancel <作业号>`(见第 6 节) |
| 启动要多久 | 5 到 8.5 分钟(各次实测 297 到 513 秒) |
| 舒适容量 | 约每分钟 6 次完整推荐(关闭思考) |

## 1. 组成

**仓库里的文件**

| 位置 | 作用 |
|---|---|
| `deploy/tacc/service/qwen-service/serve.py` | 服务主程序:启动 vLLM,探测通过才发布地址,异常时撤回并重启,到点交接 |
| `deploy/tacc/service/qwen-service/job.sh` | 作业脚本:激活课程环境,检查 GPU,选择模式,运行交接辅助循环;带 `--harness` 时还启动和停止 harness 服务 |
| `deploy/tacc/service/qwen-service/service.json` | vLLM 参数:双卡张量并行、最大长度 16384、最多 8 个并发序列、前缀缓存、CUDA graphs |
| `deploy/tacc/service/qwen-service/stage.sh` | 提交前的准备:复制客户端,写入模式 |
| `deploy/tacc/service/load-test/` | 压测和故障注入作业 |
| `deploy/tacc/service/agent-e2e/` | harness 端到端作业(mock 工具或真实知识库) |
| `inference/client.py` | 客户端:重试、总时间预算、取消、流式 |
| `inference/model_port.py` | harness 的模型适配器 `VllmModelPort` |
| `inference/harness_policy.py` | 实测推荐的 harness 运行策略 |

**集群上的运行时文件**(都在 `$TACC_USERDIR/tacc-agentforge-service/` 下)

| 文件 | 含义 |
|---|---|
| `service-url.txt` | 当前可用服务的地址。只有真实生成探测通过才会写入,服务异常或退出时删除 |
| `service-url-smoke.txt` | 冒烟测试专用地址,不会覆盖正式服务 |
| `service-<作业号>/events.jsonl` | 关键事件和时间戳:启动、就绪、宕机、重启、交接、撤回地址 |
| `service-<作业号>/probe.jsonl` | 每 30 秒一次探测:是否成功、耗时、vLLM 队列、KV cache 占用、主机内存 |
| `service-<作业号>/gpu.jsonl` | 每秒一次的 GPU 利用率、显存、功耗 |
| `service-<作业号>/server.log` | vLLM 自己的日志 |
| `service-chain.json` | 允许自动交接的剩余次数,例如 `{"remaining": 3}` |
| `service-stop` | 存在时不再交接(用于结束链条) |
| `fault-request` | 仅测试用:在 `validate` 模式下写入地址会杀掉 vLLM |
| `harness-url.txt`、`harness-token.txt` | 带 `--harness` 时:当前 harness 服务的地址和令牌(权限 600),本机后端经 SSH 读取 |
| `service-<作业号>/harness.log`、`harness-stop` | harness 服务的日志;`harness-stop` 由 `job.sh` 写入,让 harness 收尾退出 |

## 2. 启动服务

**前提**:已按 `deploy/tacc/README.md` 配置好自己的 tcloud 账户。

**模式**(由 `stage.sh` 的参数决定,写入 `mode.txt`):

| 模式 | 用途 |
|---|---|
| `serve` | 正式服务,发布到 `service-url.txt` |
| `smoke` | 冒烟测试:用真实 harness 提示跑一遍后退出,发布到 `service-url-smoke.txt` |
| `validate` | 验证用:与 `serve` 相同,但允许故障注入 |

**提交**

```bash
bash deploy/tacc/service/qwen-service/stage.sh serve
```

Windows 上用仓库的封装脚本提交(原生 SDK 在 Windows 上有路径问题):

```bash
bash deploy/tacc/tools/submit-win.sh deploy/tacc/service/qwen-service
```

Linux 或 macOS 上直接用 SDK:

```bash
cd deploy/tacc/service/qwen-service && "$TCLOUD_BIN" submit
```

**额外参数**:写在 `deploy/tacc/service/qwen-service/args.txt`(不纳入 git),一行,原样追加给 `serve.py`。例如开启交接:`--handover-lead-s 900`。测试结束后记得清空,否则会带进下一次正式运行。

**连同 harness 服务**(Task 5 的浏览器页面要用):

```bash
bash deploy/tacc/service/qwen-service/stage.sh serve --harness
bash deploy/tacc/tools/submit-win.sh deploy/tacc/service/qwen-service
```

`--harness` 把 `app/` 和 harness 打包进 `pkg/`。作业在 vLLM 加载期间装好依赖,vLLM 就绪后启动 harness 服务,发布 `harness-url.txt` 和 `harness-token.txt`。harness 通过 `service-url.txt` 调用模型,所以交接后会自动改用继任作业的 vLLM。harness 用自己的一份运行时目录(自己的 proot 根文件系统):和 `serve.py` 共用时,`serve.py` 的 proot 一退出,harness 就看不到 `/mnt/home` 下的文件(作业 28943,探测作业 28952 证实)。harness 出错不会影响 vLLM。可选的 harness 参数写在 `harness-args.txt`(不纳入 git)。

**作业资源**:1 个节点、2 块 GPU、1 小时上限,排除节点 10-0-1-10 和 10-0-1-11。这两个节点的 GPU 被 Slurm 以外的进程占着(每块 15 到 23 GiB,探测作业 28961/28962),vLLM 会内存不足(作业 28960)。

**启动前的 GPU 检查**(`job.sh`):

- 在 10-0-*-11 节点上,Slurm 给的 `CUDA_VISIBLE_DEVICES` 是 `2,3`,但作业里只看得到自己的两块卡,编号为 0 和 1,于是 vLLM 报 NVMLError(作业 28936、28955)。分配的编号超出可见卡数时,`job.sh` 改用 `0,1`,日志里会有一行 `CUDA_VISIBLE_DEVICES 2 3 is outside the 2 visible GPUs; using 0,1`。两台节点的 GPU UUID 不同,改号后用的仍是分配给本作业的卡(探测作业 28956/28957)。
- 任何一块分配到的卡在启动时已用超过 2 GiB,作业立即退出(退出码 3),并记下节点和卡号,不再花 20 分钟反复内存不足。

## 3. 判断服务是否正常

- **就绪的标志**:`service-url.txt` 出现,内容形如 `http://10.0.1.10:8000/v1`;`events.jsonl` 里有 `ready` 事件。作业状态显示"运行中"不代表服务已可用。
- **地址用的是节点 IP,不是主机名**:计算节点之间不能互相解析所有主机名。例如 10-0-1-11 解析不了 `10-0-3-10`,但通过 IP 10.0.3.10 可以正常连接(作业 28614、28616)。早期的结果目录里还能看到主机名形式的地址。
- **启动时间**:5 到 8.5 分钟,随节点不同(实测 297、305、313、367、392、443、513 秒)。启动超时设为 1200 秒。
- **运行中**:`probe.jsonl` 每 30 秒一行,`ok` 为 true 表示一次真实生成成功。连续 3 次失败会触发重启。
- **GPU**:`gpu.jsonl` 每秒一行。空闲时利用率约 11%,舒适负载下约 85%,显存约 22 GiB / 24 GiB(这是 vLLM 预留的,不代表在涨)。

查看方法(Windows,用自己的用户名和密钥):

```bash
C:/Windows/System32/OpenSSH/ssh.exe -p 30041 -i <私钥> <用户名>@gw.turing.ust.hk 'cat $HOME/USERDIR/tacc-agentforge-service/service-url.txt; tail -3 $HOME/USERDIR/tacc-agentforge-service/service-*/events.jsonl'
```

查看作业状态:`bash deploy/tacc/tools/tcloud-win.sh ps`。

## 4. 调用服务

服务只能从 TACC 集群内部访问,集群没有对外的 HTTP 入口。调用方(例如 harness)必须运行在集群上:另一个作业,或者用 `stage.sh serve --harness` 放进服务作业本身。集群外(例如本机浏览器)经 SSH 隧道访问 harness 服务,见 `app/README.md`。

**直接调用**

```python
from pathlib import Path
from inference.client import ModelClient

client = ModelClient(address_file=Path("/mnt/home/<用户名>/USERDIR/tacc-agentforge-service/service-url.txt"),
                     model="qwen3.8-27b")
r = client.chat([{"role": "user", "content": "你好"}], max_tokens=128, timeout_s=30,
                enable_thinking=False,
                on_delta=lambda kind, text: print(text, end=""))   # 去掉 on_delta 就是非流式
print(r.finish_reason, r.prompt_tokens, r.completion_tokens, r.first_token_s)
```

- 每次尝试前都会重新读取地址文件,所以服务重启或交接到新节点后,调用方不用重启。
- 连接错误、429、5xx 会在总时间预算内重试;其他 4xx 立即失败。
- 错误类型:`ServiceUnavailable`(没有可用服务)、`ServiceTimeout`(超出预算)、`ServiceRejected`(请求本身有问题)、`BadResponse`、`StreamInterrupted`(已输出部分内容后断流,不重试以免重复)、`Cancelled`。

**在 harness 里使用**

```python
from agent import RunPolicy
from inference.harness_policy import run_policy_kwargs
from inference.model_port import VllmModelPort

port = VllmModelPort(client, enable_thinking=False)            # 可选 on_delta=... 接收流式 token
policy = RunPolicy(**run_policy_kwargs(enable_thinking=False))  # 上下文 36000 字符;开启思考时输出 2048 token
```

- 建议决策调用关闭思考:一次完整推荐约 10 秒,开启思考时中位数约 32 秒(作业 28561)。
- 不要用 harness 的默认策略接真实工具:默认 512 个输出 token 会被思考耗尽,16000 字符的上下文会丢掉之前的步骤(issue #6)。
- 适配器会把动作 schema 里的 `pattern` 锚定后再交给解码器,否则 `ask_user` 的问题会只有一个字。

## 5. 自动恢复与交接

**作业内重启**(默认开启)

- vLLM 崩溃,或连续 3 次探测失败时:撤回地址,重启 vLLM,真实生成成功后重新发布。
- 每个作业最多重启 2 次(`--max-restarts`);作业剩余时间不足 600 秒时不再重启(`--min-restart-time-s`)。
- 实测(作业 28579/28580,带负载):杀掉 vLLM 后 0.57 秒撤回地址,302 秒后恢复服务。这期间调用方收到的是快速的 `ServiceUnavailable`(中位 1.7 秒、最长 2.8 秒),没有请求卡死;地址恢复后请求立即成功。

**作业交接**(默认关闭)

作业有 1 小时上限。开启交接后,作业会在到期前自行提交一个继任作业;继任作业就绪并发布地址后,旧作业等正在处理的请求结束再退出,地址不会中断。

开启方法:

1. 在集群上写入允许的交接次数:`echo '{"remaining": 3}' > $TACC_USERDIR/tacc-agentforge-service/service-chain.json`
2. 在 `args.txt` 写入 `--handover-lead-s 900`,即提前 15 分钟提交继任作业。各节点启动时间最长实测 513 秒,15 分钟留有余量。
3. 正常启动服务。每次交接会把 `remaining` 减 1。

实测(作业 28583 交接给 28584):在设定时间点提交继任作业,继任作业 3 秒后在另一个节点开始,513 秒后就绪;旧作业 0.42 秒后察觉并退出,地址全程没有中断。

**名额限制**:每人最多同时运行 2 个作业,交接时旧作业和继任作业各占一个。所以 harness 不能再单独占一个作业,否则继任作业只能排队,每小时中断 5 到 8.5 分钟。用 `stage.sh serve --harness` 把 harness 放进服务作业(issue #10)。

**带 harness 的交接**:继任作业的 vLLM 就绪后,它的 harness 随即启动并发布 `harness-url.txt`。旧作业被取代后,先等继任作业的 harness 地址出现(最多 5 分钟),再写 `harness-stop`。旧 harness 停止接收新请求,处理完正在进行的请求后退出;120 秒内没退出就由 `job.sh` 结束它的进程(只结束父进程不够:Slurm 用 `proctrack/linuxproc`,孤儿进程会在作业结束后继续占着节点,作业 28925、28943、28946 都留下过)。交接前后,旧 harness 已经通过 `service-url.txt` 改用继任作业的 vLLM。

实测(作业 28971 交接给 28973,2026-11-14,表中时间为集群日志的 UTC;本机后端经 SSH 隧道每 30 秒发一个真实请求,共 30 分钟,结果在 `app/results/handover-soak-28971.json`):

| 指标 | 结果 |
|---|---|
| 请求 | 60 个全部 HTTP 200,没有 `harness_unavailable`、`model_unavailable` |
| 切换 | 继任 harness 17:04:38 发布;旧 harness 17:04:41 停止接收,17:04:45 退出(22 个运行,0 个错误) |
| 切换前后 | 第 21 个请求由旧 harness 处理,但已在调用继任作业的 vLLM;第 22 个起由 28973 处理 |
| 节点 | 28971 在 10-0-3-10,28973 在 10-0-3-11(GPU 编号改为 0,1 后正常启动) |
| 未完成 | 23 个停在 `model_call_limit`(充电器请求 19/20),与交接无关,见 issue #21 |

上一轮(28963 交接给 28964)集群侧同样无缝,但本机后端连到了旧隧道,见 `app/README.md`。

**实现说明**:`serve.py` 运行在课程环境的 proot 隔离根文件系统里,那里没有 `sbatch` 和 Slurm 配置。所以 `serve.py` 只写一个 `handover-request` 文件,由 `job.sh` 在 proot 外运行的辅助循环代为提交,结果写回 `handover-result`。

## 6. 停止服务

1. 先阻止新的交接:`touch $TACC_USERDIR/tacc-agentforge-service/service-stop`
2. 取消作业:`scancel <作业号>`(在网关上执行;Windows 上通过 ssh)。
3. 用 `squeue -u <用户名>` 检查是否还有排队中的继任作业,有就一并取消。

作业收到停止信号后会撤回自己的地址。它只删除内容是自己地址的地址文件,不会误删继任作业的地址。

## 7. 验证与压测

| 工具 | 做什么 | 怎么用 |
|---|---|---|
| `smoke` 模式 | 用真实 harness 提示测一遍合法率、流式和延迟后退出 | `stage.sh smoke` 后提交服务 |
| `load-test/` | 开环压测:按泊松到达率发请求;`--mode runs` 按真实运行顺序回放,`--mode calls` 是不复用前缀的最坏情况;可选故障注入 | 服务用 `validate` 模式启动,压测参数写在 `load-test/args.txt`,先 `stage.sh` 再提交 |
| `agent-e2e/` | 原样的 harness 在另一个节点上调用真实服务:`e2e.py` 用 mock 工具,`e2e_kb.py` 用真实知识库 | `script.txt` 选择脚本,`args.txt` 写参数(如 `--tuned`) |

压测负载由 `scripts/make_load_workload.py` 生成,来自真实运行(作业 28561)的提示长度和输出长度。

## 8. 容量与限制

**容量**(作业 28568/28569,按真实运行回放,关闭思考,单节点双卡)

| 到达速率(次运行/秒) | 每分钟完成 | 一次推荐 p50 / p95 | GPU 利用率均值 |
|---|---|---|---|
| 0.03 | 2.3 | 15 秒 / 22 秒 | 38% |
| 0.06 | 6.3 | 12 秒 / 25 秒 | 65% |
| 0.1 | 7.2 | 17 秒 / 33 秒 | 85% |
| 0.15 | 9.1 | 37 秒 / 75 秒 | 88% |
| 0.25 | 9.5 | 23 秒 / 58 秒 | 76% |

- 舒适容量约每分钟 6 次推荐;上限约每分钟 9.5 次,再加压只会排队变长。
- 瓶颈是预填充算力:每次调用的提示平均约 4300 token,输出只有约 36 token/秒。KV cache 最高只占 23%,显存不是瓶颈。
- 前缀复用很重要:每个请求提示都不同时(作业 28566),上限只有约 0.3 次调用/秒;复用前缀后约 0.53 次调用/秒。
- 提高容量最有效的办法是缩小提示(工具结果单次最多约 8 KB)。
- 只测了关闭思考;开启思考时输出长 4 到 7 倍,容量会明显更低。

**流式**(作业 28462):首 token 中位约 0.12 秒,流式文本与非流式完全一致。

**长时间稳定性**:见下节。

## 9. 长时间稳定性(soak)

作业 28586/28587:以每秒 0.07 次运行(约每分钟 4.6 次推荐,舒适容量以内)持续回放真实运行,负载约 43 分钟。

| 指标 | 结果 |
|---|---|
| 完成情况 | 200 次运行全部完成,714 次调用,0 个错误;84 次探测,0 次失败 |
| 延迟是否漂移 | 没有。按 5 分钟分段,单次调用 p50 在 2.3 到 4.3 秒之间波动,第一段和最后一段都是 3.3 秒;一次推荐 p50 按 10 分钟分段在 8 到 15 秒之间,没有上升趋势 |
| 主机内存 | 18891 MiB 到 18908 MiB,43 分钟只涨 17 MiB(0.09%),而且主要在前 5 分钟,之后基本持平 |
| 显存 | 全程 22594 MiB,不变 |
| KV cache | 最高占用 22%,没有发生抢占(preemption) |
| GPU | 利用率均值 53% 到 63%,功耗 441 到 492 W,随负载组合波动;延迟没有变慢的迹象 |
| 日志 | vLLM 日志只有 0.18 MB |

结论:在舒适负载下,单个服务作业可以稳定运行到接近 1 小时的上限,没有发现延迟漂移或资源泄漏。

没有覆盖到的:
- 这次只跑了一个作业周期,没有覆盖超过 1 小时,也没有在负载下经历真实的整点交接。原因是服务和压测作业已占满 2 个名额,继任作业无法同时运行(见第 5 节的限制)。
- 只测了关闭思考。
- 没有直接测 GPU 温度和频率,"没有降频"是从延迟没有变慢推断的。

## 10. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 作业在运行,但没有 `service-url.txt` | 还在启动(5 到 8.5 分钟) | 等待;超过 20 分钟仍未就绪,看 `server.log` |
| 在 10-0-*-11 节点上启动失败,报 NVMLError | Slurm 给的 GPU 编号是 `2,3`,作业里只看得到编号 0、1 | 当前的 `job.sh` 会自动改号(日志里有 `using 0,1`);还报错说明用的是旧版本 |
| 作业立刻退出(退出码 3),日志里有 `GPUs already in use by processes outside this job` | 节点的 GPU 被 Slurm 以外的进程占着(10-0-1-10、10-0-1-11) | `tuxiv.conf` 已排除这两个节点;新出现的节点也加进 `exclude`,并告诉助教 |
| 交接后 harness 地址指向已停止的作业 | 旧作业没能正常收尾 | 当前的 `job.sh` 会在退出前撤回自己的 `harness-url.txt`;本机后端每次重新读取地址 |
| 作业立刻失败,日志里有 `$'\r': command not found` | 脚本被转成了 Windows 换行(CRLF) | 仓库的 `.gitattributes` 已固定为 LF;已经出问题的文件用 `sed -i 's/\r$//' 文件` 修复 |
| 调用方一直收到 `ServiceUnavailable` | 服务没在运行、正在重启或正在交接 | 看 `events.jsonl`。重启约 5 分钟;交接排队时取决于名额 |
| 偶发 `StreamInterrupted` | vLLM 在生成途中退出 | 适配器把它标记为可恢复错误(`model_unavailable`),调用方可以重试整次调用;harness 如何处理以它的实现为准 |
| 回复为空,`finish_reason` 为 `length` | 开启了思考,512 个输出 token 被推理用完 | 使用 `inference/harness_policy.py` 的策略,或关闭思考 |
| 模型反复调用同一个工具,直到调用上限 | 上下文预算太小,之前的步骤被丢掉 | 使用 `harness_policy.py` 的 36000 字符 |
| 工具调用报 `invalid_arguments`,运行直接失败 | 模型猜错品类名或属性名,而这类错误不可恢复 | 见 issue #4 |
| `ask_user` 的问题只有一个字 | schema 的 `pattern` 被解码器当成整串匹配 | 适配器已修复;新增带 `pattern` 的 schema 时注意同样的问题 |
| `events.jsonl` 里有 `handover_submit_failed` | 辅助循环没有运行,或 `sbatch` 报错 | 看 `service-<作业号>/handover-result`;确认用的是当前的 `job.sh` |
| 继任作业一直不启动 | 2 个作业名额已满 | 正常现象;它会在旧作业结束后启动 |
| 地址文件存在,但调用方一直连不上,或报 "Name or service not known" | 地址里是对方节点的主机名,本节点无法解析 | 当前的 `serve.py` 已改为发布 IP;如果看到主机名形式的地址,说明用的是旧版本 |
| harness 作业装依赖失败(找不到 pip) | 课程的 vLLM 环境没有 pip | `agent-e2e/job.sh` 已改用个人 Miniconda 引导的新版 pip,装到 `USERDIR/agent-pylib` |

## 11. 证据

| 结果目录(`inference/results/`) | 内容 |
|---|---|
| `service-stream-smoke-28462` | 流式验证 |
| `agent-e2e-28469`、`agent-e2e-kb-28561` | harness 端到端(mock 工具、真实知识库) |
| `validate-load-28566`、`validate-service-28565` | 第一轮压测(最坏情况)与故障注入 |
| `validate-load-28569`、`validate-service-28568` | 第二轮压测(按运行回放)与 GPU 采样 |
| `recover-load-28580`、`recover-service-28579` | 作业内重启 |
| `handover-service-28583`、`handover-successor-28584` | 作业交接 |
| `chain-probe-28574` | 作业内能否提交继任作业 |
| `soak-load-28587`、`soak-service-28586` | 长时间稳定性 |

| 其他结果 | 内容 |
|---|---|
| `app/results/handover-soak-28971.json` | 带 harness 的交接(28971 交接给 28973):经浏览器后端 30 分钟、60 个请求 |
| `app/results/handover-soak-28943.json`、`-28954.json` | 修复前的两轮(共用 proot 根文件系统、harness 进程残留) |
