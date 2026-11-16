# 云端 agent 服务：Cloudflare Pages + GCE + TACC vLLM（issue #25）

页面放在 Cloudflare Pages，agent 常驻在 Google Cloud 的一台虚拟机上，TACC 只跑 vLLM。默认按**完全免费方案**部署，不需要买域名，**页面公开，谁都能用**。需要登录保护或者有自己的域名时，见最后两节。

```
浏览器 ──HTTPS──▶ Cloudflare Pages：tacc-agentforge.pages.dev（app/static/index.html）
                               │ functions/api/*：同源代理，带共享密钥和访问者 IP
                               ▼
                        Tailscale Funnel：https://<虚拟机>.<tailnet>.ts.net（只有出站连接，不开入站端口）
                               ▼
  GCE 虚拟机 ─────────────────────────────────────────────────────────────────
  │ agentforge-backend   app/backend.py   127.0.0.1:8765  校验代理密钥、运行归属、按访问者 IP 限流
  │ agentforge-harness   app/harness_service.py --cloud   127.0.0.1:8100
  │                      AgentRuntime + 知识库快照 + 预算校验器 + SQLite 检查点（本地盘）
  │ agentforge-tunnel    app/model_tunnel.py  读 service-url.txt，维护 ssh -L，写本地地址文件
  ───────────────────────────────────┬────────────────────────────────────────
                                     │ ssh -L，经 gw.turing.ust.hk
                                     ▼
  TACC 服务作业：只跑 vLLM（stage.sh serve），每小时交接，地址变了隧道自动跟随

  GCS：gs://<bucket>/snapshots/<snapshot_id>.zip  每个快照只写一次
```

## 和集群内部署的区别

| | 集群内（`stage.sh serve --harness`） | 云端（本目录） |
|---|---|---|
| 页面 | 本机后端提供，只有本人能用 | Cloudflare Pages，公开，谁都能用 |
| harness | 和 vLLM 同一个作业，每小时交接 | 虚拟机上常驻，systemd 崩溃自动重启 |
| 检查点 | 内存；交接或重启时，等待回答的运行会丢 | 本地盘 SQLite；后端或 harness 重启后仍能继续（#8） |
| 运行归属 | 页面生成的会话 ID | 同样是会话 ID，但存在 `runs.sqlite`，重启后仍有效 |
| 限流 | 无 | 每个访问者 IP 同时 2 个请求、每分钟 6 次新提问；全局同时 8 个 |
| 知识库 | 仓库 zip，每个账户一份 | GCS 按快照 ID 存一份，下载时校验哈希 |
| 作业名额 | 服务作业加继任作业，两个名额都占 | 同样两个名额，作业里少了 harness 和依赖安装 |

集群内部署仍然保留，用作回退。

## 费用

| 项目 | 费用 |
|---|---|
| 虚拟机 | us-west1 的 e2-micro 和 30 GB 标准盘在永久免费额度内。官方文档说免费额度不对外部 IP 收费，但开机后仍要看一下账单明细 |
| GCS | 美国区 5 GB 以内免费；快照约 7 MB |
| Cloudflare Pages | 免费 |
| Tailscale Funnel | 个人免费版，限非商业用途；有带宽上限但官方未公布，只传文字够用 |

GCP 计费账户要求绑卡，建议在 GCP 设一个预算提醒（比如超过 1 美元就发邮件）。如果想降低到香港网关的延迟，可以用 `ZONE=asia-east2-a MACHINE=e2-small`，但这个配置收费。

## 部署步骤

### 0. 前提

- 本机已安装 gcloud 并登录（`brew install --cask google-cloud-sdk`，然后 `gcloud init`）。
- 一个开通了计费的 GCP 项目。
- 一个 Cloudflare 账户和一个 Tailscale 账户。
- TACC 账户能提交 `qwen-service`（见 `deploy/tacc/service/RUNBOOK.md`）。

### 1. 存储桶、快照和虚拟机

```bash
PROJECT=<gcp 项目> BUCKET=<全局唯一的桶名> bash deploy/gcp/create-vm.sh     # 同时创建存储桶和虚拟机
bash deploy/gcp/publish-snapshot.sh assets/knowledge/pilot-snapshot.zip <桶名>
```

- `publish-snapshot.sh` 用 `--if-generation-match=0` 上传，同名对象已经存在时会失败，所以快照不会被覆盖。脚本最后打印 `seed.url` 和 `seed.sha256`。
- 现有快照 `snapshot_5bcd00723fb0b7589cbe50c8` 迁移后，快照 ID 和 SHA256 都不变。
- 本地验证从 GCS 安装：

  ```bash
  python -m knowledge --data-root /tmp/kb bootstrap --url gs://<桶名>/snapshots/<id>.zip
  python -m knowledge --data-root /tmp/kb doctor
  ```

  `doctor` 应返回 `ready: true`、`verification.valid: true`。
- 发布新快照时，把 `configs/knowledge.json` 的 `snapshot_id`、`seed.sha256` 和 `seed.url` 一起改掉再提交，不再提交 zip。确认所有成员都能从 GCS 安装后，再从仓库删除 `assets/knowledge/pilot-snapshot.zip`。

### 2. 初始化虚拟机

```bash
gcloud compute scp deploy/gcp/setup-host.sh agentforge:~ --zone us-west1-b --project <项目>
gcloud compute ssh agentforge --zone us-west1-b --project <项目> -- \
  sudo bash setup-host.sh https://github.com/atlasmarkz/tacc-agentforge.git feat/cloud-agent-service
```

`setup-host.sh` 依次完成：

1. 安装 Python 3.12、git 和 Tailscale；
2. 创建 2 GB 交换文件（e2-micro 只有 1 GB 内存）；
3. 创建 `agentforge` 用户；
4. 克隆仓库并安装依赖；
5. 把 TACC 网关的主机公钥写入 `known_hosts`；
6. 安装三个 systemd 服务。

### 3. TACC 密钥和配置文件

在虚拟机上专门生成一把密钥，不要复用个人私钥：

```bash
sudo -u agentforge ssh-keygen -t ed25519 -N '' -f /tmp/tacc_key -C agentforge-gce
sudo install -o agentforge -g agentforge -m 600 /tmp/tacc_key /etc/agentforge/tacc_key
cat /tmp/tacc_key.pub          # 加到 TACC 账户的 ~/.ssh/authorized_keys
sudo rm /tmp/tacc_key /tmp/tacc_key.pub
```

测试新密钥：

```bash
sudo -u agentforge ssh -p 30041 -i /etc/agentforge/tacc_key <TACC 用户名>@gw.turing.ust.hk 'echo ok'
```

生成共享密钥，记下它，第 5 步要填进 Pages：

```bash
openssl rand -hex 32
```

然后按 `agentforge.env.example` 填写 `/etc/agentforge/agentforge.env`：TACC 账户、`KNOWLEDGE_SEED_URL`、`AGENTFORGE_PROXY_SECRET`。`CF_ACCESS_*` 保持注释状态，页面就是公开的。

**注意：** 是否允许把 TACC 账户的密钥放在云服务器上，需要先和助教确认（见 #25）。

### 4. Tailscale Funnel

在虚拟机上执行：

```bash
sudo tailscale up                    # 打开打印出的链接，用 Tailscale 账户登录
sudo tailscale funnel --bg 8765      # 第一次会提示到管理后台开启 HTTPS 和 Funnel，照着点开即可
tailscale funnel status              # 记下 https://<虚拟机>.<tailnet>.ts.net
```

Funnel 把公网 HTTPS 请求转到虚拟机的 127.0.0.1:8765，后端仍然只监听本机。

### 5. Cloudflare Pages

在本机执行：

```bash
npx wrangler login
bash deploy/cloudflare/deploy.sh
```

第一次运行会创建名为 `tacc-agentforge` 的 Pages 项目，地址是 `https://tacc-agentforge.pages.dev`。

然后设置两个密钥（共享密钥从文件读入，不会显示出来），再部署一次让它们生效：

```bash
cd deploy/cloudflare
npx wrangler@4 pages secret put PROXY_SECRET --project-name tacc-agentforge < <共享密钥文件>
printf %s https://<虚拟机>.<tailnet>.ts.net | npx wrangler@4 pages secret put API_ORIGIN --project-name tacc-agentforge
cd ../.. && bash deploy/cloudflare/deploy.sh
```

如果 `wrangler` 刚登录完就提示没登录：它发现 `~/.wrangler` 目录存在时，会改用这个旧位置找凭据。把 `~/Library/Preferences/.wrangler/config/default.toml` 复制到 `~/.wrangler/config/` 即可（权限 600）。

### 6. 暂存 vLLM 作业（一次性）

在本机暂存只跑 vLLM 的作业，上传到集群的 `~/WORKDIR/qwen-service/`。macOS 上没有 tcloud 时，直接用 SSH 完成，`run.sh` 和 `configurations/` 沿用之前 SDK 生成的版本。这里只上传，不提交，提交由下一步的开关负责：

```bash
cd deploy/tacc/service/qwen-service
bash stage.sh serve                                   # 不带 --harness
printf '%s\n' '--handover-lead-s 900' > args.txt      # 开启交接：到期前 15 分钟提交继任作业
tar czf - client.py job.sh main.py mode.txt serve.py service.json smoke.py smoke_requests.json stage.sh \
    tuxiv.conf args.txt | ssh -p 30041 -i ~/.ssh/id_rsa_tacc <TACC 用户名>@gw.turing.ust.hk \
  'cd ~/WORKDIR/qwen-service && rm -rf pkg harness-lib.sh harness-args.txt && tar xzf -'
```

改了 vLLM 配置（`service.json`、`serve.py` 等）之后，重新执行这一步。

### 7. 启动和停止

整个服务由一个开关控制：启动时才占用 GPU，停止时立刻释放。

在虚拟机上先登记各个组件（只需要一次）：

```bash
sudo systemctl enable agentforge-tunnel agentforge-harness agentforge-backend agentforge-cluster agentforge-keeper.timer
```

之后用总开关启停：

```bash
sudo systemctl start agentforge.target     # 启动：后端、harness、隧道；提交 vLLM 作业，开启交接链
sudo systemctl stop agentforge.target      # 停止：全部停下；取消 vLLM 作业，GPU 立刻释放
systemctl status agentforge.target         # 当前是否在运行
journalctl -u agentforge-cluster -u agentforge-keeper -f   # 集群作业的提交、补次数、取消
```

- **启动时**，`agentforge-cluster` 清除集群上的停止标记 `service-stop`，并立即提交 vLLM 作业。大约 5 到 8 分钟后模型就绪，在此之前页面会提示模型暂不可用。
- **运行中**，`app/service_keeper.py` 每 3 分钟通过 SSH 检查一次：
  - 交接次数 `service-chain.json` 少于 3 时补到 3，每小时的交接链不会用完；
  - 名下没有运行中或排队中的 `qwen-service` 作业时（继任作业没排上、作业启动就失败等），重新提交；
  - 1 小时内最多自动提交 3 次，作业一启动就失败时（例如节点 GPU 被 Slurm 以外的进程占用）不会反复重提；
  - 暂存的仍是带 harness 的版本（有 `pkg/`）时不提交，日志会提示重新执行第 6 步。
- **停止时**，`agentforge-cluster` 设置 `service-stop` 并 `scancel` 名下所有 `qwen-service` 作业（运行中和排队中的都取消）。
- **崩溃重启不会释放 GPU。** 后端或 harness 崩溃后由 systemd 自动重启，不会取消作业；只有停止总开关才会取消。
- **虚拟机重启后默认不会自动启动**，避免在没人用的时候占用 GPU。需要开机自启时执行 `sudo systemctl enable agentforge.target`。

**运行期间的代价：** 交接链会占用你的 2 个作业名额和课程 GPU。要跑评测或训练时，先停掉总开关。

## 检查

| 检查项 | 怎么看 |
|---|---|
| 模型隧道 | `cat /var/lib/agentforge/service-url.txt` 显示 `http://127.0.0.1:<端口>/v1`；日志里有 `tunnel ... -> http://10.x.x.x:8000/v1` |
| harness | `journalctl -u agentforge-harness` 里有 `READY harness service`；`checkpoint cleanup` 那一行说明检查点库已经打开 |
| 入口 | `curl https://tacc-agentforge.pages.dev/api/health` 返回 200；绕过 Pages 直接 `curl https://<虚拟机>.<tailnet>.ts.net/api/health`（不带密钥）返回 401 |
| 完整请求 | 在页面上提一个带预算的问题，流式进度正常显示，最后给出订单汇总 |
| 重启恢复（#8） | 提一个不带预算的问题，页面开始追问时执行 `sudo systemctl restart agentforge-harness agentforge-backend`，再回答，运行能继续完成 |
| 跨交接 | 在虚拟机上另起一个只监听本机、不做登录检查的后端，用它跑 soak（见下） |

跨交接 soak 绕开代理密钥检查，直接在虚拟机上运行：

```bash
sudo -u agentforge /opt/agentforge/venv/bin/python -m app.backend --port 8766 \
  --harness-url http://127.0.0.1:8100 --token-file /var/lib/agentforge/harness-token.txt &
sudo -u agentforge /opt/agentforge/venv/bin/python -m app.cluster_check --backend http://127.0.0.1:8766 \
  --soak-s 1800 --interval-s 30 --out /var/lib/agentforge/logs/handover-soak.json
```

## 故障与限制

- **vLLM 换节点。** `model_tunnel` 每 10 秒读一次 `service-url.txt`。地址一变，它在新的本机端口上建隧道，并改写本地地址文件；旧隧道再保留 180 秒，让旧 vLLM 上还没结束的流式请求跑完。`ModelClient` 每次重试前都会重读地址文件，所以 harness 不用重启。
- **TACC 上没有服务。** 本地地址文件会被撤回，请求很快失败为 `model_unavailable`，页面给出提示。正在等待回答的运行不受影响，模型恢复后可以继续回答。
- **重启后的候选集。** 检查点会保存运行状态，但知识库的 `candidate_set_id` 只存在 harness 进程的内存里（`knowledge/README.md` 第 4 节）。运行在重启后继续时，如果模型引用了旧的候选集，工具会返回可恢复的错误，模型需要重新查询商品；这会多用一次工具调用额度。
- **网关暂时连不上。** 不拆除现有隧道，下一轮轮询再试。
- **限流。** 每个访问者 IP 同时最多 2 个流式请求、每分钟最多 6 次新提问；全局同时最多 8 个（对应 vLLM 的 `max-num-seqs`）。超出时返回 429，页面给出提示。参数在 `app/backend.py` 的 `Limits` 里。IP 由 Pages Function 从 `CF-Connecting-IP` 转发，只有代理密钥正确时才被信任，浏览器伪造不了。
- **公开页面的风险。** 谁都能用，就意味着陌生人也可能占满全局并发，组员要排队。需要限制使用者时，改用下面的登录保护。
- **模型可用性。** 模型能否在线取决于 TACC：能用的节点只有 4 个，被占满时继任作业要排队。页面和 agent 可以一直在线，模型只能做到"大部分时间在线"。
- **延迟。** 拓扑变了，端到端延迟不能和 #11 冻结基线的数字直接比较，迁移后需要在新拓扑上重测。

## 可选：登录保护（Cloudflare Access）

只想让名单里的人使用时：

1. 打开 Zero Trust 控制台（one.dash.cloudflare.com）。第一次进入时要起一个团队名，套餐选 Free（最多 50 个用户）。
2. Access → Applications → Add an application → Self-hosted：
   - 名字填 `agentforge-shop`；
   - 域名添加两条：`tacc-agentforge.pages.dev`，以及 `*.tacc-agentforge.pages.dev`（后者覆盖每次部署的预览地址）；
   - 策略：Action 选 Allow，Include 用 Emails 列出允许的邮箱，登录方式用默认的一次性邮箱验证码。
3. 在虚拟机的配置文件里取消注释 `CF_ACCESS_TEAM_DOMAIN`（`<团队名>.cloudflareaccess.com`）和 `CF_ACCESS_USER_AUD`（应用 Overview 里的 AUD Tag），然后重启 `agentforge-backend`。
4. 之后运行按登录邮箱归属，限流按人计算。

## 有自己域名时：Cloudflare Tunnel

不需要 Tailscale，入口改为 Cloudflare Tunnel，代理凭证改为 Access 服务令牌：

1. 初始化虚拟机时用 `ENTRY=cloudflared sudo -E bash setup-host.sh ...`，安装 cloudflared 而不是 Tailscale。
2. 在 Zero Trust → Networks → Tunnels 新建隧道，在虚拟机上执行控制台给出的 `cloudflared service install <token>`。Public hostname 填 `api.<域名>`，Service 填 `http://127.0.0.1:8765`。
3. 在 Access → Service Auth 新建服务令牌 `pages-proxy`，记下 Client ID 和 Client Secret。
4. 为 `api.<域名>` 新建 Self-hosted 应用，只加一条策略：Action 选 Service Auth，规则选这个服务令牌。把它的 AUD 填到 `CF_ACCESS_API_AUD`，并删掉 `AGENTFORGE_PROXY_SECRET`。
5. Pages 变量改为：`API_ORIGIN=https://api.<域名>`，密钥填 `CF_ACCESS_CLIENT_ID` 和 `CF_ACCESS_CLIENT_SECRET`，删掉 `PROXY_SECRET`。页面可以另绑自定义域名 `shop.<域名>`，并把它加进页面的 Access 应用。
