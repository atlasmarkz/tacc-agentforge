# 云端 agent 服务:Cloudflare Pages + GCE + TACC vLLM(issue #25)

页面放在 Cloudflare Pages,agent 常驻在 Google Cloud 的一台虚拟机上,TACC 只跑 vLLM。

```
浏览器 ──HTTPS,Access 登录──▶ Cloudflare Pages(app/static/index.html)
                               │ functions/api/*:同源代理，带 Access 服务令牌和用户令牌
                               ▼
                        Cloudflare Tunnel(虚拟机只有出站连接，不开入站端口)
                               ▼
  GCE 虚拟机 ─────────────────────────────────────────────────────────────────
  │ agentforge-backend   app/backend.py   127.0.0.1:8765  校验 Access 令牌、运行归属、限流
  │ agentforge-harness   app/harness_service.py --cloud   127.0.0.1:8100
  │                      AgentRuntime + 知识库快照 + 预算校验器 + SQLite 检查点(本地盘)
  │ agentforge-tunnel    app/model_tunnel.py  读 service-url.txt，维护 ssh -L，写本地地址文件
  ───────────────────────────────────┬────────────────────────────────────────
                                     │ ssh -L，经 gw.turing.ust.hk
                                     ▼
  TACC 服务作业：只跑 vLLM(stage.sh serve)，每小时交接，地址变了隧道自动跟随

  GCS:gs://<bucket>/snapshots/<snapshot_id>.zip  每个快照只写一次
```

## 和集群内部署的区别

| | 集群内(`stage.sh serve --harness`) | 云端(本目录) |
|---|---|---|
| 页面 | 本机后端提供，只有本人能用 | Cloudflare Pages,团队成员登录后使用 |
| harness | 和 vLLM 同一个作业，每小时交接 | 虚拟机上常驻,systemd 崩溃自动重启 |
| 检查点 | 内存;交接或重启时，等待回答的运行会丢 | 本地盘 SQLite;后端或 harness 重启后仍能继续(#8) |
| 运行归属 | 页面生成的会话 ID | Access 登录邮箱，存在 `runs.sqlite`,重启后仍有效 |
| 知识库 | 仓库 zip,每个账户一份 | GCS 按快照 ID 存一份，下载时校验哈希 |
| 作业名额 | 服务作业加继任作业，两个名额都占 | 同样两个名额，作业里少了 harness 和依赖安装 |

集群内部署仍然保留，用作回退。

## 费用

按默认参数部署时，各项费用如下(详见 #25 的讨论):

| 项目 | 费用 |
|---|---|
| 虚拟机 | us-west1 的 e2-micro 和 30 GB 标准盘在永久免费额度内 |
| GCS | 美国区 5 GB 以内免费;快照约 7 MB |
| Cloudflare Pages、Tunnel、Access | 免费;Access 免费档最多 50 个用户 |
| 域名 | 唯一必需的开销，约 10 美元/年 |
| 虚拟机外部 IPv4 | 可能单独计费，开机后看一下账单明细 |

如果想降低到香港网关的延迟，可以用 `ZONE=asia-east2-a MACHINE=e2-small`。这个配置不在免费额度内,Google AI Pro 每月 10 美元的 Cloud 额度可以抵扣。

## 部署步骤

### 0. 前提

- GCP 项目已开通计费，并设好预算提醒。
- 一个托管在 Cloudflare 上的域名。下文假设页面用 `shop.<域名>`,API 用 `api.<域名>`。
- TACC 账户能提交 `qwen-service`(见 `deploy/tacc/service/RUNBOOK.md`)。

### 1. 存储桶和快照

```bash
PROJECT=<gcp 项目> BUCKET=<全局唯一的桶名> bash deploy/gcp/create-vm.sh     # 同时创建存储桶和虚拟机
bash deploy/gcp/publish-snapshot.sh assets/knowledge/pilot-snapshot.zip <桶名>
```

- `publish-snapshot.sh` 用 `--if-generation-match=0` 上传，同名对象已经存在时会失败，所以快照不会被覆盖。脚本最后打印 `seed.url` 和 `seed.sha256`。
- 现有快照 `snapshot_5bcd00723fb0b7589cbe50c8` 迁移后，快照 ID 和 SHA256 都不变。
- 本地验证从 GCS 安装:

  ```bash
  python -m knowledge --data-root /tmp/kb bootstrap --url gs://<桶名>/snapshots/<id>.zip
  python -m knowledge --data-root /tmp/kb doctor
  ```

  `doctor` 应返回 `ready: true`、`verification.valid: true`。
- 发布新快照时，把 `configs/knowledge.json` 的 `snapshot_id`、`seed.sha256` 和 `seed.url` 一起改掉再提交，不再提交 zip。确认所有成员都能从 GCS 安装后，再从仓库删除 `assets/knowledge/pilot-snapshot.zip`。

### 2. 虚拟机

```bash
gcloud compute scp deploy/gcp/setup-host.sh agentforge:~ --zone us-west1-b --project <项目>
gcloud compute ssh agentforge --zone us-west1-b --project <项目> -- \
  sudo bash setup-host.sh https://github.com/<owner>/tacc-agentforge.git feat/cloud-agent-service
```

`setup-host.sh` 依次完成：

1. 安装 Python 3.12、git 和 cloudflared;
2. 创建 2 GB 交换文件(e2-micro 只有 1 GB 内存);
3. 创建 `agentforge` 用户;
4. 克隆仓库并安装依赖;
5. 把 TACC 网关的主机公钥写入 `known_hosts`;
6. 安装三个 systemd 服务。

### 3. TACC 密钥

在虚拟机上专门生成一把密钥，不要复用个人私钥:

```bash
sudo -u agentforge ssh-keygen -t ed25519 -N '' -f /tmp/tacc_key -C agentforge-gce
sudo install -o agentforge -g agentforge -m 600 /tmp/tacc_key /etc/agentforge/tacc_key && sudo rm /tmp/tacc_key
cat /tmp/tacc_key.pub          # 加到 TACC 账户的 ~/.ssh/authorized_keys
```

然后按 `agentforge.env.example` 的说明填写 `/etc/agentforge/agentforge.env`,包括 TACC 账户、`KNOWLEDGE_SEED_URL` 和 Access 的三个值。

**注意:** 是否允许把 TACC 账户的密钥放在云服务器上，需要先和助教确认(见 #25)。

### 4. Cloudflare

在 Zero Trust 控制台完成以下配置:

1. **隧道**:Networks → Tunnels → Create tunnel(cloudflared)。
   - 在虚拟机上执行控制台给出的 `sudo cloudflared service install <token>`。
   - Public hostname 填 `api.<域名>`,Service 填 `http://127.0.0.1:8765`。
2. **服务令牌**:Access → Service Auth → Create service token,记下 Client ID 和 Client Secret。
3. **API 应用**:Access → Applications → Self-hosted,域名为 `api.<域名>`。
   - 只加一条策略:Action 选 **Service Auth**,规则为第 2 步的服务令牌。
   - 记下 Application Audience (AUD) tag,填到 `CF_ACCESS_API_AUD`。
4. **页面应用**:Access → Applications → Self-hosted,域名为 `shop.<域名>`。
   - 策略为 Allow,列出团队成员的邮箱，或者用邮箱域名规则。
   - 记下 AUD tag,填到 `CF_ACCESS_USER_AUD`。
5. **团队域名**:`<team>.cloudflareaccess.com` 填到 `CF_ACCESS_TEAM_DOMAIN`。

### 5. 页面

```bash
npx wrangler login
bash deploy/cloudflare/deploy.sh
```

在 Pages 项目里完成以下设置:

- **Variables and secrets:**
  - 变量 `API_ORIGIN=https://api.<域名>`;
  - 密钥 `CF_ACCESS_CLIENT_ID` 和 `CF_ACCESS_CLIENT_SECRET`。
- **Custom domains:** 添加 `shop.<域名>`。

改完变量后重新部署一次。页面本身就是 `app/static/index.html`,没有修改。它请求的相对路径 `/api/*` 由 `deploy/cloudflare/functions/api/[[path]].js` 转发，所以没有跨域请求，也不需要第三方 Cookie。

### 6. 启动

```bash
sudo systemctl enable --now agentforge-tunnel agentforge-harness agentforge-backend
journalctl -u agentforge-tunnel -u agentforge-harness -u agentforge-backend -f
```

在 TACC 上只启动 vLLM:

```bash
bash deploy/tacc/service/qwen-service/stage.sh serve          # 不带 --harness
```

要连续运行时，按 RUNBOOK 第 5 节开启交接。

## 检查

| 检查项 | 怎么看 |
|---|---|
| 模型隧道 | `cat /var/lib/agentforge/service-url.txt` 显示 `http://127.0.0.1:<端口>/v1`;日志里有 `tunnel ... -> http://10.x.x.x:8000/v1` |
| harness | `journalctl -u agentforge-harness` 里有 `READY harness service`;`checkpoint cleanup` 那一行说明检查点库已经打开 |
| 登录 | 未登录时打开 `https://shop.<域名>` 会跳到 Access 登录页;直接 `curl https://api.<域名>/api/health` 不带服务令牌，会被 Access 拦下 |
| 完整请求 | 登录后在页面上提一个带预算的问题，流式进度正常显示，最后给出订单汇总 |
| 重启恢复(#8) | 提一个不带预算的问题，页面开始追问时执行 `sudo systemctl restart agentforge-harness agentforge-backend`,再回答，运行能继续完成 |
| 跨交接 | 在虚拟机上另起一个只监听本机、不做登录检查的后端，用它跑 soak(见下) |

跨交接 soak 绕开 Access,直接在虚拟机上运行:

```bash
sudo -u agentforge /opt/agentforge/venv/bin/python -m app.backend --port 8766 \
  --harness-url http://127.0.0.1:8100 --token-file /var/lib/agentforge/harness-token.txt &
sudo -u agentforge /opt/agentforge/venv/bin/python -m app.cluster_check --backend http://127.0.0.1:8766 \
  --soak-s 1800 --interval-s 30 --out /var/lib/agentforge/logs/handover-soak.json
```

## 故障与限制

- **vLLM 换节点。** `model_tunnel` 每 10 秒读一次 `service-url.txt`。地址一变，它在新的本机端口上建隧道，并改写本地地址文件;旧隧道再保留 180 秒，让旧 vLLM 上还没结束的流式请求跑完。`ModelClient` 每次重试前都会重读地址文件，所以 harness 不用重启。
- **TACC 上没有服务。** 本地地址文件会被撤回，请求很快失败为 `model_unavailable`,页面给出提示。正在等待回答的运行不受影响，模型恢复后可以继续回答。
- **重启后的候选集。** 检查点会保存运行状态，但知识库的 `candidate_set_id` 只存在 harness 进程的内存里(`knowledge/README.md` 第 4 节)。运行在重启后继续时，如果模型引用了旧的候选集，工具会返回可恢复的错误，模型需要重新查询商品;这会多用一次工具调用额度。
- **网关暂时连不上。** 不拆除现有隧道，下一轮轮询再试。
- **限流。** 每个用户同时最多 2 个流式请求、每分钟最多 6 次新提问;全局同时最多 8 个(对应 vLLM 的 `max-num-seqs`)。超出时返回 429,页面给出提示。参数在 `app/backend.py` 的 `Limits` 里。
- **模型可用性。** 模型能否在线取决于 TACC:能用的节点只有 4 个，被占满时继任作业要排队。页面和 agent 可以一直在线，模型只能做到"大部分时间在线"。
- **延迟。** 拓扑变了，端到端延迟不能和 #11 冻结基线的数字直接比较，迁移后需要在新拓扑上重测。
