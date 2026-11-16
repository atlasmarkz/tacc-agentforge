#!/usr/bin/env bash
# Prepare the GCE VM (Ubuntu 24.04) for the agent service (issue #25). Run as root on the VM; safe to re-run.
#
#   sudo bash setup-host.sh https://github.com/<owner>/tacc-agentforge.git feat/cloud-agent-service
#
# Afterwards (once, by hand; see deploy/gcp/README.md):
#   1. /etc/agentforge/agentforge.env   from agentforge.env.example
#   2. /etc/agentforge/tacc_key         a dedicated SSH key registered with the TACC account (mode 600)
#   3. free setup: tailscale up; tailscale funnel --bg 8765
#      (own domain, ENTRY=cloudflared: cloudflared service install <tunnel token>)
#   4. systemctl enable --now agentforge-tunnel agentforge-harness agentforge-backend
set -euo pipefail
REPO_URL="${1:?repository URL}"
BRANCH="${2:-main}"
APP=/opt/agentforge
HOME_DIR=/var/lib/agentforge

apt-get update -q
apt-get install -y -q python3.12 python3.12-venv git openssh-client curl

# e2-micro has 1 GB of memory: a swap file keeps pip and a busy moment from being killed.
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  grep -q /swapfile /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

id agentforge >/dev/null 2>&1 || useradd --system --create-home --home-dir /home/agentforge --shell /usr/sbin/nologin agentforge
install -d -o agentforge -g agentforge -m 750 "$APP" "$HOME_DIR" "$HOME_DIR/logs" "$HOME_DIR/checkpoints"
install -d -o root -g agentforge -m 750 /etc/agentforge

if [ -d "$APP/repo/.git" ]; then
  sudo -u agentforge git -C "$APP/repo" fetch -q origin "$BRANCH"
  sudo -u agentforge git -C "$APP/repo" checkout -q -B "$BRANCH" "origin/$BRANCH"
else
  sudo -u agentforge git clone -q --branch "$BRANCH" "$REPO_URL" "$APP/repo"
fi
[ -d "$APP/venv" ] || sudo -u agentforge python3.12 -m venv "$APP/venv"
sudo -u agentforge "$APP/venv/bin/pip" install -q -r "$APP/repo/agent/requirements.lock" \
  -r "$APP/repo/deploy/gcp/requirements-cloud.txt"

# The TACC gateway's host key, so BatchMode ssh does not stop at the unknown-host prompt.
install -d -o agentforge -g agentforge -m 700 /home/agentforge/.ssh
ssh-keyscan -p 30041 gw.turing.ust.hk 2>/dev/null >> /home/agentforge/.ssh/known_hosts
sort -u -o /home/agentforge/.ssh/known_hosts /home/agentforge/.ssh/known_hosts
chown agentforge:agentforge /home/agentforge/.ssh/known_hosts

# Public entry. Free setup: Tailscale Funnel (an https://<vm>.<tailnet>.ts.net address, outbound only).
# Own domain: set ENTRY=cloudflared to install cloudflared instead.
if [ "${ENTRY:-tailscale}" = tailscale ] && ! command -v tailscale >/dev/null; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi
if [ "${ENTRY:-tailscale}" = cloudflared ] && ! command -v cloudflared >/dev/null; then
  mkdir -p --mode=0755 /usr/share/keyrings
  curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg -o /usr/share/keyrings/cloudflare-main.gpg
  echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' \
    > /etc/apt/sources.list.d/cloudflared.list
  apt-get update -q && apt-get install -y -q cloudflared
fi

install -m 644 "$APP/repo/deploy/gcp/systemd/"agentforge-*.service "$APP/repo/deploy/gcp/systemd/"agentforge-*.timer \
  "$APP/repo/deploy/gcp/systemd/agentforge.target" /etc/systemd/system/
systemctl daemon-reload

[ -f /etc/agentforge/agentforge.env ] || install -o root -g agentforge -m 640 \
  "$APP/repo/deploy/gcp/agentforge.env.example" /etc/agentforge/agentforge.env
echo "Host ready. Fill /etc/agentforge/agentforge.env, add /etc/agentforge/tacc_key, set up the public entry"
echo "(free setup: tailscale up, then tailscale funnel --bg 8765), then:"
echo "  systemctl enable agentforge-tunnel agentforge-harness agentforge-backend agentforge-cluster agentforge-keeper.timer"
echo "  systemctl start agentforge.target     # submits the vLLM job; systemctl stop agentforge.target frees the GPUs"
