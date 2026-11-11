#!/usr/bin/env bash
# To probe two specific nodes: copy this directory twice, write "listen" / "client" into role.txt,
# and add "- nodelist=<node>" to each tuxiv.conf (jobs 28613-28616 did this for 10-0-3-10 and 10-0-1-11).
# Network probe between compute nodes. role.txt: "listen" (HTTP on :8000 for a few minutes) or "client".
set -uo pipefail
: "${TACC_WORKDIR:?}" "${TACC_USERDIR:?}" "${SLURM_JOB_ID:?}"
cd "$TACC_WORKDIR"
ROLE="$(tr -d '[:space:]' < role.txt)"
OUT="$TACC_USERDIR/tacc-agentforge-service/net-probe"
mkdir -p "$OUT"
echo "role=$ROLE node=$(hostname) ip=$(hostname -I 2>/dev/null)"
if [ "$ROLE" = listen ]; then
    hostname > "$OUT/listener-host.txt"
    /usr/bin/python3 -c "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.connect(('10.255.255.255',1));print(s.getsockname()[0])" > "$OUT/listener-ip.txt"
    timeout 600 /usr/bin/python3 -m http.server 8000 --bind 0.0.0.0
else
    for i in $(seq 1 60); do [ -s "$OUT/listener-host.txt" ] && break; sleep 5; done
    T=$(cat "$OUT/listener-host.txt"); IP=$(cat "$OUT/listener-ip.txt"); echo "target=$T ip=$IP"; sleep 5
    getent hosts "$T" || echo "DNS: cannot resolve $T"
    for port in 8000 22; do
        /usr/bin/python3 -c "import socket;s=socket.socket();s.settimeout(5);print('tcp by ip',$port,'connect_ex =',s.connect_ex(('$IP',$port)))"
    done
    /usr/bin/python3 -c "import urllib.request;print('http by ip status',urllib.request.urlopen('http://$IP:8000/',timeout=10).status)" 2>&1 | tail -1
    echo "own /etc/hosts entries for 10-0-*:"; grep -c "10-0-" /etc/hosts; grep "10-0-" /etc/hosts | head -12
fi
