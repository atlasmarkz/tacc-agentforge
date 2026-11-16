"""Keeps an SSH tunnel from this host to the vLLM service on TACC and publishes it as a local address file
(issue #25: the harness runs off-cluster, TACC only serves the model).

  this host: ModelClient(address_file=<local file>) --> 127.0.0.1:<port> --ssh -L via the gateway--> vLLM

The vLLM job publishes its address in USERDIR/tacc-agentforge-service/service-url.txt and moves to another node
every hour (RUNBOOK section 5); `serve.py` may also pick another port when 8000 is taken. So the tunnel target
is never fixed: every `poll_s` this process reads the published address over SSH and

- opens a tunnel on a fresh local port when the address changed or the ssh process died, then rewrites the local
  address file (ModelClient re-reads it before every attempt, so callers never restart);
- keeps the previous tunnel open for `grace_s` after a move, so streams already running on the old vLLM (which
  drains during a handover) are not cut;
- withdraws the local file when the cluster has no published address (ModelClient then fails fast with
  model_unavailable instead of waiting on a dead port);
- leaves everything as it is when the gateway itself cannot be reached (a transient SSH failure is not a
  reason to drop a working tunnel).

  python -m app.model_tunnel --local-file /var/lib/agentforge/service-url.txt

Environment: TACC_SSH (user@gw.turing.ust.hk), TACC_SSH_PORT, TACC_SSH_KEY, TACC_REMOTE_USERDIR.
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

NO_ADDRESS = "__none__"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def free_port() -> int:
    # A fresh port for every tunnel: a stale ssh left on a fixed port kept forwarding to a stopped service
    # while the new tunnel failed to bind (job 28963 soak, app/README.md).
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_open(port: int, timeout: float = 2) -> bool:
    try:
        socket.create_connection(("127.0.0.1", port), timeout=timeout).close()
        return True
    except OSError:
        return False


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class Unreachable(Exception):
    """The gateway could not be asked (network, auth); says nothing about the service itself."""


@dataclass
class SshConfig:
    target: str
    port: int
    key: Path
    remote_userdir: str

    def base(self) -> list[str]:
        return ["ssh", "-p", str(self.port), "-i", str(self.key), "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]

    @property
    def remote_file(self) -> str:
        return self.remote_userdir.rstrip("/") + "/tacc-agentforge-service/service-url.txt"

    def read_address(self) -> str | None:
        """The published vLLM address, None when nothing is published; Unreachable when SSH failed."""
        command = f"test -s {self.remote_file} && cat {self.remote_file} || echo {NO_ADDRESS}"
        try:
            done = subprocess.run(self.base() + [self.target, command], capture_output=True, text=True, timeout=40)
        except subprocess.TimeoutExpired as exc:
            raise Unreachable("ssh timed out") from exc
        if done.returncode != 0:
            raise Unreachable(f"ssh exit {done.returncode}: {done.stderr.strip()[-200:]}")
        lines = [line.strip() for line in done.stdout.splitlines() if line.strip()]
        if not lines or lines[-1] == NO_ADDRESS:
            return None
        return lines[-1]

    def open_forward(self, local_port: int, host: str, port: int) -> subprocess.Popen:
        return subprocess.Popen(self.base() + ["-N", "-o", "ExitOnForwardFailure=yes",
                                               "-L", f"127.0.0.1:{local_port}:{host}:{port}", self.target],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class ModelTunnel:
    """One poll step at a time (`step`), so tests drive it without threads or SSH.

    `read_address()` -> address | None, raises Unreachable; `open_forward(local_port, host, port)` -> process
    with poll()/terminate(); `is_open(port)` -> bool.
    """

    def __init__(self, local_file: Path, read_address, open_forward, *, is_open=port_open, grace_s: float = 180,
                 connect_wait_s: float = 15, clock=time.monotonic, sleep=time.sleep):
        self.local_file, self.read_address, self.open_forward = local_file, read_address, open_forward
        self.is_open, self.grace_s, self.connect_wait_s = is_open, grace_s, connect_wait_s
        self.clock, self.sleep = clock, sleep
        self.remote: str | None = None          # published address the current tunnel points to
        self.local_port: int | None = None
        self.proc = None
        self.retiring: list[tuple[float, object]] = []    # (close at, process) of tunnels to a previous address

    def _alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and self.is_open(self.local_port)

    def _withdraw(self, why: str) -> None:
        if self.local_file.exists():
            self.local_file.unlink()
            log(f"withdrew {self.local_file}: {why}")

    def _retire_current(self) -> None:
        if self.proc is not None:
            self.retiring.append((self.clock() + self.grace_s, self.proc))
        self.proc, self.local_port = None, None

    def _close_retired(self, force: bool = False) -> None:
        keep = []
        for deadline, proc in self.retiring:
            if force or self.clock() >= deadline or proc.poll() is not None:
                if proc.poll() is None:
                    proc.terminate()
            else:
                keep.append((deadline, proc))
        self.retiring = keep

    def step(self) -> str:
        """One poll: returns what happened ("unreachable", "no_service", "kept", "opened", "failed")."""
        self._close_retired()
        try:
            address = self.read_address()
        except Unreachable as exc:
            log(f"gateway unreachable, keeping the current tunnel: {exc}")
            return "unreachable"
        if address is None:
            self._withdraw("no vLLM address is published")
            self._retire_current()
            self.remote = None
            return "no_service"
        if address == self.remote and self._alive():
            return "kept"
        u = urlsplit(address)
        if u.scheme != "http" or not u.hostname or not u.port:
            log(f"ignoring malformed published address {address!r}")
            return "failed"
        moved = address != self.remote
        if moved:
            self._retire_current()               # the old vLLM may still be finishing streams
        elif self.proc is not None:              # same address, dead tunnel
            if self.proc.poll() is None:
                self.proc.terminate()
            self.proc = None
        port = free_port()
        proc = self.open_forward(port, u.hostname, u.port)
        deadline = self.clock() + self.connect_wait_s
        while not self.is_open(port):
            if proc.poll() is not None or self.clock() >= deadline:
                if proc.poll() is None:
                    proc.terminate()
                self._withdraw(f"could not open a tunnel to {address}")
                self.remote = None
                return "failed"
            self.sleep(0.25)
        self.proc, self.local_port, self.remote = proc, port, address
        path = u.path.rstrip("/") or "/v1"
        atomic_write(self.local_file, f"http://127.0.0.1:{port}{path}\n")
        log(f"tunnel 127.0.0.1:{port} -> {address} ({'moved' if moved else 'reopened'})")
        return "opened"

    def close(self) -> None:
        self._withdraw("tunnel stopped")
        self._retire_current()
        self._close_retired(force=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local-file", type=Path, required=True, help="address file the harness's ModelClient reads")
    ap.add_argument("--poll-s", type=float, default=10)
    ap.add_argument("--grace-s", type=float, default=180, help="keep a moved-from tunnel open this long")
    a = ap.parse_args()
    user = os.environ.get("TACC_SSH")
    if not user or not os.environ.get("TACC_REMOTE_USERDIR"):
        raise SystemExit("set TACC_SSH (user@gateway) and TACC_REMOTE_USERDIR")
    ssh = SshConfig(user, int(os.environ.get("TACC_SSH_PORT", "30041")),
                    Path(os.environ.get("TACC_SSH_KEY", Path.home() / ".ssh" / "id_rsa_tacc")).expanduser(),
                    os.environ["TACC_REMOTE_USERDIR"])
    tunnel = ModelTunnel(a.local_file, ssh.read_address, ssh.open_forward, grace_s=a.grace_s)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    log(f"following {ssh.target}:{ssh.remote_file} -> {a.local_file}")
    try:
        while not stop.is_set():
            try:
                tunnel.step()
            except Exception as exc:                      # never let the loop die; systemd would restart it anyway
                log(f"poll failed: {type(exc).__name__}: {exc}")
            stop.wait(a.poll_s)
    except KeyboardInterrupt:
        pass
    finally:
        tunnel.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
