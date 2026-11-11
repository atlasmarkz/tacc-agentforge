"""宿主使用的 SQLite 检查点入口；连接和独占租约覆盖整个服务生命周期。"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


def _filesystem_type(path: Path) -> str | None:
    """在 Linux 上识别最深的挂载点；其他平台由部署者确认存储支持情况。"""
    mounts = Path("/proc/self/mountinfo")
    if not mounts.exists():
        return None
    found = []
    for line in mounts.read_text().splitlines():
        fields, filesystem = line.split(" - ", 1)
        mount = fields.split()[4]
        for escaped, value in ((r"\040", " "), (r"\011", "\t"), (r"\134", "\\")):
            mount = mount.replace(escaped, value)
        root = Path(mount)
        if path == root or root in path.parents:
            found.append((len(root.parts), filesystem.split()[0]))
    return max(found, key=lambda entry: entry[0])[1] if found else None


@asynccontextmanager
async def open_checkpointer(path: Path | str, *, create: bool = True) -> AsyncIterator[AsyncSqliteSaver]:
    """打开完整的异步 saver，拒绝已知网络盘及并行宿主；错误原样传播。

    面向 Linux/macOS 的单宿主部署。WAL 不能放在网络文件系统上；此检测
    不是文件系统支持认证，部署者仍须确认持久卷、锁和同步语义。
    """
    import fcntl

    path = Path(path).expanduser().resolve()
    filesystem = _filesystem_type(path)
    if filesystem in {
        "nfs", "nfs4", "cifs", "smbfs", "ceph", "fuse.ceph", "fuse.sshfs",
        "glusterfs", "fuse.glusterfs", "lustre", "gpfs", "panfs", "beegfs", "afs", "9p",
    }:
        raise ValueError(f"SQLite WAL does not support this shared filesystem: {filesystem}. "
                         "Use a supported persistent volume or inject another durable saver.")
    if not create and not path.is_file():
        raise FileNotFoundError(f"Checkpoint database does not exist: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    # 不删除锁文件：删除会让后续进程锁住另一个 inode，从而绕过现有租约。
    with path.with_name(path.name + ".lock").open("a+b") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Checkpoint store is in use by another host or cleanup process.") from exc
        try:
            async with aiosqlite.connect(path.as_uri() + ("?mode=rwc" if create else "?mode=rw"),
                                         uri=True, timeout=5) as conn:
                saver = AsyncSqliteSaver(conn)
                await saver.setup()
                async with conn.execute("PRAGMA synchronous=FULL"):
                    pass
                yield saver
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)
