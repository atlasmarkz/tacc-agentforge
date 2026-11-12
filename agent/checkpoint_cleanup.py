"""按最新检查点清理整个 run；只能在停止接收任务时执行。"""

import argparse
import asyncio
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime

from langgraph.checkpoint.base import BaseCheckpointSaver

from .checkpoints import open_checkpointer


@dataclass(frozen=True)
class RetentionPolicy:
    terminal_retention_s: float = 7 * 86400
    expired_question_grace_s: float = 86400

    def __post_init__(self):
        for value in (self.terminal_retention_s, self.expired_question_grace_s):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Retention durations must be finite and nonnegative.")


@dataclass(frozen=True)
class CleanupDecision:
    thread_id: str
    action: str
    reason: str


def _eligible(saved, policy: RetentionPolicy, now: float) -> tuple[bool, str]:
    state = saved.checkpoint["channel_values"]
    if state.get("state_version") != "2":
        return False, "unknown_state_version"
    if state.get("status") == "waiting_user":
        question = state.get("pending_question")
        expires = question.get("expires_at") if isinstance(question, dict) else None
        if isinstance(expires, bool) or not isinstance(expires, (int, float)) or not math.isfinite(expires):
            return False, "invalid_question_expiry"
        if now >= expires + policy.expired_question_grace_s:
            return True, "expired_question"
        return False, "question_retained"
    if state.get("status") in {"completed", "incomplete", "failed", "cancelled"} and state.get("result"):
        try:
            timestamp = datetime.fromisoformat(saved.checkpoint["ts"].replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                return False, "invalid_checkpoint_time"
        except (KeyError, TypeError, ValueError, AttributeError):
            return False, "invalid_checkpoint_time"
        if now >= timestamp.timestamp() + policy.terminal_retention_s:
            return True, "terminal_retention_elapsed"
        return False, "terminal_retained"
    return False, "unfinished_run"


async def cleanup_checkpoints(
    saver: BaseCheckpointSaver, policy: RetentionPolicy | None = None, *,
    now: float | None = None, dry_run: bool = False,
) -> list[CleanupDecision]:
    """先完成扫描以释放 saver 游标锁，再按 thread 删除全部检查点和 writes。"""
    policy = policy or RetentionPolicy()
    now = time.time() if now is None else now
    if not math.isfinite(now):
        raise ValueError("Cleanup time must be finite.")
    latest = {}
    async for saved in saver.alist(None):
        config = saved.config["configurable"]
        thread_id = config["thread_id"]
        if not thread_id.startswith("agent:") or config.get("checkpoint_ns", ""):
            continue
        previous = latest.get(thread_id)
        if previous is None or saved.checkpoint["id"] > previous.checkpoint["id"]:
            latest[thread_id] = saved
    decisions = []
    for thread_id, saved in sorted(latest.items()):
        eligible, reason = _eligible(saved, policy, now)
        action = "would_delete" if eligible and dry_run else "delete" if eligible else "keep"
        if eligible and not dry_run:
            await saver.adelete_thread(thread_id)
        decisions.append(CleanupDecision(thread_id, action, reason))
    return decisions


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Agent YAML (default: configs/agent.yaml)")
    parser.add_argument("--profile")
    parser.add_argument("--db", required=True, help="existing checkpoint database")
    parser.add_argument("--terminal-retention-s", type=float)
    parser.add_argument("--expired-question-grace-s", type=float)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from .settings import load_runtime_settings
    configured = load_runtime_settings(args.config, args.profile).checkpoint
    policy = RetentionPolicy(
        args.terminal_retention_s if args.terminal_retention_s is not None else configured.terminal_retention_s,
        args.expired_question_grace_s if args.expired_question_grace_s is not None else configured.expired_question_grace_s,
    )
    async with open_checkpointer(args.db, create=False) as saver:
        for decision in await cleanup_checkpoints(saver, policy, dry_run=args.dry_run):
            print(json.dumps(asdict(decision)), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
