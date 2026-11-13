"""验证检查点跨强杀恢复、保留策略、存储错误及独占宿主边界。"""

import asyncio
import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.serde.types import INTERRUPT
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import AgentDependencies, AgentRuntime, RunPolicy, RunStateError
from agent.checkpoint_cleanup import RetentionPolicy, cleanup_checkpoints
from agent.checkpoints import open_checkpointer
from agent.examples.mock_run import MockModel, MockPlanValidator, demo_request, demo_tools

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="SQLite host lease requires POSIX.")


def config(run_id):
    return {"configurable": {"thread_id": f"agent:{run_id}"}}


def runtime(saver, policy=None):
    return AgentRuntime(AgentDependencies(MockModel(), demo_tools(), MockPlanValidator()),
                        policy, checkpointer=saver)


def missing_budget(run_id):
    return demo_request(run_id).model_copy(update={"constraints": {"currency": "USD"}})


@pytest.mark.parametrize("original_limit,explain", [(8, False), (2, False), (8, True)])
def test_sigkill_then_fresh_process_resumes_without_replaying_search(tmp_path, original_limit, explain):
    async def scenario():
        async def launch(phase, *extra):
            return await asyncio.create_subprocess_exec(
                sys.executable, "-m", "agent.tests.persistence_worker", phase,
                "--db", str(tmp_path / "checkpoints/agent.sqlite"), "--run-id", "restart-run", *extra,
                cwd=ROOT, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )

        first = await launch("start", "--max-model-calls", str(original_limit), *(["--explain"] if explain else []))
        try:
            line = await asyncio.wait_for(first.stdout.readline(), timeout=20)
            if not line:
                pytest.fail((await first.stderr.read()).decode())
            before = json.loads(line)
            waiting = before["events"][-1]["result"]
            assert waiting["status"] == "waiting_user"
            assert before["local_tool_calls"] == 1
            assert first.returncode is None
        finally:
            if first.returncode is None:
                first.kill()
            await asyncio.wait_for(first.wait(), timeout=10)
        assert first.returncode == -9

        second = await launch("resume", "--question-id", waiting["pending_question"]["question_id"],
                              "--max-model-calls", "100")
        try:
            stdout, stderr = await asyncio.wait_for(second.communicate(), timeout=20)
        finally:
            if second.returncode is None:
                second.kill()
                await second.wait()
        assert second.returncode == 0, stderr.decode()
        after = json.loads(stdout)
        result = after["events"][-1]["result"]
        assert before["pid"] != after["pid"]
        assert result["run_id"] == waiting["run_id"]
        assert result["conversation_id"] == waiting["conversation_id"]
        assert result["snapshot_id"] == waiting["snapshot_id"]
        assert after["policy"]["max_model_calls"] == original_limit
        assert after["policy"] == before["policy"]
        assert after["local_tool_calls"] == 0
        assert result["usage"]["tool_calls"] == 1
        assert result["usage"]["elapsed_s"] >= waiting["usage"]["elapsed_s"]
        events = before["events"] + after["events"]
        assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
        assert after["events"][0]["type"] == "run_resumed"
        if original_limit == 8:
            assert result["status"] == "completed"
            assert result["usage"]["model_calls"] == 3 and after["local_model_calls"] == 1 + int(explain)
            assert result["usage"]["explain_model_calls"] == int(explain)
            if explain:
                assert result["explanation"]["verified"] and not result["explanation"]["fallback"]
            assert result["verified_plan"]["remaining_funds"] == "50.01"
            assert any('"observation"' in m["content"] for m in after["messages"])
            assert any('"clarification_answer"' in m["content"] for m in after["messages"])
        else:
            assert result["stop_reason"] == "model_call_limit"
            assert result["usage"]["model_calls"] == 2 and after["local_model_calls"] == 0
        async with open_checkpointer(tmp_path / "checkpoints/agent.sqlite", create=False) as saver:
            saved = await saver.aget_tuple(config("restart-run"))
            assert saved.checkpoint["channel_values"].get("explanation") == result["explanation"]
    asyncio.run(scenario())


def test_resume_rejections_preserve_checkpoint_across_reopen(tmp_path):
    async def scenario():
        path = tmp_path / "agent.sqlite"
        async with open_checkpointer(path) as saver:
            waiting = await runtime(saver).run(missing_budget("identity"))
            saved = await saver.aget_tuple(config("identity"))
            assert any(write[1] == INTERRUPT for write in saved.pending_writes)
        async with open_checkpointer(path, create=False) as saver:
            agent = runtime(saver)
            for run_id, question_id, code in [
                ("missing", waiting.pending_question.question_id, "run_not_found"),
                ("identity", "wrong-question", "question_mismatch"),
            ]:
                with pytest.raises(RunStateError) as exc:
                    await agent.resume(run_id, question_id, {"purchase_cap": "100.00"})
                assert exc.value.code == code
            with pytest.raises(RunStateError) as exc:
                await agent.run(missing_budget("identity"))
            assert exc.value.code == "run_exists"
            assert (await saver.aget_tuple(config("identity"))).checkpoint["id"] == saved.checkpoint["id"]
            result = await agent.resume("identity", waiting.pending_question.question_id, {"purchase_cap": "100.00"})
            assert result.status == "completed"
        async with open_checkpointer(path, create=False) as saver:
            with pytest.raises(RunStateError) as exc:
                await runtime(saver).resume("identity", waiting.pending_question.question_id, {"purchase_cap": "100.00"})
            assert exc.value.code == "not_waiting"
    asyncio.run(scenario())


def test_expired_question_is_retained_during_grace_and_resume_reports_expiry(tmp_path, monkeypatch):
    async def scenario():
        path = tmp_path / "agent.sqlite"
        async with open_checkpointer(path) as saver:
            waiting = await runtime(saver).run(missing_budget("expired"))
        now = waiting.pending_question.expires_at + 1
        async with open_checkpointer(path, create=False) as saver:
            decisions = await cleanup_checkpoints(saver, now=now)
            assert decisions[0].action == "keep"
            monkeypatch.setattr("agent.nodes.time.time", lambda: now)
            result = await runtime(saver).resume("expired", waiting.pending_question.question_id,
                                                  {"purchase_cap": "100.00"})
            assert result.status == "incomplete" and result.stop_reason == "question_expired"
            assert result.usage.model_calls == 1
    asyncio.run(scenario())


def test_cleanup_uses_latest_state_and_deletes_entire_run_with_writes(tmp_path):
    async def scenario():
        now = time.time()
        async with open_checkpointer(tmp_path / "agent.sqlite") as saver:
            waiting = await runtime(saver).run(missing_budget("expired"))
            await runtime(saver).run(demo_request("terminal"))

            async def put(thread_id, state):
                checkpoint = empty_checkpoint()
                checkpoint["ts"] = datetime.fromtimestamp(now - 100, timezone.utc).isoformat()
                checkpoint["channel_values"] = state
                await saver.aput({"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
                                 checkpoint, {"source": "loop", "step": 1, "parents": {}}, {})

            # 旧问题已过期，但最新检查点的问题仍有效，不能按历史检查点误删。
            await put("agent:reasked", {"state_version": "2", "status": "waiting_user",
                                      "pending_question": {"expires_at": 0}})
            await put("agent:reasked", {"state_version": "2", "status": "waiting_user",
                                      "pending_question": {"expires_at": now + 1e6}})
            await put("agent:unknown", {"state_version": "99", "status": "completed", "result": {"status": "completed"}})
            await put("agent:unfinished", {"state_version": "2", "status": None})
            await put("other:unrelated", {"state_version": "2", "status": "completed", "result": {"status": "completed"}})
            policy = RetentionPolicy(terminal_retention_s=10, expired_question_grace_s=10)
            early = await cleanup_checkpoints(saver, policy, now=now, dry_run=True)
            assert all(decision.action == "keep" for decision in early)
            cleanup_time = waiting.pending_question.expires_at + 11
            preview = await cleanup_checkpoints(saver, policy, now=cleanup_time, dry_run=True)
            assert {d.thread_id for d in preview if d.action == "would_delete"} == {"agent:expired", "agent:terminal"}
            assert await saver.aget_tuple(config("expired")) is not None
            assert "other:unrelated" not in {d.thread_id for d in preview}
            decisions = await cleanup_checkpoints(saver, policy, now=cleanup_time)
            assert {d.thread_id for d in decisions if d.action == "delete"} == {"agent:expired", "agent:terminal"}
            assert await saver.aget_tuple(config("reasked")) is not None
            assert await saver.aget_tuple(config("unknown")) is not None
            assert await saver.aget_tuple(config("unfinished")) is not None
            for run_id in ("expired", "terminal"):
                assert [entry async for entry in saver.alist(config(run_id))] == []
                async with saver.conn.execute("SELECT COUNT(*) FROM writes WHERE thread_id = ?", (f"agent:{run_id}",)) as cursor:
                    assert (await cursor.fetchone())[0] == 0
            with pytest.raises(RunStateError) as exc:
                await runtime(saver).resume("expired", waiting.pending_question.question_id, {})
            assert exc.value.code == "run_not_found"
    asyncio.run(scenario())


def test_store_lease_blocks_other_hosts_and_is_released_after_failure(tmp_path):
    async def scenario():
        path = tmp_path / "agent.sqlite"
        with pytest.raises(RuntimeError, match="deliberate failure"):
            async with open_checkpointer(path):
                with pytest.raises(RuntimeError, match="in use"):
                    async with open_checkpointer(path):
                        pytest.fail("concurrent store open should fail")
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "agent.checkpoint_cleanup", "--db", str(path), "--dry-run",
                    cwd=ROOT, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20)
                assert process.returncode != 0 and b"in use" in stderr
                raise RuntimeError("deliberate failure")
        async with open_checkpointer(path, create=False):
            pass
    asyncio.run(scenario())


def test_missing_corrupt_and_unsupported_stores_fail_without_memory_fallback(tmp_path, monkeypatch):
    async def scenario():
        path = tmp_path / "agent.sqlite"
        with pytest.raises(FileNotFoundError):
            async with open_checkpointer(path, create=False):
                pytest.fail("should not create missing database")
        assert not path.exists()
        path.write_text("not a database")
        with pytest.raises(sqlite3.DatabaseError):
            async with open_checkpointer(path):
                pytest.fail("corrupt database should fail")
        monkeypatch.setattr("agent.checkpoints._filesystem_type", lambda path: "nfs4")
        with pytest.raises(ValueError, match="shared filesystem"):
            async with open_checkpointer(tmp_path / "remote.sqlite"):
                pytest.fail("SQLite WAL should reject NFS")
        assert not (tmp_path / "remote.sqlite").exists()
    asyncio.run(scenario())


def test_interrupt_storage_failure_never_returns_waiting_user(tmp_path, monkeypatch):
    original = AsyncSqliteSaver.aput_writes

    async def fail_interrupt(self, config, writes, task_id, task_path=""):
        if any(channel == INTERRUPT for channel, _ in writes):
            raise sqlite3.OperationalError("simulated storage write failure")
        return await original(self, config, writes, task_id, task_path)

    async def scenario():
        async with open_checkpointer(tmp_path / "agent.sqlite") as saver:
            monkeypatch.setattr(AsyncSqliteSaver, "aput_writes", fail_interrupt)
            events = []
            with pytest.raises(sqlite3.OperationalError, match="storage write failure"):
                async for event in runtime(saver).astream(missing_budget("write-failure")):
                    events.append(event)
            assert all(event.type != "run_waiting" for event in events)
    asyncio.run(scenario())


def test_example_and_cleanup_commands(tmp_path):
    path = tmp_path / "directory with spaces" / "agent.sqlite"

    def example(*args):
        result = subprocess.run([sys.executable, "-m", "agent.examples.persistent_resume", "--db", str(path), *args],
                                cwd=ROOT, capture_output=True, text=True, check=True, timeout=20)
        return json.loads(result.stdout)

    waiting = example("start", "--run-id", "cli-example")
    assert waiting["status"] == "waiting_user"
    completed = example("resume", "--run-id", "cli-example",
                        "--question-id", waiting["pending_question"]["question_id"],
                        "--answer", '{"purchase_cap":"100.00"}')
    assert completed["status"] == "completed" and completed["usage"]["model_calls"] == 3
    command = [sys.executable, "-m", "agent.checkpoint_cleanup", "--db", str(path), "--terminal-retention-s", "0"]
    preview = subprocess.run([*command, "--dry-run"], cwd=ROOT, capture_output=True, text=True, check=True, timeout=20)
    assert json.loads(preview.stdout)["action"] == "would_delete"
    deleted = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True, timeout=20)
    assert json.loads(deleted.stdout)["action"] == "delete"

    async def check_deleted():
        async with open_checkpointer(path, create=False) as saver:
            assert await saver.aget_tuple(config("cli-example")) is None
    asyncio.run(check_deleted())
