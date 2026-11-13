"""Atomic versioned JSON sessions with durable idempotency receipts."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import closing
from typing import Callable, Protocol

from .models import (ContextError, ContextPatch, HistoryBlock, StorageError, canonical,
                     clone, identifier, nonnegative)


def _validate_state(state):
    """Reject malformed persisted JSON before it reaches runtime dictionary access."""
    try:
        if not isinstance(state, dict) or type(state.get("schema_version")) is not int or state["schema_version"] != 1:
            raise ValueError("Unsupported state schema.")
        for key in ("conversation_id", "snapshot_id"):
            identifier(state[key])
        for key in ("revision", "constraints_revision"):
            nonnegative(state[key])
        if state["constraints_revision"] > state["revision"]:
            raise ValueError("Invalid requirements revision.")
        ContextPatch(set=state["constraints"])
        boundary = state.get("snapshot_boundary", 0)
        nonnegative(boundary)
        if boundary > state["revision"] or not isinstance(state["history"], list):
            raise ValueError("Invalid history or snapshot boundary.")
        seen = set()
        fields = ("block_id", "run_id", "snapshot_id", "constraints_revision", "messages",
                  "kind", "references", "plan_items")
        for index, block in enumerate(state["history"]):
            if not isinstance(block, dict):
                raise ValueError("Invalid history block.")
            HistoryBlock(**{key: block[key] for key in fields})
            identifier(block["epoch"])
            nonnegative(block["sequence"])
            nonnegative(block["memory_revision"])
            if (block["sequence"] != index or block["memory_revision"] >= state["revision"]
                    or block["constraints_revision"] > state["constraints_revision"]
                    or block["block_id"] in seen):
                raise ValueError("Invalid history ordering or revision.")
            seen.add(block["block_id"])
    except (ValueError, TypeError, KeyError) as exc:
        raise StorageError("Context state is malformed or unsupported.") from exc
    return state


def _read_state(encoded, conversation_id):
    try:
        state = _validate_state(json.loads(encoded))
        if state["conversation_id"] != conversation_id:
            raise StorageError("Context state belongs to a different conversation.")
        return state
    except (ValueError, TypeError) as exc:
        if isinstance(exc, StorageError):
            raise
        raise StorageError() from exc


class ContextStore(Protocol):
    def get(self, conversation_id: str) -> dict | None: ...
    def mutate(self, conversation_id: str, operation_id: str, expected_revision: int,
               payload: dict, transition: Callable[[dict | None], dict | None]) -> dict | None: ...


def _apply(state, receipts, operation_id, expected_revision, payload, transition):
    identifier(operation_id)
    nonnegative(expected_revision)
    digest = hashlib.sha256(canonical(payload).encode()).hexdigest()
    if operation_id in receipts:
        receipt = receipts[operation_id]
        if (not isinstance(receipt, dict) or set(receipt) != {"digest", "result", "history_count"}
                or not isinstance(receipt["digest"], str) or len(receipt["digest"]) != 64
                or not isinstance(receipt["result"], dict)
                or type(receipt["history_count"]) is not int or state is None
                or not 0 <= receipt["history_count"] <= len(state["history"])):
            raise StorageError("Context operation receipt is malformed.")
        if receipt["digest"] != digest:
            raise ContextError("idempotency_conflict", "Operation ID already has different content.")
        result = {**clone(receipt["result"]), "history": clone(state["history"][:receipt["history_count"]])}
        if (result.get("conversation_id") != state["conversation_id"]
                or type(result.get("revision")) is not int
                or not 0 < result["revision"] <= state["revision"]
                or type(result.get("constraints_revision")) is not int
                or result["constraints_revision"] > state["constraints_revision"]):
            raise StorageError("Context operation receipt has an invalid identity or revision.")
        return _validate_state(result), None
    if state is None and payload.get("type") == "delete":
        return None, {"digest": digest, "result": None}
    if (state["revision"] if state else 0) != expected_revision:
        raise ContextError("revision_conflict", "Read the latest session before retrying.")
    result = transition(clone(state))
    if result is not None:
        result["revision"] = expected_revision + 1
        _validate_state(result)
        if state is not None and (result["conversation_id"] != state["conversation_id"]
                or result["history"][:len(state["history"])] != state["history"]):
            raise StorageError("Context mutations must preserve conversation identity and history prefixes.")
    result = clone(result)
    # History is append-only until delete. Save a prefix length instead of copying every
    # prior message into every receipt; replay still reconstructs the original result.
    if result is None:
        return result, {"digest": digest, "result": None}
    metadata = {key: value for key, value in result.items() if key != "history"}
    return result, {"digest": digest, "result": metadata, "history_count": len(result["history"])}


class InMemoryStore:
    def __init__(self):
        self._sessions = {}
        self._receipts = {}
        self._lock = threading.RLock()

    def get(self, conversation_id):
        identifier(conversation_id)
        with self._lock:
            return clone(self._sessions.get(conversation_id))

    def mutate(self, conversation_id, operation_id, expected_revision, payload, transition):
        identifier(conversation_id)
        with self._lock:
            result, receipt = _apply(self._sessions.get(conversation_id),
                                    self._receipts.get(conversation_id, {}), operation_id,
                                    expected_revision, payload, transition)
            if receipt is not None:
                if result is None:
                    self._sessions.pop(conversation_id, None)
                    self._receipts.pop(conversation_id, None)
                else:
                    self._sessions[conversation_id] = clone(result)
                    self._receipts.setdefault(conversation_id, {})[operation_id] = clone(receipt)
            return clone(result)


class SQLiteStore:
    """One connection per operation; instances can share a database across threads/processes."""
    def __init__(self, path, *, timeout_s=5.0):
        self.path = str(path)
        if self.path == ":memory:":
            raise ValueError("Use InMemoryStore or a persistent SQLite file.")
        self.timeout_s = timeout_s
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute("CREATE TABLE IF NOT EXISTS context_meta (version INTEGER NOT NULL)")
                connection.execute("INSERT INTO context_meta SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM context_meta)")
                if connection.execute("SELECT version FROM context_meta").fetchall() != [(1,)]:
                    raise StorageError("Unsupported context database version.")
                connection.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, state TEXT NOT NULL)")
                connection.execute("CREATE TABLE IF NOT EXISTS receipts (session_id TEXT, operation_id TEXT, receipt TEXT NOT NULL, PRIMARY KEY(session_id, operation_id))")
        except sqlite3.Error as exc:
            raise StorageError() from exc

    def _connect(self):
        return sqlite3.connect(self.path, timeout=self.timeout_s)

    def get(self, conversation_id):
        identifier(conversation_id)
        connection = None
        try:
            connection = self._connect()
            row = connection.execute("SELECT state FROM sessions WHERE id=?", (conversation_id,)).fetchone()
            return _read_state(row[0], conversation_id) if row else None
        except (sqlite3.Error, ValueError) as exc:
            raise StorageError() from exc
        finally:
            if connection is not None:
                connection.close()

    def mutate(self, conversation_id, operation_id, expected_revision, payload, transition):
        identifier(conversation_id)
        connection = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT state FROM sessions WHERE id=?", (conversation_id,)).fetchone()
            previous = connection.execute("SELECT receipt FROM receipts WHERE session_id=? AND operation_id=?",
                                          (conversation_id, operation_id)).fetchone()
            receipts = {operation_id: json.loads(previous[0])} if previous else {}
            result, receipt = _apply(_read_state(row[0], conversation_id) if row else None, receipts,
                                    operation_id, expected_revision, payload, transition)
            if receipt is not None:
                if result is None:
                    connection.execute("DELETE FROM sessions WHERE id=?", (conversation_id,))
                    connection.execute("DELETE FROM receipts WHERE session_id=?", (conversation_id,))
                else:
                    connection.execute("INSERT OR REPLACE INTO sessions VALUES (?,?)", (conversation_id, canonical(result)))
                    connection.execute("INSERT INTO receipts VALUES (?,?,?)", (conversation_id, operation_id, canonical(receipt)))
            connection.commit()
            return clone(result)
        except ContextError:
            raise
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            raise StorageError() from exc
        finally:
            if connection is not None:
                connection.close()  # rolls back any uncommitted transaction
