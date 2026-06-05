"""Conversation handle routing — the stateful surface's routing layer (conv-1b).

The principle (docs/conversations-api-design.md): **woollama routes conversation
*handles*; the backends own the *state*.** This module is that thin routing
layer — an in-memory handle table mapping woollama's opaque `conv_<hex>` ids to
the backend that owns the bytes plus that backend's native id, and the backend
adapters themselves. woollama never stores the conversation transcript.

Backends:
- `claude-resume` (conv-1b): `claude --resume <sid> -p`, the non-interactive
  path — verified not to hang nested. The native Claude session owns the bytes;
  woollama holds only the handle → session_id mapping.
- `stored` (conv-5): for models with NO native session (ollama, recipes, cloud
  providers), woollama itself owns the conversation — it persists the visible
  transcript in a local duckdb file (`StoredStore`) and replays it as context
  each turn. The one place woollama legitimately stores conversation bytes
  (there's no backend underneath to defer to).

Limitations (documented, not hidden):
- **Handle table is in-memory; truth for `stored` is duckdb.** `claude-resume`
  mappings (`conv_id → claude session_id`) are process-lifetime — a restart
  loses them. `stored` conversations survive a restart (rehydrated from duckdb
  into the in-memory working set at startup); `response_ids` are NOT persisted,
  so `previous_response_id` chaining is within-process only — attach-by-
  `conversation` is the durable path.
- **Chaining, not forking** — `claude --resume` continues from the session TIP
  and reports the SAME session_id, so `previous_response_id` attaches to a
  conversation and continues it; claude has no fork-from-earlier-turn primitive.
- **One writer per conversation** — turns on a given conversation serialize on a
  per-conversation lock (the backend session is single-threaded). The `stored`
  backend additionally serializes all duckdb access on a single connection lock.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field

import duckdb

from . import claude_code, responses


def _now() -> int:
    return int(time.time())


@dataclass
class Conversation:
    """A routable handle. `native_id` is the backend's own id (a claude
    session_id), None until the first turn creates the backing session."""
    id: str
    backend: str
    model: str
    native_id: str | None = None      # backend's own id (e.g. a claude session_id)
    workdir: str | None = None        # stable cwd for the backing session (resume scoping)
    response_ids: list[str] = field(default_factory=list)
    status: str = "idle"              # idle | busy | awaiting_input | dead
    title: str | None = None
    metadata: dict = field(default_factory=dict)
    created_at: int = field(default_factory=_now)
    updated_at: int = field(default_factory=_now)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ConversationStore:
    """In-memory handle table. Maps `conv_id → Conversation` and `resp_id →
    conv_id` (so `previous_response_id` resolves to its conversation)."""

    def __init__(self) -> None:
        self._convs: dict[str, Conversation] = {}
        self._resp_to_conv: dict[str, str] = {}

    def create(self, backend: str, model: str, *, metadata: dict | None = None,
               title: str | None = None) -> Conversation:
        conv = Conversation(id=responses.new_id("conv"), backend=backend,
                            model=model, metadata=metadata or {}, title=title)
        self._convs[conv.id] = conv
        return conv

    def get(self, conv_id: str) -> Conversation | None:
        return self._convs.get(conv_id)

    def list(self) -> list[Conversation]:
        return list(self._convs.values())

    def add(self, conv: Conversation) -> None:
        """Insert a fully-formed handle (e.g. one rehydrated from the `stored`
        backend at startup). Restores its response_id back-references too."""
        self._convs[conv.id] = conv
        for rid in conv.response_ids:
            self._resp_to_conv[rid] = conv.id

    def by_response(self, response_id: str) -> Conversation | None:
        cid = self._resp_to_conv.get(response_id)
        return self._convs.get(cid) if cid else None

    def record_response(self, conv: Conversation, response_id: str) -> None:
        conv.response_ids.append(response_id)
        conv.updated_at = _now()
        self._resp_to_conv[response_id] = conv.id

    def remove(self, conv_id: str) -> Conversation | None:
        """Drop the handle (and its response_id back-references). The backing's
        own teardown is the backend's `delete`; this only forgets the handle."""
        conv = self._convs.pop(conv_id, None)
        if conv is not None:
            for rid in conv.response_ids:
                self._resp_to_conv.pop(rid, None)
        return conv


class ClaudeResumeBackend:
    """Delegated, non-interactive Claude sessions via `claude --resume <sid> -p`.
    The first turn starts a session and captures its `session_id`; later turns
    resume it. woollama holds only the handle → sid mapping; Claude Code owns the
    transcript on disk."""

    name = "claude-resume"

    async def send_turn(self, conv: Conversation, messages: list[dict]) -> str:
        # A STABLE, neutral working dir per conversation — Claude scopes sessions
        # by project dir, so every turn must --resume from the same cwd. It's a
        # fresh empty temp dir (no host CLAUDE.md/settings inherited), reused for
        # the conversation's life (cleaned on delete in a later slice).
        if conv.workdir is None:
            conv.workdir = tempfile.mkdtemp(prefix="woollama-conv-")
        model = conv.model.split("/", 1)[1] if "/" in conv.model else ""
        resp, sid = await claude_code.run_resumable(
            system="", user_msgs=messages, model=model,
            session_id=conv.native_id, cwd=conv.workdir)
        if sid:
            conv.native_id = sid     # first turn sets it; resume returns the same
        return resp["choices"][0]["message"].get("content") or ""

    async def delete(self, conv: Conversation) -> None:
        """End woollama's hold on the conversation: remove the per-conversation
        workdir. The Claude session transcript on disk (~/.claude) is the user's
        data and is left intact — woollama only drops what it created."""
        if conv.workdir:
            shutil.rmtree(conv.workdir, ignore_errors=True)
            conv.workdir = None
        conv.status = "dead"


# --- the `stored` backend: server-owned conversations (conv-5) ---------------
#
# For models with no native session of their own (ollama, recipes, cloud
# providers), woollama IS the conversation's home: it persists the visible
# transcript in a local duckdb file and REPLAYS it as context on each turn
# (`complete_stateless(model, history + new)`). This is the one place woollama
# legitimately owns conversation bytes — there's no backend underneath to defer
# to. Only the visible turns are stored (the caller's messages + each assistant
# final answer); recipe tool-call internals stay hidden, exactly as
# `complete_stateless` already returns only the final text.


def _default_db_path() -> str:
    """`$WOOLLAMA_DB`, else `$XDG_DATA_HOME/woollama/conversations.duckdb`
    (defaulting XDG to `~/.local/share`). Tests point `$WOOLLAMA_DB` at a tmp
    file (or construct `StoredStore(path)` directly)."""
    env = os.environ.get("WOOLLAMA_DB")
    if env:
        return env
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    d = os.path.join(base, "woollama")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "conversations.duckdb")


class StoredStore:
    """duckdb-backed persistence for `stored` conversations. A SINGLE connection
    serialized by an `asyncio.Lock` — duckdb connections are not thread-safe, so
    every access goes through the lock (the operations are local + sub-ms, so
    serializing on the event loop is fine; no thread pool). Two tables:
    `conversations` (the handle metadata) and `messages` (the visible
    transcript). Truth lives here; the in-memory `ConversationStore` is the
    working set, rehydrated from here at startup."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or _default_db_path()
        self._lock = asyncio.Lock()
        self._con = duckdb.connect(self.path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS conversations ("
            " id VARCHAR PRIMARY KEY, backend VARCHAR, model VARCHAR,"
            " title VARCHAR, metadata VARCHAR,"
            " created_at BIGINT, updated_at BIGINT)")
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS messages ("
            " conversation_id VARCHAR, seq BIGINT, role VARCHAR,"
            " content VARCHAR, created_at BIGINT)")

    async def save_conversation(self, conv: Conversation) -> None:
        """Upsert the handle row (idempotent — called on every turn so title /
        updated_at stay current)."""
        async with self._lock:
            self._con.execute(
                "INSERT INTO conversations"
                " (id, backend, model, title, metadata, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (id) DO UPDATE SET"
                " backend=excluded.backend, model=excluded.model,"
                " title=excluded.title, metadata=excluded.metadata,"
                " updated_at=excluded.updated_at",
                [conv.id, conv.backend, conv.model, conv.title,
                 json.dumps(conv.metadata or {}), conv.created_at, conv.updated_at])

    async def append_messages(self, conv_id: str, messages: list[dict]) -> None:
        """Append visible turns, assigning monotonic per-conversation seqs."""
        if not messages:
            return
        async with self._lock:
            row = self._con.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM messages WHERE conversation_id = ?",
                [conv_id]).fetchone()
            seq = int(row[0]) + 1
            now = _now()
            for m in messages:
                self._con.execute(
                    "INSERT INTO messages"
                    " (conversation_id, seq, role, content, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    [conv_id, seq, m.get("role", "user"),
                     m.get("content") or "", now])
                seq += 1

    async def load_messages(self, conv_id: str) -> list[dict]:
        async with self._lock:
            rows = self._con.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ?"
                " ORDER BY seq", [conv_id]).fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

    async def load_conversations(self) -> list[Conversation]:
        """Rehydrate every persisted handle (transcripts load lazily via
        `load_messages`). `response_ids` are NOT persisted — `previous_response_id`
        chaining is within-process only; attach by `conversation` survives
        restart, which is the durable path."""
        async with self._lock:
            rows = self._con.execute(
                "SELECT id, backend, model, title, metadata, created_at, updated_at"
                " FROM conversations").fetchall()
        out: list[Conversation] = []
        for r in rows:
            out.append(Conversation(
                id=r[0], backend=r[1], model=r[2], title=r[3],
                metadata=json.loads(r[4]) if r[4] else {},
                created_at=r[5], updated_at=r[6]))
        return out

    async def delete(self, conv_id: str) -> None:
        async with self._lock:
            self._con.execute("DELETE FROM messages WHERE conversation_id = ?", [conv_id])
            self._con.execute("DELETE FROM conversations WHERE id = ?", [conv_id])


# Lazily-created singleton (default path). Tests inject their own via
# `monkeypatch.setattr(conversations, "_stored", StoredStore(tmp))`.
_stored: StoredStore | None = None


def stored_store() -> StoredStore:
    global _stored
    if _stored is None:
        _stored = StoredStore()
    return _stored


class StoredBackend:
    """Server-owned conversations. There's no native session to resume — woollama
    persists the visible transcript and REPLAYS it as context each turn. Works for
    any stateless model (ollama, recipes, cloud providers): `send_turn` loads the
    stored history, prepends it to the new messages, runs one stateless completion,
    then persists the new user messages + the assistant's answer."""

    name = "stored"

    async def send_turn(self, conv: Conversation, messages: list[dict]) -> str:
        store = stored_store()
        # Lazy import breaks the conversations↔router import cycle (same pattern
        # mcp_server uses for orchestrate).
        from . import router
        history = await store.load_messages(conv.id)
        text = await router.complete_stateless(conv.model, history + messages)
        conv.updated_at = _now()
        await store.save_conversation(conv)               # upsert handle (idempotent)
        await store.append_messages(
            conv.id, [*messages, {"role": "assistant", "content": text}])
        return text

    async def history(self, conv: Conversation) -> list[dict]:
        return await stored_store().load_messages(conv.id)

    async def delete(self, conv: Conversation) -> None:
        await stored_store().delete(conv.id)
        conv.status = "dead"


# Backend registry, keyed by the name stored on a Conversation.
BACKENDS: dict[str, object] = {
    ClaudeResumeBackend.name: ClaudeResumeBackend(),
    StoredBackend.name: StoredBackend(),
}


async def rehydrate_stored(store: ConversationStore,
                           stored: StoredStore | None = None) -> int:
    """Load every persisted `stored` handle from duckdb (truth) into an in-memory
    `ConversationStore` (the working set). Called once at startup so attach-by-
    `conversation` survives a restart. Returns the count rehydrated. Isolated from
    the lifespan so it's directly testable (the restart behaviour is conv-5's
    whole point)."""
    convs = await (stored or stored_store()).load_conversations()
    for conv in convs:
        store.add(conv)
    return len(convs)


def backend_for_model(model: str) -> str:
    """Which stateful backend owns conversations for this `model`.
    `claude-code/<model>` resumes a native Claude session (`claude-resume`);
    every other model (ollama, recipes, cloud providers) has no native session,
    so woollama owns it via the server-owned `stored` backend (transcript replay)."""
    provider = model.split("/", 1)[0]
    if provider == "claude-code":
        return ClaudeResumeBackend.name
    return StoredBackend.name
