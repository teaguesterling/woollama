"""Conversation handle routing — the stateful surface's routing layer (conv-1b).

The principle (docs/conversations-api-design.md): **woollama routes conversation
*handles*; the backends own the *state*.** This module is that thin routing
layer — an in-memory handle table mapping woollama's opaque `conv_<hex>` ids to
the backend that owns the bytes plus that backend's native id, and the backend
adapters themselves. **woollama never stores the conversation transcript in its
own system.** It proxies/retrieves a backend's transcript; when no backend owns
the state, the turn is stateless (the caller owns history) — woollama does not
fabricate a store of its own.

Backends (only state-OWNING backends belong here):
- `claude-resume` (conv-1b): `claude --resume <sid> -p`, the non-interactive
  path — verified not to hang nested. The native Claude session owns the bytes
  (in `~/.claude`); woollama holds only the handle → session_id mapping. Keyless
  (subscription); routed for `claude-code/<model>`.
- `managed-agents` (conv-6): Anthropic's Managed Agents API — Anthropic hosts the
  session, the loop, and a per-session container; woollama holds only the
  `session_id`. Paid (`ANTHROPIC_API_KEY`); routed for `claude-agent/<model>`.
  The first backend that also serves `history` (Anthropic exposes the event log).

- `store-backed` (conv-7, issue #2): a STORE-ONLY / BYO-inference backend — an
  external `ConversationStore` provider owns the transcript while woollama does
  assembly + STATELESS inference (design-doc §3.1, §10). Makes ollama/cloud/recipe
  models stateful WITHOUT woollama owning bytes. Registered only when a provider
  is wired in (`register_store_backend`); the first provider is fabric, pending
  its read/append contract — so none ships by default.

Models with NO state-owning backend registered (ollama, recipes, cloud providers
when no store provider is wired) have no stateful conversation here — they are
stateless (`store:false`; the caller owns history, exactly as the Anthropic
Messages API itself is stateless). A future Claude-in-tmux driver would also OWN
its state; woollama still just routes handles.

Limitations (documented, not hidden):
- **In-memory handle table** — `conv_id → claude session_id` mappings are
  process-lifetime; a restart loses them. The backend's own transcript on disk
  (`~/.claude`) is intact, but woollama's handle→sid map is gone.
- **Chaining, not forking** — `claude --resume` continues from the session TIP
  and reports the SAME session_id, so `previous_response_id` attaches to a
  conversation and continues it; claude has no fork-from-earlier-turn primitive.
- **One writer per conversation** — turns on a given conversation serialize on a
  per-conversation lock (the backend session is single-threaded).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from . import claude_code, managed_agents, responses

log = logging.getLogger("woollama.conversations")


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
    key: str | None = None            # caller's own external key (e.g. fabric's sessionName)
    workdir: str | None = None        # stable cwd for the backing session (resume scoping)
    response_ids: list[str] = field(default_factory=list)
    status: str = "idle"              # idle | busy | awaiting_input | dead
    title: str | None = None
    metadata: dict = field(default_factory=dict)
    created_at: int = field(default_factory=_now)
    updated_at: int = field(default_factory=_now)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Interactive pause state (§5): set when a backend turn stops awaiting input.
    # `required_action` is the Responses payload; `pending_tool_use_id` is the
    # backend's handle for resuming (e.g. a CMA custom_tool_use id).
    required_action: dict | None = None
    pending_tool_use_id: str | None = None


# Conversation fields that make up the durable handle (everything except the
# transient asyncio.Lock). This is ROUTING state — conv_id → (backend, native_id)
# — NOT the transcript (the backend/store owns that), so persisting it does not
# violate "woollama never owns conversation state".
_PERSISTED_FIELDS = (
    "id", "backend", "model", "native_id", "key", "workdir", "response_ids",
    "status", "title", "metadata", "created_at", "updated_at",
    "required_action", "pending_tool_use_id",
)


class ConversationStore:
    """Handle table. Maps `conv_id → Conversation` and `resp_id → conv_id` (so
    `previous_response_id` resolves to its conversation).

    Optionally **durable**: pass a `path` (or call `enable_persistence`) and the
    table is loaded on construction and atomically rewritten on every mutation, so
    conv_ids survive a woollama restart (the routing state outlives the process,
    matching the backends/stores that already outlive it). With no path it is
    purely in-memory (the default; tests rely on this)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._convs: dict[str, Conversation] = {}
        self._resp_to_conv: dict[str, str] = {}
        self._alias_to_conv: dict[str, str] = {}    # caller's key → conv_id
        self._path: Path | None = Path(path) if path else None
        if self._path is not None:
            self._load()

    def enable_persistence(self, path: str | Path) -> None:
        """Make this (already-constructed) store durable at `path`, loading any
        existing state. Used by the router lifespan so the module-level store stays
        import-side-effect-free until the server actually starts."""
        self._path = Path(path)
        self._load()

    def create(self, backend: str, model: str, *, metadata: dict | None = None,
               title: str | None = None, key: str | None = None) -> Conversation:
        conv = Conversation(id=responses.new_id("conv"), backend=backend,
                            model=model, metadata=metadata or {}, title=title,
                            key=key)
        self._convs[conv.id] = conv
        if key is not None:
            self._alias_to_conv[key] = conv.id
        self._save()
        return conv

    def by_alias(self, key: str) -> Conversation | None:
        """Resolve a caller's own external key (e.g. fabric's sessionName) to its
        conversation — woollama owns this `key → conv_id` map (durably), so the
        caller need not keep its own."""
        cid = self._alias_to_conv.get(key)
        return self._convs.get(cid) if cid else None

    def get_or_create_by_alias(self, key: str, backend: str, model: str,
                               **kw) -> tuple[Conversation, bool]:
        """Idempotent attach-by-key: return the existing conversation for `key`
        (created=False), or create a new one bound to it (created=True)."""
        existing = self.by_alias(key)
        if existing is not None:
            return existing, False
        return self.create(backend, model, key=key, **kw), True

    def get(self, conv_id: str) -> Conversation | None:
        return self._convs.get(conv_id)

    def list(self) -> list[Conversation]:
        return list(self._convs.values())

    def by_response(self, response_id: str) -> Conversation | None:
        cid = self._resp_to_conv.get(response_id)
        return self._convs.get(cid) if cid else None

    def record_response(self, conv: Conversation, response_id: str) -> None:
        conv.response_ids.append(response_id)
        conv.updated_at = _now()
        self._resp_to_conv[response_id] = conv.id
        # Called after EVERY stateful turn (router._responses_stateful), so this is
        # also the persistence point that captures native_id/status/required_action.
        self._save()

    def remove(self, conv_id: str) -> Conversation | None:
        """Drop the handle (and its response_id back-references). The backing's
        own teardown is the backend's `delete`; this only forgets the handle."""
        conv = self._convs.pop(conv_id, None)
        if conv is not None:
            for rid in conv.response_ids:
                self._resp_to_conv.pop(rid, None)
            if conv.key is not None:
                self._alias_to_conv.pop(conv.key, None)
            self._save()
        return conv

    # --- persistence (no-op unless a path is set) -----------------------------

    def _save(self) -> None:
        if self._path is None:
            return
        data = {
            "convs": [{f: getattr(c, f) for f in _PERSISTED_FIELDS}
                      for c in self._convs.values()],
            "resp_to_conv": self._resp_to_conv,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        # Owner-only from creation (no chmod-after-write window): the table
        # carries caller-supplied metadata/titles — keep it 0600 like the
        # runtime socket, not umask-default world-readable.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(data))
        tmp.replace(self._path)             # atomic swap

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            log.warning("conversation handle table at %s is unreadable (%s); "
                        "starting empty", self._path, e)
            return
        for d in data.get("convs", []):
            # A turn can't be mid-flight after a restart — a persisted 'busy' is
            # stale; reset it so the conversation is usable again.
            if d.get("status") == "busy":
                d["status"] = "idle"
            conv = Conversation(**d)
            self._convs[conv.id] = conv
            if conv.key is not None:        # rebuild the alias index from conv.key
                self._alias_to_conv[conv.key] = conv.id
        self._resp_to_conv = dict(data.get("resp_to_conv", {}))
        log.info("loaded %d conversation handle(s) from %s",
                 len(self._convs), self._path)


class ClaudeResumeBackend:
    """Delegated, non-interactive Claude sessions via `claude --resume <sid> -p`.
    The first turn starts a session and captures its `session_id`; later turns
    resume it. woollama holds only the handle → sid mapping; Claude Code owns the
    transcript on disk."""

    name = "claude-resume"

    async def send_turn(self, conv: Conversation, messages: list[dict],
                        *, options: dict | None = None) -> str:
        # `options` (e.g. ollama num_ctx) is inert here — Claude has no such knob.
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


def _latest_user_text(messages: list[dict]) -> str:
    """The new user input for a stateful turn. The backend already owns prior
    history (Anthropic's session holds it), so woollama sends only the latest
    user message — not the whole transcript."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content") or ""
    return messages[-1].get("content", "") if messages else ""


class ManagedAgentsBackend:
    """Claude-hosted stateful sessions via Anthropic's Managed Agents API
    (`/v1/agents` + `/v1/sessions`). Anthropic owns the session, the agentic
    loop, and a per-session container; woollama holds only the `session_id`. The
    purest 'backend owns state' backend — and the first that can serve `history`
    (Anthropic exposes the event log; woollama RETRIEVES, never stores).

    Minimal (§8.7): one TOOL-LESS agent per model, created lazily and cached on
    this instance (never per session, the documented anti-pattern); a single
    shared environment, created once. See managed_agents.py for the auth and the
    cost/orphan lifecycle notes (sessions are billed; a restart orphans them)."""

    name = "managed-agents"

    def __init__(self) -> None:
        self._env_id: str | None = None
        self._agents: dict[str, str] = {}        # full model id → agent_id (reused)
        self._setup_lock = asyncio.Lock()

    async def _ensure_agent(self, model: str) -> tuple[str, str]:
        """Lazily create + cache the shared environment and a per-model tool-less
        agent. Serialized so concurrent first-turns don't double-create."""
        full = managed_agents.resolve_model(model)
        async with self._setup_lock:
            if self._env_id is None:
                self._env_id = await managed_agents.create_environment("woollama-agents")
            if full not in self._agents:
                self._agents[full] = await managed_agents.create_agent(
                    name=f"woollama:{full}", model=full, system="")
        return self._agents[full], self._env_id

    def _apply(self, conv: Conversation, result: dict) -> str:
        """Fold a `{text, pending}` turn result into the conversation handle. A
        pending custom-tool call → set `required_action` + `awaiting_input` (the
        interactive pause, §5); otherwise clear any prior pause. Returns the text."""
        pending = result.get("pending")
        if pending:
            conv.pending_tool_use_id = pending["id"]
            # The ask_user tool's input IS the question payload (question[, options]).
            conv.required_action = {"type": "ask_user", "question": pending["input"]}
            conv.status = "awaiting_input"
        else:
            conv.pending_tool_use_id = None
            conv.required_action = None
        return result.get("text") or ""

    async def send_turn(self, conv: Conversation, messages: list[dict],
                        *, options: dict | None = None) -> str:
        # `options` is inert here — Anthropic owns inference for the hosted session.
        # The session (Anthropic's, holding the transcript) is created lazily on
        # the first turn; later turns reuse it via its native session_id.
        if conv.native_id is None:
            agent_id, env_id = await self._ensure_agent(conv.model)
            conv.native_id = await managed_agents.create_session(
                agent_id, env_id, title=conv.title, metadata=conv.metadata)
        result = await managed_agents.run_turn(conv.native_id, _latest_user_text(messages))
        return self._apply(conv, result)

    async def answer(self, conv: Conversation, answer: str) -> str:
        """Resume a session paused on `ask_user` by returning the user's answer
        as the pending custom-tool result (§5). May complete or pause again."""
        result = await managed_agents.answer_turn(
            conv.native_id, conv.pending_tool_use_id, answer)
        return self._apply(conv, result)

    async def history(self, conv: Conversation) -> list[dict]:
        """Retrieve the backend's transcript (Anthropic owns the bytes). Empty
        until the first turn creates the session."""
        if conv.native_id is None:
            return []
        events = await managed_agents.list_events(conv.native_id)
        return managed_agents.events_to_messages(events)

    async def delete(self, conv: Conversation) -> None:
        """Tear down the Anthropic session (a billed container — deletion matters
        more here than for claude-resume's on-disk session)."""
        if conv.native_id:
            await managed_agents.delete_session(conv.native_id)
        conv.status = "dead"


class ConversationStoreProvider(Protocol):
    """**PROVISIONAL contract proposal — issue #2 / design-doc §10; NOT yet agreed
    with fabric.** A pluggable external owner of conversation transcripts that
    woollama is a CLIENT to — woollama never holds the bytes. (Distinct from the
    `ConversationStore` handle table above, which only maps woollama's opaque
    `conv_id`s to backends — that is routing state, not transcript storage.) The
    first intended provider is fabric / the cosmic-fabricd session daemon; this
    `create/get/append/delete` shape is woollama's PROPOSED read/append surface
    and may change once that cross-repo contract is settled. An MCP
    conversation-store or a JSONL reader can implement the same Protocol later."""

    async def create(self) -> str: ...
    async def get(self, thread_id: str) -> list[dict]: ...
    async def append(self, thread_id: str, messages: list[dict]) -> None: ...
    async def delete(self, thread_id: str) -> None: ...


# Inference callable injected into a StoreBackedBackend: (model, messages, *,
# options) -> assistant text. Injected (not imported) so this module never imports
# `router` (which owns inferencer routing) — avoids a conversations↔router cycle.
# (Loosely typed because of the keyword-only `options`.)
CompleteFn = Callable[..., Awaitable[str]]


class StoreBackedBackend:
    """Store-only / BYO-inference backend (design-doc §3.1, §10): an external
    `ConversationStore` owns the transcript; woollama assembles prior history +
    the new turn, runs the STATELESS inferencer, and writes the turn back. Makes
    non-claude models (ollama, cloud, recipes) stateful WITHOUT woollama ever
    owning the bytes.

    `options` (e.g. ollama `num_ctx`) is forwarded to the injected `complete`, so a
    store-backed ollama turn DOES honor a requested context size — the router's
    `complete_stateless` routes ollama natively when `num_ctx` is present (the
    #1↔#2 seam is closed)."""

    def __init__(self, name: str, store: ConversationStoreProvider, complete: CompleteFn):
        self.name = name
        self._store = store
        self._complete = complete

    async def send_turn(self, conv: Conversation, messages: list[dict],
                        *, options: dict | None = None) -> str:
        if conv.native_id is None:
            conv.native_id = await self._store.create()    # store mints the thread
        prior = await self._store.get(conv.native_id)      # bytes owned by the store
        answer = await self._complete(conv.model, prior + messages, options=options)
        await self._store.append(                          # write the turn back
            conv.native_id, list(messages) + [{"role": "assistant", "content": answer}])
        return answer

    async def history(self, conv: Conversation) -> list[dict]:
        if conv.native_id is None:
            return []
        return await self._store.get(conv.native_id)

    async def delete(self, conv: Conversation) -> None:
        if conv.native_id:
            await self._store.delete(conv.native_id)
        conv.status = "dead"


# Backend registry, keyed by the name stored on a Conversation. ONLY backends
# that own (or defer to an external owner of) their conversation state belong
# here — woollama routes handles to them, it never owns conversation state itself.
BACKENDS: dict[str, object] = {
    ClaudeResumeBackend.name: ClaudeResumeBackend(),
    ManagedAgentsBackend.name: ManagedAgentsBackend(),
}

# Name under which a store-backed backend registers, IF a conversation-store
# provider is wired in (register_store_backend). None ships by default — the
# fabric provider contract (§10.2) is pending — so non-claude models stay
# stateless until then.
STORE_BACKEND_NAME = "store-backed"


def register_store_backend(store: ConversationStoreProvider, complete: CompleteFn,
                           *, name: str = STORE_BACKEND_NAME) -> None:
    """Wire a `ConversationStoreProvider` in as the state owner for non-claude
    models (so `/v1/responses` + `/v1/conversations` become stateful for them).
    Until called, those models are STATELESS (`backend_for_model` → None) — no
    provider ships by default. The router passes `complete_stateless` as the
    inference fn."""
    BACKENDS[name] = StoreBackedBackend(name, store, complete)


# An injected MCP call: (tool_name, args) -> the already-parsed JSON value the
# convstore server returned. The router supplies a wrapper that invokes the
# server via `ServerManager.call_tool`, parses its text blocks, and converts any
# transport/parse failure into a clean `OrchestrationError(..., 502)` — so a
# flaky store surfaces as a gateway error through `_responses_stateful`, never a
# 500. Injected (not imported) so this module never imports `manager`/`router`.
StoreCallFn = Callable[..., Awaitable[Any]]


class McpStoreProvider:
    """A `ConversationStoreProvider` backed by an MCP conversation-store server
    (the reference implementation is `examples/mcp-convstore/server.py`). Holds
    NO transcript bytes itself — every operation is one MCP tool call to the
    external server, which owns the state. This keeps woollama a *client* to the
    store, the core principle of the conversations design.

    The tool contract (the server returns explicit `json.dumps(...)` strings):
      - `create_thread()`                  -> thread_id (str)
      - `get_thread(thread_id)`            -> list[message dict]  ([] when fresh)
      - `append_turn(thread_id, messages)` -> {"ok", "count"}
      - `delete_thread(thread_id)`         -> {"ok": True}

    The injected `call` returns the already-parsed value and raises
    `OrchestrationError` on transport/parse failure (built by the router; see
    `_mcp_store_call`)."""

    def __init__(self, call: StoreCallFn):
        self._call = call

    async def create(self) -> str:
        return await self._call("create_thread", {})

    async def get(self, thread_id: str) -> list[dict]:
        return await self._call("get_thread", {"thread_id": thread_id})

    async def append(self, thread_id: str, messages: list[dict]) -> None:
        await self._call("append_turn", {"thread_id": thread_id, "messages": messages})

    async def delete(self, thread_id: str) -> None:
        await self._call("delete_thread", {"thread_id": thread_id})


class HttpStoreProvider:
    """A `ConversationStoreProvider` backed by a REST conversation-store endpoint
    (the reference implementation is `examples/rest-convstore/server.py`). Sibling
    to `McpStoreProvider` — the same four ops, plain HTTP instead of MCP; holds no
    transcript bytes. Two reference providers over one transport-agnostic seam.

    The provider mints the thread id (a UUID) and `PUT`s it, so the store needs no
    id-minting logic and create is idempotent. The injected `call(method, path,
    body)` (built by the router; see `_http_store_call`) issues the request,
    returns parsed JSON (or None for an empty/204 body), and raises
    `OrchestrationError(502)` on any non-2xx / transport failure.

    REST mapping (`examples/rest-convstore/server.py`):
      - create  -> `PUT    /threads/{uuid}`  (provider mints the uuid)
      - get     -> `GET    /threads/{id}`    -> list[message dict]  ([] when fresh)
      - append  -> `PATCH  /threads/{id}`    (body = the new messages)
      - delete  -> `DELETE /threads/{id}`
    """

    def __init__(self, call: StoreCallFn):
        self._call = call

    async def create(self) -> str:
        thread_id = uuid.uuid4().hex
        await self._call("PUT", f"/threads/{thread_id}", None)
        return thread_id

    async def get(self, thread_id: str) -> list[dict]:
        return await self._call("GET", f"/threads/{thread_id}", None) or []

    async def append(self, thread_id: str, messages: list[dict]) -> None:
        await self._call("PATCH", f"/threads/{thread_id}", messages)

    async def delete(self, thread_id: str) -> None:
        await self._call("DELETE", f"/threads/{thread_id}", None)


def backend_for_model(model: str) -> str | None:
    """Which state-owning backend (if any) backs conversations for this `model`:
    - `claude-code/<model>` → `claude-resume` (the native Claude session owns the
      bytes on disk; keyless/subscription).
    - `claude-agent/<model>` → `managed-agents` (Anthropic hosts the session;
      `ANTHROPIC_API_KEY`, paid).
    Every other model (ollama, recipes, cloud providers) has NO state owner, so it
    has no stateful backend — those conversations are stateless (`store:false`,
    the caller owns history). woollama does not store transcripts itself."""
    provider = model.split("/", 1)[0]
    if provider == "claude-code":
        return ClaudeResumeBackend.name
    if provider == "claude-agent":
        return ManagedAgentsBackend.name
    # Any other model (ollama, cloud, recipe) is stateful ONLY when a
    # conversation-store provider has been registered (§10); else stateless —
    # woollama owns no store of its own.
    if STORE_BACKEND_NAME in BACKENDS:
        return STORE_BACKEND_NAME
    return None
