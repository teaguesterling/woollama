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
  (in `~/.claude`); woollama holds only the handle → session_id mapping.

Models with NO state-owning backend (ollama, recipes, cloud providers) have no
stateful conversation here — they are stateless (`store:false`; the caller owns
history, exactly as the Anthropic Messages API itself is stateless). Future
state-owning backends — a Claude-in-tmux driver, and Anthropic Managed Agents
(see the design doc) — also OWN their state; woollama still just routes handles.

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
import shutil
import tempfile
import time
from dataclasses import dataclass, field

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


# Backend registry, keyed by the name stored on a Conversation. ONLY backends
# that own their conversation state belong here — woollama routes handles to
# them, it never owns conversation state itself.
BACKENDS: dict[str, object] = {ClaudeResumeBackend.name: ClaudeResumeBackend()}


def backend_for_model(model: str) -> str | None:
    """Which state-owning backend (if any) backs conversations for this `model`.
    Only `claude-code/<model>` has one in this build — `claude-resume`, where the
    native Claude session owns the bytes. Every other model (ollama, recipes,
    cloud providers) has NO state owner, so it has no stateful backend: those
    conversations are stateless (`store:false`, the caller owns history).
    woollama does not store transcripts in its own system."""
    provider = model.split("/", 1)[0]
    if provider == "claude-code":
        return ClaudeResumeBackend.name
    return None
