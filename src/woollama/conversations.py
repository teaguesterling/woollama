"""Conversation handle routing — the stateful surface's routing layer (conv-1b).

The principle (docs/conversations-api-design.md): **woollama routes conversation
*handles*; the backends own the *state*.** This module is that thin routing
layer — an in-memory handle table mapping woollama's opaque `conv_<hex>` ids to
the backend that owns the bytes plus that backend's native id, and the backend
adapters themselves. woollama never stores the conversation transcript.

Scope (conv-1b): the `claude-resume` backend (`claude --resume <sid> -p`, the
non-interactive path — verified not to hang nested). The §3 interface is kept to
what `/v1/responses` actually needs — `send_turn` — per the design review; the
interactive `poll`/`answer` methods arrive with the driver/interactive slices,
and `history`/`delete` with the `/v1/conversations` endpoints (conv-2).

Limitations (documented, not hidden):
- **In-memory** — the handle table is process-lifetime; a restart loses the
  `conv_id → claude session_id` mappings (a persistent store is a later slice).
- **Chaining, not forking** — `claude --resume` continues from the session TIP
  and reports the SAME session_id, so `previous_response_id` attaches to a
  conversation and continues it; claude has no fork-from-earlier-turn primitive.
- **One writer per conversation** — turns on a given conversation serialize on a
  per-conversation lock (the backend session is single-threaded).
"""
from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass, field

from . import claude_code, responses


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
    status: str = "idle"
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class ConversationStore:
    """In-memory handle table. Maps `conv_id → Conversation` and `resp_id →
    conv_id` (so `previous_response_id` resolves to its conversation)."""

    def __init__(self) -> None:
        self._convs: dict[str, Conversation] = {}
        self._resp_to_conv: dict[str, str] = {}

    def create(self, backend: str, model: str) -> Conversation:
        conv = Conversation(id=responses.new_id("conv"), backend=backend, model=model)
        self._convs[conv.id] = conv
        return conv

    def get(self, conv_id: str) -> Conversation | None:
        return self._convs.get(conv_id)

    def by_response(self, response_id: str) -> Conversation | None:
        cid = self._resp_to_conv.get(response_id)
        return self._convs.get(cid) if cid else None

    def record_response(self, conv: Conversation, response_id: str) -> None:
        conv.response_ids.append(response_id)
        self._resp_to_conv[response_id] = conv.id


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


# Backend registry, keyed by the name stored on a Conversation.
BACKENDS: dict[str, object] = {ClaudeResumeBackend.name: ClaudeResumeBackend()}


def backend_for_model(model: str) -> str | None:
    """Which stateful backend (if any) owns conversations for this `model`. Only
    `claude-code/<model>` has a stateful backend in this build; ollama/recipe
    conversations need the server-owned `stored` backend (a later slice)."""
    provider = model.split("/", 1)[0]
    if provider == "claude-code":
        return ClaudeResumeBackend.name
    return None
