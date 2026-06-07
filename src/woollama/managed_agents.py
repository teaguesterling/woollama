"""Anthropic Managed Agents — the SDK wrapper for the `managed-agents`
conversation backend (conv-6).

woollama routes conversation *handles*; the backend owns the *state*. Here the
owner is Anthropic: a Managed Agents **session** holds the transcript, the loop,
and a per-session container — woollama holds only the `session_id`. This is the
purest "backend owns state" backend (docs/conversations-api-design.md §8.7), and
the first one that can also serve `history` (Anthropic exposes the event log).

Auth is `ANTHROPIC_API_KEY` (NOT the keyless subscription path that
claude-resume/claude-code use) over the official `anthropic` SDK, which sets the
`managed-agents-2026-04-01` beta header itself.

The SDK is synchronous; every call here runs in a worker thread
(`asyncio.to_thread`) so the event loop is never blocked — the same isolation
`claude_code` uses for its subprocess. `_client()` is the single seam the
hermetic tests patch, so the suite needs neither the `anthropic` package nor a
key.

Minimal slice (§8.7): one TOOL-LESS agent per model, created lazily and cached on
the backend instance (an agent is a persisted, versioned, REUSABLE object —
never created per session, the documented anti-pattern). No recipe→agent MCP
mapping, no vaults, no file/repo resources yet.

Cost / lifecycle (documented, not hidden): unlike claude-resume (an orphan is
just disk in ~/.claude), a Managed Agents session is a BILLED container and an
agent is a persistent account object. `delete_session()` tears a session down,
but the in-memory handle table means a woollama restart orphans live sessions
(billed containers woollama has lost the handle to), and each fresh process
re-creates its per-model agent (slow accumulation of agent objects). Acceptable
for the prototype; the fix (reuse-by-name / the `ant` YAML control plane) is
deferred.
"""
from __future__ import annotations

import asyncio
import os
import time

# Short aliases for the `claude-agent/<alias>` namespace; anything else passes
# through unchanged (so a full id like `claude-agent/claude-opus-4-8` works too).
DEFAULT_MODEL = "claude-opus-4-8"
_MODEL_ALIASES = {
    "opus": "claude-opus-4-8",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
}

_DEADLINE = 300.0   # wall-clock seconds for one turn's stream-collect


class ManagedAgentsError(RuntimeError):
    """Any failure talking to the Managed Agents API: missing key/package, an SDK
    error, or a turn that exceeded the wall-clock deadline."""


def resolve_model(model: str) -> str:
    """Map a `claude-agent/<model>` value to a full Anthropic model id. Short
    aliases (opus/sonnet/haiku) expand; anything else passes through (a full id
    is used as-is); empty → the default."""
    name = model.split("/", 1)[1] if "/" in model else model
    if not name:
        return DEFAULT_MODEL
    return _MODEL_ALIASES.get(name, name)


def _client():
    """The single mock seam. Returns a sync `anthropic.Anthropic` client, or
    raises ManagedAgentsError if the package or the key is absent. Hermetic tests
    patch this to inject a fake client, so they need neither."""
    try:
        import anthropic
    except ModuleNotFoundError as e:                       # pragma: no cover
        raise ManagedAgentsError(
            "the `anthropic` package is required for the managed-agents backend "
            "(install the `agents` extra: `uv sync --extra agents`).") from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ManagedAgentsError(
            "ANTHROPIC_API_KEY is not set — the managed-agents backend is a paid, "
            "key-authenticated path, distinct from the keyless claude-resume / "
            "claude-code subscription backends.")
    return anthropic.Anthropic()


def _wrap(fn):
    """Run a blocking SDK callable in a worker thread, normalizing any non-
    ManagedAgentsError into one."""
    async def run(*args, **kwargs):
        try:
            return await asyncio.to_thread(fn, *args, **kwargs)
        except ManagedAgentsError:
            raise
        except Exception as e:                             # SDK / network error
            raise ManagedAgentsError(f"managed-agents API error: {e}") from e
    return run


# --- environment + agent (control plane: created once, reused) ----------------

def _create_environment_sync(name: str) -> str:
    env = _client().beta.environments.create(
        name=name,
        config={"type": "cloud", "networking": {"type": "unrestricted"}},
    )
    return env.id


def _create_agent_sync(name: str, model: str, system: str) -> str:
    # Tool-less agent (§8.7 minimal): only `name` + `model` are required; no
    # toolset means no per-session container provisioning we don't need yet.
    kwargs = {"name": name, "model": model}
    if system:
        kwargs["system"] = system
    agent = _client().beta.agents.create(**kwargs)
    return agent.id


# --- session (data plane: one per conversation) -------------------------------

def _create_session_sync(agent_id: str, env_id: str, *,
                         title: str | None, metadata: dict | None) -> str:
    kwargs = {"agent": agent_id, "environment_id": env_id}   # string id → latest version
    if title:
        kwargs["title"] = title
    if metadata:
        kwargs["metadata"] = metadata
    session = _client().beta.sessions.create(**kwargs)
    return session.id


def _delete_session_sync(session_id: str) -> None:
    _client().beta.sessions.delete(session_id=session_id)


def _event_text(event) -> str:
    """Join the text blocks of an `agent.message` event."""
    parts = []
    for block in getattr(event, "content", None) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts)


def _run_turn_sync(session_id: str, text: str, deadline: float) -> str:
    """Send one user message and collect the agent's reply, streaming events
    until the session goes idle. Stream-first (open before send, per the SDK
    guidance), with a per-event wall-clock deadline (SDK stream timeouts are
    per-chunk, not total)."""
    client = _client()
    start = time.monotonic()
    chunks: list[str] = []
    with client.beta.sessions.events.stream(session_id=session_id) as stream:
        client.beta.sessions.events.send(
            session_id=session_id,
            events=[{"type": "user.message",
                     "content": [{"type": "text", "text": text}]}],
        )
        for event in stream:
            if time.monotonic() - start > deadline:
                raise ManagedAgentsError(
                    f"managed-agents turn exceeded its {deadline:.0f}s deadline")
            etype = getattr(event, "type", None)
            if etype == "agent.message":
                chunks.append(_event_text(event))
            elif etype in ("session.status_idle", "session.status_terminated"):
                break
    return "".join(chunks)


def _list_events_sync(session_id: str) -> list:
    return list(_client().beta.sessions.events.list(session_id=session_id).data)


# --- async entry points -------------------------------------------------------

create_environment = _wrap(_create_environment_sync)
create_agent = _wrap(_create_agent_sync)
create_session = _wrap(_create_session_sync)
delete_session = _wrap(_delete_session_sync)
list_events = _wrap(_list_events_sync)


async def run_turn(session_id: str, text: str, *, deadline: float = _DEADLINE) -> str:
    """One conversational turn. Wraps the blocking stream-collect in a hard
    wall-clock `wait_for` (in case the stream produces NO events at all, where
    the per-event deadline can't fire), atop the per-event deadline inside."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_run_turn_sync, session_id, text, deadline),
            timeout=deadline + 30.0)
    except asyncio.TimeoutError as e:
        raise ManagedAgentsError(
            f"managed-agents turn timed out after {deadline + 30:.0f}s "
            "(no events received)") from e
    except ManagedAgentsError:
        raise
    except Exception as e:
        raise ManagedAgentsError(f"managed-agents turn failed: {e}") from e


def events_to_messages(events: list) -> list[dict]:
    """Parse a Managed Agents event list into woollama transcript messages
    (`{role, content}`), the shape `responses.item_object` consumes. Anthropic
    owns the bytes; woollama RETRIEVES and reshapes — it does not store them.
    Only the conversational turns are surfaced (`user.message` → user,
    `agent.message` → assistant); tool/status events are skipped for this v1."""
    out: list[dict] = []
    for event in events:
        etype = getattr(event, "type", None)
        if etype == "user.message":
            out.append({"role": "user", "content": _event_text(event)})
        elif etype == "agent.message":
            out.append({"role": "assistant", "content": _event_text(event)})
    return out
