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
import logging
import os
import time

log = logging.getLogger("woollama.managed_agents")

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

ENV_NETWORKING = "WOOLLAMA_AGENT_NETWORKING"


def _networking_config() -> dict:
    """Least-privilege default for the hosted environment: `limited` networking
    (no outbound hosts, no package registries — the agent woollama provisions is
    tool-less, so it needs none). `WOOLLAMA_AGENT_NETWORKING=unrestricted` is
    the explicit opt-out for setups that add network-using capabilities."""
    mode = (os.environ.get(ENV_NETWORKING) or "limited").strip().lower()
    if mode == "unrestricted":
        return {"type": "unrestricted"}
    if mode != "limited":
        log.warning("unknown %s=%r; using 'limited'", ENV_NETWORKING, mode)
    return {"type": "limited"}


def _create_environment_sync(name: str) -> str:
    env = _client().beta.environments.create(
        name=name,
        config={"type": "cloud", "networking": _networking_config()},
    )
    return env.id


# One client-side custom tool so the hosted agent can PAUSE for user input — the
# interactive `requires_action` path (§5). Custom tools are executed by us, not on
# a container, so this adds no provisioning over the tool-less agent. When the
# model calls it the session goes idle with stop_reason `requires_action`; woollama
# surfaces the question and resumes with the answer as a `user.custom_tool_result`.
ASK_USER_TOOL = {
    "type": "custom",
    "name": "ask_user",
    "description": "Ask the user a question and pause until they answer. Use when "
                   "you need clarification or a decision only the user can make.",
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question to ask."},
            "options": {"type": "array", "items": {"type": "string"},
                        "description": "Optional choices to offer."},
        },
        "required": ["question"],
    },
}


def _create_agent_sync(name: str, model: str, system: str) -> str:
    # `name` + `model` are the only required fields; we add ONE custom tool
    # (ask_user) so the agent can pause for input (§5). Custom tools are
    # client-executed → no per-session container provisioning.
    kwargs = {"name": name, "model": model, "tools": [ASK_USER_TOOL]}
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


def _collect_sync(session_id: str, send_events: list, deadline: float) -> dict:
    """Send `send_events` (a user.message OR a user.custom_tool_result), then
    stream until the session goes idle, returning `{text, pending}`. `pending` is
    None for a completed turn, or `{id, name, input}` if the agent paused on a
    custom tool (the interactive `requires_action` signal — §5). Stream-first
    (open before send), per-event wall-clock deadline (SDK timeouts are
    per-chunk)."""
    client = _client()
    start = time.monotonic()
    chunks: list[str] = []
    pending: dict | None = None
    with client.beta.sessions.events.stream(session_id=session_id) as stream:
        client.beta.sessions.events.send(session_id=session_id, events=send_events)
        for event in stream:
            if time.monotonic() - start > deadline:
                raise ManagedAgentsError(
                    f"managed-agents turn exceeded its {deadline:.0f}s deadline")
            etype = getattr(event, "type", None)
            if etype == "agent.message":
                chunks.append(_event_text(event))
            elif etype == "agent.custom_tool_use":
                # The agent invoked our client-side tool → it will idle awaiting
                # the result. Capture the call; the session is now paused.
                pending = {"id": event.id,
                           "name": getattr(event, "name", ""),
                           "input": getattr(event, "input", None) or {}}
            elif etype in ("session.status_idle", "session.status_terminated"):
                break
    return {"text": "".join(chunks), "pending": pending}


def _list_events_sync(session_id: str) -> list:
    return list(_client().beta.sessions.events.list(session_id=session_id).data)


# --- async entry points -------------------------------------------------------

create_environment = _wrap(_create_environment_sync)
create_agent = _wrap(_create_agent_sync)
create_session = _wrap(_create_session_sync)
delete_session = _wrap(_delete_session_sync)
list_events = _wrap(_list_events_sync)


async def _collect(session_id: str, send_events: list, deadline: float) -> dict:
    """Async wrapper around `_collect_sync`: a hard wall-clock `wait_for` (covers
    a stream that produces NO events, where the per-event deadline can't fire) on
    top of the per-event deadline inside. Returns `{text, pending}`."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_collect_sync, session_id, send_events, deadline),
            timeout=deadline + 30.0)
    except asyncio.TimeoutError as e:
        raise ManagedAgentsError(
            f"managed-agents turn timed out after {deadline + 30:.0f}s "
            "(no events received)") from e
    except ManagedAgentsError:
        raise
    except Exception as e:
        raise ManagedAgentsError(f"managed-agents turn failed: {e}") from e


async def run_turn(session_id: str, text: str, *, deadline: float = _DEADLINE) -> dict:
    """One conversational turn (a new user message). Returns `{text, pending}` —
    `pending` set if the agent paused for input (requires_action)."""
    return await _collect(
        session_id,
        [{"type": "user.message", "content": [{"type": "text", "text": text}]}],
        deadline)


async def answer_turn(session_id: str, tool_use_id: str, answer: str, *,
                      deadline: float = _DEADLINE) -> dict:
    """Resume a paused session by returning the user's answer as the result of the
    pending custom-tool call (`user.custom_tool_result`). Returns `{text,
    pending}` — the agent may complete, or pause again."""
    return await _collect(
        session_id,
        [{"type": "user.custom_tool_result", "custom_tool_use_id": tool_use_id,
          "content": [{"type": "text", "text": answer}]}],
        deadline)


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
