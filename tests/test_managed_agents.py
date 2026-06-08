"""conv-6 — the managed-agents backend (Anthropic Managed Agents).

Proves handle routing onto Anthropic-hosted sessions WITHOUT a real API key or
the `anthropic` package's network: the SDK client seam (`managed_agents._client`)
is patched with a fake that records control-plane calls (agents/environments)
and scripts the session event stream. These assert that woollama creates the
agent/environment ONCE and reuses them, creates a session per conversation,
sends only the new turn (Anthropic owns prior history), retrieves the transcript
via `history`, and tears the session down on delete. The live counterpart is the
opt-in, PAID integration test.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from woollama import conversations, managed_agents, router

# --- a fake `anthropic` client mirroring the managed-agents SDK surface --------

def _event(type_: str, text: str | None = None):
    content = [SimpleNamespace(type="text", text=text)] if text is not None else []
    return SimpleNamespace(type=type_, content=content)


def _tool_use_event(id_: str, name: str, input_: dict):
    """An agent.custom_tool_use event — the interactive pause signal (its fields
    mirror the installed SDK's BetaManagedAgentsAgentCustomToolUseEvent)."""
    return SimpleNamespace(type="agent.custom_tool_use", id=id_, name=name,
                           input=input_, content=[])


class _FakeStream:
    """Context manager + iterable over a session's pending events. Reads the
    queue at iteration time, so the stream-first pattern (open, THEN send) sees
    the events `send` produced."""
    def __init__(self, session):
        self._session = session

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self._session["pending"])


class FakeClient:
    def __init__(self):
        self.rec = {"agents": [], "envs": [], "sessions": [], "deleted": []}
        self._sessions: dict[str, dict] = {}
        self._n = 0
        # Interactive scripting: when True, the NEXT user.message makes the agent
        # call ask_user (pause) instead of replying. The id it later receives in a
        # custom_tool_result is recorded for round-trip assertions.
        self.pause_next = False
        self.received_tool_result_id = None
        events = SimpleNamespace(send=self._send, stream=self._stream, list=self._list)
        self.beta = SimpleNamespace(
            agents=SimpleNamespace(create=self._agent_create),
            environments=SimpleNamespace(create=self._env_create),
            sessions=SimpleNamespace(create=self._session_create,
                                     delete=self._session_delete,
                                     events=events),
        )

    def _agent_create(self, **kw):
        self.rec["agents"].append((kw["name"], kw["model"]))
        self._n += 1
        return SimpleNamespace(id=f"agent_{self._n}", version=1)

    def _env_create(self, **kw):
        self.rec["envs"].append(kw["name"])
        self._n += 1
        return SimpleNamespace(id=f"env_{self._n}")

    def _session_create(self, **kw):
        self.rec["sessions"].append((kw["agent"], kw["environment_id"]))
        self._n += 1
        sid = f"sesn_{self._n}"
        self._sessions[sid] = {"log": [], "pending": []}
        return SimpleNamespace(id=sid, status="idle")

    def _session_delete(self, session_id):
        self.rec["deleted"].append(session_id)

    def _send(self, session_id, events):
        s = self._sessions[session_id]
        for ev in events:
            if ev["type"] == "user.custom_tool_result":
                # Resuming a paused turn: record the id, then the agent answers.
                self.received_tool_result_id = ev["custom_tool_use_id"]
                answer = ev["content"][0]["text"]
                reply = f"hello {answer}"
                s["log"].append(("assistant", reply))
                s["pending"] = [_event("agent.message", reply),
                                _event("session.status_idle")]
                return
            text = ev["content"][0]["text"]
            s["log"].append(("user", text))
        if self.pause_next:
            # The agent calls ask_user instead of replying → session pauses.
            self.pause_next = False
            s["pending"] = [_tool_use_event("tu_1", "ask_user",
                                            {"question": "What's your name?"}),
                            _event("session.status_idle")]
            return
        reply = f"echo: {s['log'][-1][1]}"
        s["log"].append(("assistant", reply))
        s["pending"] = [_event("agent.message", reply), _event("session.status_idle")]

    def _stream(self, session_id):
        return _FakeStream(self._sessions[session_id])

    def _list(self, session_id):
        s = self._sessions[session_id]
        data = [_event("user.message" if role == "user" else "agent.message", text)
                for role, text in s["log"]]
        return SimpleNamespace(data=data)


class FakeRequest:
    def __init__(self, body: dict):
        self._body = body

    async def json(self) -> dict:
        return self._body


def fresh(monkeypatch):
    """Patch the SDK seam with a fake client + install a fresh handle table and a
    fresh (cache-empty) managed-agents backend into the registry."""
    fake = FakeClient()
    monkeypatch.setattr(managed_agents, "_client", lambda: fake)
    monkeypatch.setattr(router, "conversation_store", conversations.ConversationStore())
    backend = conversations.ManagedAgentsBackend()
    monkeypatch.setattr(conversations, "BACKENDS",
                        {**conversations.BACKENDS, "managed-agents": backend})
    return fake


# --- model resolution + routing -----------------------------------------------

def test_resolve_model_aliases_passthrough_and_default():
    assert managed_agents.resolve_model("claude-agent/opus") == "claude-opus-4-8"
    assert managed_agents.resolve_model("claude-agent/haiku") == "claude-haiku-4-5"
    # A full id passes through unchanged.
    assert managed_agents.resolve_model("claude-agent/claude-opus-4-8") == "claude-opus-4-8"
    # Empty suffix → the default.
    assert managed_agents.resolve_model("claude-agent/") == managed_agents.DEFAULT_MODEL


def test_backend_for_model_and_registry():
    assert conversations.backend_for_model("claude-agent/opus") == "managed-agents"
    assert conversations.backend_for_model("claude-code/haiku") == "claude-resume"
    assert "managed-agents" in conversations.BACKENDS


def test_client_requires_api_key(monkeypatch):
    pytest.importorskip("anthropic")    # the no-key branch only after the import
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(managed_agents.ManagedAgentsError, match="ANTHROPIC_API_KEY"):
        managed_agents._client()


# --- backend behavior (unit) --------------------------------------------------

async def test_send_turn_creates_session_lazily_then_reuses(monkeypatch):
    fake = fresh(monkeypatch)
    backend = conversations.BACKENDS["managed-agents"]
    conv = router.conversation_store.create("managed-agents", "claude-agent/opus")

    # Turn 1 — creates env + agent + session, sends the user text, collects reply.
    out1 = await backend.send_turn(conv, [{"role": "user", "content": "hi"}])
    assert out1 == "echo: hi"
    assert conv.native_id == "sesn_3"            # env(1) + agent(2) + session(3)
    assert fake.rec["envs"] == ["woollama-agents"]
    assert len(fake.rec["agents"]) == 1 and len(fake.rec["sessions"]) == 1

    # Turn 2 — REUSES the session; no new env/agent/session.
    out2 = await backend.send_turn(conv, [{"role": "user", "content": "again"}])
    assert out2 == "echo: again"
    assert len(fake.rec["sessions"]) == 1 and len(fake.rec["agents"]) == 1


async def test_agent_and_env_created_once_across_conversations(monkeypatch):
    fake = fresh(monkeypatch)
    backend = conversations.BACKENDS["managed-agents"]
    a = router.conversation_store.create("managed-agents", "claude-agent/opus")
    b = router.conversation_store.create("managed-agents", "claude-agent/opus")
    await backend.send_turn(a, [{"role": "user", "content": "1"}])
    await backend.send_turn(b, [{"role": "user", "content": "2"}])
    # Same model → ONE agent and ONE environment, but a session per conversation.
    assert len(fake.rec["agents"]) == 1
    assert len(fake.rec["envs"]) == 1
    assert len(fake.rec["sessions"]) == 2
    assert a.native_id != b.native_id


async def test_history_retrieves_transcript(monkeypatch):
    fresh(monkeypatch)
    backend = conversations.BACKENDS["managed-agents"]
    conv = router.conversation_store.create("managed-agents", "claude-agent/opus")
    assert await backend.history(conv) == []          # no session yet
    await backend.send_turn(conv, [{"role": "user", "content": "ping"}])
    hist = await backend.history(conv)
    assert hist == [{"role": "user", "content": "ping"},
                    {"role": "assistant", "content": "echo: ping"}]


async def test_delete_tears_down_session(monkeypatch):
    fake = fresh(monkeypatch)
    backend = conversations.BACKENDS["managed-agents"]
    conv = router.conversation_store.create("managed-agents", "claude-agent/opus")
    await backend.send_turn(conv, [{"role": "user", "content": "x"}])
    sid = conv.native_id
    await backend.delete(conv)
    assert fake.rec["deleted"] == [sid] and conv.status == "dead"


# --- through the router surfaces -----------------------------------------------

async def test_stateful_responses_routes_to_managed_agents(monkeypatch):
    fresh(monkeypatch)
    r = await router.responses_create(FakeRequest({
        "model": "claude-agent/opus", "input": "hello", "store": True}))
    body = json.loads(r.body)
    cid = body["conversation"]["id"]
    assert body["output"][0]["content"][0]["text"] == "echo: hello"
    conv = router.conversation_store.get(cid)
    assert conv.backend == "managed-agents" and conv.native_id is not None


async def test_conversations_items_returns_transcript_for_managed(monkeypatch):
    """The capability win: managed-agents exposes `history`, so /items serves the
    transcript (200) — unlike claude-resume, which still 501s."""
    fresh(monkeypatch)
    created = json.loads((await router.conversations_create(
        FakeRequest({"model": "claude-agent/opus"}))).body)
    cid = created["id"]
    assert created["backend"] == "managed-agents"
    # Drive one turn so the session (and its transcript) exists.
    await router.responses_create(FakeRequest({
        "model": "claude-agent/opus", "input": "ping", "conversation": cid}))
    items = await router.conversations_items(cid)
    assert items.status_code == 200
    data = json.loads(items.body)["data"]
    assert [d["role"] for d in data] == ["user", "assistant"]
    assert data[0]["content"][0]["text"] == "ping"


async def test_backend_error_maps_to_502(monkeypatch):
    fresh(monkeypatch)

    def boom():
        raise managed_agents.ManagedAgentsError("agent service down")
    monkeypatch.setattr(managed_agents, "_client", boom)

    r = await router.responses_create(FakeRequest({
        "model": "claude-agent/opus", "input": "hi", "store": True}))
    assert r.status_code == 502
    assert "managed-agents backend" in json.loads(r.body)["error"]["message"]


# --- interactive requires_action round-trip (§5) ------------------------------

async def test_requires_action_pause_then_answer_round_trip(monkeypatch):
    """The model calls ask_user → Response status:requires_action carrying the
    question + conv awaiting_input; the client continues with the answer → the
    EXACT custom_tool_use_id is returned and the Response completes."""
    fake = fresh(monkeypatch)
    fake.pause_next = True

    # Turn 1 — the agent pauses on ask_user.
    r1 = await router.responses_create(FakeRequest({
        "model": "claude-agent/opus", "input": "help me", "store": True}))
    b1 = json.loads(r1.body)
    cid = b1["conversation"]["id"]
    assert b1["status"] == "requires_action"
    assert b1["required_action"] == {"type": "ask_user",
                                     "question": {"question": "What's your name?"}}
    conv = router.conversation_store.get(cid)
    assert conv.status == "awaiting_input" and conv.pending_tool_use_id == "tu_1"

    # Turn 2 — continue with the answer; resumes via user.custom_tool_result.
    r2 = await router.responses_create(FakeRequest({
        "model": "claude-agent/opus", "input": "Alice", "conversation": cid}))
    b2 = json.loads(r2.body)
    assert b2["status"] == "completed"
    assert "hello alice" in b2["output"][0]["content"][0]["text"].lower()
    # The resume sent back the EXACT id from the pause event (the conv-6 lesson).
    assert fake.received_tool_result_id == "tu_1"
    # Pending cleared → a third turn is a normal send_turn again.
    assert conv.status == "idle" and conv.required_action is None
    assert "required_action" not in b2


async def test_resume_routes_to_answer_not_send_turn(monkeypatch):
    """The routing discriminator: an attached turn on an awaiting_input conv hits
    backend.answer (custom_tool_result), NOT a fresh send_turn (user.message)."""
    fake = fresh(monkeypatch)
    fake.pause_next = True
    r1 = await router.responses_create(FakeRequest({
        "model": "claude-agent/opus", "input": "go", "store": True}))
    cid = json.loads(r1.body)["conversation"]["id"]
    sid = router.conversation_store.get(cid).native_id

    await router.responses_create(FakeRequest({
        "model": "claude-agent/opus", "input": "Bob", "conversation": cid}))
    # The answer arrived as a custom_tool_result (id recorded); had it gone through
    # send_turn it would have been a user.message and recorded nothing.
    assert fake.received_tool_result_id == "tu_1"
    # The transcript's user turns are the first message + (assistant) — the answer
    # went via the tool-result path, not appended as a user.message.
    log_roles = [r for r, _ in fake._sessions[sid]["log"]]
    assert log_roles.count("user") == 1     # only the initial "go"
