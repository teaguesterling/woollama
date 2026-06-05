"""conv-1b — stateful /v1/responses via the claude-resume backend.

Proves handle routing without a real `claude`: the subprocess seam
(`claude_code._invoke`) is patched, so these assert that woollama mints a
conversation handle, captures the backend's session_id on the first turn, and
RESUMES that same session (`--resume <sid>`) when the conversation is continued —
by handle or by previous_response_id. The live counterpart is the opt-in
claude-code integration test.
"""
from __future__ import annotations

import json

from woollama import claude_code, conversations, router


class FakeRequest:
    def __init__(self, body: dict):
        self._body = body

    async def json(self) -> dict:
        return self._body


def _result(text: str, sid: str) -> bytes:
    """A `claude -p --output-format json` result event carrying a session_id."""
    return json.dumps([
        {"type": "result", "subtype": "success", "is_error": False,
         "result": text, "session_id": sid},
    ]).encode()


def patch_claude(monkeypatch, sid: str = "sid-1"):
    """Patch the subprocess seam; record each invocation's args. The result
    always reports `sid` (claude resume returns the SAME session_id)."""
    calls: list[list[str]] = []

    async def fake_invoke(args, env, cwd, timeout):
        calls.append(list(args))
        return 0, _result(f"turn{len(calls)}", sid), b""

    monkeypatch.setattr(claude_code, "_invoke", fake_invoke)
    return calls


def fresh_store(monkeypatch):
    store = conversations.ConversationStore()
    monkeypatch.setattr(router, "conversation_store", store)
    return store


# ---------------------------------------------------------------------------
# Handle table (unit)
# ---------------------------------------------------------------------------

def test_store_create_get_and_response_mapping():
    store = conversations.ConversationStore()
    conv = store.create("claude-resume", "claude-code/haiku")
    assert conv.id.startswith("conv_") and conv.native_id is None
    assert store.get(conv.id) is conv
    store.record_response(conv, "resp_1")
    assert store.by_response("resp_1") is conv
    assert store.by_response("resp_unknown") is None


def test_backend_for_model():
    assert conversations.backend_for_model("claude-code/haiku") == "claude-resume"
    assert conversations.backend_for_model("ollama/x") is None
    assert conversations.backend_for_model("woollama/streamer") is None


# ---------------------------------------------------------------------------
# Stateful /v1/responses — create then continue (handle routing)
# ---------------------------------------------------------------------------

async def test_stateful_create_captures_session_then_resumes_by_conversation(monkeypatch):
    calls = patch_claude(monkeypatch, sid="sid-xyz")
    fresh_store(monkeypatch)

    # Turn 1 — store:true with a claude-code model creates a conversation.
    r1 = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "hello", "store": True}))
    b1 = json.loads(r1.body)
    conv_id = b1["conversation"]["id"]         # response carries a conversation object
    assert conv_id.startswith("conv_")
    assert b1["output"][0]["content"][0]["text"] == "turn1"
    assert "--resume" not in calls[0]          # first turn STARTS the session

    # Turn 2 — attach by conversation id (a bare string) → RESUMES the session.
    r2 = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "again", "conversation": conv_id}))
    b2 = json.loads(r2.body)
    assert b2["conversation"]["id"] == conv_id
    assert "--resume" in calls[1]
    assert calls[1][calls[1].index("--resume") + 1] == "sid-xyz"


async def test_stateful_continue_via_previous_response_id(monkeypatch):
    calls = patch_claude(monkeypatch, sid="sid-prev")
    fresh_store(monkeypatch)

    r1 = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "hi", "store": True}))
    resp_id = json.loads(r1.body)["id"]

    r2 = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "more",
        "previous_response_id": resp_id}))
    b2 = json.loads(r2.body)
    # chained onto the SAME conversation, resuming its session
    assert b2["conversation"]["id"] == json.loads(r1.body)["conversation"]["id"]
    assert "--resume" in calls[1] and calls[1][calls[1].index("--resume") + 1] == "sid-prev"


async def test_stateful_response_parses_in_openai_sdk(monkeypatch):
    from openai.types.responses import Response
    patch_claude(monkeypatch, sid="s1")
    fresh_store(monkeypatch)
    r = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "hi", "store": True}))
    parsed = Response.model_validate(json.loads(r.body))
    assert parsed.output_text == "turn1"
    assert parsed.conversation.id == json.loads(r.body)["conversation"]["id"]


# ---------------------------------------------------------------------------
# Routing errors
# ---------------------------------------------------------------------------

async def test_stateful_non_claude_model_is_501(monkeypatch):
    fresh_store(monkeypatch)
    r = await router.responses_create(FakeRequest({
        "model": "ollama/qwen3", "input": "hi", "store": True}))
    assert r.status_code == 501
    assert "stateful" in json.loads(r.body)["error"]["message"]


async def test_stateful_prev_response_belonging_to_other_conversation_is_400(monkeypatch):
    patch_claude(monkeypatch)
    fresh_store(monkeypatch)
    # Two independent conversations.
    a = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "a", "store": True}))
    b = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "b", "store": True}))
    conv_a = json.loads(a.body)["conversation"]["id"]
    resp_b = json.loads(b.body)["id"]
    # prev belongs to B but we name conversation A → conflict.
    r = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "x",
        "conversation": conv_a, "previous_response_id": resp_b}))
    assert r.status_code == 400


async def test_stateful_backend_error_is_502(monkeypatch):
    fresh_store(monkeypatch)

    async def boom(args, env, cwd, timeout):
        return 1, b"", b"claude exploded"
    monkeypatch.setattr(claude_code, "_invoke", boom)

    r = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "hi", "store": True}))
    assert r.status_code == 502
    assert "claude-resume backend" in json.loads(r.body)["error"]["message"]
