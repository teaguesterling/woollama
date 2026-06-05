"""Unit tests for the stateful surface — /v1/responses (slice conv-1a).

conv-1a is the STATELESS subset: a Responses-shaped superset of
/v1/chat/completions. These tests prove (a) the wire shape parses in the real
`openai` SDK — the whole reason to adopt the shape — (b) input parsing, (c)
routing parity with chat-completions, and (d) the stateful opt-in is cleanly
deferred (501) rather than half-implemented.
"""
from __future__ import annotations

import json

from woollama import conversations, recipes, responses, router
from woollama.manager import Registry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeRequest:
    def __init__(self, body: dict):
        self._body = body

    async def json(self) -> dict:
        return self._body


class _Stub:
    def __init__(self, payload, status=200):
        self.status_code = status
        self._p = payload

    def json(self):
        return self._p


def mock_post(monkeypatch, payload):
    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        async def get(self, *a, **k): return _Stub({})
        async def post(self, *a, **k): return _Stub(payload)
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)


# ---------------------------------------------------------------------------
# Pure shaping
# ---------------------------------------------------------------------------

def test_parse_input_string_is_one_user_turn():
    assert responses.parse_input("hi") == [{"role": "user", "content": "hi"}]


def test_parse_input_list_with_string_and_part_content():
    out = responses.parse_input([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": [{"type": "input_text", "text": "a"},
                                     {"type": "input_text", "text": "b"}]},
    ])
    assert out == [{"role": "system", "content": "be brief"},
                   {"role": "user", "content": "ab"}]


def test_build_response_parses_in_the_openai_sdk():
    """The wire shape must round-trip through the REAL openai SDK and expose the
    SDK-computed `.output_text` — this is the contract conv-1a exists to meet."""
    from openai.types.responses import Response
    d = responses.build_response(responses.new_id("resp"), "ollama/x", "hello world")
    r = Response.model_validate(d)
    assert r.id.startswith("resp_")
    assert r.status == "completed"
    assert r.output_text == "hello world"     # SDK aggregates our output_text part


# ---------------------------------------------------------------------------
# /v1/responses — stateless routing (parity with chat-completions)
# ---------------------------------------------------------------------------

async def test_responses_stateless_recipe_returns_responses_shape(monkeypatch, tmp_path):
    """woollama/<recipe> with store:false runs orchestrate and returns a valid
    Responses object (verified by parsing it back through the SDK)."""
    from openai.types.responses import Response
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "recipes.toml").write_text(
        '[recipes.chat]\ninferencer="ollama/x"\ntools=[]\nsystem="be brief"\n')
    recipes.reload()
    monkeypatch.setattr(router, "registry", Registry())
    mock_post(monkeypatch, {"choices": [{"message": {"content": "the answer"}}]})

    resp = await router.responses_create(FakeRequest({
        "model": "woollama/chat", "input": "hello", "store": False}))
    body = json.loads(resp.body)
    assert Response.model_validate(body).output_text == "the answer"
    assert body["id"].startswith("resp_") and body["status"] == "completed"


async def test_responses_stateless_passthrough_provider(monkeypatch):
    """A known inferencer (ollama/<model>) routes through passthrough; the
    assistant content lands in the Responses output."""
    from openai.types.responses import Response
    mock_post(monkeypatch, {"choices": [{"message": {"content": "pong"}}]})
    resp = await router.responses_create(FakeRequest({
        "model": "ollama/qwen3", "input": [{"role": "user", "content": "ping"}]}))
    body = json.loads(resp.body)
    assert Response.model_validate(body).output_text == "pong"
    assert body["model"] == "ollama/qwen3"


# ---------------------------------------------------------------------------
# Stateful opt-in routes to a backend (claude-resume or stored) — conv-1b/conv-5
# ---------------------------------------------------------------------------

async def test_responses_store_true_routes_to_stored(monkeypatch, tmp_path):
    """store:true on a non-claude model now creates a server-owned `stored`
    conversation (conv-5) — no longer a 501."""
    monkeypatch.setattr(router, "conversation_store", conversations.ConversationStore())
    monkeypatch.setattr(conversations, "_stored",
                        conversations.StoredStore(str(tmp_path / "c.duckdb")))

    async def fake_complete(model, messages):
        return "hi back"
    monkeypatch.setattr(router, "complete_stateless", fake_complete)

    resp = await router.responses_create(FakeRequest({
        "model": "ollama/x", "input": "hi", "store": True}))
    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert body["conversation"]["id"].startswith("conv_")
    assert body["output"][0]["content"][0]["text"] == "hi back"


async def test_responses_attach_unknown_conversation_is_404():
    # conv-1b: attach is implemented; an unknown handle is a 404, not a 501.
    resp = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "hi", "conversation": "conv_nope"}))
    assert resp.status_code == 404


async def test_responses_unknown_previous_response_id_is_404():
    resp = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "hi", "previous_response_id": "resp_nope"}))
    assert resp.status_code == 404


async def test_responses_stream_true_is_400():
    resp = await router.responses_create(FakeRequest({
        "model": "ollama/x", "input": "hi", "stream": True}))
    assert resp.status_code == 400
    assert json.loads(resp.body)["error"]["type"] == "invalid_request_error"


async def test_responses_unknown_recipe_404(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()
    resp = await router.responses_create(FakeRequest({
        "model": "woollama/nope", "input": "hi"}))
    assert resp.status_code == 404


async def test_responses_unknown_namespace_400():
    resp = await router.responses_create(FakeRequest({
        "model": "bogus/x", "input": "hi"}))
    assert resp.status_code == 400
    assert "unknown model namespace" in json.loads(resp.body)["error"]["message"]
