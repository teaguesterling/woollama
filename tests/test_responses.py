"""Unit tests for the stateful surface — /v1/responses (slice conv-1a).

conv-1a is the STATELESS subset: a Responses-shaped superset of
/v1/chat/completions. These tests prove (a) the wire shape parses in the real
`openai` SDK — the whole reason to adopt the shape — (b) input parsing, (c)
routing parity with chat-completions, and (d) the stateful opt-in is cleanly
deferred (501) rather than half-implemented.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from woollama import recipes, responses, router
from woollama.manager import Registry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeRequest:
    def __init__(self, body: dict):
        self._body = body

    async def json(self) -> dict:
        return self._body


def mock_post(monkeypatch, payload):
    """Serve `payload` for every inferencer POST from a mock HTTP server + point
    $WOOLLAMA_OLLAMA_URL at it. The recipe loop / complete run in the Rust core
    (reqwest), which an in-process httpx patch can't intercept."""
    class _H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("WOOLLAMA_OLLAMA_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    return srv


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
# Stateful opt-in: claude-code routes to a backend; non-owners are 501 (conv-1b)
# ---------------------------------------------------------------------------

async def test_responses_store_true_non_owner_is_501():
    """store:true on a non-claude model is a 501: woollama owns no conversation
    storage, so a model with no state-owning backend has no stateful path — the
    caller owns history (store:false)."""
    resp = await router.responses_create(FakeRequest({
        "model": "ollama/x", "input": "hi", "store": True}))
    assert resp.status_code == 501
    body = json.loads(resp.body)
    assert body["error"]["type"] == "not_implemented"
    assert "store:false" in body["error"]["message"]


async def test_responses_attach_unknown_conversation_is_404():
    # conv-1b: attach is implemented; an unknown handle is a 404, not a 501.
    resp = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "hi", "conversation": "conv_nope"}))
    assert resp.status_code == 404


async def test_responses_unknown_previous_response_id_is_404():
    resp = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "hi", "previous_response_id": "resp_nope"}))
    assert resp.status_code == 404


async def test_responses_stateful_stream_is_400():
    """Stateless `stream:true` now streams Responses SSE (see
    test_responses_stream.py); STATEFUL streaming is the still-deferred case → 400."""
    resp = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "hi", "stream": True, "store": True}))
    assert resp.status_code == 400
    assert json.loads(resp.body)["error"]["type"] == "invalid_request_error"


async def test_responses_bad_input_type_is_400():
    """`input` that is neither a string nor a list → parse_input raises, mapped
    to a 400 (the negative half of the well-tested positive parse cases)."""
    resp = await router.responses_create(FakeRequest({
        "model": "ollama/x", "input": 123}))
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
