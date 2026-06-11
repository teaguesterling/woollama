"""Issue #1 — native Ollama /api/chat routing so `num_ctx` is honored.

The pure translators are tested against the wire shapes captured live from
ollama; the router wiring is tested with a mocked httpx so it asserts that an
`ollama/<model>` request carrying `options.num_ctx` is POSTed to /api/chat (not
/v1/chat/completions) with num_ctx intact, and that the native response is
translated back to the OpenAI chat-completions shape. The live acceptance
(`ollama ps` shows the requested context) is the opt-in integration test.
"""
from __future__ import annotations

import json

import httpx

from woollama import router
from woollama import ollama_native


class FakeRequest:
    def __init__(self, body: dict):
        self._body = body

    async def json(self) -> dict:
        return self._body


# --- pure translation ---------------------------------------------------------

def test_wants_native_only_with_num_ctx_and_no_tools():
    assert ollama_native.wants_native({"options": {"num_ctx": 16384}}) is True
    assert ollama_native.wants_native({"options": {"temperature": 0}}) is False
    assert ollama_native.wants_native({}) is False
    # tools present → stay on /v1 (tool-calling works there; num_ctx not honored)
    assert ollama_native.wants_native(
        {"options": {"num_ctx": 8192}, "tools": [{"type": "function"}]}) is False


def test_native_chat_url_strips_v1():
    assert ollama_native.native_chat_url("http://localhost:11434/v1") \
        == "http://localhost:11434/api/chat"
    assert ollama_native.native_chat_url("http://host:11434/v1/") \
        == "http://host:11434/api/chat"


def test_to_native_request_folds_params_and_keeps_num_ctx():
    req = ollama_native.to_native_request({
        "model": "qwen3:14b", "messages": [{"role": "user", "content": "hi"}],
        "stream": True, "temperature": 0.5, "max_tokens": 256,
        "options": {"num_ctx": 16384}})
    assert req["model"] == "qwen3:14b"
    assert req["stream"] is True
    assert req["options"]["num_ctx"] == 16384      # preserved (the whole point)
    assert req["options"]["temperature"] == 0.5    # folded from top-level
    assert req["options"]["num_predict"] == 256    # max_tokens → num_predict
    assert req["messages"] == [{"role": "user", "content": "hi"}]


def test_to_native_request_does_not_clobber_caller_options():
    req = ollama_native.to_native_request({
        "model": "m", "temperature": 0.9,
        "options": {"num_ctx": 4096, "temperature": 0.1}})
    assert req["options"]["temperature"] == 0.1    # caller's options win


def test_from_native_response_maps_shape_and_usage():
    # Captured live from ollama /api/chat (stream:false).
    native = {
        "model": "qwen3.5:4b",
        "created_at": "2026-06-07T22:46:47.476910632Z",
        "message": {"role": "assistant", "content": "hello"},
        "done": True, "done_reason": "stop",
        "prompt_eval_count": 19, "eval_count": 20,
    }
    out = ollama_native.from_native_response(native, "qwen3.5:4b")
    assert out["object"] == "chat.completion"
    assert isinstance(out["created"], int)         # NOT the RFC3339 string
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"] == {"prompt_tokens": 19, "completion_tokens": 20,
                            "total_tokens": 39}


def test_from_native_response_maps_length_finish():
    out = ollama_native.from_native_response(
        {"message": {"content": ""}, "done": True, "done_reason": "length"}, "m")
    assert out["choices"][0]["finish_reason"] == "length"


def test_sse_translator_against_captured_frames():
    # NDJSON frames in the exact shape ollama streams (content deltas + a final
    # done frame). Role rides the first content chunk; done → finish + [DONE].
    translate = ollama_native.sse_translator("qwen3.5:4b")
    lines = [
        '{"message":{"role":"assistant","content":"one"},"done":false}',
        '{"message":{"role":"assistant","content":" two"},"done":false}',
        '{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}',
    ]
    chunks = [c for ln in lines for c in translate(ln)]
    texts = [c.decode() for c in chunks]
    # first content chunk carries role
    first = json.loads(texts[0][len("data: "):])
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": "one"}
    assert first["object"] == "chat.completion.chunk"
    second = json.loads(texts[1][len("data: "):])
    assert second["choices"][0]["delta"] == {"content": " two"}
    # terminal chunk has finish_reason then a [DONE] sentinel
    term = json.loads(texts[2][len("data: "):])
    assert term["choices"][0]["finish_reason"] == "stop"
    assert texts[3] == "data: [DONE]\n\n"


# --- router wiring (mocked httpx) ---------------------------------------------

class _Resp:
    def __init__(self, payload: dict, status: int = 200):
        self.status_code = status
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


def _mock_httpx(monkeypatch, posts: list, native_payload: dict | None = None):
    """Capture every POST (url, json). Non-stream POSTs return native_payload."""
    payload = native_payload or {
        "model": "qwen3:14b", "message": {"role": "assistant", "content": "ok"},
        "done": True, "done_reason": "stop",
        "prompt_eval_count": 5, "eval_count": 3,
    }

    class _Client:
        def __init__(self, *_a, **_kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_a): return None
        async def get(self, *_a, **_kw): return _Resp({})
        async def post(self, url, json=None, **_kw):
            posts.append((url, json))
            return _Resp(payload)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)


async def test_passthrough_routes_num_ctx_to_native_endpoint(monkeypatch):
    posts: list = []
    _mock_httpx(monkeypatch, posts)
    r = await router.chat_completions(FakeRequest({
        "model": "ollama/qwen3:14b",
        "messages": [{"role": "user", "content": "long input"}],
        "options": {"num_ctx": 16384}}))
    assert r.status_code == 200
    # Hit the NATIVE endpoint, with num_ctx intact.
    url, sent = posts[-1]
    assert url.endswith("/api/chat")
    assert sent["options"]["num_ctx"] == 16384
    assert sent["model"] == "qwen3:14b"            # bare, not namespaced
    # Response translated back to OpenAI chat.completion.
    body = json.loads(r.body)
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "ok"


async def test_passthrough_without_num_ctx_stays_on_v1(monkeypatch):
    posts: list = []
    # /v1 returns an OpenAI-shaped body (passthrough returns it verbatim).
    _mock_httpx(monkeypatch, posts,
                native_payload={"choices": [{"message": {"content": "x"}}]})
    await router.chat_completions(FakeRequest({
        "model": "ollama/qwen3:14b",
        "messages": [{"role": "user", "content": "hi"}]}))
    assert posts[-1][0].endswith("/v1/chat/completions")


async def test_passthrough_num_ctx_with_tools_stays_on_v1(monkeypatch):
    posts: list = []
    _mock_httpx(monkeypatch, posts,
                native_payload={"choices": [{"message": {"content": "x"}}]})
    await router.chat_completions(FakeRequest({
        "model": "ollama/qwen3:14b",
        "messages": [{"role": "user", "content": "hi"}],
        "options": {"num_ctx": 8192},
        "tools": [{"type": "function", "function": {"name": "f"}}]}))
    assert posts[-1][0].endswith("/v1/chat/completions")  # tools win; native skipped


async def test_complete_stateless_routes_num_ctx_native(monkeypatch):
    """complete_stateless (backs /v1/responses + the store-backed path) honors
    num_ctx for ollama by going native — closing the #1↔#2 seam."""
    posts: list = []
    _mock_httpx(monkeypatch, posts,
                native_payload={"message": {"content": "ok"}, "done": True})
    out = await router.complete_stateless(
        "ollama/qwen3", [{"role": "user", "content": "hi"}],
        options={"num_ctx": 16384})
    assert out == "ok"
    url, sent = posts[-1]
    assert url.endswith("/api/chat") and sent["options"]["num_ctx"] == 16384


async def test_complete_stateless_without_num_ctx_uses_v1(monkeypatch):
    posts: list = []
    _mock_httpx(monkeypatch, posts,
                native_payload={"choices": [{"message": {"content": "x"}}]})
    out = await router.complete_stateless(
        "ollama/qwen3", [{"role": "user", "content": "hi"}])
    assert out == "x"
    assert posts[-1][0].endswith("/v1/chat/completions")


async def test_native_stream_translates_ndjson_to_sse(monkeypatch):
    sent: dict = {}
    frames = [
        '{"message":{"role":"assistant","content":"he"},"done":false}',
        '{"message":{"role":"assistant","content":"llo"},"done":false}',
        '{"message":{"content":""},"done":true,"done_reason":"stop"}',
    ]

    class _Stream:
        def __init__(self, url, json=None):
            sent["url"] = url
            sent["json"] = json
            self.status_code = 200
        async def __aenter__(self): return self
        async def __aexit__(self, *_a): return None
        async def aiter_lines(self):
            for f in frames:
                yield f

    class _Client:
        def __init__(self, *_a, **_kw): pass
        async def aclose(self): return None
        def stream(self, _method, url, json=None, **_kw):
            return _Stream(url, json)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    r = await router.chat_completions(FakeRequest({
        "model": "ollama/qwen3:14b", "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
        "options": {"num_ctx": 16384}}))
    body = b"".join([chunk async for chunk in r.body_iterator]).decode()
    assert sent["url"].endswith("/api/chat")
    assert sent["json"]["options"]["num_ctx"] == 16384
    assert sent["json"]["stream"] is True
    assert '"content": "he"' in body and '"content": "llo"' in body
    assert "data: [DONE]\n\n" in body
