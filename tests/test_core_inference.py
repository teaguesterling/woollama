"""core.inference — the server-free stateless inference primitives.

Exercises `complete` / `complete_stream` DIRECTLY (embedder-style, no router),
including the new per-call `api_key` / `base_url` overrides that the extraction
adds. The router→core delegation for the existing surfaces is covered by
test_router / test_ollama_native / test_responses_stream.
"""
from __future__ import annotations

import httpx
import pytest

from woollama.core import inference


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


class _Client:
    """Fake httpx.AsyncClient capturing the POST (url/json/headers)."""
    calls: list = []
    resp = None

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    async def post(self, url, json=None, headers=None):
        _Client.calls.append({"url": url, "json": json, "headers": headers})
        return _Client.resp


def _use_fake(monkeypatch, payload):
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    _Client.calls = []
    _Client.resp = _Resp(payload)


async def test_complete_returns_text(monkeypatch):
    _use_fake(monkeypatch, {"choices": [{"message": {"content": "hi"}}]})
    out = await inference.complete("ollama/qwen3", [{"role": "user", "content": "x"}])
    assert out == "hi"
    call = _Client.calls[-1]
    assert call["url"].endswith("/v1/chat/completions")       # ollama base ends /v1
    assert call["json"] == {"model": "qwen3",
                            "messages": [{"role": "user", "content": "x"}],
                            "stream": False}


async def test_complete_api_key_and_base_url_override(monkeypatch):
    """The new library knobs: a per-call key/url bypass env + configured base_url
    (so an embedder drives multiple keys/endpoints without touching global config)."""
    _use_fake(monkeypatch, {"choices": [{"message": {"content": "ok"}}]})
    out = await inference.complete(
        "openai/gpt-x", [{"role": "user", "content": "x"}],
        api_key="sk-override", base_url="http://proxy:9000/v1")
    assert out == "ok"
    call = _Client.calls[-1]
    assert call["url"] == "http://proxy:9000/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer sk-override"


async def test_complete_num_ctx_routes_native(monkeypatch):
    """options.num_ctx for ollama → native /api/chat (not /v1)."""
    _use_fake(monkeypatch, {"message": {"content": "sized"}})
    out = await inference.complete("ollama/qwen3", [{"role": "user", "content": "x"}],
                                   options={"num_ctx": 16384})
    assert out == "sized"
    assert _Client.calls[-1]["url"].endswith("/api/chat")
    assert _Client.calls[-1]["json"]["options"]["num_ctx"] == 16384


async def test_complete_unknown_provider_raises():
    with pytest.raises(inference.InferenceError) as ei:
        await inference.complete("nope/x", [{"role": "user", "content": "x"}])
    assert ei.value.status == 400


async def test_complete_upstream_error_carries_payload(monkeypatch):
    _use_fake(monkeypatch, {"error": "boom"})                 # no "choices"
    with pytest.raises(inference.InferenceError) as ei:
        await inference.complete("ollama/x", [{"role": "user", "content": "x"}])
    assert ei.value.status == 502 and ei.value.payload == {"error": "boom"}


# --- streaming ----------------------------------------------------------------

class _StreamCM:
    def __init__(self, lines, status=200):
        self._lines = lines
        self.status_code = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return b'{"error": "bad"}'


class _StreamClient:
    cm = None

    def __init__(self, *a, **k):
        pass

    def stream(self, method, url, json=None, headers=None):
        _StreamClient.last = {"url": url, "json": json}
        return _StreamClient.cm

    async def aclose(self):
        return None


async def test_complete_stream_yields_deltas(monkeypatch):
    lines = ['data: {"choices":[{"delta":{"content":"a"}}]}',
             'data: {"choices":[{"delta":{"content":"b"}}]}',
             'data: [DONE]']
    monkeypatch.setattr(httpx, "AsyncClient", _StreamClient)
    _StreamClient.cm = _StreamCM(lines)
    out = [d async for d in
           inference.complete_stream("ollama/x", [{"role": "user", "content": "x"}])]
    assert out == ["a", "b"]
    assert _StreamClient.last["json"]["stream"] is True


async def test_complete_stream_upstream_status_raises(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", _StreamClient)
    _StreamClient.cm = _StreamCM([], status=503)
    with pytest.raises(inference.InferenceError) as ei:
        async for _ in inference.complete_stream("ollama/x", [{"role": "user", "content": "x"}]):
            pass
    assert ei.value.status == 503 and ei.value.payload == {"error": "bad"}
