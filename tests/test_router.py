"""Unit tests for `woollama.router`'s route handlers.

We call the handler functions directly instead of going through FastAPI's
TestClient. This avoids the lifespan running (which would spawn real MCP
subprocesses) and keeps the tests purely about the dispatch logic.

The tests:
  * /v1/models — Ollama call mocked; verifies envelope shape + recipe presence
  * /v1/tools  — registry contents surfaced as plain JSON
  * /v1/chat/completions:
      - unknown model namespace → 400
      - woollama/<unknown recipe> → 404
      - ollama/<X> → pass-through (Ollama call mocked)
      - woollama/<recipe>, no tool_calls in Ollama response → final answer
      - woollama/<recipe>, tool_calls present → loop dispatches + composes
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from woollama import recipes, router
from woollama.manager import Registry, ServerManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeRequest:
    """Stand-in for Starlette's Request — only the body matters for these
    handlers; we don't need URL routing because we're calling functions."""

    def __init__(self, body: dict):
        self._body = body

    async def json(self) -> dict:
        return self._body


class HttpxResponseStub:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def mock_httpx(monkeypatch, post_responses: list[dict] | dict | None = None,
               get_payload: dict | None = None):
    """Monkeypatch httpx.AsyncClient with predetermined responses. Use a
    list for post_responses to script multi-turn loops; a single dict for
    repeated identical responses."""

    if post_responses is None:
        post_responses = {}
    if isinstance(post_responses, dict):
        post_script = None
        single_post = post_responses
    else:
        post_script = list(post_responses)
        single_post = None

    class _FakeClient:
        def __init__(self, *_a, **_kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return None

        async def get(self, _url, **_kw):
            return HttpxResponseStub(200, get_payload or {})

        async def post(self, _url, json=None, **_kw):
            if post_script is not None:
                return HttpxResponseStub(200, post_script.pop(0))
            return HttpxResponseStub(200, single_post)

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)


def _tool(name: str, description: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema={"type": "object", "properties": {}},
    )


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

async def test_models_lists_ollama_prefixed_plus_recipes(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()
    mock_httpx(monkeypatch, get_payload={"data": [
        {"id": "qwen3:14b-iq4xs"}, {"id": "llama3:8b"},
    ]})
    resp = await router.list_models()
    data = json.loads(resp.body)["data"]
    ids = [d["id"] for d in data]
    assert "ollama/qwen3:14b-iq4xs" in ids
    assert "ollama/llama3:8b" in ids
    assert "woollama/streamer" in ids
    assert "woollama/textcounter" in ids


async def test_models_survives_ollama_unreachable(monkeypatch, tmp_path):
    """If Ollama is down, /v1/models still returns recipes (just no
    ollama/* entries)."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()
    # FakeClient that raises on get()
    class _Boom:
        def __init__(self, *_a, **_kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_a): return None
        async def get(self, *_a, **_kw):
            raise ConnectionError("ollama unreachable")
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Boom)
    resp = await router.list_models()
    data = json.loads(resp.body)["data"]
    ids = [d["id"] for d in data]
    assert not any(i.startswith("ollama/") for i in ids)
    assert any(i.startswith("woollama/") for i in ids)


# ---------------------------------------------------------------------------
# /v1/tools (the introspection endpoint)
# ---------------------------------------------------------------------------

async def test_tools_endpoint_lists_namespaced_registry(monkeypatch):
    fake_reg = Registry()
    mgr = ServerManager("hello", "echo", [])
    mgr.tools = [_tool("count_to"), _tool("ask_user")]
    fake_reg.add(mgr)
    monkeypatch.setattr(router, "registry", fake_reg)
    resp = await router.list_tools()
    body = json.loads(resp.body)
    assert sorted(body["tools"]) == ["hello.ask_user", "hello.count_to"]


# ---------------------------------------------------------------------------
# /v1/chat/completions — error paths
# ---------------------------------------------------------------------------

async def test_chat_unknown_model_namespace_returns_400():
    resp = await router.chat_completions(
        FakeRequest({"model": "unknown/foo", "messages": []}))
    assert resp.status_code == 400
    body = json.loads(resp.body)
    assert body["error"]["type"] == "invalid_request_error"
    assert "unknown model namespace" in body["error"]["message"]


async def test_chat_unknown_woollama_recipe_returns_404(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()
    resp = await router.chat_completions(
        FakeRequest({"model": "woollama/does-not-exist", "messages": []}))
    assert resp.status_code == 404
    body = json.loads(resp.body)
    assert body["error"]["type"] == "not_found"


# ---------------------------------------------------------------------------
# /v1/chat/completions — pass-through
# ---------------------------------------------------------------------------

async def test_chat_ollama_passthrough_strips_prefix_and_forwards(monkeypatch):
    """ollama/<X> path should strip the prefix before forwarding."""
    captured: dict = {}

    class _SpyClient:
        def __init__(self, *_a, **_kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_a): return None
        async def post(self, url, json=None, **_kw):
            captured["url"] = url
            captured["body"] = json
            return HttpxResponseStub(
                200, {"choices": [{"message": {"content": "ok"}}]})
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _SpyClient)

    await router.chat_completions(FakeRequest({
        "model": "ollama/qwen3:14b-iq4xs",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }))
    assert "/v1/chat/completions" in captured["url"]
    # prefix stripped, stream forced to False
    assert captured["body"]["model"] == "qwen3:14b-iq4xs"
    assert captured["body"]["stream"] is False


# ---------------------------------------------------------------------------
# /v1/chat/completions — recipe orchestration
# ---------------------------------------------------------------------------

async def test_chat_recipe_no_tool_calls_returns_final_answer(monkeypatch, tmp_path):
    """A recipe whose first inferencer turn produces final content (no tool
    calls) returns that content directly — single loop iteration."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()

    # Use a registry with the right tools available so the recipe's allow-list resolves
    fake_reg = Registry()
    mgr = ServerManager("hello", "echo", [])
    mgr.tools = [_tool("count_to")]
    fake_reg.add(mgr)
    monkeypatch.setattr(router, "registry", fake_reg)

    mock_httpx(monkeypatch, post_responses={
        "choices": [{"message": {"content": "the final answer"}}],
    })

    resp = await router.chat_completions(FakeRequest({
        "model": "woollama/streamer",
        "messages": [{"role": "user", "content": "count to 3"}],
    }))
    body = json.loads(resp.body)
    assert body["choices"][0]["message"]["content"] == "the final answer"


async def test_chat_recipe_with_tool_calls_loops_to_completion(monkeypatch, tmp_path):
    """First turn emits a tool_call; daemon dispatches; second turn emits the
    final content. The client gets back the final answer."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()

    # Need a registry whose 'hello.count_to' dispatches via a stub manager
    fake_reg = Registry()
    mgr = ServerManager("hello", "echo", [])
    mgr.tools = [_tool("count_to")]

    # Override .call_tool to a stub — bypass the queue/task machinery for
    # this dispatch test (manager-internal mechanics are covered in
    # test_manager.py).
    async def stub_call_tool(name, args):
        return SimpleNamespace(content=[
            SimpleNamespace(text=f'{{"count":{args.get("n")},"done":true}}'),
        ])
    mgr.call_tool = stub_call_tool  # type: ignore[method-assign]
    fake_reg.add(mgr)
    monkeypatch.setattr(router, "registry", fake_reg)

    # Two Ollama responses: turn 1 emits a tool_call; turn 2 produces the answer.
    mock_httpx(monkeypatch, post_responses=[
        {"choices": [{"message": {
            "content": "",
            "tool_calls": [{"id": "c1", "function": {
                "name": "hello.count_to",
                "arguments": '{"n": 3}',
            }}],
        }}]},
        {"choices": [{"message": {"content": "Counted to 3."}}]},
    ])

    resp = await router.chat_completions(FakeRequest({
        "model": "woollama/streamer",
        "messages": [{"role": "user", "content": "count to 3"}],
    }))
    body = json.loads(resp.body)
    assert body["choices"][0]["message"]["content"] == "Counted to 3."


async def test_chat_recipe_unknown_inferencer_returns_501(monkeypatch, tmp_path):
    """A recipe pointing at a provider woollama doesn't know should fail clearly,
    not silently. (ollama + anthropic are known; this uses a made-up one.)"""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "recipes.toml").write_text("""
[recipes.bogus]
inferencer = "no-such-provider/some-model"
tools = []
system = "test"
""")
    recipes.reload()

    fake_reg = Registry()
    monkeypatch.setattr(router, "registry", fake_reg)

    resp = await router.chat_completions(FakeRequest({
        "model": "woollama/bogus",
        "messages": [{"role": "user", "content": "hi"}],
    }))
    assert resp.status_code == 501
    body = json.loads(resp.body)
    assert body["error"]["type"] == "not_implemented"
    assert "unsupported inferencer" in body["error"]["message"]
