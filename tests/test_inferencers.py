"""Tests for the OpenAI-compat inferencer seam (inferencers.py + router wiring).

Unit tests of the registry, plus routing assertions that woollama sends the
RIGHT request to the RIGHT backend (URL + auth header + model + provider-specific
body) for both orchestration and pass-through. Upstream is mocked — these prove
what woollama *emits*, not what Anthropic accepts (the docs confirm tools work;
a real round-trip is the opt-in @needs_anthropic live test in test_integration).
"""
from __future__ import annotations

import json

import pytest

from woollama import inferencers, recipes, router
from woollama.manager import Registry


# ---------------------------------------------------------------------------
# Registry unit tests
# ---------------------------------------------------------------------------

def test_builtins_present():
    assert {"ollama", "anthropic"} <= set(inferencers.names())
    assert inferencers.get("no-such-provider") is None


def test_ollama_url_no_auth(monkeypatch):
    monkeypatch.setenv("WOOLLAMA_OLLAMA_URL", "http://box:1234")
    inf = inferencers.get("ollama")
    assert inf.chat_url() == "http://box:1234/v1/chat/completions"
    assert inf.headers() == {}                      # no auth for local ollama
    assert inf.extra_body == {"options": {"temperature": 0}}


def test_anthropic_url_and_bearer_auth(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xyz")
    inf = inferencers.get("anthropic")
    assert inf.chat_url() == "https://api.anthropic.com/v1/chat/completions"
    assert inf.headers() == {"Authorization": "Bearer sk-ant-xyz"}
    assert inf.extra_body.get("max_tokens")          # anthropic gets a default


def test_anthropic_missing_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(inferencers.InferencerError, match="ANTHROPIC_API_KEY"):
        inferencers.get("anthropic").headers()


# ---------------------------------------------------------------------------
# Routing: a capturing fake httpx so we can assert URL + headers + body
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, payload, status=200):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def _capture_httpx(monkeypatch, payload):
    """Patch httpx.AsyncClient; record the last POST (url, json, headers)."""
    seen = {}

    class _Client:
        def __init__(self, *_a, **_kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_a): return None
        async def get(self, *_a, **_kw): return _Resp({})
        async def post(self, url, json=None, headers=None, **_kw):
            seen["url"], seen["json"], seen["headers"] = url, json, headers or {}
            return _Resp(payload)

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return seen


async def test_orchestrate_routes_to_anthropic_with_auth(monkeypatch, tmp_path):
    """A woollama recipe with an anthropic inferencer orchestrates against the
    Anthropic compat endpoint: right URL, Bearer auth, bare model, and the
    provider's extra_body (max_tokens) merged in."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    (tmp_path / "recipes.toml").write_text(
        '[recipes.cloud]\ninferencer="anthropic/claude-sonnet-4-6"\ntools=[]\n'
        'system="be brief"\n')
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path)); recipes.reload()

    seen = _capture_httpx(monkeypatch, {"choices": [{"message": {"content": "ok"}}]})

    resp = await router.orchestrate(
        recipes.get("cloud"), [{"role": "user", "content": "hi"}], Registry())

    assert resp["choices"][0]["message"]["content"] == "ok"
    assert seen["url"] == "https://api.anthropic.com/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer sk-ant-test"
    assert seen["json"]["model"] == "claude-sonnet-4-6"        # prefix stripped
    assert seen["json"]["max_tokens"]                          # extra_body merged
    assert seen["json"]["messages"][0] == {"role": "system", "content": "be brief"}


async def test_orchestrate_ollama_unchanged(monkeypatch, tmp_path):
    """Regression: the ollama path still posts to the ollama URL with no auth and
    its native `options` body (the generalization must not change ollama)."""
    monkeypatch.setenv("WOOLLAMA_OLLAMA_URL", "http://localhost:11434")
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path)); recipes.reload()
    seen = _capture_httpx(monkeypatch, {"choices": [{"message": {"content": "done"}}]})

    await router.orchestrate(
        recipes.get("streamer"), [{"role": "user", "content": "count to 3"}], Registry())

    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["headers"] == {}
    assert seen["json"]["options"] == {"temperature": 0}


async def test_passthrough_routes_to_anthropic(monkeypatch):
    """A direct `anthropic/<model>` request pass-through goes to the Anthropic
    endpoint with auth and the bare model, no orchestration."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-pt")
    seen = _capture_httpx(monkeypatch, {"choices": [{"message": {"content": "pong"}}]})

    class FakeRequest:
        def __init__(self, body): self._b = body
        async def json(self): return self._b

    await router.chat_completions(FakeRequest({
        "model": "anthropic/claude-haiku-4-5",
        "messages": [{"role": "user", "content": "hi"}], "stream": True}))

    assert seen["url"] == "https://api.anthropic.com/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer sk-ant-pt"
    assert seen["json"]["model"] == "claude-haiku-4-5"
    assert seen["json"]["stream"] is False
