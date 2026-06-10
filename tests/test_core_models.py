"""Phase 4 ergonomics — ModelRegistry, complete_sync, make_recipe.

Proves an embedder can build its provider set IN MEMORY (no config files) and
drive inference against it, call the sync wrapper, and construct recipes in code.
"""
from __future__ import annotations

import httpx
import pytest

from woollama.core import (
    InferenceError,
    Inferencer,
    ModelRegistry,
    complete,
    complete_sync,
    make_recipe,
)


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


class _Client:
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


# --- ModelRegistry ------------------------------------------------------------

async def test_in_memory_registry_resolves_and_routes(monkeypatch):
    reg = ModelRegistry()
    reg.add(Inferencer(name="local", base_url="http://local:1234/v1"))
    _use_fake(monkeypatch, {"choices": [{"message": {"content": "hi"}}]})
    out = await complete("local/m", [{"role": "user", "content": "x"}], registry=reg)
    assert out == "hi"
    assert _Client.calls[-1]["url"] == "http://local:1234/v1/chat/completions"
    assert reg.names() == ["local"] and reg.get("local").base_url.endswith("/v1")


async def test_in_memory_registry_unknown_provider_is_400(monkeypatch):
    with pytest.raises(InferenceError) as ei:
        await complete("nope/m", [{"role": "user", "content": "x"}],
                       registry=ModelRegistry())
    assert ei.value.status == 400


def test_from_config_has_builtins():
    reg = ModelRegistry.from_config()
    assert reg.get("ollama") is not None        # built-in present
    assert "ollama" in reg.names()


# --- complete_sync ------------------------------------------------------------

def test_complete_sync(monkeypatch):
    _use_fake(monkeypatch, {"choices": [{"message": {"content": "sync"}}]})
    out = complete_sync("ollama/x", [{"role": "user", "content": "x"}])
    assert out == "sync"


# --- make_recipe --------------------------------------------------------------

def test_make_recipe_defaults_and_params():
    assert make_recipe("ollama/x") == {
        "inferencer": "ollama/x", "system": "", "tools": [], "params": {}}
    r = make_recipe("ollama/x", "sys", tools=("a.b",), params={"temperature": 0.2})
    assert r == {"inferencer": "ollama/x", "system": "sys",
                 "tools": ["a.b"], "params": {"temperature": 0.2}}
