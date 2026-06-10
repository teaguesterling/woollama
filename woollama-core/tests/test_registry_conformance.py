"""Conformance tests for `ModelRegistry` + `inferencers.toml` loading — it must
behave like `woollama.core.inferencers` (the registry merge), the oracle.

The merge inherits per-field with TWO different idioms (inferencers.py:110-126):
`base_url`/`extra_body` inherit on FALSY (`spec or base`); `api_key_env` inherits on
ABSENCE (`spec.get(k, base)`). The discriminating tests are the built-in extensions
that exercise both. `from_config()` reads `$WOOLLAMA_CONFIG_DIR/inferencers.toml`,
driven here via a monkeypatched temp dir.

Registry resolution is also wired into the loop: `orchestrate(..., registry=reg)`
resolves a config-only provider — tested with NO `base_url` override, so the test
actually proves registry resolution fed the request (an override would mask it).
"""
from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from woollama import core as wc

BUILTINS = {"ollama", "anthropic", "openai", "groq", "together", "openrouter"}


def _write_cfg(tmp_path, toml_text, monkeypatch):
    (tmp_path / "inferencers.toml").write_text(toml_text)
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))


# --- from_config / merge -----------------------------------------------------

def test_from_config_has_builtins_plus_new_provider(tmp_path, monkeypatch):
    _write_cfg(tmp_path, """
[inferencers.vllm]
base_url = "http://localhost:8000/v1"
api_key_env = "VLLM_KEY"
""", monkeypatch)

    reg = wc.ModelRegistry.from_config()

    assert BUILTINS <= set(reg.names())
    vllm = reg.get("vllm")
    assert vllm["base_url"] == "http://localhost:8000/v1"
    assert vllm["api_key_env"] == "VLLM_KEY"


def test_extend_builtin_absence_inherits_api_key_env_and_base_url(tmp_path, monkeypatch):
    # set ONLY extra_body on anthropic → api_key_env inherits (ABSENCE), base_url retained
    _write_cfg(tmp_path, """
[inferencers.anthropic]
extra_body = { temperature = 0.5 }
""", monkeypatch)

    a = wc.ModelRegistry.from_config().get("anthropic")

    assert a["api_key_env"] == "ANTHROPIC_API_KEY"             # absence-inherit
    assert a["base_url"] == "https://api.anthropic.com/v1"     # retained
    assert a["extra_body"] == {"temperature": 0.5}            # overridden


def test_empty_extra_body_inherits_builtin_falsy_or(tmp_path, monkeypatch):
    # extra_body = {} is FALSY → inherits the built-in's, doesn't blank it
    _write_cfg(tmp_path, """
[inferencers.anthropic]
extra_body = {}
""", monkeypatch)

    a = wc.ModelRegistry.from_config().get("anthropic")

    assert a["extra_body"] == {"temperature": 0, "max_tokens": 4096}   # the built-in's


def test_set_api_key_env_overrides_builtin(tmp_path, monkeypatch):
    _write_cfg(tmp_path, """
[inferencers.anthropic]
api_key_env = "MY_OWN_KEY"
""", monkeypatch)

    assert wc.ModelRegistry.from_config().get("anthropic")["api_key_env"] == "MY_OWN_KEY"


def test_new_provider_without_base_url_raises(tmp_path, monkeypatch):
    _write_cfg(tmp_path, """
[inferencers.broken]
api_key_env = "X"
""", monkeypatch)

    with pytest.raises(wc.InferenceError, match="base_url"):
        wc.ModelRegistry.from_config()


def test_env_var_expansion_in_values(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_HOST", "example.com")
    _write_cfg(tmp_path, """
[inferencers.custom]
base_url = "https://${MY_HOST}/v1"
""", monkeypatch)

    assert wc.ModelRegistry.from_config().get("custom")["base_url"] == "https://example.com/v1"


def test_missing_config_file_is_builtins_only(tmp_path, monkeypatch):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))   # empty dir, no toml

    assert set(wc.ModelRegistry.from_config().names()) == BUILTINS


def test_base_url_override_on_builtin(tmp_path, monkeypatch):
    # falsy-or for base_url: a non-empty value on a built-in replaces it.
    _write_cfg(tmp_path, """
[inferencers.openai]
base_url = "https://proxy.internal/v1"
""", monkeypatch)

    o = wc.ModelRegistry.from_config().get("openai")
    assert o["base_url"] == "https://proxy.internal/v1"
    assert o["api_key_env"] == "OPENAI_API_KEY"               # absence-inherit


@pytest.mark.parametrize("env_url", ["http://h:9", "http://h:9/", "http://h:9/v1"])
def test_ollama_url_env_normalized_to_v1(tmp_path, monkeypatch, env_url):
    # Python takes $WOOLLAMA_OLLAMA_URL as the ROOT and appends /v1; we normalize a
    # trailing / and /v1 so any of these forms resolves to <root>/v1. (Without this,
    # a remote ollama URL would build a request MISSING /v1.)
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))   # no toml
    monkeypatch.setenv("WOOLLAMA_OLLAMA_URL", env_url)

    assert wc.ModelRegistry.from_config().get("ollama")["base_url"] == "http://h:9/v1"


def test_registry_add_and_accessors():
    reg = wc.ModelRegistry()
    reg.add("local", "http://x/v1", api_key_env=None, extra_body={"temperature": 0})

    assert reg.names() == ["local"]
    assert reg.get("local")["base_url"] == "http://x/v1"
    assert reg.get("local")["api_key_env"] is None
    assert reg.get("nope") is None
    assert reg.all()["local"]["extra_body"] == {"temperature": 0}


# --- registry wired into the loop --------------------------------------------

class _Mock(BaseHTTPRequestHandler):
    record: dict = {}

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Mock.record = {"path": self.path, "body": body,
                        "auth": self.headers.get("Authorization")}
        raw = json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield srv
    finally:
        srv.shutdown()


class _NoTools:
    def tools_for(self, allow):
        return []

    async def dispatch(self, name, args):
        raise AssertionError("no tools should be dispatched")


def test_orchestrate_resolves_inferencer_via_registry(server):
    # A config-only provider (not a built-in) — it can ONLY resolve via the registry,
    # and we pass NO base_url override, so the request landing at the registry's URL
    # proves registry resolution fed the loop.
    port = server.server_address[1]
    reg = wc.ModelRegistry()
    reg.add("myinf", f"http://127.0.0.1:{port}/v1", api_key_env=None)
    recipe = {"inferencer": "myinf/some-model", "system": "s", "tools": []}

    async def go():
        return await wc.orchestrate(recipe, [{"role": "user", "content": "x"}],
                                    _NoTools(), registry=reg)

    out = asyncio.run(go())

    assert out["choices"][0]["message"]["content"] == "ok"
    assert _Mock.record["path"] == "/v1/chat/completions"
    assert _Mock.record["body"]["model"] == "some-model"
    assert _Mock.record["auth"] is None                       # myinf has no api_key_env → no auth


def test_orchestrate_unknown_provider_in_registry_raises():
    reg = wc.ModelRegistry()
    reg.add("only", "http://x/v1")
    recipe = {"inferencer": "missing/m", "system": "s", "tools": []}
    # eager setup → raises on the call (before the awaitable)
    with pytest.raises(wc.InferenceError):
        wc.orchestrate(recipe, [{"role": "user", "content": "x"}], _NoTools(), registry=reg)
