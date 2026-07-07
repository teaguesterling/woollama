"""Regression tests for surface access control (binding + HTTP auth + dispatch).

The invariants under test:

  * **Fail-closed bind** — a non-loopback `WOOLLAMA_ADDRESS` must refuse to
    start unless `WOOLLAMA_TOKEN` is configured. Loopback binding (the default
    dev workflow) keeps working with no token.
  * **Surface auth** — requests to `/v1/*` and the mounted `/mcp` are served
    only when the peer is local (loopback TCP, or the mode-0600 Unix socket) or
    presents the configured bearer token. When a token is configured it is
    required on every TCP request.
  * **Dispatch-time allow-list** — the Python dispatch path (`Registry.dispatch`
    / `RegistryToolProvider.dispatch`) blocks a tool that is configured but not
    in the active recipe allow-list, independent of what the compiled core does
    at offer time.

These are permanent regression guards; they were written to fail on the code
that predates the hardening.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from woollama import binding, manager
from woollama.manager import Registry, RegistryToolProvider, ServerManager

# ---------------------------------------------------------------------------
# Fail-closed binding
# ---------------------------------------------------------------------------

def _close(listeners) -> None:
    for s in listeners.sockets:
        s.close()
    binding.cleanup(listeners)


def test_open_sockets_refuses_nonloopback_bind_without_token(tmp_path, monkeypatch):
    """A non-loopback bind with no token must fail closed at startup — never
    silently serve an unauthenticated surface beyond loopback."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("WOOLLAMA_ADDRESS", "0.0.0.0:0")
    monkeypatch.delenv("WOOLLAMA_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        _close(binding.open_sockets())      # cleanup only runs if it wrongly bound


def test_open_sockets_nonloopback_bind_with_token_is_allowed(tmp_path, monkeypatch):
    """With a token configured, the explicit non-loopback opt-in works."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("WOOLLAMA_ADDRESS", "0.0.0.0:0")
    monkeypatch.setenv("WOOLLAMA_TOKEN", "s3cret")
    listeners = binding.open_sockets()
    try:
        assert listeners.tcp_host == "0.0.0.0"
    finally:
        _close(listeners)


def test_open_sockets_loopback_bind_needs_no_token(tmp_path, monkeypatch):
    """The default local workflow (loopback, no token) must keep working."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.delenv("WOOLLAMA_ADDRESS", raising=False)
    monkeypatch.delenv("WOOLLAMA_TOKEN", raising=False)
    listeners = binding.open_sockets()
    try:
        assert listeners.tcp_host == "127.0.0.1"
    finally:
        _close(listeners)


def test_open_sockets_hostname_bind_without_token_fails_closed(tmp_path, monkeypatch):
    """A hostname that isn't provably loopback is treated as non-loopback
    (fail closed) — only `localhost` and loopback IPs are exempt."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv("WOOLLAMA_ADDRESS", "example.internal:0")
    monkeypatch.delenv("WOOLLAMA_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        _close(binding.open_sockets())


# ---------------------------------------------------------------------------
# HTTP surface auth (middleware; no lifespan — TestClient used un-entered so
# no MCP subprocesses spawn; the routes under test don't need the registry)
# ---------------------------------------------------------------------------

def _client(peer_host: str, token: str | None = None) -> TestClient:
    from woollama.router import app
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return TestClient(app, client=(peer_host, 51515), headers=headers)


def test_nonloopback_request_without_token_is_refused(monkeypatch):
    """No token configured → only local peers are served. A non-loopback peer
    gets 401, on /v1/* and on the mounted /mcp alike."""
    monkeypatch.delenv("WOOLLAMA_TOKEN", raising=False)
    c = _client("203.0.113.9")
    assert c.get("/v1/tools").status_code == 401
    assert c.get("/mcp").status_code == 401
    assert c.post("/v1/chat/completions", json={"model": "x"}).status_code == 401


def test_loopback_request_without_token_is_served(monkeypatch):
    """The loopback dev path stays open when no token is configured."""
    monkeypatch.delenv("WOOLLAMA_TOKEN", raising=False)
    r = _client("127.0.0.1").get("/v1/tools")
    assert r.status_code == 200
    assert r.json() == {"tools": []}


def test_token_configured_requires_bearer_even_on_loopback(monkeypatch):
    """Once a token is configured, every TCP request must present it —
    a token-bearing deployment is uniformly authenticated."""
    monkeypatch.setenv("WOOLLAMA_TOKEN", "s3cret")
    r = _client("127.0.0.1").get("/v1/tools")
    assert r.status_code == 401
    assert "WWW-Authenticate" in r.headers or "www-authenticate" in r.headers


def test_correct_bearer_token_is_accepted(monkeypatch):
    monkeypatch.setenv("WOOLLAMA_TOKEN", "s3cret")
    assert _client("203.0.113.9", token="s3cret").get("/v1/tools").status_code == 200
    assert _client("127.0.0.1", token="s3cret").get("/v1/tools").status_code == 200


def test_wrong_bearer_token_is_refused(monkeypatch):
    monkeypatch.setenv("WOOLLAMA_TOKEN", "s3cret")
    assert _client("127.0.0.1", token="nope").get("/v1/tools").status_code == 401


def test_uds_peer_is_exempt(monkeypatch):
    """A Unix-socket peer (ASGI `client` is None) is authorized by the socket's
    0600 mode — filesystem permissions are its credential. Pure decision-level
    check (TestClient can't fake a None client)."""
    from woollama import auth
    monkeypatch.delenv("WOOLLAMA_TOKEN", raising=False)
    assert auth.authorize(None, None) is None                    # no token
    monkeypatch.setenv("WOOLLAMA_TOKEN", "s3cret")
    assert auth.authorize(None, None) is None                    # token set


def test_ipv6_and_mapped_loopback_are_recognized():
    from woollama import auth
    assert auth.is_loopback_host("::1")
    assert auth.is_loopback_host("::ffff:127.0.0.1")
    assert auth.is_loopback_host("127.0.0.1")
    assert auth.is_loopback_host("localhost")
    assert not auth.is_loopback_host("0.0.0.0")
    assert not auth.is_loopback_host("192.168.1.10")
    assert not auth.is_loopback_host("example.internal")
    assert not auth.is_loopback_host(None)


# ---------------------------------------------------------------------------
# Dispatch-time allow-list (Python-side; independent of the compiled core)
# ---------------------------------------------------------------------------

def _tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, description="",
                           inputSchema={"type": "object", "properties": {}})


def _stub_session_factory(tools):
    class _StubSession:
        def __init__(self, _read, _write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def initialize(self):
            pass

        async def list_tools(self):
            return SimpleNamespace(tools=tools)

        async def call_tool(self, name: str, args: dict):
            from mcp.types import TextContent
            return SimpleNamespace(
                content=[TextContent(type="text", text=f"ok:{name}")])

    return _StubSession


@asynccontextmanager
async def _stub_stdio_client(_params):
    yield (object(), object())


@pytest.fixture
async def hello_registry(monkeypatch):
    """A started Registry with one stub server 'hello' offering greet + delete."""
    monkeypatch.setattr(manager, "stdio_client", _stub_stdio_client)
    monkeypatch.setattr(manager, "ClientSession",
                        _stub_session_factory([_tool("greet"), _tool("delete")]))
    reg = Registry()
    mgr = ServerManager("hello", "echo", [])
    reg.add(mgr)
    await mgr.start()
    yield reg
    await mgr.stop()


async def test_registry_dispatch_enforces_allow_list(hello_registry):
    """`Registry.dispatch` with an allow-list blocks a configured-but-not-listed
    tool in Python — regardless of who asked."""
    reg = hello_registry
    with pytest.raises(PermissionError):
        await reg.dispatch("hello.delete", {}, allow=["hello.greet"])
    result = await reg.dispatch("hello.greet", {}, allow=["hello.greet"])
    assert result.content[0].text == "ok:greet"


async def test_registry_tool_provider_dispatch_blocks_unoffered_tool(hello_registry):
    """`RegistryToolProvider` constructed with the recipe allow-list refuses to
    dispatch a tool outside it — the defense-in-depth boundary the recipe loop
    relies on, enforced here in Python."""
    provider = RegistryToolProvider(hello_registry, allow=["hello.greet"])
    with pytest.raises(PermissionError):
        await provider.dispatch("hello.delete", {})
    ok = await provider.dispatch("hello.greet", {})
    assert not ok.is_error


async def test_registry_dispatch_without_allow_list_stays_open(hello_registry):
    """`allow=None` (the MCP aggregator/proxy surface, which re-exports every
    tool by design) keeps full dispatch — the allow-list is per-recipe."""
    result = await hello_registry.dispatch("hello.delete", {})
    assert result.content[0].text == "ok:delete"


async def test_orchestrate_wires_recipe_allowlist_into_provider(hello_registry, monkeypatch):
    """`router.orchestrate_events` must hand the core a provider that carries the
    RECIPE's allow-list, so dispatch-time enforcement is wired, not optional."""
    from woollama import router

    captured: dict = {}

    async def fake_core_events(recipe, msgs, *, tools=None, registry=None,
                               stream=False):
        captured["tools"] = tools
        yield {"type": "final",
               "response": {"choices": [{"message": {"content": "hi"}}]}}

    monkeypatch.setattr(router.core, "orchestrate_events", fake_core_events)
    recipe = {"inferencer": "ollama/llama3", "tools": ["hello.greet"],
              "system": "s", "params": {}}
    events = [ev async for ev in router.orchestrate_events(
        recipe, [{"role": "user", "content": "x"}], hello_registry)]
    assert events[-1]["type"] == "final"
    provider = captured["tools"]
    assert isinstance(provider, RegistryToolProvider)
    with pytest.raises(PermissionError):
        await provider.dispatch("hello.delete", {})
