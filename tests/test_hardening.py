"""Regression tests for the hardening/correctness items shipped alongside the
surface-auth work (tracked in issue #8):

  * `mcp.json` `env` is forwarded to the spawned MCP server (previously parsed
    but dropped — the only workaround put secrets in process argv).
  * `ServerManager.call_tool` has a timeout, and a timed-out call does not wedge
    the connection's worker task for subsequent calls.
  * managed-agents environments default to `limited` networking (least
    privilege); `unrestricted` is an explicit opt-in.
  * the durable conversation handle table is not world-readable.
"""
from __future__ import annotations

import asyncio
import stat
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from woollama import managed_agents, manager
from woollama.manager import ServerManager

# ---------------------------------------------------------------------------
# mcp.json env forwarding
# ---------------------------------------------------------------------------

def _tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, description="",
                           inputSchema={"type": "object", "properties": {}})


def _stub_session_factory(tools, call_tool=None):
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
            if call_tool is not None:
                return await call_tool(name, args)
            from mcp.types import TextContent
            return SimpleNamespace(
                content=[TextContent(type="text", text=f"ok:{name}")])

    return _StubSession


async def test_server_manager_forwards_env_to_stdio_params(monkeypatch):
    """The `env` block a user writes in mcp.json must reach the spawned server
    via `StdioServerParameters.env` (the SDK merges it over its safe default
    environment) — NOT be silently dropped."""
    captured: dict = {}

    @asynccontextmanager
    async def capture_stdio(params):
        captured["params"] = params
        yield (object(), object())

    monkeypatch.setattr(manager, "stdio_client", capture_stdio)
    monkeypatch.setattr(manager, "ClientSession",
                        _stub_session_factory([_tool("t")]))
    mgr = ServerManager("s", "echo", [], env={"DOWNSTREAM_KEY": "v1"})
    await mgr.start()
    try:
        assert captured["params"].env == {"DOWNSTREAM_KEY": "v1"}
    finally:
        await mgr.stop()


async def test_server_manager_no_env_keeps_sdk_default(monkeypatch):
    """No `env` in config → `StdioServerParameters.env` stays None, so the MCP
    SDK applies its own default environment (unchanged behavior)."""
    captured: dict = {}

    @asynccontextmanager
    async def capture_stdio(params):
        captured["params"] = params
        yield (object(), object())

    monkeypatch.setattr(manager, "stdio_client", capture_stdio)
    monkeypatch.setattr(manager, "ClientSession",
                        _stub_session_factory([_tool("t")]))
    mgr = ServerManager("s", "echo", [])
    await mgr.start()
    try:
        assert captured["params"].env is None
    finally:
        await mgr.stop()


def test_registry_wiring_passes_env_through(monkeypatch):
    """Both spawn sites (the router lifespan and the stdio `build_registry`)
    hand mcp.json's `env` to the ServerManager."""
    from woollama import mcp_server

    monkeypatch.setattr(
        mcp_server.config, "load_mcp_servers",
        lambda: {"srv": {"command": "echo", "args": [], "env": {"K": "v"}}})
    reg = mcp_server.build_registry()
    assert reg.servers["srv"].env == {"K": "v"}


# ---------------------------------------------------------------------------
# call_tool timeout
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _stub_stdio(_params):
    yield (object(), object())


async def test_call_tool_times_out_and_worker_stays_alive(monkeypatch):
    """A hung downstream tool call must time out (bounded turn) AND must not
    wedge the connection's worker task — the next call still serves."""
    monkeypatch.setenv("WOOLLAMA_TOOL_TIMEOUT", "0.2")

    async def handler(name: str, args: dict):
        if name == "slow":
            await asyncio.sleep(30)
        from mcp.types import TextContent
        return SimpleNamespace(
            content=[TextContent(type="text", text=f"ok:{name}")])

    monkeypatch.setattr(manager, "stdio_client", _stub_stdio)
    monkeypatch.setattr(manager, "ClientSession",
                        _stub_session_factory([_tool("slow"), _tool("fast")],
                                              call_tool=handler))
    mgr = ServerManager("s", "echo", [])
    await mgr.start()
    try:
        with pytest.raises(TimeoutError):
            # Outer guard bounds the test itself; the fix must trip well inside it.
            await asyncio.wait_for(mgr.call_tool("slow", {}), timeout=5)
        # The worker must not be stuck behind the hung call.
        result = await asyncio.wait_for(mgr.call_tool("fast", {}), timeout=5)
        assert result.content[0].text == "ok:fast"
    finally:
        await mgr.stop()


# ---------------------------------------------------------------------------
# managed-agents networking default
# ---------------------------------------------------------------------------

def _fake_agents_client(captured: dict):
    def create(*, name, config):
        captured["name"] = name
        captured["config"] = config
        return SimpleNamespace(id="env_1")

    return SimpleNamespace(
        beta=SimpleNamespace(environments=SimpleNamespace(create=create)))


def test_managed_agents_environment_defaults_to_limited_networking(monkeypatch):
    """Least privilege: a tool-less hosted agent gets `limited` networking by
    default, not `unrestricted`."""
    monkeypatch.delenv("WOOLLAMA_AGENT_NETWORKING", raising=False)
    captured: dict = {}
    monkeypatch.setattr(managed_agents, "_client",
                        lambda: _fake_agents_client(captured))
    managed_agents._create_environment_sync("woollama-test")
    assert captured["config"]["networking"]["type"] == "limited"


def test_managed_agents_unrestricted_networking_is_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("WOOLLAMA_AGENT_NETWORKING", "unrestricted")
    captured: dict = {}
    monkeypatch.setattr(managed_agents, "_client",
                        lambda: _fake_agents_client(captured))
    managed_agents._create_environment_sync("woollama-test")
    assert captured["config"]["networking"]["type"] == "unrestricted"


# ---------------------------------------------------------------------------
# conversation handle table file mode
# ---------------------------------------------------------------------------

def test_conversation_state_file_is_not_world_readable(tmp_path):
    """`conversations.json` carries caller-supplied metadata/titles; keep it
    owner-only like the rest of woollama's runtime artifacts."""
    from woollama.conversations import ConversationStore

    path = tmp_path / "conversations.json"
    store = ConversationStore(path)
    store.create("claude-resume", "claude-code/haiku")
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode & 0o077 == 0, oct(mode)
