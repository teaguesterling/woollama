"""Unit tests for `woollama.manager`:

  * ServerManager — its task lifecycle + queue-mediated tool calls, exercised
    via a stub ClientSession (no real MCP subprocess, no anyio cancel scope
    concerns).
  * Registry — naming, lookup parsing, dispatch routing across multiple
    managers, OpenAI schema translation, allow-list filtering.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from woollama import manager
from woollama.manager import Registry, ServerManager


# ---------------------------------------------------------------------------
# Stubs for the MCP client surfaces ServerManager wraps
# ---------------------------------------------------------------------------

def _tool(name: str, description: str = "", params: dict | None = None) -> SimpleNamespace:
    """Build an MCP-tool-shaped object (SimpleNamespace with .name, .description,
    .inputSchema) — what `sess.list_tools().tools[i]` looks like in production."""
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=params or {"type": "object", "properties": {}},
    )


def make_stub_session_factory(tools, call_handler=None, init_delay: float = 0.0,
                              fail_on_init: bool = False):
    """Return a class drop-in for `mcp.client.session.ClientSession`. Each
    test configures its own (tools, behavior); we then monkeypatch
    `woollama.manager.ClientSession` to this class for the test's duration.
    """
    class _StubSession:
        def __init__(self, _read, _write):
            self.calls: list[tuple[str, dict]] = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def initialize(self):
            if init_delay:
                import asyncio
                await asyncio.sleep(init_delay)
            if fail_on_init:
                raise RuntimeError("simulated init failure")

        async def list_tools(self):
            return SimpleNamespace(tools=tools)

        async def call_tool(self, name: str, args: dict):
            self.calls.append((name, args))
            from mcp.types import TextContent
            handler = call_handler or (lambda n, a: f"ok:{n}")
            return SimpleNamespace(content=[TextContent(type="text", text=handler(name, args))])

    return _StubSession


@asynccontextmanager
async def _stub_stdio_client(_params):
    """Replacement for `mcp.client.stdio.stdio_client` — yields dummy
    (read, write); the stub session ignores them anyway."""
    yield (object(), object())


@pytest.fixture
def patch_mcp(monkeypatch):
    """Helper fixture: a function the test calls with (tools, handler) to
    install the stubs. Returns the configured StubSession class so the test
    can inspect its `.calls` list if needed."""
    def _install(tools, call_handler=None, init_delay=0.0, fail_on_init=False):
        stub = make_stub_session_factory(tools, call_handler, init_delay, fail_on_init)
        monkeypatch.setattr(manager, "stdio_client", _stub_stdio_client)
        monkeypatch.setattr(manager, "ClientSession", stub)
        return stub
    return _install


# ---------------------------------------------------------------------------
# ServerManager — lifecycle + queue mechanics
# ---------------------------------------------------------------------------

async def test_server_manager_start_caches_tools(patch_mcp):
    patch_mcp(tools=[_tool("foo"), _tool("bar")])
    mgr = ServerManager("test", "echo", [])
    await mgr.start()
    try:
        assert [t.name for t in mgr.tools] == ["foo", "bar"]
    finally:
        await mgr.stop()


async def test_server_manager_call_tool_dispatches_through_queue(patch_mcp):
    """The queue + task setup correctly marshals a call_tool through the
    owning task; the response comes back via the future."""
    patch_mcp(tools=[_tool("echo")],
              call_handler=lambda n, a: f"got:{n}({a})")
    mgr = ServerManager("test", "echo", [])
    await mgr.start()
    try:
        result = await mgr.call_tool("echo", {"x": 1})
        assert result.content[0].text == "got:echo({'x': 1})"
    finally:
        await mgr.stop()


async def test_server_manager_init_failure_unblocks_start(patch_mcp):
    """If the connection or initialize() fails, start() shouldn't hang
    forever — the _ready event is set, then start() raises."""
    patch_mcp(tools=[], fail_on_init=True)
    mgr = ServerManager("bad", "echo", [])
    with pytest.raises(RuntimeError, match="simulated init failure"):
        await mgr.start()
    # No await mgr.stop() — start() failed before the queue loop started.


async def test_server_manager_stop_cleanly_shuts_down(patch_mcp):
    patch_mcp(tools=[_tool("t")])
    mgr = ServerManager("test", "echo", [])
    await mgr.start()
    await mgr.stop()
    # If stop() returns without timing out, the task exited cleanly.
    assert mgr._task is not None
    assert mgr._task.done()


# ---------------------------------------------------------------------------
# Registry — naming, lookup, dispatch, schema translation
# ---------------------------------------------------------------------------

def test_registry_add_rejects_duplicate_name():
    reg = Registry()
    reg.add(ServerManager("a", "cmd", []))
    with pytest.raises(ValueError, match="already registered"):
        reg.add(ServerManager("a", "cmd", []))


def test_registry_lookup_requires_namespaced():
    reg = Registry()
    with pytest.raises(KeyError, match="namespaced"):
        reg.lookup_tool("bare_name")


def test_registry_lookup_unknown_server():
    reg = Registry()
    with pytest.raises(KeyError, match="unknown server"):
        reg.lookup_tool("nope.tool")


def test_registry_lookup_unknown_tool_on_known_server(patch_mcp):
    """Server is registered but the tool name doesn't exist on it."""
    patch_mcp(tools=[_tool("foo")])
    reg = Registry()
    mgr = ServerManager("srv", "echo", [])
    # Skip the live start; populate tools directly for this test
    mgr.tools = [_tool("foo")]
    reg.add(mgr)
    with pytest.raises(KeyError, match="not found on server"):
        reg.lookup_tool("srv.bar")


async def test_registry_dispatch_routes_to_correct_server(patch_mcp):
    """Two registered servers, both with overlapping tool names — dispatch
    must route based on the namespace prefix, not just tool name."""
    patch_mcp(tools=[_tool("echo")],
              call_handler=lambda n, a: f"server-a:{n}")
    reg = Registry()
    mgr_a = ServerManager("a", "echo", [])
    await mgr_a.start()
    reg.add(mgr_a)

    patch_mcp(tools=[_tool("echo")],
              call_handler=lambda n, a: f"server-b:{n}")
    mgr_b = ServerManager("b", "echo", [])
    await mgr_b.start()
    reg.add(mgr_b)

    try:
        r_a = await reg.dispatch("a.echo", {})
        r_b = await reg.dispatch("b.echo", {})
        assert r_a.content[0].text == "server-a:echo"
        assert r_b.content[0].text == "server-b:echo"
    finally:
        await reg.stop_all()


def test_registry_openai_tools_for_filters_to_allow_list(patch_mcp):
    """openai_tools_for returns only the namespaced tools in the allow-list,
    in OpenAI function-calling schema shape."""
    reg = Registry()
    mgr = ServerManager("hello", "echo", [])
    mgr.tools = [
        _tool("count_to", "count to n", {"type": "object",
                                          "properties": {"n": {"type": "integer"}}}),
        _tool("ask_user", "ask the user"),
    ]
    reg.add(mgr)

    schemas = reg.openai_tools_for(["hello.count_to"])
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "hello.count_to"
    assert schemas[0]["function"]["description"] == "count to n"
    assert "n" in schemas[0]["function"]["parameters"]["properties"]


def test_registry_openai_tools_for_silently_skips_unknown(patch_mcp, caplog):
    """A recipe referencing a tool we don't have shouldn't blow up — it just
    isn't exposed (with a warning logged)."""
    reg = Registry()
    mgr = ServerManager("hello", "echo", [])
    mgr.tools = [_tool("count_to")]
    reg.add(mgr)

    schemas = reg.openai_tools_for(["hello.count_to", "hello.does_not_exist",
                                     "ghost.anything"])
    names = [s["function"]["name"] for s in schemas]
    assert names == ["hello.count_to"]


def test_registry_all_tool_names_lists_namespaced(patch_mcp):
    reg = Registry()
    for name, tools in [("a", ["x", "y"]), ("b", ["x"])]:
        mgr = ServerManager(name, "echo", [])
        mgr.tools = [_tool(t) for t in tools]
        reg.add(mgr)
    assert sorted(reg.all_tool_names()) == ["a.x", "a.y", "b.x"]
