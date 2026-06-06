"""Unit tests for woollama's MCP server surface (slice e).

woollama-as-MCP-server projects its inbound OpenAI machinery onto an outbound
MCP surface so MCP clients (Claude Desktop, the cosmic-fabric panel) can drive
it natively:

  * recipes        → MCP `prompts` (prompts/list, prompts/get)
  * the chat verb  → MCP `tool`    (tools/list, tools/call)
  * capabilities   → advertised on initialize

These are in-memory tests: we drive the FastMCP server through `fastmcp.Client`
passed the server object directly — no subprocess, no stdio. That keeps these
in the default suite and fast. The real stdio transport (and a started
registry) is exercised by the opt-in integration test in test_integration.py.

Registry note: all five tests inject a *bare* `Registry()` (zero servers), so
the server's start_all/stop_all lifespan hooks are genuine no-ops and no real
MCP subprocess is spawned. The orchestration test (5) mocks the inferencer so
its first turn returns final content — tool dispatch is never reached, so an
empty registry is fine.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastmcp import Client
from mcp.types import TextContent

from woollama import mcp_server, recipes
from woollama.manager import Registry, ServerManager

# fastmcp's Client is async; mark the whole module.
pytestmark = pytest.mark.asyncio


@pytest.fixture
def server(monkeypatch, tmp_path):
    """A FastMCP server built from the bundled-default recipes, over an empty
    registry. WOOLLAMA_CONFIG_DIR is pointed at an empty dir so the packaged
    defaults load; recipes are reloaded so prompts snapshot them at build."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()
    return mcp_server.build_server(Registry())


# ---------------------------------------------------------------------------
# initialize — capability negotiation
# ---------------------------------------------------------------------------

async def test_initialize_advertises_expected_capabilities(server):
    async with Client(server) as c:
        caps = c.initialize_result.capabilities
        assert caps.tools is not None, "server must advertise tools capability"
        assert caps.prompts is not None, "server must advertise prompts capability"
        assert c.initialize_result.serverInfo.name == "woollama"


# ---------------------------------------------------------------------------
# prompts/list — each recipe is a prompt
# ---------------------------------------------------------------------------

async def test_prompts_list_returns_loaded_recipes(server):
    async with Client(server) as c:
        names = {p.name for p in await c.list_prompts()}
    assert "streamer" in names
    assert "textcounter" in names
    assert names == set(recipes.names())


# ---------------------------------------------------------------------------
# prompts/get — returns the recipe's rendered system message
# ---------------------------------------------------------------------------

async def test_prompts_get_returns_rendered_system_message(server):
    async with Client(server) as c:
        got = await c.get_prompt("streamer")
    text = got.messages[0].content.text
    assert text == recipes.get("streamer")["system"]
    assert "counting assistant" in text


# ---------------------------------------------------------------------------
# tools/list — includes the `chat` orchestration verb
# ---------------------------------------------------------------------------

async def test_tools_list_includes_chat_orchestration_verb(server):
    async with Client(server) as c:
        tools = {t.name: t for t in await c.list_tools()}
    assert "chat" in tools
    props = set((tools["chat"].inputSchema or {}).get("properties", {}))
    # input schema mirrors the OpenAI surface: a recipe selector + messages,
    # with `model` accepted for symmetry.
    assert {"recipe", "messages", "model"} <= props


# ---------------------------------------------------------------------------
# tools/call chat — orchestrates end-to-end, loop hidden (inferencer mocked)
# ---------------------------------------------------------------------------

async def test_chat_tool_orchestrates_end_to_end(server, monkeypatch):
    """`chat {recipe, messages}` runs the shared orchestration loop and returns
    only the final assistant message — same contract as /v1/chat/completions.
    The inferencer is mocked (unit test, not integration): turn 1 returns final
    content, so the tool-dispatch path is never reached."""

    class _Resp:
        def __init__(self, payload):
            self.status_code = 200
            self._payload = payload

        def json(self):
            return self._payload

    class _FakeClient:
        def __init__(self, *_a, **_kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return None

        async def post(self, _url, json=None, **_kw):
            return _Resp({"choices": [{"message": {"content": "Counted to 3."}}]})

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    async with Client(server) as c:
        result = await c.call_tool("chat", {
            "recipe": "streamer",
            "messages": [{"role": "user", "content": "Count to 3."}],
        })
    assert result.data == "Counted to 3."


# ---------------------------------------------------------------------------
# tools/list — re-exports discovered downstream tools (decision #3 / aggregator)
# ---------------------------------------------------------------------------

async def test_tools_list_reexports_discovered_downstream_tools(monkeypatch, tmp_path):
    """With a STARTED registry, tools/list is the union of the `chat` verb and
    every discovered downstream tool, namespaced — and a re-exported tool
    dispatches through the registry. Registration happens in the lifespan
    (tools are only known post-start), so this drives it through a real Client
    connection rather than calling the builder directly.

    The ServerManager is stubbed (start = no-op, call_tool = canned) so no real
    subprocess spawns; manager-internal mechanics are covered in test_manager.py.
    """
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()

    reg = Registry()
    mgr = ServerManager("hello", "echo", [])
    mgr.tools = [SimpleNamespace(
        name="count_to", description="count to n",
        inputSchema={"type": "object",
                     "properties": {"n": {"type": "integer"}},
                     "required": ["n"]},
    )]

    async def _noop_start() -> None:  # avoid spawning a real subprocess
        return None

    async def _call(bare: str, args: dict):
        return SimpleNamespace(
            content=[TextContent(type="text", text=f"counted {args['n']}")],
            isError=False,
        )

    mgr.start = _noop_start          # type: ignore[method-assign]
    mgr.call_tool = _call            # type: ignore[method-assign]
    reg.add(mgr)

    server = mcp_server.build_server(reg)
    async with Client(server) as c:
        names = {t.name for t in await c.list_tools()}
        assert "chat" in names, "the orchestration verb is still present"
        assert "hello.count_to" in names, "downstream tool re-exported, namespaced"

        # The re-exported tool's schema is the downstream tool's own schema.
        tool = next(t for t in await c.list_tools() if t.name == "hello.count_to")
        assert "n" in (tool.inputSchema or {}).get("properties", {})
        # The downstream tool declares NO output schema → the proxy must not
        # fabricate one (the negative half of output_schema mirroring).
        assert tool.outputSchema is None

        # And it dispatches through the registry end-to-end.
        result = await c.call_tool("hello.count_to", {"n": 3})
    # Exact content fidelity — not a substring-in-repr that passes on mangled output.
    assert result.content[0].text == "counted 3"


# ---------------------------------------------------------------------------
# tools/list — re-export mirrors the downstream output_schema (output_schema slice)
# ---------------------------------------------------------------------------

def _stub_registry(monkeypatch, tmp_path, *, out_schema, call):
    """A started registry with one stubbed downstream tool that declares
    `out_schema` and whose dispatch returns `call(args)`. No subprocess."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()
    reg = Registry()
    mgr = ServerManager("svc", "echo", [])
    mgr.tools = [SimpleNamespace(
        name="op", description="an op",
        inputSchema={"type": "object", "properties": {"n": {"type": "integer"}}},
        outputSchema=out_schema,
    )]

    async def _noop_start() -> None:
        return None

    async def _call(bare: str, args: dict):
        return call(args)

    mgr.start = _noop_start          # type: ignore[method-assign]
    mgr.call_tool = _call            # type: ignore[method-assign]
    reg.add(mgr)
    return reg


async def test_reexport_mirrors_output_schema_and_forwards_structured(monkeypatch, tmp_path):
    """A downstream tool that declares an output_schema has it MIRRORED onto the
    re-exported proxy (advertised on tools/list), and a conforming structured
    result is forwarded — validating cleanly against that advertised schema. This
    is safe because the downstream already validated its own output before we got
    it (proven live against hello.count_to in the stdio integration test)."""
    out_schema = {"type": "object",
                  "properties": {"count": {"type": "integer"}, "done": {"type": "boolean"}},
                  "required": ["count", "done"]}

    def call(args):
        return SimpleNamespace(
            content=[TextContent(type="text", text="{...}")],
            structuredContent={"count": args["n"], "done": True},
            isError=False)

    reg = _stub_registry(monkeypatch, tmp_path, out_schema=out_schema, call=call)
    server = mcp_server.build_server(reg)
    async with Client(server) as c:
        tool = next(t for t in await c.list_tools() if t.name == "svc.op")
        assert tool.outputSchema == out_schema          # mirrored onto tools/list
        result = await c.call_tool("svc.op", {"n": 3})
        # structured payload forwarded + validated against the advertised schema
        # (the client deserializes .data into a model now that a schema exists,
        # so assert on the raw structured_content).
        assert result.structured_content == {"count": 3, "done": True}


async def test_reexport_nonconforming_output_surfaces_clear_error(monkeypatch, tmp_path):
    """Deliberate, documented behaviour: a downstream that DECLARES an
    output_schema but returns content-only (violating its own contract) surfaces
    a clear output-validation error through the proxy, rather than the proxy
    silently dropping the declared schema. The faithful-proxy choice."""
    from fastmcp.exceptions import ToolError
    out_schema = {"type": "object", "properties": {"x": {"type": "integer"}},
                  "required": ["x"]}

    def call(args):   # declares a schema above but returns NO structured content
        return SimpleNamespace(content=[TextContent(type="text", text="oops")],
                               structuredContent=None, isError=False)

    reg = _stub_registry(monkeypatch, tmp_path, out_schema=out_schema, call=call)
    server = mcp_server.build_server(reg)
    async with Client(server) as c:
        with pytest.raises(ToolError, match=r"[Oo]utput"):
            await c.call_tool("svc.op", {"n": 1})


async def test_reexport_proxy_surfaces_dispatch_exception_as_toolerror(monkeypatch, tmp_path):
    """If the downstream dispatch RAISES, the proxy surfaces a clear ToolError
    ('dispatch failed') rather than leaking the raw exception."""
    from fastmcp.exceptions import ToolError

    def call(args):
        raise RuntimeError("connection reset")

    reg = _stub_registry(monkeypatch, tmp_path, out_schema=None, call=call)
    server = mcp_server.build_server(reg)
    async with Client(server) as c:
        with pytest.raises(ToolError, match="dispatch failed"):
            await c.call_tool("svc.op", {"n": 1})


async def test_reexport_proxy_surfaces_downstream_iserror_as_toolerror(monkeypatch, tmp_path):
    """A downstream result with isError=True becomes a ToolError carrying the
    downstream's error text — not a silent success."""
    from fastmcp.exceptions import ToolError

    def call(args):
        return SimpleNamespace(
            content=[TextContent(type="text", text="downstream said no")],
            isError=True)

    reg = _stub_registry(monkeypatch, tmp_path, out_schema=None, call=call)
    server = mcp_server.build_server(reg)
    async with Client(server) as c:
        with pytest.raises(ToolError, match="downstream said no"):
            await c.call_tool("svc.op", {"n": 1})


async def test_chat_tool_unknown_recipe_raises(server):
    """The chat verb's negative path: an unknown recipe selector surfaces as a
    ToolError, not a silent empty/crash (mcp_server `_chat_tool`)."""
    from fastmcp.exceptions import ToolError
    async with Client(server) as c:
        with pytest.raises(ToolError, match="unknown recipe"):
            await c.call_tool("chat", {"recipe": "_no_such_recipe_",
                                       "messages": [{"role": "user", "content": "hi"}]})


async def test_chat_tool_requires_a_recipe_selector(server):
    """No recipe and no model → a clear ToolError, not a crash."""
    from fastmcp.exceptions import ToolError
    async with Client(server) as c:
        with pytest.raises(ToolError, match="requires a 'recipe'"):
            await c.call_tool("chat", {"messages": [{"role": "user", "content": "hi"}]})
