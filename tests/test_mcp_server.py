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

import pytest
from fastmcp import Client

from woollama import mcp_server, recipes
from woollama.manager import Registry


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
