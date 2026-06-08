"""conv-7 / issue #2 — the MCP conversation-store provider (woollama as a CLIENT).

`McpStoreProvider` adapts an external MCP convstore server (the reference impl is
examples/mcp-convstore/server.py) to the `ConversationStoreProvider` contract.
woollama holds NO transcript bytes — every op is one MCP tool call. These tests
cover, hermetically:

  - the provider maps create/get/append/delete to the right tool names + args;
  - the router's `_mcp_store_call` wrapper parses the real MCP CallToolResult
    shape (text blocks holding `json.dumps(...)` strings) — grounded live against
    the example server before being hard-coded here;
  - a flaky store (call_tool raises, or returns unparseable text) surfaces as a
    clean OrchestrationError(502), and through /v1/responses as a 502 — never a 500.

The full real-server round-trip (convstore + ollama) lives in test_integration.py.
"""
from __future__ import annotations

import json

import pytest

from woollama import conversations, router

# --- fakes mimicking the grounded MCP CallToolResult shape --------------------

class _Block:
    """A TextContent-like block: has `.text`. (Non-text blocks omit it, so the
    wrapper's `hasattr(c, 'text')` filter is exercised by mixing one in.)"""
    def __init__(self, text: str):
        self.text = text


class _Result:
    """A CallToolResult-like value: `.content` is a list of blocks. Mirrors what
    fastmcp returns — each tool's `json.dumps(...)` string in one TextContent."""
    def __init__(self, content: list):
        self.content = content


class FakeManager:
    """Stands in for a ServerManager: records calls, returns a canned MCP result
    (or raises) so `_mcp_store_call`'s parse + error-wrap path is covered."""
    name = "convstore"

    def __init__(self, replies: dict | None = None, raises: bool = False):
        self.replies = replies or {}
        self.raises = raises
        self.calls: list = []

    async def call_tool(self, tool: str, args: dict):
        self.calls.append((tool, args))
        if self.raises:
            raise RuntimeError("server crashed")
        payload = self.replies.get(tool, "null")
        # Mix in a non-text block to prove the hasattr filter holds.
        return _Result([object(), _Block(payload)])


# --- provider maps to the right tool calls ------------------------------------

async def test_provider_maps_ops_to_tool_calls():
    """McpStoreProvider delegates each op to the injected call with the contract
    tool name + args (no transcript bytes held provider-side)."""
    seen: list = []

    async def call(tool, args):
        seen.append((tool, args))
        return {"create_thread": "t1", "get_thread": [], "append_turn": {"ok": True},
                "delete_thread": {"ok": True}}[tool]

    p = conversations.McpStoreProvider(call)
    assert await p.create() == "t1"
    assert await p.get("t1") == []
    await p.append("t1", [{"role": "user", "content": "hi"}])
    await p.delete("t1")
    assert seen == [
        ("create_thread", {}),
        ("get_thread", {"thread_id": "t1"}),
        ("append_turn", {"thread_id": "t1", "messages": [{"role": "user", "content": "hi"}]}),
        ("delete_thread", {"thread_id": "t1"}),
    ]


# --- the router wrapper parses the grounded MCP result shape -------------------

async def test_mcp_store_call_parses_text_blocks():
    """`_mcp_store_call` joins the result's text block(s) and json.loads them —
    the shape grounded live against examples/mcp-convstore (TextContent text is a
    `json.dumps(...)` string; non-text blocks ignored)."""
    mgr = FakeManager(replies={
        "create_thread": json.dumps("thread-1"),
        "get_thread": json.dumps([{"role": "user", "content": "hi"}]),
        "append_turn": json.dumps({"ok": True, "count": 2}),
    })
    call = router._mcp_store_call(mgr)
    assert await call("create_thread", {}) == "thread-1"
    assert await call("get_thread", {"thread_id": "thread-1"}) == [
        {"role": "user", "content": "hi"}]
    assert await call("append_turn", {"thread_id": "thread-1", "messages": []}) == {
        "ok": True, "count": 2}


async def test_mcp_store_call_wraps_transport_failure_as_502():
    """call_tool raising → clean OrchestrationError(502), not a bare exception."""
    call = router._mcp_store_call(FakeManager(raises=True))
    with pytest.raises(router.OrchestrationError) as ei:
        await call("get_thread", {"thread_id": "x"})
    assert ei.value.status == 502


async def test_mcp_store_call_wraps_parse_failure_as_502():
    """Unparseable text (not JSON) → 502, not an escaping JSONDecodeError."""
    call = router._mcp_store_call(FakeManager(replies={"get_thread": "not json{"}))
    with pytest.raises(router.OrchestrationError) as ei:
        await call("get_thread", {"thread_id": "x"})
    assert ei.value.status == 502


# --- end-to-end: a flaky store surfaces as 502 through /v1/responses ----------

class FakeRequest:
    def __init__(self, body: dict):
        self._body = body

    async def json(self) -> dict:
        return self._body


async def test_flaky_store_surfaces_as_502_through_responses(monkeypatch):
    """A store-backed turn whose provider raises OrchestrationError (built by the
    router wrapper) must surface as a 502 via _responses_stateful, not a 500.
    Registers a StoreBackedBackend over an McpStoreProvider whose call raises."""
    monkeypatch.setattr(router, "conversation_store", conversations.ConversationStore())
    call = router._mcp_store_call(FakeManager(raises=True))  # every op → 502
    backend = conversations.StoreBackedBackend(
        conversations.STORE_BACKEND_NAME, conversations.McpStoreProvider(call),
        router.complete_stateless)
    monkeypatch.setattr(conversations, "BACKENDS",
                        {**conversations.BACKENDS,
                         conversations.STORE_BACKEND_NAME: backend})
    r = await router.responses_create(FakeRequest({
        "model": "ollama/qwen3", "input": "hi", "store": True}))
    assert r.status_code == 502
