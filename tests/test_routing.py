"""Routing topology — executable documentation of how woollama routes a request.

This is the map of "what can come in, and where it goes":

  inbound surface     verb / model        routes to
  ───────────────     ────────────        ─────────────────────────────────────
  HTTP /v1/chat       ollama/<model>      passthrough to Ollama (no tools)
  HTTP /v1/chat       woollama/<recipe>   orchestrate → Registry.dispatch per tool
  MCP  tools/call     chat                orchestrate (same core, different wire)
  MCP  tools/call     <server>.<tool>     proxy → Registry.dispatch to that server

The HEADLINE: one chat to a recipe whose allow-list spans TWO providers fans
its tool calls out to two SEPARATE long-lived MCP sessions (hello + textops),
routed by namespace prefix. Below that, a rejection matrix for the things that
must NOT work — including the recipe allow-list boundary.

Everything here is hermetic: the inferencer (Ollama) is mocked with scripted
turns, and each downstream MCP session is a stubbed ServerManager that records
what IT received — so we assert the RIGHT call reached the RIGHT session, not
merely that some calls fired. The live, against-real-Ollama counterpart is
test_integration.py::test_*two_provider*; the watch-it-happen version is
examples/routing_demo.py.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from mcp.types import TextContent

from woollama import claude_code, mcp_server, recipes, router
from woollama.manager import Registry, ServerManager

# ---------------------------------------------------------------------------
# Fakes — a scripted inferencer + two recording downstream sessions
# ---------------------------------------------------------------------------

class _Resp:
    def __init__(self, payload: dict, status: int = 200):
        self.status_code = status
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def mock_inferencer(monkeypatch, turns: list[dict]):
    """Monkeypatch httpx.AsyncClient so each POST to Ollama returns the next
    scripted turn. A turn is a full OpenAI-shaped response dict."""
    script = list(turns)

    class _Client:
        def __init__(self, *_a, **_kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_a): return None
        async def get(self, *_a, **_kw): return _Resp({})
        async def post(self, _url, json=None, **_kw):
            return _Resp(script.pop(0))

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)


def _assistant_tool_call(name: str, args: dict, call_id: str = "c1") -> dict:
    return {"choices": [{"message": {
        "content": "",
        "tool_calls": [{"id": call_id, "function": {
            "name": name, "arguments": json.dumps(args)}}],
    }}]}


def _assistant_final(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def _recording_manager(server: str, tool: str) -> tuple[ServerManager, list]:
    """A stubbed ServerManager for `server` exposing one `tool`. Its call_tool
    records (bare_name, args) into the returned list and returns canned content.
    `start` is a no-op so no real subprocess spawns."""
    calls: list[tuple[str, dict]] = []
    mgr = ServerManager(server, "echo", [])
    mgr.tools = [SimpleNamespace(
        name=tool, description=f"{server}.{tool}",
        inputSchema={"type": "object", "properties": {}})]

    async def _start() -> None:
        return None

    async def _call(bare: str, args: dict):
        calls.append((bare, args))
        return SimpleNamespace(
            content=[TextContent(type="text", text=f"{server}.{bare} ok")],
            isError=False)

    mgr.start = _start          # type: ignore[method-assign]
    mgr.call_tool = _call       # type: ignore[method-assign]
    return mgr, calls


def two_provider_registry() -> tuple[Registry, list, list]:
    """A Registry with two distinct long-lived sessions: `hello` (count_to) and
    `textops` (word_count). Returns (registry, hello_calls, textops_calls)."""
    reg = Registry()
    hello, hello_calls = _recording_manager("hello", "count_to")
    textops, textops_calls = _recording_manager("textops", "word_count")
    reg.add(hello)
    reg.add(textops)
    return reg, hello_calls, textops_calls


class FakeRequest:
    def __init__(self, body: dict): self._body = body
    async def json(self) -> dict: return self._body


# ===========================================================================
# HEADLINE — one chat, tools from two different providers, across two sessions
# ===========================================================================

async def test_http_chat_fans_out_across_two_provider_sessions(monkeypatch, tmp_path):
    """woollama/textcounter allow-lists textops.word_count AND hello.count_to.
    The model calls word_count (session A) then count_to (session B); each lands
    on its OWN session. This is "proxying MCP tools across sessions"."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))  # bundled defaults
    recipes.reload()
    assert recipes.get("textcounter")["tools"] == ["textops.word_count", "hello.count_to"]

    reg, hello_calls, textops_calls = two_provider_registry()
    monkeypatch.setattr(router, "registry", reg)
    mock_inferencer(monkeypatch, [
        _assistant_tool_call("textops.word_count", {"text": "a b c d"}, "t1"),
        _assistant_tool_call("hello.count_to", {"n": 4}, "t2"),
        _assistant_final("Four words; counted to 4."),
    ])

    resp = await router.chat_completions(FakeRequest({
        "model": "woollama/textcounter",
        "messages": [{"role": "user", "content": "Count words in 'a b c d'."}],
    }))

    # The RIGHT call reached the RIGHT session — that's the routing proof.
    assert textops_calls == [("word_count", {"text": "a b c d"})]
    assert hello_calls == [("count_to", {"n": 4})]
    assert json.loads(resp.body)["choices"][0]["message"]["content"] == \
        "Four words; counted to 4."


async def test_mcp_chat_fans_out_across_two_provider_sessions(monkeypatch, tmp_path):
    """Same cross-session fan-out, but driven through the MCP `chat` tool — the
    other transport reuses the same orchestrate() core."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()

    reg, hello_calls, textops_calls = two_provider_registry()
    mock_inferencer(monkeypatch, [
        _assistant_tool_call("textops.word_count", {"text": "a b c d"}, "t1"),
        _assistant_tool_call("hello.count_to", {"n": 4}, "t2"),
        _assistant_final("Four words; counted to 4."),
    ])

    server = mcp_server.build_server(reg)
    async with Client(server) as c:
        result = await c.call_tool("chat", {
            "recipe": "textcounter",
            "messages": [{"role": "user", "content": "Count words in 'a b c d'."}],
        })
    assert textops_calls == [("word_count", {"text": "a b c d"})]
    assert hello_calls == [("count_to", {"n": 4})]
    assert result.data == "Four words; counted to 4."


async def test_mcp_chat_emits_tool_progress_notifications(monkeypatch, tmp_path):
    """slice streaming-3: while the hidden tool loop runs, the MCP `chat` tool
    emits a `ctx.info` notification per tool call/result, so a connected client
    gets live progress through the otherwise-invisible tool turns. The tool's
    RETURN value is unchanged — just the final answer, no tool JSON."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()

    reg, hello_calls, textops_calls = two_provider_registry()
    mock_inferencer(monkeypatch, [
        _assistant_tool_call("textops.word_count", {"text": "a b c d"}, "t1"),
        _assistant_tool_call("hello.count_to", {"n": 4}, "t2"),
        _assistant_final("Four words; counted to 4."),
    ])

    logs: list[str] = []

    async def log_handler(params):
        data = params.data
        logs.append(data["msg"] if isinstance(data, dict) and "msg" in data else str(data))

    server = mcp_server.build_server(reg)
    async with Client(server, log_handler=log_handler) as c:
        result = await c.call_tool("chat", {
            "recipe": "textcounter",
            "messages": [{"role": "user", "content": "Count words in 'a b c d'."}],
        })

    assert result.data == "Four words; counted to 4."      # return value unchanged
    joined = "\n".join(logs)
    # a "→" call line and a "← ok" result line for EACH tool, in dispatch order
    for marker in ("→ textops.word_count", "← textops.word_count: ok",
                   "→ hello.count_to", "← hello.count_to: ok"):
        assert marker in joined, f"missing progress line: {marker}\nlogs={logs}"
    assert joined.index("→ textops.word_count") < joined.index("→ hello.count_to")


# ===========================================================================
# Direct proxy routing — a re-exported tool goes straight to its own session
# ===========================================================================

async def test_proxy_tool_routes_to_its_own_session(monkeypatch, tmp_path):
    """tools/call hello.count_to and textops.word_count each route to the
    matching session by namespace — no orchestration, raw passthrough."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()
    reg, hello_calls, textops_calls = two_provider_registry()

    server = mcp_server.build_server(reg)
    async with Client(server) as c:
        names = {t.name for t in await c.list_tools()}
        assert {"hello.count_to", "textops.word_count"} <= names  # both re-exported
        await c.call_tool("hello.count_to", {"n": 2})
        await c.call_tool("textops.word_count", {"text": "x y"})

    assert hello_calls == [("count_to", {"n": 2})]
    assert textops_calls == [("word_count", {"text": "x y"})]


# ===========================================================================
# Passthrough — ollama/<model> bypasses orchestration entirely
# ===========================================================================

async def test_ollama_passthrough_strips_prefix_no_tools(monkeypatch):
    captured: dict = {}

    class _Spy:
        def __init__(self, *_a, **_kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_a): return None
        async def post(self, url, json=None, **_kw):
            captured["url"], captured["body"] = url, json
            return _Resp({"choices": [{"message": {"content": "pong"}}]})

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Spy)
    await router.chat_completions(FakeRequest({
        "model": "ollama/qwen3", "messages": [{"role": "user", "content": "hi"}]}))
    assert captured["body"]["model"] == "qwen3"      # prefix stripped
    assert "tools" not in captured["body"]            # no orchestration


# ===========================================================================
# REJECTION MATRIX — things that must NOT work
# ===========================================================================

async def test_reject_unknown_model_namespace(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()
    resp = await router.chat_completions(FakeRequest({"model": "bogus/x", "messages": []}))
    assert resp.status_code == 400


async def test_reject_unknown_recipe(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()
    resp = await router.chat_completions(
        FakeRequest({"model": "woollama/nope", "messages": []}))
    assert resp.status_code == 404


async def test_reject_unknown_inferencer(monkeypatch, tmp_path):
    """An unknown provider (not ollama/anthropic/claude-code) → 501."""
    (tmp_path / "recipes.toml").write_text(
        '[recipes.bogus]\ninferencer="no-such-provider/m"\ntools=[]\nsystem="x"\n')
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()
    monkeypatch.setattr(router, "registry", Registry())
    resp = await router.chat_completions(
        FakeRequest({"model": "woollama/bogus", "messages": []}))
    assert resp.status_code == 501


async def test_reject_anthropic_without_api_key(monkeypatch, tmp_path):
    """anthropic IS a supported inferencer now, but with no ANTHROPIC_API_KEY it
    must fail with a clear credential error (400), distinct from unknown-provider
    (501). The key check fails fast before any network call."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    (tmp_path / "recipes.toml").write_text(
        '[recipes.cloud]\ninferencer="anthropic/claude-sonnet-4-6"\ntools=[]\nsystem="x"\n')
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()
    monkeypatch.setattr(router, "registry", Registry())
    resp = await router.chat_completions(
        FakeRequest({"model": "woollama/cloud", "messages": [{"role": "user", "content": "hi"}]}))
    assert resp.status_code == 400
    assert "ANTHROPIC_API_KEY" in json.loads(resp.body)["error"]["message"]


async def test_reject_mcp_chat_unknown_recipe(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()
    server = mcp_server.build_server(Registry())
    async with Client(server) as c:
        with pytest.raises(ToolError, match="unknown recipe"):
            await c.call_tool("chat", {"recipe": "nope", "messages": []})


async def test_reject_dispatch_to_unknown_provider():
    """The Registry refuses to route a namespaced name it doesn't own."""
    reg, _, _ = two_provider_registry()
    with pytest.raises(KeyError):
        await reg.dispatch("ghost.tool", {})
    with pytest.raises(KeyError):
        await reg.dispatch("hello.does_not_exist", {})  # known server, unknown tool


async def test_claude_code_inferencer_delegates_to_backend(monkeypatch, tmp_path):
    """A tool-less `claude-code/<model>` recipe routes to the Claude Code
    backend (NOT the ollama loop), and its completion comes back to the client.
    The backend is mocked — real `claude` invocation is the opt-in live test."""
    (tmp_path / "recipes.toml").write_text(
        '[recipes.cc]\ninferencer="claude-code/haiku"\ntools=[]\n'
        'system="You are concise."\n')
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()

    seen = {}

    async def fake_run(system, user_msgs, model):
        seen["system"], seen["model"] = system, model
        return {"choices": [{"message": {"role": "assistant", "content": "hi from claude"}}]}

    monkeypatch.setattr(claude_code, "run_completion", fake_run)

    resp = await router.chat_completions(FakeRequest({
        "model": "woollama/cc",
        "messages": [{"role": "user", "content": "hello"}]}))

    assert json.loads(resp.body)["choices"][0]["message"]["content"] == "hi from claude"
    assert seen == {"system": "You are concise.", "model": "haiku"}


async def test_claude_code_recipe_with_tools_delegates(monkeypatch, tmp_path):
    """A claude-code recipe WITH tools now DELEGATES (executor): woollama hands
    Claude the recipe's allow-listed tools (and ONLY the servers they reference)
    and returns Claude's answer — no more 501. The backend is mocked; the real
    multi-turn invocation is the opt-in plain-terminal live test."""
    (tmp_path / "recipes.toml").write_text(
        '[recipes.cctools]\ninferencer="claude-code/haiku"\n'
        'tools=["hello.count_to"]\nsystem="count please"\n')
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))   # mcp falls back to bundled (has hello)
    recipes.reload()

    seen = {}

    async def fake_delegated(system, user_msgs, model, *, allowed_tools,
                             mcp_servers, **kw):
        seen["system"], seen["model"] = system, model
        seen["allowed_tools"] = allowed_tools
        seen["mcp_servers"] = mcp_servers
        return {"choices": [{"message": {"role": "assistant", "content": "counted to 3"}}]}

    monkeypatch.setattr(claude_code, "run_delegated", fake_delegated)

    resp = await router.chat_completions(FakeRequest({
        "model": "woollama/cctools", "messages": [{"role": "user", "content": "count to 3"}]}))
    assert resp.status_code == 200
    assert json.loads(resp.body)["choices"][0]["message"]["content"] == "counted to 3"
    # Routed with the recipe's allow-list and ONLY the hello server's launch spec.
    assert seen["allowed_tools"] == ["hello.count_to"]
    assert set(seen["mcp_servers"]) == {"hello"}
    assert "command" in seen["mcp_servers"]["hello"]
    assert seen["model"] == "haiku"


async def test_claude_code_delegation_missing_server_is_400(monkeypatch, tmp_path):
    """A delegated recipe referencing a tool whose MCP server isn't configured
    fails clearly (400) — woollama never hands Claude a partial toolset."""
    (tmp_path / "recipes.toml").write_text(
        '[recipes.bad]\ninferencer="claude-code/haiku"\n'
        'tools=["ghost.do_thing"]\nsystem="x"\n')
    (tmp_path / "mcp.json").write_text(
        '{"mcpServers": {"hello": {"command": "python", "args": ["h.py"]}}}')
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()

    resp = await router.chat_completions(FakeRequest({
        "model": "woollama/bad", "messages": [{"role": "user", "content": "x"}]}))
    assert resp.status_code == 400
    assert "not configured" in json.loads(resp.body)["error"]["message"]


async def test_claude_code_delegation_rejects_comma_in_tool_name(monkeypatch, tmp_path):
    """A recipe tool name with a comma would inject an extra --allowedTools entry
    (a same-server sibling grant) — rejected at routing with a 400."""
    (tmp_path / "recipes.toml").write_text(
        '[recipes.inj]\ninferencer="claude-code/haiku"\n'
        'tools=["hello.count_to,mcp__hello__hello"]\nsystem="x"\n')
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()

    resp = await router.chat_completions(FakeRequest({
        "model": "woollama/inj", "messages": [{"role": "user", "content": "x"}]}))
    assert resp.status_code == 400
    assert "invalid tool name" in json.loads(resp.body)["error"]["message"]


async def test_reject_tool_outside_recipe_allowlist(monkeypatch, tmp_path):
    """THE boundary test: a recipe scoped to hello.count_to must NOT be able to
    reach textops.word_count even though that session is connected. The model
    emits the out-of-list call; orchestrate refuses it WITHOUT dispatching, the
    refusal is fed back, and the loop still completes."""
    (tmp_path / "recipes.toml").write_text(
        '[recipes.helloonly]\ninferencer="ollama/qwen3"\n'
        'tools=["hello.count_to"]\nsystem="hello only"\n')
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()

    reg, hello_calls, textops_calls = two_provider_registry()
    monkeypatch.setattr(router, "registry", reg)
    mock_inferencer(monkeypatch, [
        _assistant_tool_call("textops.word_count", {"text": "secret"}, "x1"),  # out of list
        _assistant_final("I can't use that tool."),
    ])

    resp = await router.chat_completions(FakeRequest({
        "model": "woollama/helloonly",
        "messages": [{"role": "user", "content": "count the words"}]}))

    # The discriminating check: the forbidden session was NEVER invoked.
    assert textops_calls == [], "out-of-list tool must not be dispatched"
    assert hello_calls == []
    # ...and the chat still completed (refusal fed back, model recovered).
    assert json.loads(resp.body)["choices"][0]["message"]["content"] == \
        "I can't use that tool."
