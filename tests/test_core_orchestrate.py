"""core.orchestrate + core.tooling — the recipe loop and tool seam, server-free.

Drives `orchestrate_events` DIRECTLY over a fake `ToolProvider` (the embedder path
— no router, no MCP), proving the generic inferencer↔tool loop, the lossless
ToolResult render, and the isError fix (a tool that *reports* failure is no longer
mistaken for success). The router→core delegation + the real MCP adapter are
covered by test_routing / test_responses_stream.
"""
from __future__ import annotations

import httpx

from woollama.core import orchestrate
from woollama.core.tooling import ToolResult, ToolSpec, render_tool_result


class _Block:
    def __init__(self, text):
        self.text = text

    def model_dump(self):
        return {"type": "text", "text": self.text}


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


class _SeqClient:
    """httpx.AsyncClient fake: returns queued chat.completion payloads per POST,
    recording each request body."""
    queue: list = []
    posts: list = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    async def post(self, url, json=None, headers=None):
        _SeqClient.posts.append(json)
        return _Resp(_SeqClient.queue.pop(0))


class FakeTools:
    """A ToolProvider over a canned ToolResult; records dispatch calls."""
    def __init__(self, result: ToolResult):
        self._result = result
        self.calls: list = []

    def tools_for(self, allow):
        return [ToolSpec(name=n, schema={"type": "function", "function": {
            "name": n, "description": "", "parameters": {}}}) for n in allow]

    async def dispatch(self, name, args):
        self.calls.append((name, args))
        return self._result


def _seq(monkeypatch, payloads):
    monkeypatch.setattr(httpx, "AsyncClient", _SeqClient)
    _SeqClient.queue = list(payloads)
    _SeqClient.posts = []


def _tool_call_turn(name, args_json, cid):
    return {"choices": [{"message": {"role": "assistant", "content": "",
            "tool_calls": [{"id": cid, "type": "function",
                            "function": {"name": name, "arguments": args_json}}]}}]}


def _final_turn(text):
    return {"choices": [{"message": {"role": "assistant", "content": text}}]}


RECIPE = {"inferencer": "ollama/qwen3", "system": "sys", "tools": ["srv.tool"]}


async def _drain(recipe, tools, **kw):
    events = []
    async for ev in orchestrate.orchestrate_events(recipe, [{"role": "user", "content": "hi"}],
                                                    tools=tools, **kw):
        events.append(ev)
    return events


# --- the loop -----------------------------------------------------------------

async def test_orchestrate_runs_tool_then_finalizes(monkeypatch):
    _seq(monkeypatch, [_tool_call_turn("srv.tool", '{"x": 1}', "c1"),
                       _final_turn("done")])
    tools = FakeTools(ToolResult(blocks=[_Block("42")]))
    events = await _drain(RECIPE, tools)

    # the tool was dispatched with the model's args
    assert tools.calls == [("srv.tool", {"x": 1})]
    # progress events + a single final
    assert [e["type"] for e in events] == ["tool_call", "tool_result", "final"]
    assert events[1] == {"type": "tool_result", "turn": 1, "name": "srv.tool", "ok": True}
    assert events[-1]["response"]["choices"][0]["message"]["content"] == "done"
    # the RENDERED tool result reached the model as the tool message (turn-2 req)
    tool_msg = _SeqClient.posts[1]["messages"][-1]
    assert tool_msg == {"role": "tool", "content": "42", "tool_call_id": "c1"}
    # system prompt was prepended
    assert _SeqClient.posts[0]["messages"][0] == {"role": "system", "content": "sys"}


async def test_orchestrate_merges_recipe_params(monkeypatch):
    """A recipe's optional `params` (e.g. temperature) are merged into the request."""
    _seq(monkeypatch, [_final_turn("ok")])
    recipe = {"inferencer": "ollama/qwen3", "system": "s", "tools": [],
              "params": {"temperature": 0.5}}
    await _drain(recipe, FakeTools(ToolResult()))
    assert _SeqClient.posts[0]["temperature"] == 0.5


async def test_orchestrate_no_tools_single_turn(monkeypatch):
    _seq(monkeypatch, [_final_turn("hello")])
    events = await _drain({"inferencer": "ollama/qwen3", "system": "s", "tools": []},
                          FakeTools(ToolResult()))
    assert [e["type"] for e in events] == ["final"]
    assert events[0]["response"]["choices"][0]["message"]["content"] == "hello"


async def test_orchestrate_is_error_marks_not_ok(monkeypatch):
    """The fix: a tool that returns is_error=True (no exception) → ok=False AND the
    rendered content is prefixed, so the model can tell it failed."""
    _seq(monkeypatch, [_tool_call_turn("srv.tool", "{}", "c1"), _final_turn("ok")])
    tools = FakeTools(ToolResult(blocks=[_Block("boom")], is_error=True))
    events = await _drain(RECIPE, tools)
    assert events[1]["ok"] is False
    assert _SeqClient.posts[1]["messages"][-1]["content"] == "[tool error] boom"


# --- the renderer -------------------------------------------------------------

def test_render_text_blocks_joined():
    assert render_tool_result(ToolResult(blocks=[_Block("a"), _Block("b")])) == "a\nb"


class _ImageBlock:
    def model_dump(self):
        return {"type": "image", "mimeType": "image/png"}


def test_render_no_text_falls_back_to_json_dump():
    # A non-text block (no .text) → JSON-dump the raw blocks (matches the original
    # fallback), so a structured/non-text result still reaches the model.
    out = render_tool_result(ToolResult(blocks=[_ImageBlock()]))
    assert out.startswith("[") and '"type": "image"' in out


def test_render_is_error_prefix():
    assert render_tool_result(ToolResult(blocks=[_Block("nope")], is_error=True)) \
        == "[tool error] nope"
    assert render_tool_result(ToolResult(is_error=True)) == "[tool error]"
