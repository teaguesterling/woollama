"""Conformance tests for the Rust `orchestrate` — the recipe↔tool loop. It must
behave like `woollama.core.orchestrate` (the drainer over `orchestrate_events`),
the oracle: prepend the system prompt, offer the allow-listed tools, dispatch the
ones the model calls through a Python `ToolProvider`, feed results back, repeat
(≤8 turns), return the final OpenAI response dict.

Two mocks stand in for the two seams: a threaded HTTP server scripted to drive the
inferencer turns, and a Python `ToolProvider` returning genuinely `ToolResult`-
shaped objects (a block with `.text`, an `is_error` case, a raising dispatch) so
the Rust render path and the `[tool error]` / `ERROR: …` handling are exercised,
not just the happy join.
"""
from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from woollama import core as wc

orchestrate = wc.orchestrate
MSGS = [{"role": "user", "content": "go"}]


# --- ToolProvider mock (ToolResult/ToolSpec-shaped) --------------------------

class _Block:
    """A text content block (mirrors an MCP text block: has `.text`)."""
    def __init__(self, text: str):
        self.text = text


class _Result:
    """A faithful `ToolResult` shape: `.blocks` list + `.is_error`."""
    def __init__(self, blocks, is_error: bool = False):
        self.blocks = blocks
        self.is_error = is_error
        self.structured = None
        self.meta = None


class _Spec:
    """A `ToolSpec` shape: the loop reads only `.schema`."""
    def __init__(self, name: str):
        self.name = name
        self.schema = {"type": "function", "function": {
            "name": name, "description": "d",
            "parameters": {"type": "object", "properties": {}}}}


class MockProvider:
    """`tools_for(allow)` advertises one function per allowed name; async
    `dispatch` records calls and returns (or raises) per a name->handler map."""
    def __init__(self, handlers):
        self.handlers = handlers          # name -> callable(args) -> _Result (may raise)
        self.dispatched: list = []

    def tools_for(self, allow):
        return [_Spec(n) for n in allow]

    async def dispatch(self, name, args):
        await asyncio.sleep(0)            # a real suspension point
        self.dispatched.append((name, args))
        return self.handlers[name](args)


# --- inferencer mock (scripted OpenAI chat.completion responses) -------------

def _tool_call_resp(name, args, call_id="c1"):
    return {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "",
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)}}]}}]}


def _final_resp(text):
    return {"choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}]}


class _LoopMock(BaseHTTPRequestHandler):
    script: list = []        # responses popped in order, one per inferencer turn
    requests: list = []      # request bodies seen

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _LoopMock.requests.append(body)
        payload = _LoopMock.script[len(_LoopMock.requests) - 1]
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def base_url():
    _LoopMock.script, _LoopMock.requests = [], []
    srv = HTTPServer(("127.0.0.1", 0), _LoopMock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    finally:
        srv.shutdown()


def _run(recipe, prov, base_url, **kw):
    # NB: the awaitable binds to the running loop at creation (pyo3-async-runtimes),
    # so it must be BUILT inside the loop — call + await together in a coroutine.
    async def go():
        return await orchestrate(recipe, MSGS, prov, base_url=base_url, **kw)
    return asyncio.run(go())


def _tool_msgs(body):
    return [m for m in body["messages"] if m.get("role") == "tool"]


# --- tests -------------------------------------------------------------------

def test_dispatches_tool_then_returns_final(base_url):
    _LoopMock.script = [_tool_call_resp("hello.count_to", {"n": 3}), _final_resp("counted to 3")]
    prov = MockProvider({"hello.count_to": lambda a: _Result([_Block("1 2 3")])})
    recipe = {"inferencer": "openai/gpt-x", "system": "sys", "tools": ["hello.count_to"]}

    out = _run(recipe, prov, base_url, api_key="k")

    assert out["choices"][0]["message"]["content"] == "counted to 3"
    assert prov.dispatched == [("hello.count_to", {"n": 3})]   # args JSON-string -> dict
    # The 2nd request carried system, the assistant tool_call, and the rendered result.
    second = _LoopMock.requests[1]["messages"]
    assert second[0] == {"role": "system", "content": "sys"}
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in second)
    tm = _tool_msgs(_LoopMock.requests[1])
    assert tm and tm[0]["content"] == "1 2 3" and tm[0]["tool_call_id"] == "c1"


def test_refuses_out_of_list_tool_without_dispatch(base_url):
    # The model (mock) calls a tool NOT on the recipe's allow-list. The loop must
    # refuse it at the boundary — never reaching dispatch — and feed the refusal
    # back so it can recover. (The adversarial unit gate.)
    _LoopMock.script = [_tool_call_resp("evil.rm", {}), _final_resp("done")]
    prov = MockProvider({})
    recipe = {"inferencer": "openai/gpt-x", "system": "s", "tools": ["hello.count_to"]}

    out = _run(recipe, prov, base_url, api_key="k")

    assert out["choices"][0]["message"]["content"] == "done"
    assert prov.dispatched == []                       # forbidden tool NEVER dispatched
    tm = _tool_msgs(_LoopMock.requests[1])[0]
    assert "not permitted" in tm["content"] and "evil.rm" in tm["content"]


def test_is_error_result_gets_tool_error_prefix(base_url):
    _LoopMock.script = [_tool_call_resp("hello.count_to", {}), _final_resp("ok")]
    prov = MockProvider({"hello.count_to": lambda a: _Result([_Block("boom")], is_error=True)})
    recipe = {"inferencer": "openai/gpt-x", "system": "s", "tools": ["hello.count_to"]}

    _run(recipe, prov, base_url, api_key="k")

    assert _tool_msgs(_LoopMock.requests[1])[0]["content"] == "[tool error] boom"


def test_dispatch_exception_becomes_error_result_and_loop_recovers(base_url):
    def raiser(a):
        raise ValueError("nope")

    _LoopMock.script = [_tool_call_resp("hello.count_to", {}), _final_resp("recovered")]
    prov = MockProvider({"hello.count_to": raiser})
    recipe = {"inferencer": "openai/gpt-x", "system": "s", "tools": ["hello.count_to"]}

    out = _run(recipe, prov, base_url, api_key="k")

    assert out["choices"][0]["message"]["content"] == "recovered"   # did NOT propagate
    assert _tool_msgs(_LoopMock.requests[1])[0]["content"] == "ERROR: ValueError: nope"


def test_merges_extra_body_and_params(base_url):
    # cloud extra_body is {"temperature": 0}; the recipe param overrides it.
    _LoopMock.script = [_final_resp("hi")]
    prov = MockProvider({})
    recipe = {"inferencer": "openai/gpt-x", "system": "s", "tools": [],
              "params": {"temperature": 0.7}}

    _run(recipe, prov, base_url, api_key="k")

    b = _LoopMock.requests[0]
    assert b["model"] == "gpt-x" and b["stream"] is False and b["tools"] == []
    assert b["temperature"] == 0.7                       # param wins over extra_body's 0
    assert b["messages"][0] == {"role": "system", "content": "s"}


def test_ollama_extra_body_nests_options(base_url):
    _LoopMock.script = [_final_resp("hi")]
    prov = MockProvider({})
    recipe = {"inferencer": "ollama/qwen", "system": "s", "tools": []}

    _run(recipe, prov, base_url)                          # no api_key — ollama needs none

    assert _LoopMock.requests[0]["options"] == {"temperature": 0}


def test_unsupported_inferencer_raises_before_await():
    # setup runs synchronously, so this raises on the call (before the awaitable).
    prov = MockProvider({})
    recipe = {"inferencer": "bogus/m", "system": "s", "tools": []}
    with pytest.raises(wc.InferenceError):
        orchestrate(recipe, MSGS, prov)
