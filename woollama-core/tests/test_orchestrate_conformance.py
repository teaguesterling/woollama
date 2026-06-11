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


def _multi_tool_call_resp(calls):
    # calls: list of (name, args, call_id) — a single assistant message, N tool_calls
    return {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "",
        "tool_calls": [{"id": cid, "type": "function",
                        "function": {"name": n, "arguments": json.dumps(a)}}
                       for (n, a, cid) in calls]}}]}


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
    # `tools` is OMITTED when the recipe allow-lists none (Anthropic rejects `[]`).
    assert b["model"] == "gpt-x" and b["stream"] is False and "tools" not in b
    assert b["temperature"] == 0.7                       # param wins over extra_body's 0
    assert b["messages"][0] == {"role": "system", "content": "s"}


def test_ollama_extra_body_nests_options(base_url):
    _LoopMock.script = [_final_resp("hi")]
    prov = MockProvider({})
    recipe = {"inferencer": "ollama/qwen", "system": "s", "tools": []}

    _run(recipe, prov, base_url)                          # no api_key — ollama needs none

    assert _LoopMock.requests[0]["options"] == {"temperature": 0}


def test_multiple_tool_calls_in_one_turn(base_url):
    # Models routinely emit PARALLEL tool calls in one assistant message. All must
    # be dispatched (in order), and the next request must carry the single assistant
    # message with BOTH calls plus one matching `tool` message per call_id.
    _LoopMock.script = [
        _multi_tool_call_resp([("hello.count_to", {"n": 1}, "c1"),
                               ("math.add", {"a": 2, "b": 3}, "c2")]),
        _final_resp("both done"),
    ]
    prov = MockProvider({
        "hello.count_to": lambda a: _Result([_Block("one")]),
        "math.add": lambda a: _Result([_Block("five")]),
    })
    recipe = {"inferencer": "openai/gpt-x", "system": "s",
              "tools": ["hello.count_to", "math.add"]}

    out = _run(recipe, prov, base_url, api_key="k")

    assert out["choices"][0]["message"]["content"] == "both done"
    assert prov.dispatched == [("hello.count_to", {"n": 1}), ("math.add", {"a": 2, "b": 3})]
    second = _LoopMock.requests[1]["messages"]
    asst = [m for m in second if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(asst) == 1 and len(asst[0]["tool_calls"]) == 2     # one msg, both calls
    tms = _tool_msgs(_LoopMock.requests[1])
    assert [(m["tool_call_id"], m["content"]) for m in tms] == [("c1", "one"), ("c2", "five")]


def test_max_turns_exceeded_raises(base_url):
    # The model never stops calling tools → the loop must give up at 8 turns.
    _LoopMock.script = [_tool_call_resp("hello.count_to", {}, f"c{i}") for i in range(8)]
    prov = MockProvider({"hello.count_to": lambda a: _Result([_Block("x")])})
    recipe = {"inferencer": "openai/gpt-x", "system": "s", "tools": ["hello.count_to"]}

    with pytest.raises(wc.InferenceError, match="max turns"):
        _run(recipe, prov, base_url, api_key="k")
    assert len(_LoopMock.requests) == 8                            # exactly 8 turns, no 9th


def test_unsupported_inferencer_raises_before_await():
    # setup runs synchronously, so this raises on the call (before the awaitable).
    prov = MockProvider({})
    recipe = {"inferencer": "bogus/m", "system": "s", "tools": []}
    with pytest.raises(wc.InferenceError):
        orchestrate(recipe, MSGS, prov)


# --- orchestrate_events (the per-event generator the server consumes) ---------

def _events(recipe, prov, base_url, **kw):
    async def go():
        return [ev async for ev in
                wc.orchestrate_events(recipe, MSGS, prov, base_url=base_url, **kw)]
    return asyncio.run(go())


def test_events_emits_tool_call_result_final_in_order(base_url):
    _LoopMock.script = [_tool_call_resp("hello.count_to", {"n": 3}), _final_resp("counted to 3")]
    prov = MockProvider({"hello.count_to": lambda a: _Result([_Block("1 2 3")])})
    recipe = {"inferencer": "openai/gpt-x", "system": "s", "tools": ["hello.count_to"]}

    evs = _events(recipe, prov, base_url, api_key="k")

    assert [e["type"] for e in evs] == ["tool_call", "tool_result", "final"]
    tc, tr, fin = evs
    assert (tc["turn"], tc["name"], tc["args"]) == (1, "hello.count_to", {"n": 3})
    assert (tr["turn"], tr["name"], tr["ok"]) == (1, "hello.count_to", True)
    assert fin["response"]["choices"][0]["message"]["content"] == "counted to 3"


def test_events_tool_result_ok_false_on_is_error(base_url):
    _LoopMock.script = [_tool_call_resp("hello.count_to", {}), _final_resp("ok")]
    prov = MockProvider({"hello.count_to": lambda a: _Result([_Block("boom")], is_error=True)})
    recipe = {"inferencer": "openai/gpt-x", "system": "s", "tools": ["hello.count_to"]}

    evs = _events(recipe, prov, base_url, api_key="k")

    assert [e for e in evs if e["type"] == "tool_result"][0]["ok"] is False


def test_events_parallel_calls_emit_two_pairs_in_order(base_url):
    _LoopMock.script = [
        _multi_tool_call_resp([("hello.count_to", {}, "c1"), ("math.add", {}, "c2")]),
        _final_resp("done"),
    ]
    prov = MockProvider({"hello.count_to": lambda a: _Result([_Block("x")]),
                         "math.add": lambda a: _Result([_Block("y")])})
    recipe = {"inferencer": "openai/gpt-x", "system": "s",
              "tools": ["hello.count_to", "math.add"]}

    evs = _events(recipe, prov, base_url, api_key="k")

    assert [e["type"] for e in evs] == \
        ["tool_call", "tool_result", "tool_call", "tool_result", "final"]
    assert [e["name"] for e in evs if e["type"] == "tool_call"] == ["hello.count_to", "math.add"]


def test_events_refusal_emits_tool_result_ok_false_without_dispatch(base_url):
    _LoopMock.script = [_tool_call_resp("evil.rm", {}), _final_resp("done")]
    prov = MockProvider({})
    recipe = {"inferencer": "openai/gpt-x", "system": "s", "tools": ["hello.count_to"]}

    evs = _events(recipe, prov, base_url, api_key="k")

    assert prov.dispatched == []
    tr = [e for e in evs if e["type"] == "tool_result"][0]
    assert tr["name"] == "evil.rm" and tr["ok"] is False


def test_events_max_turns_raises_during_iteration(base_url):
    _LoopMock.script = [_tool_call_resp("hello.count_to", {}, f"c{i}") for i in range(8)]
    prov = MockProvider({"hello.count_to": lambda a: _Result([_Block("x")])})
    recipe = {"inferencer": "openai/gpt-x", "system": "s", "tools": ["hello.count_to"]}

    with pytest.raises(wc.InferenceError, match="max turns"):
        _events(recipe, prov, base_url, api_key="k")


def test_events_unsupported_inferencer_raises_before_iter():
    # setup is eager (a deliberate divergence from Python's lazy generator) — raises
    # on the call, not on first `__anext__`.
    prov = MockProvider({})
    recipe = {"inferencer": "bogus/m", "system": "s", "tools": []}
    with pytest.raises(wc.InferenceError):
        wc.orchestrate_events(recipe, MSGS, prov)


# --- streaming (stream=True): delta events + SSE per-turn tool_call reassembly ---

def _sse(*chunks):
    return "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"


def _content_chunk(text):
    return {"choices": [{"delta": {"content": text}}]}


def _tc_chunk(idx, **fields):
    tc = {"index": idx}
    tc.update(fields)
    return {"choices": [{"delta": {"tool_calls": [tc]}}]}


class _SSELoopMock(BaseHTTPRequestHandler):
    script: list = []        # (status, sse_body) per streamed turn
    requests: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _SSELoopMock.requests.append(body)
        status, sse = _SSELoopMock.script[len(_SSELoopMock.requests) - 1]
        raw = sse.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def sse_url():
    _SSELoopMock.script, _SSELoopMock.requests = [], []
    srv = HTTPServer(("127.0.0.1", 0), _SSELoopMock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    finally:
        srv.shutdown()


def _stream_events(recipe, prov, sse_url, **kw):
    async def go():
        return [ev async for ev in
                wc.orchestrate_events(recipe, MSGS, prov, base_url=sse_url, stream=True, **kw)]
    return asyncio.run(go())


def test_stream_yields_delta_events_then_synthesized_final(sse_url):
    _SSELoopMock.script = [(200, _sse(_content_chunk("Hel"), _content_chunk("lo"), _content_chunk("!")))]
    prov = MockProvider({})
    recipe = {"inferencer": "openai/gpt-x", "system": "s", "tools": []}

    evs = _stream_events(recipe, prov, sse_url, api_key="k")

    assert [e["content"] for e in evs if e["type"] == "delta"] == ["Hel", "lo", "!"]
    fin = [e for e in evs if e["type"] == "final"][0]["response"]
    # the final is SYNTHESIZED from the deltas (no single upstream object exists)
    assert fin["object"] == "chat.completion"
    choice = fin["choices"][0]
    assert choice["message"]["content"] == "Hello!"
    assert choice["message"]["tool_calls"] is None       # `calls or None`
    assert choice["finish_reason"] == "stop"


def test_stream_reassembles_fragmented_tool_call_and_dispatches(sse_url):
    # turn 1 streams ONE tool call: id+name in one chunk, `arguments` split across two
    # (the bug-prone path). turn 2 streams the final text.
    _SSELoopMock.script = [
        (200, _sse(
            _tc_chunk(0, id="cZ", function={"name": "hello.count_to"}),
            _tc_chunk(0, function={"arguments": '{"n":'}),
            _tc_chunk(0, function={"arguments": "3}"}),
        )),
        (200, _sse(_content_chunk("did 3"))),
    ]
    prov = MockProvider({"hello.count_to": lambda a: _Result([_Block("123")])})
    recipe = {"inferencer": "openai/gpt-x", "system": "s", "tools": ["hello.count_to"]}

    evs = _stream_events(recipe, prov, sse_url, api_key="k")

    assert prov.dispatched == [("hello.count_to", {"n": 3})]   # args reassembled, then parsed
    assert [e["type"] for e in evs] == ["tool_call", "tool_result", "delta", "final"]
    tc = [e for e in evs if e["type"] == "tool_call"][0]
    assert tc["name"] == "hello.count_to" and tc["args"] == {"n": 3}
    # turn 2's request carried the assistant tool_call (reassembled id "cZ") + tool msg
    tool_msgs = [m for m in _SSELoopMock.requests[1]["messages"] if m.get("role") == "tool"]
    assert tool_msgs[0]["tool_call_id"] == "cZ" and tool_msgs[0]["content"] == "123"
    assert [e for e in evs if e["type"] == "final"][0]["response"]["choices"][0]["message"]["content"] == "did 3"


def test_stream_upstream_error_raises(sse_url):
    _SSELoopMock.script = [(503, '{"error": "down"}')]
    prov = MockProvider({})
    recipe = {"inferencer": "openai/gpt-x", "system": "s", "tools": []}

    with pytest.raises(wc.InferenceError):
        _stream_events(recipe, prov, sse_url, api_key="k")
