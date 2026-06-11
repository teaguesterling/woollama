"""Unit tests for `woollama.router`'s route handlers.

We call the handler functions directly instead of going through FastAPI's
TestClient. This avoids the lifespan running (which would spawn real MCP
subprocesses) and keeps the tests purely about the dispatch logic.

The tests:
  * /v1/models — Ollama call mocked; verifies envelope shape + recipe presence
  * /v1/tools  — registry contents surfaced as plain JSON
  * /v1/chat/completions:
      - unknown model namespace → 400
      - woollama/<unknown recipe> → 404
      - ollama/<X> → pass-through (Ollama call mocked)
      - woollama/<recipe>, no tool_calls in Ollama response → final answer
      - woollama/<recipe>, tool_calls present → loop dispatches + composes
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from woollama import router
from woollama import recipes
from woollama.manager import Registry, ServerManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeRequest:
    """Stand-in for Starlette's Request — only the body matters for these
    handlers; we don't need URL routing because we're calling functions."""

    def __init__(self, body: dict):
        self._body = body

    async def json(self) -> dict:
        return self._body


class HttpxResponseStub:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def mock_httpx(monkeypatch, post_responses: list[dict] | dict | None = None,
               get_payload: dict | None = None):
    """Monkeypatch httpx.AsyncClient with predetermined responses. Use a
    list for post_responses to script multi-turn loops; a single dict for
    repeated identical responses."""

    if post_responses is None:
        post_responses = {}
    if isinstance(post_responses, dict):
        post_script = None
        single_post = post_responses
    else:
        post_script = list(post_responses)
        single_post = None

    class _FakeClient:
        def __init__(self, *_a, **_kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return None

        async def get(self, _url, **_kw):
            return HttpxResponseStub(200, get_payload or {})

        async def post(self, _url, json=None, **_kw):
            if post_script is not None:
                return HttpxResponseStub(200, post_script.pop(0))
            return HttpxResponseStub(200, single_post)

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)


def _tool(name: str, description: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema={"type": "object", "properties": {}},
    )


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------

async def test_models_lists_ollama_prefixed_plus_recipes(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()
    mock_httpx(monkeypatch, get_payload={"data": [
        {"id": "qwen3:14b-iq4xs"}, {"id": "llama3:8b"},
    ]})
    resp = await router.list_models()
    data = json.loads(resp.body)["data"]
    ids = [d["id"] for d in data]
    assert "ollama/qwen3:14b-iq4xs" in ids
    assert "ollama/llama3:8b" in ids
    assert "woollama/streamer" in ids
    assert "woollama/textcounter" in ids


async def test_models_survives_ollama_unreachable(monkeypatch, tmp_path):
    """If Ollama is down, /v1/models still returns recipes (just no
    ollama/* entries)."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()
    # FakeClient that raises on get()
    class _Boom:
        def __init__(self, *_a, **_kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_a): return None
        async def get(self, *_a, **_kw):
            raise ConnectionError("ollama unreachable")
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Boom)
    resp = await router.list_models()
    data = json.loads(resp.body)["data"]
    ids = [d["id"] for d in data]
    assert not any(i.startswith("ollama/") for i in ids)
    assert any(i.startswith("woollama/") for i in ids)


# ---------------------------------------------------------------------------
# /v1/tools (the introspection endpoint)
# ---------------------------------------------------------------------------

async def test_tools_endpoint_lists_namespaced_registry(monkeypatch):
    fake_reg = Registry()
    mgr = ServerManager("hello", "echo", [])
    mgr.tools = [_tool("count_to"), _tool("ask_user")]
    fake_reg.add(mgr)
    monkeypatch.setattr(router, "registry", fake_reg)
    resp = await router.list_tools()
    body = json.loads(resp.body)
    assert sorted(body["tools"]) == ["hello.ask_user", "hello.count_to"]


# ---------------------------------------------------------------------------
# /v1/chat/completions — error paths
# ---------------------------------------------------------------------------

async def test_chat_unknown_model_namespace_returns_400():
    resp = await router.chat_completions(
        FakeRequest({"model": "unknown/foo", "messages": []}))
    assert resp.status_code == 400
    body = json.loads(resp.body)
    assert body["error"]["type"] == "invalid_request_error"
    assert "unknown model namespace" in body["error"]["message"]


async def test_chat_unknown_woollama_recipe_returns_404(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()
    resp = await router.chat_completions(
        FakeRequest({"model": "woollama/does-not-exist", "messages": []}))
    assert resp.status_code == 404
    body = json.loads(resp.body)
    assert body["error"]["type"] == "not_found"


# ---------------------------------------------------------------------------
# /v1/chat/completions — pass-through
# ---------------------------------------------------------------------------

async def test_chat_ollama_passthrough_strips_prefix_and_forwards(monkeypatch):
    """ollama/<X> path should strip the prefix before forwarding."""
    captured: dict = {}

    class _SpyClient:
        def __init__(self, *_a, **_kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_a): return None
        async def post(self, url, json=None, **_kw):
            captured["url"] = url
            captured["body"] = json
            return HttpxResponseStub(
                200, {"choices": [{"message": {"content": "ok"}}]})
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _SpyClient)

    await router.chat_completions(FakeRequest({
        "model": "ollama/qwen3:14b-iq4xs",
        "messages": [{"role": "user", "content": "hi"}],
    }))
    assert "/v1/chat/completions" in captured["url"]
    # prefix stripped; non-streaming request forced to stream=False
    assert captured["body"]["model"] == "qwen3:14b-iq4xs"
    assert captured["body"]["stream"] is False


async def test_chat_passthrough_streams_upstream_sse_verbatim(monkeypatch):
    """stream:true on a passthrough model relays the upstream SSE byte-for-byte
    (chunk framing + `[DONE]` sentinel preserved) and keeps stream=true on the
    forwarded body so the upstream actually streams."""
    captured: dict = {}
    chunks = [b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
              b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
              b"data: [DONE]\n\n"]

    class _StreamCM:
        def __init__(self, body):
            captured["body"] = body
            self.status_code = 200
        async def __aenter__(self): return self
        async def __aexit__(self, *_a): return None
        async def aiter_bytes(self):
            for c in chunks:
                yield c

    class _StreamClient:
        def __init__(self, *_a, **_kw): pass
        async def aclose(self): captured["closed"] = True
        def stream(self, _method, url, json=None, **_kw):
            captured["url"] = url
            return _StreamCM(json)
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _StreamClient)

    resp = await router.chat_completions(FakeRequest({
        "model": "ollama/qwen3:14b-iq4xs",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }))
    assert resp.media_type == "text/event-stream"
    body = b"".join([c async for c in resp.body_iterator])
    assert body == b"".join(chunks)
    assert captured["body"]["model"] == "qwen3:14b-iq4xs"
    assert captured["body"]["stream"] is True   # NOT forced off — upstream streams
    assert captured["closed"] is True           # client closed after relay


async def test_chat_passthrough_stream_upstream_error_is_json_not_empty_stream(monkeypatch):
    """An upstream 4xx during a streaming request surfaces as a JSON error with
    the upstream status — never an empty 200 stream (status can't change once a
    StreamingResponse starts)."""
    class _StreamCM:
        def __init__(self): self.status_code = 401
        async def __aenter__(self): return self
        async def __aexit__(self, *_a): return None
        async def aread(self):
            return b'{"error":{"message":"bad key","type":"auth"}}'

    class _StreamClient:
        def __init__(self, *_a, **_kw): pass
        async def aclose(self): pass
        def stream(self, *_a, **_kw): return _StreamCM()
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _StreamClient)

    resp = await router.chat_completions(FakeRequest({
        "model": "ollama/qwen3:14b-iq4xs",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }))
    assert resp.status_code == 401
    assert json.loads(bytes(resp.body))["error"]["message"] == "bad key"


# ---------------------------------------------------------------------------
# /v1/chat/completions — recipe orchestration
# ---------------------------------------------------------------------------

async def test_chat_recipe_no_tool_calls_returns_final_answer(monkeypatch, tmp_path):
    """A recipe whose first inferencer turn produces final content (no tool
    calls) returns that content directly — single loop iteration."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()

    # Use a registry with the right tools available so the recipe's allow-list resolves
    fake_reg = Registry()
    mgr = ServerManager("hello", "echo", [])
    mgr.tools = [_tool("count_to")]
    fake_reg.add(mgr)
    monkeypatch.setattr(router, "registry", fake_reg)

    mock_httpx(monkeypatch, post_responses={
        "choices": [{"message": {"content": "the final answer"}}],
    })

    resp = await router.chat_completions(FakeRequest({
        "model": "woollama/streamer",
        "messages": [{"role": "user", "content": "count to 3"}],
    }))
    body = json.loads(resp.body)
    assert body["choices"][0]["message"]["content"] == "the final answer"


async def test_chat_recipe_with_tool_calls_loops_to_completion(monkeypatch, tmp_path):
    """First turn emits a tool_call; daemon dispatches; second turn emits the
    final content. The client gets back the final answer."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()

    # Need a registry whose 'hello.count_to' dispatches via a stub manager
    fake_reg = Registry()
    mgr = ServerManager("hello", "echo", [])
    mgr.tools = [_tool("count_to")]

    # Override .call_tool to a stub — bypass the queue/task machinery for
    # this dispatch test (manager-internal mechanics are covered in
    # test_manager.py).
    async def stub_call_tool(name, args):
        return SimpleNamespace(content=[
            SimpleNamespace(text=f'{{"count":{args.get("n")},"done":true}}'),
        ])
    mgr.call_tool = stub_call_tool  # type: ignore[method-assign]
    fake_reg.add(mgr)
    monkeypatch.setattr(router, "registry", fake_reg)

    # Two Ollama responses: turn 1 emits a tool_call; turn 2 produces the answer.
    mock_httpx(monkeypatch, post_responses=[
        {"choices": [{"message": {
            "content": "",
            "tool_calls": [{"id": "c1", "function": {
                "name": "hello.count_to",
                "arguments": '{"n": 3}',
            }}],
        }}]},
        {"choices": [{"message": {"content": "Counted to 3."}}]},
    ])

    resp = await router.chat_completions(FakeRequest({
        "model": "woollama/streamer",
        "messages": [{"role": "user", "content": "count to 3"}],
    }))
    body = json.loads(resp.body)
    assert body["choices"][0]["message"]["content"] == "Counted to 3."


async def test_orchestrate_sends_system_prompt_and_feeds_tool_result(monkeypatch, tmp_path):
    """The two halves of a recipe the scripted-mock tests never check, because the
    inferencer mock discards the outgoing `json=`: (1) the recipe's SYSTEM PROMPT
    is sent on turn 1, and (2) a tool's RESULT is fed back into the next turn's
    messages. Capture the posted payloads and assert on them — deleting the
    system prepend or dropping the tool result would otherwise leave the suite
    green."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()

    fake_reg = Registry()
    mgr = ServerManager("hello", "echo", [])
    mgr.tools = [_tool("count_to")]

    async def stub_call_tool(name, args):
        return SimpleNamespace(content=[SimpleNamespace(text="COUNTED_TO_THREE_RESULT")])
    mgr.call_tool = stub_call_tool  # type: ignore[method-assign]
    fake_reg.add(mgr)
    monkeypatch.setattr(router, "registry", fake_reg)

    posted: list[dict] = []
    script = [
        {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "hello.count_to",
                                      "arguments": '{"n": 3}'}}]}}]},
        {"choices": [{"message": {"content": "done"}}]},
    ]

    class _Capture:
        def __init__(self, *_a, **_kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_a): return None
        async def post(self, _url, json=None, **_kw):
            posted.append(json)
            return HttpxResponseStub(200, script.pop(0))
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Capture)

    await router.chat_completions(FakeRequest({
        "model": "woollama/streamer",
        "messages": [{"role": "user", "content": "count to 3"}]}))

    # (1) turn 1 carries the recipe's system prompt as the leading system message.
    turn1 = posted[0]["messages"]
    assert turn1[0]["role"] == "system"
    assert turn1[0]["content"] == recipes.get("streamer")["system"]
    # (2) turn 2 feeds the tool's RESULT back to the model as a tool message.
    turn2 = posted[1]["messages"]
    assert any(m.get("role") == "tool" and "COUNTED_TO_THREE_RESULT" in (m.get("content") or "")
               for m in turn2), turn2


async def test_chat_recipe_inferencer_error_passes_payload_through_502(monkeypatch, tmp_path):
    """A choice-less upstream response (an inferencer error) surfaces as a 502
    that passes the raw upstream payload through — not a generic 200/500. Covers
    the OrchestrationError payload branch the no-payload error tests miss."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()
    monkeypatch.setattr(router, "registry", Registry())
    # No "choices" key → orchestrate raises an inferencer error carrying payload.
    mock_httpx(monkeypatch, post_responses=[{"error": {"message": "model overloaded"}}])

    resp = await router.chat_completions(FakeRequest({
        "model": "woollama/streamer",
        "messages": [{"role": "user", "content": "hi"}]}))
    assert resp.status_code == 502
    assert json.loads(resp.body)["error"]["message"] == "model overloaded"


async def test_orchestrate_continues_when_a_tool_dispatch_raises(monkeypatch, tmp_path):
    """A tool that throws during dispatch doesn't abort the loop: the failure is
    fed back as an ERROR tool result (ok=False) and the model still produces a
    final answer. Distinct from the allow-list-deny path (which never dispatches)."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()

    fake_reg = Registry()
    mgr = ServerManager("hello", "echo", [])
    mgr.tools = [_tool("count_to")]

    async def boom(name, args):
        raise RuntimeError("downstream blew up")
    mgr.call_tool = boom  # type: ignore[method-assign]
    fake_reg.add(mgr)
    monkeypatch.setattr(router, "registry", fake_reg)

    posted: list[dict] = []
    script = [
        {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "hello.count_to",
                                      "arguments": "{}"}}]}}]},
        {"choices": [{"message": {"content": "recovered"}}]},
    ]

    class _Capture:
        def __init__(self, *_a, **_kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_a): return None
        async def post(self, _url, json=None, **_kw):
            posted.append(json)
            return HttpxResponseStub(200, script.pop(0))
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Capture)

    resp = await router.chat_completions(FakeRequest({
        "model": "woollama/streamer",
        "messages": [{"role": "user", "content": "count"}]}))
    assert json.loads(resp.body)["choices"][0]["message"]["content"] == "recovered"
    # the dispatch failure was fed back as an ERROR tool result, not raised.
    tool_msgs = [m for m in posted[1]["messages"] if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["content"].startswith("ERROR: RuntimeError")


async def test_chat_recipe_unknown_inferencer_returns_501(monkeypatch, tmp_path):
    """A recipe pointing at a provider woollama doesn't know should fail clearly,
    not silently. (ollama + anthropic are known; this uses a made-up one.)"""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "recipes.toml").write_text("""
[recipes.bogus]
inferencer = "no-such-provider/some-model"
tools = []
system = "test"
""")
    recipes.reload()

    fake_reg = Registry()
    monkeypatch.setattr(router, "registry", fake_reg)

    resp = await router.chat_completions(FakeRequest({
        "model": "woollama/bogus",
        "messages": [{"role": "user", "content": "hi"}],
    }))
    assert resp.status_code == 501
    body = json.loads(resp.body)
    assert body["error"]["type"] == "not_implemented"
    assert "unsupported inferencer" in body["error"]["message"]


# ---------------------------------------------------------------------------
# /v1/chat/completions — recipe orchestration, STREAMING (slice streaming-2)
# ---------------------------------------------------------------------------

def mock_inferencer_stream(monkeypatch, turns: list[list[dict]]):
    """Monkeypatch httpx.AsyncClient so each `.stream(POST)` plays the next
    scripted SSE turn. A turn is a list of chunk dicts; each becomes a
    `data: {...}` line and a trailing `data: [DONE]` is appended automatically —
    so the per-turn DONE is something the router must SWALLOW, not relay."""
    script = list(turns)

    class _StreamCM:
        def __init__(self, turn):
            # A turn is normally a list of chunk dicts; a dict with `_status`
            # makes that turn return an upstream error (for mid-loop-error tests).
            if isinstance(turn, dict) and "_status" in turn:
                self.status_code = turn["_status"]
                self._lines = []
                self._body = json.dumps(turn.get("_body", {})).encode()
            else:
                self.status_code = 200
                self._lines = ([f"data: {json.dumps(c)}" for c in turn]
                               + ["", "data: [DONE]", ""])
                self._body = b""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return None

        async def aiter_lines(self):
            for ln in self._lines:
                yield ln

        async def aread(self):
            return self._body

    class _Client:
        def __init__(self, *_a, **_kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return None

        async def aclose(self):
            pass

        def stream(self, _method, _url, json=None, **_kw):
            return _StreamCM(script.pop(0))

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)


async def _collect_sse(resp) -> tuple[list[dict], bool, list[dict]]:
    """Drain a StreamingResponse into (chat.completion.chunk dicts, saw_done,
    non-chunk frames e.g. errors)."""
    raw = b""
    async for piece in resp.body_iterator:
        raw += piece if isinstance(piece, bytes) else piece.encode()
    chunks, errors, done = [], [], False
    for frame in raw.decode().split("\n\n"):
        frame = frame.strip()
        if not frame.startswith("data:"):
            continue
        data = frame[len("data:"):].strip()
        if data == "[DONE]":
            done = True
            continue
        obj = json.loads(data)
        (chunks if obj.get("object") == "chat.completion.chunk" else errors).append(obj)
    return chunks, done, errors


def _deltas(chunks: list[dict]) -> str:
    return "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)


def _finishes(chunks: list[dict]) -> list[str]:
    return [c["choices"][0]["finish_reason"] for c in chunks
            if c["choices"][0]["finish_reason"] is not None]


def _delta(text: str, **extra) -> dict:
    return {"choices": [{"delta": {"content": text, **extra}}]}


async def test_chat_recipe_stream_tool_less_streams_content(monkeypatch, tmp_path):
    """A tool-less recipe streams its single final turn as content chunks: one
    role chunk, the deltas verbatim, exactly one finish_reason, then [DONE]."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "recipes.toml").write_text(
        '[recipes.chat]\ninferencer="ollama/qwen3"\ntools=[]\nsystem="be brief"\n')
    recipes.reload()
    monkeypatch.setattr(router, "registry", Registry())

    mock_inferencer_stream(monkeypatch, [[_delta("Hel"), _delta("lo!")]])

    resp = await router.chat_completions(FakeRequest({
        "model": "woollama/chat", "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }))
    assert resp.media_type == "text/event-stream"
    chunks, done, errors = await _collect_sse(resp)
    assert not errors and done
    assert _deltas(chunks) == "Hello!"            # streamed once, in order
    assert _finishes(chunks) == ["stop"]          # exactly one terminator
    # role announced exactly once, before any content
    roles = [c["choices"][0]["delta"].get("role") for c in chunks]
    assert roles.count("assistant") == 1 and roles[0] == "assistant"


async def test_chat_recipe_stream_tool_turn_is_invisible(monkeypatch, tmp_path):
    """The tool turn must not leak: the model emits a tool_call (fragmented
    across chunks, no content) on turn 1, woollama dispatches it, turn 2 streams
    the answer. The client sees ONLY the final answer and ONE finish_reason —
    no tool JSON, and the per-turn `tool_calls` finish/[DONE] are swallowed."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()                              # bundled `streamer` (hello.count_to)

    dispatched: dict = {}
    fake_reg = Registry()
    mgr = ServerManager("hello", "echo", [])
    mgr.tools = [_tool("count_to")]

    async def stub_call_tool(name, args):
        dispatched["name"], dispatched["args"] = name, args
        return SimpleNamespace(content=[SimpleNamespace(text='{"done":true}')])
    mgr.call_tool = stub_call_tool  # type: ignore[method-assign]
    fake_reg.add(mgr)
    monkeypatch.setattr(router, "registry", fake_reg)

    # Turn 1: a tool_call split across chunks (id+name first, arguments in
    # pieces) and NO content. Turn 2: the streamed final answer.
    mock_inferencer_stream(monkeypatch, [
        [
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1",
                 "function": {"name": "hello.count_to"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"n":'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": " 3}"}}]}}]},
        ],
        [_delta("Counted "), _delta("to 3.")],
    ])

    resp = await router.chat_completions(FakeRequest({
        "model": "woollama/streamer", "stream": True,
        "messages": [{"role": "user", "content": "count to 3"}],
    }))
    chunks, done, errors = await _collect_sse(resp)
    assert not errors and done
    # dispatched with the REASSEMBLED fragmented arguments (Registry routes by
    # the `hello.` prefix and hands the manager the bare tool name).
    assert dispatched == {"name": "count_to", "args": {"n": 3}}
    # client saw only the final answer — no tool JSON leaked
    out = _deltas(chunks)
    assert out == "Counted to 3."
    assert "tool_call" not in out and "count_to" not in out
    assert _finishes(chunks) == ["stop"]          # ONE terminator across 2 turns


async def test_chat_recipe_stream_empty_final_is_valid_stream(monkeypatch, tmp_path):
    """A final turn with empty content still yields a well-formed stream: a role
    chunk, one finish_reason:stop, and [DONE] — no content chunks."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "recipes.toml").write_text(
        '[recipes.chat]\ninferencer="ollama/qwen3"\ntools=[]\nsystem="x"\n')
    recipes.reload()
    monkeypatch.setattr(router, "registry", Registry())

    mock_inferencer_stream(monkeypatch, [[{"choices": [{"delta": {}}]}]])

    resp = await router.chat_completions(FakeRequest({
        "model": "woollama/chat", "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }))
    chunks, done, errors = await _collect_sse(resp)
    assert not errors and done
    assert _deltas(chunks) == ""
    assert _finishes(chunks) == ["stop"]


async def test_chat_recipe_stream_setup_error_is_json_status_not_stream(monkeypatch, tmp_path):
    """An error BEFORE any output (unsupported inferencer) must come back as a
    proper HTTP status JSON, never an empty 200 event-stream."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "recipes.toml").write_text(
        '[recipes.bogus]\ninferencer="no-such-provider/m"\ntools=[]\nsystem="x"\n')
    recipes.reload()
    monkeypatch.setattr(router, "registry", Registry())

    resp = await router.chat_completions(FakeRequest({
        "model": "woollama/bogus", "stream": True,
        "messages": [{"role": "user", "content": "hi"}],
    }))
    assert resp.status_code == 501                # JSON error, not a stream
    assert resp.media_type != "text/event-stream"
    assert json.loads(resp.body)["error"]["type"] == "not_implemented"


async def test_chat_recipe_stream_mid_loop_error_is_sse_frame_not_status(monkeypatch, tmp_path):
    """Once streaming has begun, the HTTP status can't change. A tool turn
    succeeds (committing 200), then a later turn's inferencer error surfaces as
    an SSE error frame + a clean terminator — NOT an HTTP error. (Since slice
    streaming-3 the first surfaced event is the turn-1 tool_call, so 200 commits
    before the turn-2 error; pre-streaming-3 this same case mapped to a status.)"""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()                              # bundled streamer (hello.count_to)

    fake_reg = Registry()
    mgr = ServerManager("hello", "echo", [])
    mgr.tools = [_tool("count_to")]

    async def stub_call_tool(name, args):
        return SimpleNamespace(content=[SimpleNamespace(text='{"done":true}')])
    mgr.call_tool = stub_call_tool  # type: ignore[method-assign]
    fake_reg.add(mgr)
    monkeypatch.setattr(router, "registry", fake_reg)

    # Turn 1: a tool_call (commits the 200). Turn 2: upstream errors.
    mock_inferencer_stream(monkeypatch, [
        [{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1",
            "function": {"name": "hello.count_to", "arguments": "{}"}}]}}]}],
        {"_status": 500, "_body": {"error": {"message": "upstream boom"}}},
    ])

    resp = await router.chat_completions(FakeRequest({
        "model": "woollama/streamer", "stream": True,
        "messages": [{"role": "user", "content": "count to 3"}],
    }))
    assert resp.media_type == "text/event-stream"   # a stream, not a status JSON
    chunks, done, errors = await _collect_sse(resp)
    assert done                                     # still terminated cleanly
    assert _finishes(chunks) == ["stop"]            # one terminator
    assert errors and errors[0]["error"]["message"] == "upstream boom"


# ---------------------------------------------------------------------------
# orchestrate_events — tool-progress events (slice streaming-3)
# ---------------------------------------------------------------------------

async def test_orchestrate_events_emits_tool_progress_in_order(monkeypatch, tmp_path):
    """The core loop yields tool_call → tool_result (ok=True) around each
    dispatch, then a final event — the protocol the MCP chat tool turns into
    ctx.info notifications. Emitted even in stream=False mode."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    recipes.reload()                              # bundled streamer (hello.count_to)

    fake_reg = Registry()
    mgr = ServerManager("hello", "echo", [])
    mgr.tools = [_tool("count_to")]

    async def stub_call_tool(name, args):
        return SimpleNamespace(content=[SimpleNamespace(text='{"done":true}')])
    mgr.call_tool = stub_call_tool  # type: ignore[method-assign]
    fake_reg.add(mgr)

    mock_httpx(monkeypatch, post_responses=[
        {"choices": [{"message": {"content": "", "tool_calls": [{"id": "c1",
            "function": {"name": "hello.count_to", "arguments": '{"n": 3}'}}]}}]},
        {"choices": [{"message": {"content": "done"}}]},
    ])

    events = [ev async for ev in router.orchestrate_events(
        recipes.get("streamer"),
        [{"role": "user", "content": "count to 3"}], fake_reg, stream=False)]

    assert [e["type"] for e in events] == ["tool_call", "tool_result", "final"]
    assert events[0] == {"type": "tool_call", "turn": 1,
                         "name": "hello.count_to", "args": {"n": 3}}
    assert events[1] == {"type": "tool_result", "turn": 1,
                         "name": "hello.count_to", "ok": True}
    assert events[2]["response"]["choices"][0]["message"]["content"] == "done"


async def test_orchestrate_events_tool_result_ok_false_when_denied(monkeypatch, tmp_path):
    """A tool the recipe's allow-list forbids is refused: tool_result.ok is
    False, and the loop still continues to a final answer."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "recipes.toml").write_text(
        '[recipes.locked]\ninferencer="ollama/q"\ntools=[]\nsystem="x"\n')
    recipes.reload()

    mock_httpx(monkeypatch, post_responses=[
        {"choices": [{"message": {"content": "", "tool_calls": [{"id": "c1",
            "function": {"name": "hello.count_to", "arguments": "{}"}}]}}]},
        {"choices": [{"message": {"content": "sorry"}}]},
    ])

    events = [ev async for ev in router.orchestrate_events(
        recipes.get("locked"),
        [{"role": "user", "content": "go"}], Registry(), stream=False)]

    result = next(e for e in events if e["type"] == "tool_result")
    assert result["ok"] is False                  # denied by the empty allow-list
    assert events[-1]["type"] == "final"          # loop recovered to a final answer
