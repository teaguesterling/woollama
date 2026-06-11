"""Streaming /v1/responses — OpenAI Responses SSE (conv-1a streaming).

Proves a `stream:true` stateless /v1/responses turn emits the canonical Responses
event sequence (created → output_item.added → content_part.added →
output_text.delta* → output_text.done → content_part.done → output_item.done →
completed), sourcing deltas from a plain inferencer's chat SSE (mocked httpx) or a
recipe (mocked orchestrate_events). The emitted frames are validated against the
real `openai` SDK event models — the whole point of speaking the shape. Stateful
streaming is cleanly deferred (400); setup errors surface as HTTP status, not an
empty 200 stream.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from woollama import recipes, router


class FakeRequest:
    def __init__(self, body: dict):
        self._body = body

    async def json(self) -> dict:
        return self._body


async def _collect(resp) -> list[dict]:
    """Drain a StreamingResponse into a list of {event, data} SSE frames."""
    body = "".join([c if isinstance(c, str) else c.decode()
                    async for c in resp.body_iterator])
    frames = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        event = data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        frames.append({"event": event, "data": data})
    return frames


def _mock_inferencer_stream(monkeypatch, deltas, status=200):
    """Stream chat.completion.chunk SSE from a mock HTTP server + point
    $WOOLLAMA_OLLAMA_URL at it. The Rust core streams via reqwest, which an in-process
    httpx patch can't intercept."""
    if status >= 400:
        body = '{"error": {"message": "upstream"}}'
    else:
        body = "".join(
            f'data: {json.dumps({"choices": [{"delta": {"content": d}}]})}\n\n'
            for d in deltas) + "data: [DONE]\n\n"
    raw = body.encode()

    class _H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("WOOLLAMA_OLLAMA_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    return srv


# --- inferencer streaming -----------------------------------------------------

async def test_stateless_inferencer_streams_responses_sse(monkeypatch):
    _mock_inferencer_stream(monkeypatch, ["He", "llo"])
    r = await router.responses_create(FakeRequest({
        "model": "ollama/qwen3", "input": "hi", "stream": True}))
    frames = await _collect(r)
    types = [f["event"] for f in frames]
    assert types == [
        "response.created", "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta", "response.output_text.delta",
        "response.output_text.done", "response.content_part.done",
        "response.output_item.done", "response.completed",
    ]
    # sequence numbers are monotonic from 0
    assert [f["data"]["sequence_number"] for f in frames] == list(range(len(frames)))
    # deltas + accumulated text
    deltas = [f["data"]["delta"] for f in frames
              if f["event"] == "response.output_text.delta"]
    assert deltas == ["He", "llo"]
    done = next(f for f in frames if f["event"] == "response.output_text.done")
    assert done["data"]["text"] == "Hello"
    completed = next(f for f in frames if f["event"] == "response.completed")
    assert completed["data"]["response"]["output"][0]["content"][0]["text"] == "Hello"
    assert completed["data"]["response"]["status"] == "completed"


async def test_emitted_events_validate_against_openai_sdk(monkeypatch):
    """The emitted frames parse as the real openai Responses event models — the
    grounding that we speak the actual wire shape."""
    from openai.types.responses import (
        ResponseCompletedEvent,
        ResponseCreatedEvent,
        ResponseTextDeltaEvent,
    )
    _mock_inferencer_stream(monkeypatch, ["x"])
    frames = await _collect(await router.responses_create(FakeRequest({
        "model": "ollama/qwen3", "input": "hi", "stream": True})))
    by = {f["event"]: f["data"] for f in frames}
    ResponseCreatedEvent.model_validate(by["response.created"])
    ResponseTextDeltaEvent.model_validate(by["response.output_text.delta"])
    ResponseCompletedEvent.model_validate(by["response.completed"])


# --- recipe streaming ---------------------------------------------------------

async def test_recipe_streams_responses_sse(monkeypatch):
    monkeypatch.setattr(recipes, "get", lambda name: {"name": name})

    async def fake_oe(recipe, messages, registry, stream=False):
        yield {"type": "delta", "content": "A"}
        yield {"type": "delta", "content": "B"}
        yield {"type": "final", "response": {}}
    monkeypatch.setattr(router, "orchestrate_events", fake_oe)

    frames = await _collect(await router.responses_create(FakeRequest({
        "model": "woollama/streamer", "input": "go", "stream": True})))
    deltas = [f["data"]["delta"] for f in frames
              if f["event"] == "response.output_text.delta"]
    assert deltas == ["A", "B"]
    completed = next(f for f in frames if f["event"] == "response.completed")
    assert completed["data"]["response"]["output"][0]["content"][0]["text"] == "AB"


# --- guards -------------------------------------------------------------------

async def test_stateful_stream_is_400(monkeypatch):
    r = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "hi", "stream": True, "store": True}))
    assert r.status_code == 400
    assert "STATEFUL" in json.loads(r.body)["error"]["message"]


async def test_stream_unknown_model_is_http_error_not_stream(monkeypatch):
    # Setup error before the first delta → a JSON HTTP error, not a 200 stream.
    r = await router.responses_create(FakeRequest({
        "model": "bogus/x", "input": "hi", "stream": True}))
    assert r.status_code == 400
    assert "unknown model namespace" in json.loads(r.body)["error"]["message"]


async def test_stream_unknown_recipe_is_404_not_stream(monkeypatch):
    monkeypatch.setattr(recipes, "get", lambda name: None)
    r = await router.responses_create(FakeRequest({
        "model": "woollama/nope", "input": "hi", "stream": True}))
    assert r.status_code == 404
