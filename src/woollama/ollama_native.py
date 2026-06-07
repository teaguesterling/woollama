"""Native Ollama `/api/chat` translation (issue #1).

Ollama's OpenAI-compatible `/v1/chat/completions` endpoint **ignores `num_ctx`**
— a request with `options.num_ctx=N` still loads the model at the default
context and silently truncates long inputs. Ollama only honors `num_ctx` on its
NATIVE `POST /api/chat` via `options.num_ctx`.

So when a passthrough `ollama/<model>` request asks for a context size, woollama
routes it to `/api/chat` instead of `/v1/chat/completions`, translating the
request TO ollama's native shape and the response BACK to the OpenAI
chat-completions shape (so the client — an OpenAI SDK — parses it unchanged). The
native wire shapes encoded here were captured live from ollama (localhost:11434),
not recalled:

  non-stream: {"model","created_at","message":{"role","content"[,"thinking"]},
               "done":true,"done_reason","prompt_eval_count","eval_count",...}
  stream:     one JSON object per line (NDJSON); content arrives as
              `message.content` deltas; the final frame is `done:true` with
              `done_reason` + the eval counts.

`created_at` is an RFC3339 STRING, so OpenAI `created` uses our own
`int(time.time())`.

Scope (this slice): the chat/content path. A request that ALSO carries `tools`
stays on the `/v1` path (tool-calling works there; `num_ctx` is not honored) —
we don't half-translate tool_calls. Ollama's `thinking` (reasoning) field is not
surfaced (OpenAI chat-completions has no standard slot for it).
"""
from __future__ import annotations

import json
import time
import uuid

# OpenAI top-level sampling params folded into ollama's native `options` block
# (num_ctx etc. already live under `options`).
_OPTION_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "seed": "seed",
    "stop": "stop",
}


def wants_native(body: dict) -> bool:
    """True if this ollama request needs the native endpoint: a context size is
    requested (`options.num_ctx`) AND it carries no `tools` (those stay on `/v1`,
    where tool-calling is handled — at the cost of num_ctx not being honored)."""
    opts = body.get("options")
    has_ctx = isinstance(opts, dict) and opts.get("num_ctx") is not None
    return bool(has_ctx) and not body.get("tools")


def native_chat_url(base_url: str) -> str:
    """Derive the native `/api/chat` URL from ollama's OpenAI-compat `base_url`
    (which is `<root>/v1`)."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return f"{root}/api/chat"


def to_native_request(body: dict) -> dict:
    """OpenAI chat-completions body → ollama `/api/chat` body. `body['model']` is
    already the bare model name. Folds top-level OpenAI sampling params into
    `options` (without clobbering ones the caller already set there) and passes
    `messages` through unchanged."""
    options = dict(body.get("options") or {})
    for oai, native in _OPTION_MAP.items():
        if oai in body and native not in options:
            options[native] = body[oai]
    for cap in ("max_completion_tokens", "max_tokens"):     # → ollama's output cap
        if body.get(cap) is not None and "num_predict" not in options:
            options["num_predict"] = body[cap]
            break
    req = {
        "model": body["model"],
        "messages": body.get("messages", []),
        "stream": bool(body.get("stream")),
        "options": options,
    }
    if body.get("format") is not None:        # JSON-mode passthrough
        req["format"] = body["format"]
    return req


def _chatcmpl_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def _finish(done_reason: str | None) -> str:
    return "length" if done_reason == "length" else "stop"


def from_native_response(native: dict, model: str) -> dict:
    """ollama `/api/chat` (stream:false) response → OpenAI `chat.completion`."""
    msg = native.get("message") or {}
    prompt = native.get("prompt_eval_count") or 0
    completion = native.get("eval_count") or 0
    return {
        "id": _chatcmpl_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": msg.get("content") or ""},
            "finish_reason": _finish(native.get("done_reason")),
        }],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


def sse_translator(model: str):
    """Return a `translate(line) -> list[bytes]` that turns ollama NDJSON frames
    into OpenAI `chat.completion.chunk` SSE byte strings. The first content chunk
    also carries `role: assistant` (matching OpenAI's stream); the `done:true`
    frame emits a terminal finish chunk followed by `data: [DONE]`."""
    cid = _chatcmpl_id()
    created = int(time.time())
    state = {"role_sent": False}

    def _chunk(delta: dict, finish: str | None) -> bytes:
        payload = {
            "id": cid, "object": "chat.completion.chunk",
            "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return b"data: " + json.dumps(payload).encode() + b"\n\n"

    def translate(line: str) -> list[bytes]:
        line = line.strip()
        if not line:
            return []
        frame = json.loads(line)
        if frame.get("done"):
            return [_chunk({}, _finish(frame.get("done_reason"))), b"data: [DONE]\n\n"]
        content = (frame.get("message") or {}).get("content") or ""
        if not content:
            return []
        if not state["role_sent"]:
            state["role_sent"] = True
            return [_chunk({"role": "assistant", "content": content}, None)]
        return [_chunk({"content": content}, None)]

    return translate
