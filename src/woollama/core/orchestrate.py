"""The recipe chat-loop — server-free, the SINGLE source of truth for the
inferencer↔tool loop (do NOT reimplement it elsewhere).

`orchestrate_events` prepends the recipe's system prompt and runs the generic
loop: offer the recipe's allow-listed tools to the model, dispatch the ones it
calls (through a `ToolProvider`), feed results back, repeat (≤8 turns). It handles
ONLY a directly-addressed inferencer (`<provider>/<model>`); woollama's
claude-code executor branch lives in the server, which dispatches to this for
everything else.

Tools cross the seam losslessly: `tools.tools_for()` returns `ToolSpec`s (the
model sees only `.schema`), `tools.dispatch()` returns a `ToolResult` (mirrors the
MCP result, `is_error` included), and `render_tool_result` narrows it to the tool
message content for the target model.
"""
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from . import inferencers, recipes
from .inference import InferenceError
from .tooling import DEFAULT_CAPS, ToolProvider, render_tool_result

log = logging.getLogger("woollama.core.orchestrate")

_HTTP_TIMEOUT = 180.0


async def orchestrate(recipe: recipes.Recipe, user_msgs: list[dict], *,
                      tools: ToolProvider) -> dict:
    """Run a recipe end-to-end and return the final OpenAI-shaped response dict.
    A thin drainer over `orchestrate_events` (keeps the terminal `final`)."""
    final: dict | None = None
    async for ev in orchestrate_events(recipe, user_msgs, tools=tools, stream=False):
        if ev["type"] == "final":
            final = ev["response"]
    assert final is not None, "orchestrate_events always yields a final or raises"
    return final


async def orchestrate_events(recipe: recipes.Recipe, user_msgs: list[dict], *,
                             tools: ToolProvider,
                             stream: bool = False) -> AsyncIterator[dict]:
    """The recipe loop as an async generator. Yields `delta` (content, stream
    only), `tool_call` / `tool_result` (progress, both modes), and one terminal
    `final` (the OpenAI response dict). Raises `InferenceError` on unsupported
    inferencer / inferencer error / max turns. (See module docstring.)"""
    inferencer = recipe["inferencer"]
    provider = inferencer.split("/", 1)[0]

    inf = inferencers.get(provider)
    if inf is None:
        raise InferenceError(
            f"unsupported inferencer '{inferencer}' (supported providers: "
            f"{', '.join(inferencers.names())}, claude-code)", "not_implemented", 501)
    try:
        headers = inf.headers()           # fail fast on a missing API key
    except inferencers.InferencerError as e:
        raise InferenceError(str(e), "invalid_request_error", 400) from e

    messages = [{"role": "system", "content": recipe["system"]}] + list(user_msgs)
    inferencer_model = inferencer.split("/", 1)[1]

    specs = tools.tools_for(recipe["tools"])
    schemas = [s.schema for s in specs]
    # The recipe's allow-list is a BOUNDARY, not a hint: only these tools are
    # offered AND only these may be dispatched. A tool_call for anything else is
    # refused below. The schema's function name is the namespaced allow-list name,
    # so membership matches the emitted name directly.
    allowed = set(recipe["tools"])
    log.info("orchestrating: tools=%s inferencer=%s stream=%s",
             [s.name for s in specs], inferencer, stream)

    for turn in range(1, 9):
        req = {
            "model": inferencer_model,
            "messages": messages,
            "tools": schemas,
            "stream": bool(stream),
            **inf.extra_body,             # provider-specific (Ollama options / Anthropic max_tokens)
        }
        if stream:
            acc: dict = {}
            async for piece in _stream_turn(inf, req, headers, acc):
                yield {"type": "delta", "content": piece}
            content, calls, resp = acc["content"], acc["calls"], acc["response"]
        else:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as c:
                r = await c.post(inf.chat_url(), json=req, headers=headers)
                resp = r.json()
            if "choices" not in resp:
                log.warning("inferencer error: %s", resp)
                raise InferenceError("inferencer error", "server_error", 502,
                                     payload=resp)
            msg = resp["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            content = msg.get("content") or ""

        log.info("turn %d: content[%d] tool_calls=%d", turn, len(content), len(calls))

        if not calls:
            yield {"type": "final", "response": resp}
            return

        messages.append({"role": "assistant", "content": content,
                         "tool_calls": calls})
        for call in calls:
            fn = call.get("function") or {}
            namespaced = fn.get("name", "")
            raw_args = fn.get("arguments") or "{}"
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            log.info("  → %s(%s)", namespaced, json.dumps(args)[:120])
            yield {"type": "tool_call", "turn": turn, "name": namespaced, "args": args}
            if namespaced not in allowed:
                # Refuse, but feed the refusal back as the tool result (every
                # tool_call needs a matching tool message) so the loop can recover.
                log.warning("recipe denied out-of-list tool '%s' (allow-list: %s)",
                            namespaced, sorted(allowed))
                ok = False
                result = (f"ERROR: tool '{namespaced}' is not permitted by this "
                          f"recipe (allowed: {sorted(allowed)})")
            else:
                try:
                    tr = await tools.dispatch(namespaced, args)
                    result = render_tool_result(tr, caps=DEFAULT_CAPS)
                    ok = not tr.is_error
                except Exception as e:
                    ok = False
                    result = f"ERROR: {type(e).__name__}: {e}"
            preview = (result[:80] + "…") if len(result) > 80 else result
            log.info("  ← %s", preview)
            yield {"type": "tool_result", "turn": turn, "name": namespaced, "ok": ok}
            messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": call.get("id", f"call_{turn}_{namespaced}"),
            })

    raise InferenceError("max turns (8) exceeded", "server_error", 500)


async def _stream_turn(inf, req: dict, headers: dict, acc: dict):
    """Stream ONE inferencer turn over SSE. Yields assistant content deltas; on
    completion fills `acc` with `{content, calls, response}`. Owns: (1) the
    upstream per-turn `finish_reason`/`[DONE]` are CONSUMED here, never surfaced —
    the loop is closed by exactly one synthesized terminator; (2) tool_call deltas
    arrive FRAGMENTED (id in one chunk, arguments piecemeal), reassembled by
    `index`."""
    content_parts: list[str] = []
    calls_by_index: dict[int, dict] = {}
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as c:
        async with c.stream("POST", inf.chat_url(), json=req, headers=headers) as r:
            if r.status_code >= 400:
                raw = await r.aread()
                try:
                    payload = json.loads(raw)
                except (ValueError, TypeError):
                    payload = {"error": {"message": raw.decode("utf-8", "replace")
                                         or "inferencer error", "type": "server_error"}}
                raise InferenceError("inferencer error", "server_error", 502,
                                     payload=payload)
            async for line in r.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    content_parts.append(piece)
                    yield piece
                for tc in delta.get("tool_calls") or []:
                    slot = calls_by_index.setdefault(
                        tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]

    content = "".join(content_parts)
    calls = [
        {"id": s["id"] or f"call_{idx}", "type": "function",
         "function": {"name": s["name"], "arguments": s["arguments"]}}
        for idx, s in sorted(calls_by_index.items())
    ]
    acc["content"] = content
    acc["calls"] = calls
    acc["response"] = {
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content,
                        "tool_calls": calls or None},
            "finish_reason": "tool_calls" if calls else "stop",
        }],
    }
