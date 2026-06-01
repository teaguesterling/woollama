"""The router — woollama's OpenAI-compatible HTTP surface.

Endpoints:
  * `GET  /v1/models`         — list Ollama models (prefixed) + recipes
  * `POST /v1/chat/completions`
      - model = "ollama/X"     → pass-through to local Ollama
      - model = "woollama/X"   → recipe orchestration with multi-MCP-server
                                  tool dispatch through the Registry

Connections to MCP servers are long-lived (one task per server, queue-
mediated) so we don't pay subprocess-spawn cost per request and we sidestep
FastAPI lifespan's split startup/shutdown task scope.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import config, recipes
from .manager import Registry, ServerManager


log = logging.getLogger("woollama.router")


class OrchestrationError(Exception):
    """Raised by the transport-agnostic orchestration loop. Each transport
    maps it to its own error surface (HTTP status / MCP error).

    `payload` carries the raw upstream response when the inferencer itself
    errored, so the HTTP surface can pass it through verbatim."""

    def __init__(self, message: str, kind: str, status: int,
                 payload: dict | None = None):
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.status = status
        self.payload = payload

OLLAMA_URL = os.environ.get("WOOLLAMA_OLLAMA_URL", "http://localhost:11434")


# Module-level registry; populated by lifespan.
registry = Registry()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Server bundle from mcp.json (user config or bundled defaults).
    for name, cfg in config.load_mcp_servers().items():
        registry.add(ServerManager(name, cfg["command"], cfg["args"]))
    await registry.start_all()
    log.info("registry ready: %s", registry.all_tool_names())
    try:
        yield
    finally:
        await registry.stop_all()


app = FastAPI(title="woollama", version="0.1.0", lifespan=lifespan)


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    data: list[dict] = []
    async with httpx.AsyncClient(timeout=10) as c:
        try:
            r = await c.get(f"{OLLAMA_URL}/v1/models")
            for m in r.json().get("data", []):
                data.append({"id": f"ollama/{m['id']}", "object": "model",
                             "owned_by": "ollama"})
        except Exception as e:
            log.warning("ollama /v1/models failed: %s", e)
    for name in recipes.names():
        data.append({"id": f"woollama/{name}", "object": "model",
                     "owned_by": "woollama"})
    return JSONResponse({"object": "list", "data": data})


@app.get("/v1/tools")
async def list_tools() -> JSONResponse:
    """Non-OpenAI introspection: what tools across all servers we know about.
    Useful for debugging multi-server discovery."""
    return JSONResponse({"tools": registry.all_tool_names()})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    body = await request.json()
    model = body.get("model", "")

    if model.startswith("ollama/"):
        return await _passthrough_ollama(body)

    if model.startswith("woollama/"):
        name = model[len("woollama/"):]
        recipe = recipes.get(name)
        if recipe is None:
            return _error(f"unknown recipe '{name}'", "not_found", 404)
        return await _orchestrate_recipe(recipe, body)

    return _error(
        f"unknown model namespace: '{model}'. Use 'ollama/<name>' or "
        f"'woollama/<recipe>'.",
        "invalid_request_error", 400,
    )


async def _passthrough_ollama(body: dict) -> JSONResponse:
    body = dict(body)
    body["model"] = body["model"][len("ollama/"):]
    body["stream"] = False  # v0.1: non-streaming
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(f"{OLLAMA_URL}/v1/chat/completions", json=body)
        return JSONResponse(r.json(), status_code=r.status_code)


async def orchestrate(recipe: recipes.Recipe, user_msgs: list[dict],
                      reg: Registry) -> dict:
    """Transport-agnostic recipe chat-loop. Prepends the recipe's system
    prompt, runs the inferencer ↔ tool-dispatch loop (≤8 turns), and returns
    the final OpenAI-shaped response dict (the one with `choices`).

    Both the HTTP `/v1/chat/completions` handler and the MCP `chat` tool call
    this — do NOT reimplement the loop. Tool dispatch routes through `reg`
    (the caller's `Registry`), so each transport owns its own registry
    lifecycle. Raises `OrchestrationError` for the unsupported-inferencer,
    inferencer-error, and max-turns-exceeded cases."""
    messages = [{"role": "system", "content": recipe["system"]}] + list(user_msgs)

    inferencer = recipe["inferencer"]
    if not inferencer.startswith("ollama/"):
        raise OrchestrationError(
            f"v0.1 supports ollama/ inferencers only (got '{inferencer}')",
            "not_implemented", 501,
        )
    inferencer_model = inferencer[len("ollama/"):]

    tools = reg.openai_tools_for(recipe["tools"])
    log.info("orchestrating: tools=%s inferencer=%s",
             [t["function"]["name"] for t in tools], inferencer_model)

    for turn in range(1, 9):
        req = {
            "model": inferencer_model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "options": {"temperature": 0},
        }
        async with httpx.AsyncClient(timeout=180) as c:
            r = await c.post(f"{OLLAMA_URL}/v1/chat/completions", json=req)
            resp = r.json()
        if "choices" not in resp:
            log.warning("inferencer error: %s", resp)
            raise OrchestrationError("inferencer error", "server_error", 502,
                                     payload=resp)

        msg = resp["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""
        log.info("turn %d: content[%d] tool_calls=%d",
                 turn, len(content), len(calls))

        if not calls:
            return resp

        messages.append({"role": "assistant", "content": content,
                         "tool_calls": calls})
        for call in calls:
            fn = call.get("function") or {}
            namespaced = fn.get("name", "")
            raw_args = fn.get("arguments") or "{}"
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            log.info("  → %s(%s)", namespaced, json.dumps(args)[:120])
            try:
                r2 = await reg.dispatch(namespaced, args)
                parts = [c.text for c in r2.content if hasattr(c, "text")]
                result = "\n".join(parts) if parts else json.dumps(
                    [c.model_dump() for c in r2.content], default=str)
            except Exception as e:
                result = f"ERROR: {type(e).__name__}: {e}"
            preview = (result[:80] + "…") if len(result) > 80 else result
            log.info("  ← %s", preview)
            messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": call.get("id", f"call_{turn}_{namespaced}"),
            })

    raise OrchestrationError("max turns (8) exceeded", "server_error", 500)


async def _orchestrate_recipe(recipe: recipes.Recipe, body: dict) -> JSONResponse:
    """HTTP adapter: run the shared loop, map results/errors onto JSON."""
    try:
        resp = await orchestrate(recipe, body.get("messages", []), registry)
    except OrchestrationError as e:
        if e.payload is not None:
            return JSONResponse(e.payload, status_code=e.status)
        return _error(e.message, e.kind, e.status)
    return JSONResponse(resp)


def _error(message: str, kind: str, status: int) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": kind}},
        status_code=status,
    )
