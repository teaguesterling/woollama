"""The router — woollama's OpenAI-compatible HTTP surface.

What it does:
  * `GET  /v1/models`         — enumerate Ollama models (prefixed) + recipes
  * `POST /v1/chat/completions`
      - model = "ollama/X"     → pass-through to local Ollama
      - model = "woollama/X"   → resolve recipe; orchestrate chat-loop with
                                  MCP tools; return final answer transparently

Per-request MCP subprocess spawn (correct but not optimal — long-lived
connection pooling is a follow-on). Non-streaming on both sides in v0.1
(streaming is queued for v0.2).
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from . import recipes


log = logging.getLogger("woollama.router")


# v0.1: hardcoded backend addresses. Move to config in v0.2.
OLLAMA_URL = os.environ.get("WOOLLAMA_OLLAMA_URL", "http://localhost:11434")

# v0.1: the bundled hello MCP server is the only tool source. v0.2 adds
# multi-server discovery via mcp.json.
HELLO_SERVER_PATH = str(
    Path(__file__).resolve().parent.parent.parent
    / "examples" / "mcp-hello" / "server.py"
)


@asynccontextmanager
async def _mcp_session():
    """Spawn the bundled hello MCP server, initialize a client, yield it,
    clean up. Per-request to sidestep FastAPI's lifespan task-scope split
    that breaks anyio cancel scopes (see docs/architecture.md, "What v0.1
    does not include")."""
    params = StdioServerParameters(command="python", args=[HELLO_SERVER_PATH])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as sess:
            await sess.initialize()
            yield sess


def _mcp_tools_for_openai(tools_result, allow_list: list[str]) -> list[dict]:
    """Translate MCP ToolSpec → OpenAI function-calling format.
    MCP's `inputSchema` is already JSON-Schema; the OpenAI shape is one
    level of nesting deeper."""
    out = []
    for t in tools_result.tools:
        if t.name not in allow_list:
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema or {"type": "object", "properties": {}},
            },
        })
    return out


app = FastAPI(title="woollama", version="0.1.0")


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    """OpenAI-compatible model list: Ollama models (prefixed with `ollama/`)
    + woollama recipes (prefixed with `woollama/`)."""
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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    """The dispatch verb. Parses `model` and routes:
      - `ollama/X`    → pass-through
      - `woollama/X`  → recipe orchestration
    """
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


async def _orchestrate_recipe(recipe: recipes.Recipe, body: dict) -> JSONResponse:
    """The chat-loop. Build messages from system + user turns; call the
    recipe's inferencer with the recipe's tool allow-list; dispatch any
    tool_calls to the MCP session; loop until no more tool_calls or
    max_turns hit; return the final assistant message in OpenAI format.
    """
    user_msgs = body.get("messages", [])
    messages = [{"role": "system", "content": recipe["system"]}] + list(user_msgs)

    inferencer = recipe["inferencer"]
    if not inferencer.startswith("ollama/"):
        return _error(
            f"v0.1 supports ollama/ inferencers only (got '{inferencer}')",
            "not_implemented", 501,
        )
    inferencer_model = inferencer[len("ollama/"):]

    async with _mcp_session() as sess:
        tools_list = await sess.list_tools()
        tools = _mcp_tools_for_openai(tools_list, recipe["tools"])
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
                return JSONResponse(resp, status_code=502)

            msg = resp["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            content = msg.get("content") or ""
            log.info("turn %d: content[%d] tool_calls=%d",
                     turn, len(content), len(calls))

            if not calls:
                # Final answer — return as-is in OpenAI shape.
                return JSONResponse(resp)

            # Tool dispatch: echo the assistant decision, then append each
            # tool result. Loop continues.
            messages.append({"role": "assistant", "content": content,
                             "tool_calls": calls})
            for call in calls:
                fn = call.get("function") or {}
                name = fn.get("name", "")
                raw_args = fn.get("arguments") or "{}"
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                log.info("  → %s(%s)", name, json.dumps(args)[:120])
                try:
                    r2 = await sess.call_tool(name, args)
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
                    "tool_call_id": call.get("id", f"call_{turn}_{name}"),
                })

    return _error("max turns (8) exceeded", "server_error", 500)


def _error(message: str, kind: str, status: int) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": kind}},
        status_code=status,
    )
