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
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import claude_code, config, inferencers, recipes
from .manager import Registry, ServerManager
from .mcp_server import build_server, register_reexported_tools

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


# Module-level registry; SHARED by both surfaces — the OpenAI orchestration
# path (below) and the mounted MCP server. Populated + started once by the
# lifespan, so there is a single connection layer to the downstream MCP servers.
registry = Registry()

# The MCP server, mounted onto this same FastAPI app so woollama exposes BOTH
# surfaces on one port: /v1/* (OpenAI-compatible) and /mcp (MCP over Streamable
# HTTP). Built with manage_registry=False — the FastAPI lifespan owns the shared
# registry, so the MCP server must not start/stop it (double-start). The MCP app
# carries its own (session-manager) lifespan, composed into ours below.
_mcp = build_server(registry, manage_registry=False)
_mcp_app = _mcp.http_app(path="/", transport="http")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Server bundle from mcp.json (user config or bundled defaults).
    for name, cfg in config.load_mcp_servers().items():
        registry.add(ServerManager(name, cfg["command"], cfg["args"]))
    await registry.start_all()
    log.info("registry ready: %s", registry.all_tool_names())
    # Downstream tools are known now → re-export them onto the MCP surface
    # (same dynamic registration the stdio path does in build_server's lifespan).
    register_reexported_tools(_mcp, registry)
    # Run the mounted MCP app's own lifespan (Streamable HTTP session manager)
    # for the duration of the server, then tear the registry down.
    async with _mcp_app.lifespan(app):
        try:
            yield
        finally:
            await registry.stop_all()


app = FastAPI(title="woollama", version="0.1.0", lifespan=lifespan)
app.mount("/mcp", _mcp_app)


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    data: list[dict] = []
    async with httpx.AsyncClient(timeout=10) as c:
        try:
            r = await c.get(f"{inferencers.get('ollama').base_url}/models")
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
async def chat_completions(request: Request) -> Response:
    body = await request.json()
    model = body.get("model", "")

    if model.startswith("woollama/"):
        name = model[len("woollama/"):]
        recipe = recipes.get(name)
        if recipe is None:
            return _error(f"unknown recipe '{name}'", "not_found", 404)
        return await _orchestrate_recipe(recipe, body)

    # `<provider>/<model>` against a known OpenAI-compat inferencer → pass-through.
    provider = model.split("/", 1)[0]
    if inferencers.get(provider) is not None:
        return await _passthrough(body)

    return _error(
        f"unknown model namespace: '{model}'. Use 'woollama/<recipe>' or "
        f"'<provider>/<model>' for a known inferencer ({', '.join(inferencers.names())}).",
        "invalid_request_error", 400,
    )


async def _passthrough(body: dict) -> Response:
    """Forward `<provider>/<model>` straight to that inferencer's OpenAI-compat
    endpoint (no orchestration). The client owns the body; we only swap the
    namespaced model for the bare name and add auth. `stream:true` is honoured —
    we relay the upstream SSE verbatim (slice: streaming-1)."""
    body = dict(body)
    provider, _, bare = body["model"].partition("/")
    inf = inferencers.get(provider)        # caller verified it's known
    body["model"] = bare
    try:
        headers = inf.headers()
    except inferencers.InferencerError as e:
        return _error(str(e), "invalid_request_error", 400)
    if body.get("stream"):
        return await _passthrough_stream(inf, body, headers)
    body["stream"] = False
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(inf.chat_url(), json=body, headers=headers)
        return JSONResponse(r.json(), status_code=r.status_code)


async def _passthrough_stream(inf: inferencers.Inferencer, body: dict,
                              headers: dict) -> Response:
    """Relay the upstream OpenAI SSE stream byte-for-byte (preserves chunk
    framing and the `data: [DONE]` sentinel for free).

    We open the upstream connection and check its status BEFORE returning a
    StreamingResponse: once a 200 stream begins its status can't be changed, so
    an upstream 4xx/5xx must surface as a JSON error (matching the non-streaming
    path), not an empty 200. That forces manual context management — on success
    the generator owns closing the stream and client; on error we close here."""
    client = httpx.AsyncClient(timeout=180)
    cm = client.stream("POST", inf.chat_url(), json=body, headers=headers)
    r = await cm.__aenter__()
    if r.status_code >= 400:
        raw = await r.aread()
        await cm.__aexit__(None, None, None)
        await client.aclose()
        try:
            return JSONResponse(json.loads(raw), status_code=r.status_code)
        except (ValueError, TypeError):
            return _error(raw.decode("utf-8", "replace") or "upstream error",
                          "server_error", r.status_code)

    async def relay():
        try:
            async for chunk in r.aiter_bytes():
                yield chunk
        finally:
            await cm.__aexit__(None, None, None)
            await client.aclose()

    return StreamingResponse(relay(), status_code=r.status_code,
                             media_type="text/event-stream")


async def orchestrate(recipe: recipes.Recipe, user_msgs: list[dict],
                      reg: Registry) -> dict:
    """Transport-agnostic recipe chat-loop. Prepends the recipe's system
    prompt, runs the inferencer ↔ tool-dispatch loop (≤8 turns), and returns
    the final OpenAI-shaped response dict (the one with `choices`).

    Both the HTTP `/v1/chat/completions` handler and the MCP `chat` tool call
    this — do NOT reimplement the loop. Tool dispatch routes through `reg`
    (the caller's `Registry`), so each transport owns its own registry
    lifecycle. Raises `OrchestrationError` for the unsupported-inferencer,
    inferencer-error, and max-turns-exceeded cases.

    Inferencer dispatch by `<provider>/`:
      * `claude-code/<model>` → delegate a TOOL-LESS completion to the local
        Claude Code CLI (keyless, uses the user's Claude auth). Recipes with a
        non-empty tools list are rejected — tool delegation is a later slice.
      * `ollama/<model>`      → the woollama-owned inferencer ↔ tool loop below.
      * anything else         → unsupported (501)."""
    inferencer = recipe["inferencer"]
    provider = inferencer.split("/", 1)[0]

    if provider == "claude-code":
        if recipe["tools"]:
            raise OrchestrationError(
                "tool delegation to claude-code is not yet supported "
                "(route tool-using recipes to an ollama/ inferencer, or use a "
                "tool-less claude-code recipe)", "not_implemented", 501)
        model = inferencer.split("/", 1)[1] if "/" in inferencer else ""
        try:
            return await claude_code.run_completion(
                recipe["system"], user_msgs, model)
        except claude_code.ClaudeCodeError as e:
            raise OrchestrationError(
                f"claude-code backend: {e}", "server_error", 502) from e

    inf = inferencers.get(provider)
    if inf is None:
        raise OrchestrationError(
            f"unsupported inferencer '{inferencer}' (supported providers: "
            f"{', '.join(inferencers.names())}, claude-code)", "not_implemented", 501)
    try:
        headers = inf.headers()           # fail fast on a missing API key
    except inferencers.InferencerError as e:
        raise OrchestrationError(str(e), "invalid_request_error", 400) from e

    messages = [{"role": "system", "content": recipe["system"]}] + list(user_msgs)
    inferencer_model = inferencer.split("/", 1)[1]

    tools = reg.openai_tools_for(recipe["tools"])
    # The recipe's allow-list is a BOUNDARY, not a hint: only these tools are
    # offered to the model AND only these may be dispatched. If the model emits
    # a tool_call for anything else (hallucination, or a name it shouldn't know),
    # we refuse it below rather than reaching across to a provider the recipe was
    # never granted. `openai_tools_for` preserves the namespaced name as the
    # function name, so membership matches the emitted name directly.
    allowed = set(recipe["tools"])
    log.info("orchestrating: tools=%s inferencer=%s",
             [t["function"]["name"] for t in tools], inferencer)

    for turn in range(1, 9):
        req = {
            "model": inferencer_model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            **inf.extra_body,             # provider-specific (Ollama options / Anthropic max_tokens)
        }
        async with httpx.AsyncClient(timeout=180) as c:
            r = await c.post(inf.chat_url(), json=req, headers=headers)
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
            if namespaced not in allowed:
                # Refuse and feed the refusal back as the tool result (every
                # tool_call needs a matching tool message, including denied
                # ones) so the loop continues and the model can recover.
                log.warning("recipe denied out-of-list tool '%s' (allow-list: %s)",
                            namespaced, sorted(allowed))
                result = (f"ERROR: tool '{namespaced}' is not permitted by this "
                          f"recipe (allowed: {sorted(allowed)})")
            else:
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
