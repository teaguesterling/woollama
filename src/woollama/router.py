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
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import (
    __version__,
    claude_code,
    config,
    conversations,
    inferencers,
    managed_agents,
    ollama_native,
    recipes,
    responses,
)
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

# In-memory conversation handle table for the stateful /v1/responses surface
# (conv-1b). woollama routes handles → backends; it does not store transcripts.
conversation_store = conversations.ConversationStore()

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


app = FastAPI(title="woollama", version=__version__, lifespan=lifespan)
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
        if body.get("stream"):
            return await _orchestrate_recipe_stream(recipe, body)
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


@app.post("/v1/responses")
async def responses_create(request: Request) -> Response:
    """Stateful surface — OpenAI *Responses* shape (docs/conversations-api-design).

    conv-1a covers the STATELESS subset (`store:false`): a superset of
    /v1/chat/completions that speaks the Responses wire format, routed by `model`
    identically. Stateful conversations (handle routing + the claude-resume
    backend) arrive in conv-1b; the server-owned `stored` backend is a later
    slice. The principle holds throughout: woollama routes conversation handles,
    backends own the bytes — it never becomes a conversation database."""
    body = await request.json()
    model = body.get("model", "")

    if body.get("stream"):
        return _error("streaming is not yet supported on /v1/responses "
                      "(use /v1/chat/completions for SSE; a Responses streaming "
                      "shape is a later slice)", "invalid_request_error", 400)

    try:
        messages = responses.parse_input(body.get("input", ""))
    except ValueError as e:
        return _error(str(e), "invalid_request_error", 400)

    # Stateful opt-in (conv-1b): a backing conversation is involved.
    if body.get("store") or body.get("conversation") or body.get("previous_response_id"):
        return await _responses_stateful(body, model, messages)

    # Stateless (conv-1a): a Responses-shaped /v1/chat/completions.
    try:
        text = await complete_stateless(model, messages, options=body.get("options"))
    except OrchestrationError as e:
        if e.payload is not None:
            return JSONResponse(e.payload, status_code=e.status)
        return _error(e.message, e.kind, e.status)
    return JSONResponse(responses.build_response(
        responses.new_id("resp"), model, text))


async def _responses_stateful(body: dict, model: str,
                              messages: list[dict]) -> Response:
    """Route a stateful turn: resolve/attach the conversation handle, run the
    turn on its backend under a per-conversation write lock, and return the
    Responses object carrying the conversation id.

    Attach precedence: an explicit `conversation` wins; else `previous_response_id`
    resolves to its conversation (chaining off a prior turn); else a NEW
    conversation is created, its backend chosen by `model`."""
    conv_id = body.get("conversation")
    prev = body.get("previous_response_id")

    if conv_id:
        conv = conversation_store.get(conv_id)
        if conv is None:
            return _error(f"unknown conversation '{conv_id}'", "not_found", 404)
        if prev and conversation_store.by_response(prev) is not conv:
            return _error(
                f"previous_response_id '{prev}' does not belong to conversation "
                f"'{conv_id}'", "invalid_request_error", 400)
    elif prev:
        conv = conversation_store.by_response(prev)
        if conv is None:
            return _error(f"unknown previous_response_id '{prev}'", "not_found", 404)
    else:
        backend = conversations.backend_for_model(model)
        if backend is None:
            return _error(
                f"no stateful backend for model '{model}': only claude-code "
                "(claude-resume) and claude-agent (managed-agents) models have a "
                "state-owning backend in this build. woollama does not store "
                "conversations in its own system — use `store:false` (the caller "
                "owns history) for ollama/recipe/cloud models.",
                "not_implemented", 501)
        conv = conversation_store.create(backend, model)

    backend_impl = conversations.BACKENDS[conv.backend]
    async with conv.lock:                       # one writer per conversation
        conv.status = "busy"
        try:
            text = await backend_impl.send_turn(conv, messages, options=body.get("options"))
        except (claude_code.ClaudeCodeError, managed_agents.ManagedAgentsError) as e:
            conv.status = "idle"
            return _error(f"{conv.backend} backend: {e}", "server_error", 502)
        except OrchestrationError as e:
            # A store-backed backend runs inference via complete_stateless, which
            # raises OrchestrationError (bad model, inferencer down, recipe error).
            # Surface it cleanly instead of letting it 500.
            conv.status = "idle"
            if e.payload is not None:
                return JSONResponse(e.payload, status_code=e.status)
            return _error(e.message, e.kind, e.status)
        conv.status = "idle"

    resp_id = responses.new_id("resp")
    conversation_store.record_response(conv, resp_id)
    return JSONResponse(responses.build_response(
        resp_id, conv.model, text, conversation=conv.id))


# --- /v1/conversations — discovery + attach + teardown (conv-2) --------------

@app.post("/v1/conversations")
async def conversations_create(request: Request) -> Response:
    """Create a conversation handle. The backend is taken explicitly or derived
    from `model`; the backing session itself is created lazily on the first turn
    (woollama routes the handle, the backend owns the bytes)."""
    body = await request.json()
    model = body.get("model", "")
    if not model:
        return _error("`model` is required to create a conversation "
                      "(e.g. 'claude-code/haiku')", "invalid_request_error", 400)
    backend = body.get("backend") or conversations.backend_for_model(model)
    if backend is None or backend not in conversations.BACKENDS:
        return _error(
            f"no state-owning backend for model '{model}': only claude-code "
            "(claude-resume) and claude-agent (managed-agents) models have one in "
            "this build. woollama does not store conversations in its own system — "
            "ollama/recipe/cloud models are stateless (use `store:false`).",
            "not_implemented", 501)
    conv = conversation_store.create(backend, model,
                                     metadata=body.get("metadata") or {},
                                     title=body.get("title"))
    return JSONResponse(responses.conversation_object(conv), status_code=201)


@app.get("/v1/conversations")
async def conversations_list() -> JSONResponse:
    """List known conversation handles — the discovery surface cosmic-fabric
    binds to."""
    return JSONResponse({"object": "list",
                         "data": [responses.conversation_object(c)
                                  for c in conversation_store.list()]})


@app.get("/v1/conversations/{conv_id}")
async def conversations_get(conv_id: str) -> Response:
    conv = conversation_store.get(conv_id)
    if conv is None:
        return _error(f"unknown conversation '{conv_id}'", "not_found", 404)
    return JSONResponse(responses.conversation_object(conv))


@app.get("/v1/conversations/{conv_id}/items")
async def conversations_items(conv_id: str) -> Response:
    """The transcript. A backend that exposes its event log (managed-agents:
    Anthropic owns the bytes, woollama RETRIEVES them via `history`) serves items
    directly. For `claude-resume`, reading the transcript means parsing the
    backend's own session log — that's the session driver's job (a later slice),
    so it has no `history` and still 501s."""
    conv = conversation_store.get(conv_id)
    if conv is None:
        return _error(f"unknown conversation '{conv_id}'", "not_found", 404)
    backend_impl = conversations.BACKENDS[conv.backend]
    if not hasattr(backend_impl, "history"):
        return _error(
            f"conversation transcript items are not available for the "
            f"'{conv.backend}' backend yet — reading its transcript is the "
            "session driver's job (a later slice).",
            "not_implemented", 501)
    data = [responses.item_object(m) for m in await backend_impl.history(conv)]
    return JSONResponse({
        "object": "list",
        "data": data,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
        "has_more": False,
    })


@app.delete("/v1/conversations/{conv_id}")
async def conversations_delete(conv_id: str) -> Response:
    """End woollama's hold on the conversation: tear down the backend's local
    state (best-effort) and forget the handle."""
    conv = conversation_store.get(conv_id)
    if conv is None:
        return _error(f"unknown conversation '{conv_id}'", "not_found", 404)
    try:
        await conversations.BACKENDS[conv.backend].delete(conv)
    except Exception as e:               # teardown is best-effort; still forget it
        log.warning("backend delete for '%s' failed: %s", conv_id, e)
    conversation_store.remove(conv_id)
    return JSONResponse({"id": conv_id, "object": "conversation.deleted",
                         "deleted": True})


async def complete_stateless(model: str, messages: list[dict], *,
                             options: dict | None = None) -> str:
    """Run one stateless turn, return the assistant text. Routes by `model`
    exactly like /v1/chat/completions (woollama/<recipe> → orchestrate; a known
    inferencer → passthrough), raising OrchestrationError for the error cases so
    the caller maps them onto the response surface. Backs the stateless
    `/v1/responses` path (conv-1a) and the store-backed stateful path (conv-7).

    `options` carries ollama-native knobs (e.g. `num_ctx`); when present for the
    ollama provider the turn goes through the native /api/chat (which honors them)
    — the same routing as the chat-completions passthrough (#1), closing the
    #1↔#2 seam so a stateful ollama turn sizes its context too."""
    if model.startswith("woollama/"):
        name = model[len("woollama/"):]
        recipe = recipes.get(name)
        if recipe is None:
            raise OrchestrationError(f"unknown recipe '{name}'", "not_found", 404)
        resp = await orchestrate(recipe, messages, registry)
        return resp["choices"][0]["message"].get("content") or ""

    provider = model.split("/", 1)[0]
    inf = inferencers.get(provider)
    if inf is None:
        raise OrchestrationError(
            f"unknown model namespace: '{model}'. Use 'woollama/<recipe>' or "
            f"'<provider>/<model>' for a known inferencer "
            f"({', '.join(inferencers.names())}).", "invalid_request_error", 400)
    bare = model.split("/", 1)[1] if "/" in model else ""
    try:
        headers = inf.headers()
    except inferencers.InferencerError as e:
        raise OrchestrationError(str(e), "invalid_request_error", 400) from e

    # Ollama honors num_ctx only on its native /api/chat (#1) — route there when a
    # context size is requested, translating the native reply back to text.
    if provider == "ollama" and options and options.get("num_ctx") is not None:
        req = ollama_native.to_native_request(
            {"model": bare, "messages": messages, "options": options, "stream": False})
        async with httpx.AsyncClient(timeout=_OLLAMA_NATIVE_TIMEOUT) as c:
            r = await c.post(ollama_native.native_chat_url(inf.base_url),
                             json=req, headers=headers)
            data = r.json()
        if "message" not in data:
            raise OrchestrationError("inferencer error", "server_error", 502, payload=data)
        return (data.get("message") or {}).get("content") or ""

    body = {"model": bare, "messages": messages, "stream": False}
    if options:
        body["options"] = options
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(inf.chat_url(), json=body, headers=headers)
        data = r.json()
    if "choices" not in data:
        raise OrchestrationError("inferencer error", "server_error", 502, payload=data)
    return data["choices"][0]["message"].get("content") or ""


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
    # Ollama ignores `num_ctx` on its OpenAI-compat /v1 endpoint — honor it by
    # routing to the native /api/chat when a context size is requested (#1).
    if provider == "ollama" and ollama_native.wants_native(body):
        return await _passthrough_ollama_native(inf, body, headers)
    if body.get("stream"):
        return await _passthrough_stream(inf, body, headers)
    body["stream"] = False
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(inf.chat_url(), json=body, headers=headers)
        return JSONResponse(r.json(), status_code=r.status_code)


# Large num_ctx means a long prompt-eval before the first byte (and that's
# correlated with exactly the requests that take this path), so the native
# endpoint gets a generous read timeout, not the /v1 path's 180s.
_OLLAMA_NATIVE_TIMEOUT = httpx.Timeout(600.0, connect=10.0)


async def _passthrough_ollama_native(inf: inferencers.Inferencer, body: dict,
                                     headers: dict) -> Response:
    """Issue #1: route an ollama request that asks for a context size to the
    NATIVE /api/chat (which honors options.num_ctx), translating the request to
    native shape and the response back to the OpenAI chat-completions shape."""
    url = ollama_native.native_chat_url(inf.base_url)
    req = ollama_native.to_native_request(body)
    if body.get("stream"):
        return await _ollama_native_stream(url, req, headers, body["model"])
    async with httpx.AsyncClient(timeout=_OLLAMA_NATIVE_TIMEOUT) as c:
        r = await c.post(url, json=req, headers=headers)
        if r.status_code >= 400:
            try:
                return JSONResponse(r.json(), status_code=r.status_code)
            except (ValueError, TypeError):
                return _error(r.text or "ollama error", "server_error", r.status_code)
        return JSONResponse(ollama_native.from_native_response(r.json(), body["model"]))


async def _ollama_native_stream(url: str, req: dict, headers: dict,
                                model: str) -> Response:
    """Stream native NDJSON from /api/chat and translate each frame to OpenAI
    SSE. Mirrors `_passthrough_stream`'s status-check-before-streaming structure
    (an upstream 4xx must surface as JSON, not an empty 200)."""
    client = httpx.AsyncClient(timeout=_OLLAMA_NATIVE_TIMEOUT)
    cm = client.stream("POST", url, json=req, headers=headers)
    r = await cm.__aenter__()
    if r.status_code >= 400:
        raw = await r.aread()
        await cm.__aexit__(None, None, None)
        await client.aclose()
        try:
            return JSONResponse(json.loads(raw), status_code=r.status_code)
        except (ValueError, TypeError):
            return _error(raw.decode("utf-8", "replace") or "ollama error",
                          "server_error", r.status_code)
    translate = ollama_native.sse_translator(model)

    async def relay():
        try:
            async for line in r.aiter_lines():
                for chunk in translate(line):
                    yield chunk
        finally:
            await cm.__aexit__(None, None, None)
            await client.aclose()

    return StreamingResponse(relay(), media_type="text/event-stream")


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


def _delegate_mcp_servers(tools: list[str]) -> dict[str, dict]:
    """Build the per-recipe MCP config for delegation: the launch spec of EACH
    downstream server referenced by the recipe's ``<server>.<tool>`` tools, taken
    from the active mcp config. Only ``command``/``args`` are forwarded (a
    minimal, clean config for the child). Raises ``OrchestrationError`` (400) if a
    referenced server isn't configured — woollama never hands Claude a partial
    toolset."""
    available = config.load_mcp_servers()
    servers: dict[str, dict] = {}
    for t in tools:
        # A comma/whitespace in a tool name would inject extra entries into the
        # comma-joined --allowedTools (a same-server sibling grant). Reject it.
        if "," in t or any(ch.isspace() for ch in t):
            raise OrchestrationError(
                f"invalid tool name in recipe allow-list: {t!r} "
                "(commas/whitespace are not allowed)", "invalid_request_error", 400)
        server = t.split(".", 1)[0]
        if server not in available:
            raise OrchestrationError(
                f"recipe references tool '{t}' but its MCP server '{server}' is "
                f"not configured (known: {sorted(available)})",
                "invalid_request_error", 400)
        cfg = available[server]
        servers[server] = {"command": cfg["command"], "args": cfg.get("args", [])}
    return servers


async def orchestrate(recipe: recipes.Recipe, user_msgs: list[dict],
                      reg: Registry) -> dict:
    """Run a recipe end-to-end and return the final OpenAI-shaped response dict.

    A thin drainer over `orchestrate_events` (the single source of truth): it
    ignores the streamed content deltas and keeps the terminal `final` event.
    The contract is unchanged from before streaming-2 — the MCP `chat` tool and
    the HTTP non-streaming handler both call this and must keep working as-is.
    Raises `OrchestrationError` (unsupported inferencer / inferencer error /
    max turns) exactly as the underlying loop does."""
    final: dict | None = None
    async for ev in orchestrate_events(recipe, user_msgs, reg, stream=False):
        if ev["type"] == "final":
            final = ev["response"]
    assert final is not None, "orchestrate_events always yields a final or raises"
    return final


async def orchestrate_events(recipe: recipes.Recipe, user_msgs: list[dict],
                             reg: Registry, *, stream: bool = False):
    """The recipe chat-loop as an async generator — the SINGLE source of truth
    for orchestration (do NOT reimplement the loop elsewhere). Prepends the
    recipe's system prompt and runs the inferencer ↔ tool-dispatch loop (≤8
    turns). Yields:

      * `{"type": "delta", "content": str}` — assistant content to surface to
        the client. Emitted only when `stream=True`.
      * `{"type": "tool_call", "turn", "name", "args"}` — a tool is about to be
        dispatched. `{"type": "tool_result", "turn", "name", "ok"}` — it
        returned (`ok=False` for a denied/errored tool). Emitted in BOTH modes
        for progress surfacing (the MCP `chat` tool turns these into
        `ctx.info(...)`); the HTTP adapters ignore them.
      * `{"type": "final", "response": dict}` — the final OpenAI response dict.

    What is hidden in BOTH modes (the correctness invariant): the tool-call
    JSON and the tool results never appear in the surfaced stream, and the
    upstream per-turn `finish_reason`/`[DONE]` are consumed, never relayed —
    the transport synthesizes exactly one terminator.

    Deliberate divergence (a product choice, not a bug): when streaming, the
    content of *every* turn is surfaced as one continuous assistant message, so
    a tool-using recipe streams any pre-tool narration. Non-streaming returns
    only the final turn's response dict (intermediate content is dropped), so
    the same recipe can show more text when streamed. Truly-invisible
    intermediate content is incompatible with live-streaming the final turn (we
    can't know a turn is final until its `finish_reason` arrives), and tool
    turns usually carry no content anyway — so this is the right trade.

    Only the per-turn inferencer fetch differs by mode (`stream:false` POST vs.
    SSE accumulation in `_stream_turn`); the system-prompt prepend, allow-list
    boundary, tool dispatch, and max-turns guard are shared below.

    Dispatch by `<provider>/`:
      * `claude-code/<model>` → TOOL-LESS completion via the local Claude Code
        CLI (keyless). Recipes with a non-empty tools list are rejected.
      * a known inferencer    → the inferencer ↔ tool loop below.
      * anything else         → unsupported (501)."""
    inferencer = recipe["inferencer"]
    provider = inferencer.split("/", 1)[0]

    if provider == "claude-code":
        model = inferencer.split("/", 1)[1] if "/" in inferencer else ""
        try:
            if recipe["tools"]:
                # DELEGATION (executor): Claude owns the agentic loop and calls
                # the recipe's allow-listed MCP tools itself. Hand it ONLY the
                # downstream servers those tools reference; the allow-list stays
                # a hard boundary via --allowedTools (see claude_code).
                mcp_servers = _delegate_mcp_servers(recipe["tools"])
                resp = await claude_code.run_delegated(
                    recipe["system"], user_msgs, model,
                    allowed_tools=recipe["tools"], mcp_servers=mcp_servers)
            else:
                resp = await claude_code.run_completion(
                    recipe["system"], user_msgs, model)
        except claude_code.ClaudeCodeError as e:
            raise OrchestrationError(
                f"claude-code backend: {e}", "server_error", 502) from e
        # claude-code is non-streaming: surface its whole answer as one delta.
        if stream:
            content = resp["choices"][0]["message"].get("content") or ""
            if content:
                yield {"type": "delta", "content": content}
        yield {"type": "final", "response": resp}
        return

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
    log.info("orchestrating: tools=%s inferencer=%s stream=%s",
             [t["function"]["name"] for t in tools], inferencer, stream)

    for turn in range(1, 9):
        req = {
            "model": inferencer_model,
            "messages": messages,
            "tools": tools,
            "stream": bool(stream),
            **inf.extra_body,             # provider-specific (Ollama options / Anthropic max_tokens)
        }
        if stream:
            # Surface content deltas live; `acc` is filled with the turn's full
            # message once the upstream stream ends.
            acc: dict = {}
            async for piece in _stream_turn(inf, req, headers, acc):
                yield {"type": "delta", "content": piece}
            content, calls, resp = acc["content"], acc["calls"], acc["response"]
        else:
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
            # Tool-phase progress events (slice streaming-3): surfaced by the MCP
            # `chat` tool as `ctx.info(...)` notifications. The HTTP adapters and
            # the non-streaming drainer ignore every non-delta/final event, so
            # these are free for those paths. Emitted in BOTH modes — the loop is
            # mode-agnostic here, and the MCP path runs stream=False.
            yield {"type": "tool_call", "turn": turn,
                   "name": namespaced, "args": args}
            if namespaced not in allowed:
                # Refuse and feed the refusal back as the tool result (every
                # tool_call needs a matching tool message, including denied
                # ones) so the loop continues and the model can recover.
                log.warning("recipe denied out-of-list tool '%s' (allow-list: %s)",
                            namespaced, sorted(allowed))
                ok = False
                result = (f"ERROR: tool '{namespaced}' is not permitted by this "
                          f"recipe (allowed: {sorted(allowed)})")
            else:
                try:
                    r2 = await reg.dispatch(namespaced, args)
                    parts = [c.text for c in r2.content if hasattr(c, "text")]
                    result = "\n".join(parts) if parts else json.dumps(
                        [c.model_dump() for c in r2.content], default=str)
                    ok = True
                except Exception as e:
                    ok = False
                    result = f"ERROR: {type(e).__name__}: {e}"
            preview = (result[:80] + "…") if len(result) > 80 else result
            log.info("  ← %s", preview)
            yield {"type": "tool_result", "turn": turn,
                   "name": namespaced, "ok": ok}
            messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": call.get("id", f"call_{turn}_{namespaced}"),
            })

    raise OrchestrationError("max turns (8) exceeded", "server_error", 500)


async def _stream_turn(inf: inferencers.Inferencer, req: dict, headers: dict,
                       acc: dict):
    """Stream ONE inferencer turn over SSE. Yields assistant content delta
    strings as they arrive; on completion fills `acc` with the turn's
    accumulated `{content, calls, response}`.

    Two things this owns: (1) the upstream per-turn `finish_reason` and `[DONE]`
    are CONSUMED here and never surfaced — the orchestration stream is closed by
    exactly one synthesized terminator regardless of turn count. (2) tool_call
    deltas arrive FRAGMENTED across chunks (the `id` in one, `arguments`
    piecemeal), so they're reassembled by `index`."""
    content_parts: list[str] = []
    calls_by_index: dict[int, dict] = {}
    async with httpx.AsyncClient(timeout=180) as c:
        async with c.stream("POST", inf.chat_url(), json=req, headers=headers) as r:
            if r.status_code >= 400:
                raw = await r.aread()
                try:
                    payload = json.loads(raw)
                except (ValueError, TypeError):
                    payload = {"error": {"message": raw.decode("utf-8", "replace")
                                         or "inferencer error", "type": "server_error"}}
                raise OrchestrationError("inferencer error", "server_error", 502,
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


async def _orchestrate_recipe(recipe: recipes.Recipe, body: dict) -> JSONResponse:
    """HTTP adapter (non-streaming): run the shared loop, map results/errors
    onto JSON."""
    try:
        resp = await orchestrate(recipe, body.get("messages", []), registry)
    except OrchestrationError as e:
        if e.payload is not None:
            return JSONResponse(e.payload, status_code=e.status)
        return _error(e.message, e.kind, e.status)
    return JSONResponse(resp)


async def _orchestrate_recipe_stream(recipe: recipes.Recipe, body: dict) -> Response:
    """HTTP adapter (streaming): drive `orchestrate_events(stream=True)` and emit
    OpenAI `chat.completion.chunk` SSE frames — a canonical role chunk, the
    content deltas, then exactly ONE `finish_reason:"stop"` chunk + `[DONE]`,
    no matter how many tool turns ran.

    Like the passthrough streamer, we PRIME the generator before returning a
    StreamingResponse: any error before the first surfaced output (missing key,
    unsupported inferencer, a first-turn inferencer error) maps to a proper HTTP
    status rather than an empty 200 stream. Errors after streaming has begun can
    no longer change the status, so they go out as a best-effort error frame."""
    model = body.get("model", "")
    agen = orchestrate_events(recipe, body.get("messages", []), registry, stream=True)
    try:
        first = await agen.__anext__()
    except StopAsyncIteration:
        first = None
    except OrchestrationError as e:
        await agen.aclose()
        if e.payload is not None:
            return JSONResponse(e.payload, status_code=e.status)
        return _error(e.message, e.kind, e.status)

    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def frame(delta: dict, finish: str | None = None) -> str:
        return "data: " + json.dumps({
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }) + "\n\n"

    async def sse():
        yield frame({"role": "assistant"})        # canonical first chunk
        try:
            for ev in ([first] if first is not None else []):
                if ev["type"] == "delta":
                    yield frame({"content": ev["content"]})
            async for ev in agen:                 # 'final' events terminate below
                if ev["type"] == "delta":
                    yield frame({"content": ev["content"]})
        except OrchestrationError as e:
            payload = e.payload if e.payload is not None else {
                "error": {"message": e.message, "type": e.kind}}
            yield "data: " + json.dumps(payload) + "\n\n"
        yield frame({}, finish="stop")            # the one and only terminator
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


def _error(message: str, kind: str, status: int) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": kind}},
        status_code=status,
    )
