"""Stateless inference primitives — server-free.

`complete()` / `complete_stream()` run ONE turn against a directly-addressed
inferencer (`<provider>/<model>`) and return the assistant text / yield text
deltas. They are the model-management core that the router's stateless path and
embedders (lackpy) build on. Recipe orchestration (`woollama/<recipe>`) lives in
`core.orchestrate`, not here — these handle only a concrete provider/model.

Per-call `api_key` / `base_url` override the inferencer's configured values, so an
embedder can drive multiple keys / endpoints without mutating global config (the
env-var-only model was a real library limitation).
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx

from . import inferencers, ollama_native

# Ollama's native /api/chat may need to load a large model on the first call; the
# /v1 path is the usual OpenAI request timeout.
NATIVE_TIMEOUT = httpx.Timeout(600.0, connect=10.0)
_HTTP_TIMEOUT = 180.0


class InferenceError(Exception):
    """Raised by the inference / orchestration layer. Each transport maps it to its
    own error surface (HTTP status / MCP error). `payload` carries the raw upstream
    response when the inferencer itself errored, so the surface can pass it through
    verbatim. (The router re-exports this as `OrchestrationError`.)"""

    def __init__(self, message: str, kind: str, status: int,
                 payload: dict | None = None):
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.status = status
        self.payload = payload


def _resolve(model: str, base_url: str | None,
             registry: "inferencers.ModelRegistry | None") -> tuple:
    """Return `(inferencer, bare_model, effective_base)` for a `<provider>/<model>`
    id, or raise `InferenceError` if the provider is unknown. Resolves against an
    explicit `registry` when given, else the module-level (config) lookup."""
    provider = model.split("/", 1)[0]
    inf = registry.get(provider) if registry is not None else inferencers.get(provider)
    if inf is None:
        known = registry.names() if registry is not None else inferencers.names()
        raise InferenceError(
            f"unknown model namespace: '{model}'. Use 'woollama/<recipe>' or "
            f"'<provider>/<model>' for a known inferencer "
            f"({', '.join(known)}).", "invalid_request_error", 400)
    bare = model.split("/", 1)[1] if "/" in model else ""
    return inf, bare, (base_url or inf.base_url).rstrip("/")


def _headers(inf, api_key: str | None) -> dict[str, str]:
    if api_key is not None:                 # per-call override skips the env lookup
        return {"Authorization": f"Bearer {api_key}"}
    try:
        return inf.headers()
    except inferencers.InferencerError as e:
        raise InferenceError(str(e), "invalid_request_error", 400) from e


async def complete(model: str, messages: list[dict], *, options: dict | None = None,
                   api_key: str | None = None, base_url: str | None = None,
                   registry: "inferencers.ModelRegistry | None" = None) -> str:
    """Run one stateless turn against `<provider>/<model>` and return the assistant
    text. `options` carries ollama-native knobs (e.g. `num_ctx`): when present for
    the ollama provider the turn goes through the native `/api/chat` (which honors
    them), translating the native reply back to text."""
    provider = model.split("/", 1)[0]
    inf, bare, base = _resolve(model, base_url, registry)
    headers = _headers(inf, api_key)

    if provider == "ollama" and options and options.get("num_ctx") is not None:
        req = ollama_native.to_native_request(
            {"model": bare, "messages": messages, "options": options, "stream": False})
        async with httpx.AsyncClient(timeout=NATIVE_TIMEOUT) as c:
            r = await c.post(ollama_native.native_chat_url(base), json=req, headers=headers)
            data = r.json()
        if "message" not in data:
            raise InferenceError("inferencer error", "server_error", 502, payload=data)
        return (data.get("message") or {}).get("content") or ""

    body = {"model": bare, "messages": messages, "stream": False}
    if options:
        body["options"] = options
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as c:
        r = await c.post(f"{base}/chat/completions", json=body, headers=headers)
        data = r.json()
    if "choices" not in data:
        raise InferenceError("inferencer error", "server_error", 502, payload=data)
    return data["choices"][0]["message"].get("content") or ""


async def complete_stream(model: str, messages: list[dict], *,
                          api_key: str | None = None,
                          base_url: str | None = None,
                          registry: "inferencers.ModelRegistry | None" = None
                          ) -> AsyncIterator[str]:
    """Yield assistant TEXT DELTAS for one stateless turn against
    `<provider>/<model>` (the inferencer's `/v1` SSE; `num_ctx`-native routing is
    non-stream only for now). Raises `InferenceError` before the first yield for the
    setup / upstream-status error cases, so the caller can map it to a status."""
    inf, bare, base = _resolve(model, base_url, registry)
    headers = _headers(inf, api_key)
    req = {"model": bare, "messages": messages, "stream": True}
    client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT)
    cm = client.stream("POST", f"{base}/chat/completions", json=req, headers=headers)
    r = await cm.__aenter__()
    if r.status_code >= 400:
        raw = await r.aread()
        await cm.__aexit__(None, None, None)
        await client.aclose()
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            payload = None
        raise InferenceError("inferencer error", "server_error",
                             r.status_code, payload=payload)
    try:
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                yield delta
    finally:
        await cm.__aexit__(None, None, None)
        await client.aclose()


def complete_sync(model: str, messages: list[dict], **kwargs) -> str:
    """Synchronous wrapper over `complete` for non-async embedders — spins a fresh
    event loop via `asyncio.run`. Raises `RuntimeError` if called from inside a
    running loop (use the async `complete` there). Accepts the same keyword args
    (`options` / `api_key` / `base_url` / `registry`)."""
    return asyncio.run(complete(model, messages, **kwargs))
