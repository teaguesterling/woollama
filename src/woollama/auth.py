"""Surface access control for woollama's HTTP surfaces (`/v1/*` and `/mcp`).

The router holds provider API keys and can dispatch local MCP tools, so its
surfaces are access-controlled, not open (docs/security.md):

  * **No token configured** (the default): only *local* peers are served —
    loopback TCP and the mode-0600 Unix socket. The bind layer additionally
    refuses to bind a non-loopback address at all (`check_bind_allowed`), so
    fail-closed holds at startup AND per-request (the per-request check also
    covers a reverse proxy / port forward that re-exposes a loopback bind).
  * **`WOOLLAMA_TOKEN` configured**: every TCP request must present
    `Authorization: Bearer <token>` (constant-time compared) — a token-bearing
    deployment is uniformly authenticated, loopback included. This is what
    makes an explicit non-loopback `WOOLLAMA_ADDRESS` bind acceptable.
  * **Unix-socket peers** (ASGI `client is None`): exempt. The socket is bound
    mode 0600 (`binding.py`), so the filesystem is the credential; uvicorn
    reports no peer address for UDS connections, which is how we recognize
    them. TCP connections always carry a peer address.

`authorize` is the single pure decision function; `SurfaceAuthMiddleware` is
the thin ASGI wrapper that applies it to the whole app (including mounted
sub-apps like `/mcp`). It is pure-ASGI, not `BaseHTTPMiddleware`, so SSE
streaming responses pass through unbuffered.
"""
from __future__ import annotations

import ipaddress
import json
import os
import secrets

ENV_TOKEN = "WOOLLAMA_TOKEN"


def configured_token() -> str | None:
    """The configured surface token, or None (unset/empty ⇒ no token)."""
    return os.environ.get(ENV_TOKEN) or None


def is_loopback_host(host: str | None) -> bool:
    """True iff `host` provably refers to loopback: `localhost`, 127.0.0.0/8,
    `::1`, or an IPv4-mapped loopback (`::ffff:127.x.x.x`). Any other hostname
    is treated as NOT loopback (fail closed — we don't resolve names)."""
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return (mapped or ip).is_loopback


class ExposureError(RuntimeError):
    """Refusing to expose the surface: a non-loopback bind with no token."""


def check_bind_allowed(host: str) -> None:
    """Fail closed at startup: a non-loopback bind target requires a configured
    token. Raises `ExposureError` instead of silently serving an
    unauthenticated surface beyond loopback."""
    if is_loopback_host(host):
        return
    if configured_token() is None:
        raise ExposureError(
            f"refusing to bind non-loopback address {host!r} without an auth "
            f"token: set {ENV_TOKEN} (clients then send "
            f"'Authorization: Bearer <token>'), or bind loopback (unset "
            f"WOOLLAMA_ADDRESS / use 127.0.0.1).")


def authorize(client_host: str | None, authorization: str | None) -> str | None:
    """The per-request decision. Returns None when the request is authorized,
    else a short refusal reason.

    `client_host` is the ASGI peer address (None for a Unix-socket peer);
    `authorization` is the raw Authorization header value, if any."""
    if client_host is None:
        # Unix-socket peer: the 0600 socket mode is the credential.
        return None
    token = configured_token()
    if token is not None:
        supplied = ""
        if authorization and authorization.startswith("Bearer "):
            supplied = authorization[len("Bearer "):]
        if secrets.compare_digest(supplied.encode(), token.encode()):
            return None
        return "missing or invalid bearer token"
    if is_loopback_host(client_host):
        return None
    return (f"no {ENV_TOKEN} is configured, so only local (loopback / unix "
            "socket) clients are served")


class SurfaceAuthMiddleware:
    """Pure-ASGI middleware applying `authorize` to every http/websocket
    request, before routing (so mounted sub-apps are covered too)."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        client = scope.get("client")
        authorization = None
        for key, value in scope.get("headers") or []:
            if key == b"authorization":
                authorization = value.decode("latin-1")
                break
        reason = authorize(client[0] if client else None, authorization)
        if reason is None:
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({"error": {
            "message": f"unauthorized: {reason}",
            "type": "authentication_error",
        }}).encode()
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"www-authenticate", b"Bearer"),
            ],
        })
        await send({"type": "http.response.body", "body": body})
