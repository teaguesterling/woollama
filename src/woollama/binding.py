"""Ephemeral local-only binding.

Same pattern as a local `fabric --serve` instance, with TWO coexisting local
surfaces (docs/architecture.md §"Binding"):

  * **Unix socket** — `$XDG_RUNTIME_DIR/woollama.sock`, the default for local
    MCP clients (the panel, the CLI). No network at all.
  * **HTTP loopback** — a random free port on `127.0.0.1`, for OpenAI-compatible
    clients that need HTTP. The chosen `host:port` is persisted to
    `$XDG_RUNTIME_DIR/woollama.addr` so clients can discover it.

We *never* bind to `0.0.0.0` without an explicit opt-in via `WOOLLAMA_ADDRESS`:
the router holds API keys and routes to local resources — it must not be
LAN-reachable by default.

The router holds API keys, so each surface is access-controlled the same way:
loopback for TCP, and **mode 0600** for the Unix socket (a connectable socket is
as good as the keys — anyone who can `connect()` can spend them). The socket is
bound here (not handed to uvicorn as a `uds=` path) so we can read the *actual*
bound port back from the socket and persist exactly that — no resolve-then-bind
TOCTOU where the persisted `.addr` could disagree with the real port.
"""
from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass

log = logging.getLogger("woollama.binding")

ADDR_FILENAME = "woollama.addr"
SOCK_FILENAME = "woollama.sock"
ENV_OVERRIDE = "WOOLLAMA_ADDRESS"


def _runtime_dir() -> str:
    """`$XDG_RUNTIME_DIR` (0700, per-user) or `/tmp` as a fallback."""
    return os.environ.get("XDG_RUNTIME_DIR") or "/tmp"


def addr_file_path() -> str:
    """`$XDG_RUNTIME_DIR/woollama.addr` (falls back to /tmp)."""
    return os.path.join(_runtime_dir(), ADDR_FILENAME)


def sock_file_path() -> str:
    """`$XDG_RUNTIME_DIR/woollama.sock` (falls back to /tmp)."""
    return os.path.join(_runtime_dir(), SOCK_FILENAME)


def resolve_tcp_target() -> tuple[str, int]:
    """The TCP host/port to bind — pure (no socket, no side effects).

    Precedence:
      1. `$WOOLLAMA_ADDRESS=host[:port]` (explicit override; the only way to ever
         bind a non-loopback host like `0.0.0.0` — opt-in). A missing/zero port
         means "pick a free one".
      2. `127.0.0.1:0` — loopback, free port (default).

    Port 0 is resolved to a real port by the OS at bind time in `open_sockets`;
    that real port is what gets persisted, so there's no resolve/bind race."""
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        host, _, port = override.partition(":")
        return (host or "127.0.0.1", int(port) if port else 0)
    return ("127.0.0.1", 0)


@dataclass
class Listeners:
    """Bound listen sockets plus where they ended up, for the banner/cleanup.
    `sock_path` is None when the Unix socket couldn't be bound (degraded to
    TCP-only)."""
    sockets: list[socket.socket]
    tcp_host: str
    tcp_port: int
    sock_path: str | None


def open_sockets() -> Listeners:
    """Bind both local surfaces and persist their discovery files.

    Always binds the TCP loopback (or `$WOOLLAMA_ADDRESS` override). Also binds
    the Unix socket best-effort: if it can't (e.g. an unwritable runtime dir, or
    a path over the ~108-char `sun_path` limit), we log and serve TCP-only
    rather than failing to start. Callers serve `listeners.sockets` and must
    `cleanup()` on shutdown to remove the socket file."""
    host, port = resolve_tcp_target()
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcp.bind((host, port))
    real_host, real_port = tcp.getsockname()[:2]
    _persist(addr_file_path(), f"{real_host}:{real_port}")

    # The TCP port is random, so it's persisted to .addr for discovery. The
    # Unix socket path is deterministic ($XDG_RUNTIME_DIR/woollama.sock), so its
    # mere existence IS the discovery artifact — no separate file to write.
    sockets = [tcp]
    sock_path: str | None = sock_file_path()
    try:
        sockets.insert(0, _open_unix_socket(sock_path))
    except OSError as e:
        log.warning("unix socket unavailable (%s); serving TCP-only", e)
        sock_path = None

    return Listeners(sockets=sockets, tcp_host=real_host, tcp_port=real_port,
                     sock_path=sock_path)


def _open_unix_socket(path: str) -> socket.socket:
    """Bind a stream Unix socket at `path`, mode 0600. Unlinks a stale socket
    file first (local single-instance daemon: a leftover from a dead run, or our
    own previous run). The umask narrows the create-time mode so there is no
    window where the socket is world-connectable before the chmod lands."""
    try:
        os.unlink(path)             # clear a stale socket from a prior run
    except FileNotFoundError:
        pass
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old_umask = os.umask(0o177)     # → file mode 0600 at creation
    try:
        s.bind(path)
    finally:
        os.umask(old_umask)
    os.chmod(path, 0o600)           # explicit: don't rely on umask alone
    return s


def cleanup(listeners: Listeners) -> None:
    """Remove the Unix socket file on shutdown. uvicorn cleans a `uds=` path it
    created itself, but not a pre-bound socket handed to it via `sockets=`, so we
    own this."""
    if listeners.sock_path:
        try:
            os.unlink(listeners.sock_path)
        except FileNotFoundError:
            pass


def _persist(path: str, contents: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError:
        pass
    try:
        with open(path, "w") as f:
            f.write(contents + "\n")
    except OSError:
        pass


def discover_addr() -> str | None:
    """Client-side helper: read the persisted TCP address. None if absent."""
    try:
        return open(addr_file_path()).read().strip()
    except FileNotFoundError:
        return None


def discover_sock() -> str | None:
    """Client-side helper: the Unix socket path if the daemon bound one (the
    socket's existence at the well-known path is the discovery signal). None if
    no socket is present."""
    path = sock_file_path()
    return path if os.path.exists(path) else None
