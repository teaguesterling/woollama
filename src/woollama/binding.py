"""Ephemeral local-only binding.

Same pattern as a local `fabric --serve` instance: bind to a random free
loopback port at startup, persist the chosen address to
`$XDG_RUNTIME_DIR/woollama.addr` so clients can discover it, and *never*
bind to `0.0.0.0` without an explicit opt-in via `WOOLLAMA_ADDRESS`. The
router holds API keys and routes to local resources — it must not be
LAN-reachable.
"""
from __future__ import annotations

import os
import socket

ADDR_FILENAME = "woollama.addr"
ENV_OVERRIDE = "WOOLLAMA_ADDRESS"


def addr_file_path() -> str:
    """`$XDG_RUNTIME_DIR/woollama.addr` (falls back to /tmp)."""
    base = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    return os.path.join(base, ADDR_FILENAME)


def _free_loopback_port() -> int:
    """Ask the OS for an unused loopback TCP port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def resolve_bind() -> tuple[str, int]:
    """Where to bind the HTTP server.

    Precedence (highest first):
      1. `$WOOLLAMA_ADDRESS=host:port` env var (explicit override; the only
         way to ever bind to `0.0.0.0` — opt-in)
      2. Random free loopback port (default)

    The chosen address is persisted to `addr_file_path()` so clients can
    discover it without us being on a well-known port.
    """
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        host, _, port = override.partition(":")
        host = host or "127.0.0.1"
        port = int(port or "0") or _free_loopback_port()
        _persist(f"{host}:{port}")
        return host, port

    port = _free_loopback_port()
    _persist(f"127.0.0.1:{port}")
    return "127.0.0.1", port


def _persist(addr: str) -> None:
    path = addr_file_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError:
        pass
    try:
        with open(path, "w") as f:
            f.write(addr + "\n")
    except OSError:
        pass


def discover_addr() -> str | None:
    """Client-side helper: read the persisted address. Returns None if absent."""
    try:
        return open(addr_file_path()).read().strip()
    except FileNotFoundError:
        return None
