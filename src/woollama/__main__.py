"""`python -m woollama` and the `woollama` console script entrypoint.

Two modes:
  * `woollama`        — the OpenAI-compatible HTTP server (default). Resolves
                        the bind address, prints it, starts uvicorn.
  * `woollama mcp`    — woollama as an MCP server over stdio (slice e). This
                        is what an MCP client puts in its mcp.json:
                        { "command": "woollama", "args": ["mcp"] }.
"""
from __future__ import annotations

import logging
import sys

import uvicorn

from . import __version__
from .binding import addr_file_path, resolve_bind
from .router import app


def _run_mcp() -> int:
    """`woollama mcp` — stdio MCP server. stdout is the JSON-RPC channel, so
    logging must go to stderr and nothing else may print to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    from .mcp_server import serve
    serve()
    return 0


def main() -> int:
    if "--version" in sys.argv:
        print(f"woollama {__version__}")
        return 0

    if len(sys.argv) > 1 and sys.argv[1] == "mcp":
        return _run_mcp()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    host, port = resolve_bind()
    print(f"woollama {__version__} — listening on http://{host}:{port}", flush=True)
    print(f"  address persisted to: {addr_file_path()}", flush=True)
    print(f"  OpenAI base_url:      http://{host}:{port}/v1", flush=True)
    print(f"  models:               GET /v1/models", flush=True)
    print(f"  chat:                 POST /v1/chat/completions", flush=True)
    print(f"  MCP (Streamable HTTP): http://{host}:{port}/mcp", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
