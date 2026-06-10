"""The core/router boundary contract: `import woollama.core` must NOT pull in the
server stack (FastAPI / uvicorn / the MCP server) or the router. This test IS the
guarantee that keeps `woollama.core` embeddable (see docs/core-extraction.md) — if
it fails, something in core grew a server dependency and the split leaked.

It runs in a FRESH interpreter on purpose: inside the pytest process fastapi /
uvicorn / woollama.router are already imported by the rest of the suite, so an
in-process sys.modules check would always see them and prove nothing.
"""
from __future__ import annotations

import subprocess
import sys

FORBIDDEN = ("fastapi", "uvicorn", "mcp", "woollama.router", "woollama.manager",
             "woollama.mcp_server", "woollama.conversations")


def test_core_import_is_server_free():
    code = (
        "import importlib, sys\n"
        "importlib.import_module('woollama.core')\n"
        f"forbidden = {FORBIDDEN!r}\n"
        "leaked = [m for m in forbidden if m in sys.modules]\n"
        "print(','.join(leaked))\n"
        "sys.exit(1 if leaked else 0)\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, (
        f"woollama.core leaked server modules: {r.stdout.strip()!r}\n"
        f"stderr:\n{r.stderr}")
