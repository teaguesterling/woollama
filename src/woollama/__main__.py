"""`python -m woollama` and the `woollama` console script entrypoint.

Resolves the bind address, prints it (and the persisted addr-file path),
starts uvicorn.
"""
from __future__ import annotations

import logging
import sys

import uvicorn

from . import __version__
from .binding import addr_file_path, resolve_bind
from .router import app


def main() -> int:
    if "--version" in sys.argv:
        print(f"woollama {__version__}")
        return 0

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
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
