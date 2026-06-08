"""rest-convstore — a reference REST conversation-store server (issue #2).

A second reference store alongside `examples/mcp-convstore`, proving woollama's
`ConversationStoreProvider` seam is transport-agnostic — the same four ops over
plain HTTP instead of MCP. This one is **file-backed**, so transcripts persist
across restarts (the in-memory MCP example does not).

woollama is a *client* to it (`conversations.HttpStoreProvider`); the bytes live
HERE, one JSON file per thread under `$CONVSTORE_DIR`. woollama holds only the
opaque thread id.

REST surface (the `ConversationStoreProvider` contract):

    PUT    /threads/{id}   create an empty thread (idempotent)      -> 204
    GET    /threads/{id}   the message list ([] if absent)          -> 200
    PATCH  /threads/{id}   append messages (JSON array body)        -> 200 {"count"}
    DELETE /threads/{id}   delete the thread (idempotent)           -> 204

The provider mints the thread id (a UUID) and PUTs it, so this server needs no
id-minting logic and create is idempotent.

Run with:
    CONVSTORE_DIR=/tmp/threads python server.py --port 9000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

# The bytes live here — owned by THIS server, never by woollama.
STORE_DIR = Path(os.environ.get("CONVSTORE_DIR")
                 or (Path(tempfile.gettempdir()) / "woollama-rest-convstore"))
STORE_DIR.mkdir(parents=True, exist_ok=True)

# Thread ids come from the URL path; constrain them so a crafted id can't escape
# STORE_DIR (path traversal). The provider mints uuid4().hex, which matches.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")

app = FastAPI(title="rest-convstore")


def _path(thread_id: str) -> Path:
    if not _SAFE_ID.match(thread_id):
        raise HTTPException(status_code=400, detail="invalid thread id")
    return STORE_DIR / f"{thread_id}.json"


def _read(p: Path) -> list[dict]:
    return json.loads(p.read_text()) if p.exists() else []


@app.put("/threads/{thread_id}")
def create_thread(thread_id: str) -> Response:
    """Create an empty thread (idempotent — re-PUT leaves an existing one)."""
    p = _path(thread_id)
    if not p.exists():
        p.write_text("[]")
    return Response(status_code=204)


@app.get("/threads/{thread_id}")
def get_thread(thread_id: str) -> JSONResponse:
    """The thread's message list; [] for an unknown/fresh thread."""
    return JSONResponse(_read(_path(thread_id)))


@app.patch("/threads/{thread_id}")
async def append_turn(thread_id: str, request: Request) -> JSONResponse:
    """Append the JSON-array body's messages to the thread (read-modify-write).
    Creates the thread implicitly if absent — forgiving of races."""
    new = await request.json()
    if not isinstance(new, list):
        raise HTTPException(status_code=400, detail="body must be a JSON array")
    p = _path(thread_id)
    messages = _read(p)
    messages.extend(new)
    p.write_text(json.dumps(messages))
    return JSONResponse({"ok": True, "count": len(messages)})


@app.delete("/threads/{thread_id}")
def delete_thread(thread_id: str) -> Response:
    """Delete the thread and its file (idempotent)."""
    _path(thread_id).unlink(missing_ok=True)
    return Response(status_code=204)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("CONVSTORE_PORT", "9000")))
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
