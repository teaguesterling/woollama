"""mcp-convstore — a reference MCP **conversation-store** server.

This is the load-bearing example for woollama's core principle: *woollama
routes conversation handles; backends OWN the state*. woollama must never
store transcript bytes in its own process. A conversation store running as
its own MCP server is the sanctioned pattern — woollama stays a client.

The four tools form the `ConversationStoreProvider` contract:

  - create_thread()                 -> thread_id (str)
  - get_thread(thread_id)           -> list[message dict]
  - append_turn(thread_id, msgs)    -> {"ok": True, "count": N}
  - delete_thread(thread_id)        -> {"ok": True}

The transcript bytes live in THIS process (a plain in-memory dict here; a
real provider would back it with sqlite/duckdb/postgres/a file). woollama
holds only the opaque `thread_id` handle and asks this server for the
history each turn.

Every tool returns an explicit JSON string (`json.dumps(...)`) so the wire
shape is unambiguous: woollama's `McpStoreProvider` parses these back. The
empty-thread case is exercised by design — `create_thread` then
`get_thread` returns `[]`.

Run with:
    python server.py
    # or: fastmcp run server.py:mcp --transport stdio
"""
from __future__ import annotations

import json

from fastmcp import FastMCP

mcp = FastMCP("mcp-convstore")

# The transcript bytes — owned HERE, in this server's process. woollama never
# sees this dict; it only holds the thread_id keys.
_THREADS: dict[str, list[dict]] = {}
_NEXT_ID = {"n": 0}


@mcp.tool()
def create_thread() -> str:
    """Create a new, empty conversation thread. Returns its opaque id as a
    JSON string. The thread starts with no messages — get_thread on a fresh
    id returns []."""
    _NEXT_ID["n"] += 1
    thread_id = f"thread-{_NEXT_ID['n']}"
    _THREADS[thread_id] = []
    return json.dumps(thread_id)


@mcp.tool()
def get_thread(thread_id: str) -> str:
    """Return the full message list for a thread as a JSON-string array of
    {role, content} dicts. Unknown or freshly-created threads return []."""
    return json.dumps(_THREADS.get(thread_id, []))


@mcp.tool()
def append_turn(thread_id: str, messages: list[dict]) -> str:
    """Append messages (a list of {role, content} dicts) to a thread.
    Returns {"ok": True, "count": <new total>} as a JSON string. Creating
    the thread implicitly if it does not exist keeps the provider forgiving
    of races."""
    thread = _THREADS.setdefault(thread_id, [])
    thread.extend(messages)
    return json.dumps({"ok": True, "count": len(thread)})


@mcp.tool()
def delete_thread(thread_id: str) -> str:
    """Delete a thread and its transcript. Idempotent — deleting an unknown
    thread still returns {"ok": True}."""
    _THREADS.pop(thread_id, None)
    return json.dumps({"ok": True})


if __name__ == "__main__":
    mcp.run(transport="stdio")
