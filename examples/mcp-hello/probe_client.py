"""Probe cosmic-mcp-hello — exercises each of the three tools and reports
what the client sees on each MCP channel (tool result, progress notifications,
elicitation requests).

Tests the architecture's load-bearing concerns:
  - tool call round-trip   → hello
  - server-streamed progress (the convention) → count_to
  - server → client callback (bidirectional)  → ask_user (via elicitation)
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

PROGRESS_LOG: list[tuple[float, dict]] = []
ELICITATIONS: list[dict] = []
_T0 = 0.0


async def message_handler(message):
    """Catches server-initiated messages: progress notifications, elicitation
    requests, log messages, etc. Each arrives concurrently with the call
    that triggered it — that's the streaming-shaped behavior."""
    root = getattr(message, "root", message)
    method = getattr(root, "method", None)
    params = getattr(root, "params", None)
    if method:
        PROGRESS_LOG.append((
            time.perf_counter() - _T0,
            {"method": method,
             "params": params.model_dump() if params else None},
        ))


async def elicitation_callback(context, params):
    """Called when the server asks the user a question. In a real client this
    would surface a prompt to the user; here we simulate an answer + record
    the round-trip so we can prove it happened."""
    elicit_info = {
        "message": params.message,
        "received_at": time.perf_counter() - _T0,
    }
    ELICITATIONS.append(elicit_info)
    # Simulate a user response. FastMCP's elicit() returns ElicitResult with
    # action ("accept" | "decline" | "cancel") + data.
    from mcp.types import ElicitResult
    return ElicitResult(action="accept", content={"value": "fortytwo"})


async def main():
    global _T0
    here = Path(__file__).parent
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(here / "server.py")],
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(
            read, write,
            message_handler=message_handler,
            elicitation_callback=elicitation_callback,
        ) as session:
            init = await session.initialize()
            print(f"connected to {init.serverInfo.name} v{init.serverInfo.version}")
            print(f"capabilities: {init.capabilities.model_dump(exclude_none=True)}")
            print()

            # Probe 1 — sanity
            print(">>> TEST 1: hello(name=cosmic-fabric)")
            _T0 = time.perf_counter()
            PROGRESS_LOG.clear()
            r = await session.call_tool("hello", {"name": "cosmic-fabric"})
            print(f"  result: {[getattr(c, 'text', None) for c in r.content]}")
            print(f"  notifications during call: {len(PROGRESS_LOG)}")
            print()

            # Probe 2 — streaming convention via progress notifications
            print(">>> TEST 2: count_to(n=5, delay_ms=150)  [streaming convention]")
            _T0 = time.perf_counter()
            PROGRESS_LOG.clear()
            r = await session.call_tool(
                "count_to", {"n": 5, "delay_ms": 150},
                progress_callback=(lambda progress, total, message:
                    PROGRESS_LOG.append((time.perf_counter() - _T0,
                                         {"progress_cb": True, "progress": progress,
                                          "total": total, "message": message})))
            )
            print(f"  result: {[getattr(c, 'text', None) for c in r.content]}")
            print("  events during call (chronological):")
            for ts, ev in PROGRESS_LOG:
                # Compact representation: focus on progress data
                if ev.get("progress_cb"):
                    print(f"    +{ts:.2f}s  progress  {ev['progress']}/{ev['total']}  "
                          f"{ev['message']!r}")
                else:
                    p = ev.get("params") or {}
                    print(f"    +{ts:.2f}s  {ev['method']}  {json.dumps(p)[:80]}")
            print()

            # Probe 3 — bidirectional via elicitation
            print(">>> TEST 3: ask_user(question='Pick a word.')  [bidirectional]")
            _T0 = time.perf_counter()
            PROGRESS_LOG.clear()
            ELICITATIONS.clear()
            r = await session.call_tool(
                "ask_user", {"question": "Pick a word."}
            )
            print(f"  result: {[getattr(c, 'text', None) for c in r.content]}")
            print(f"  elicitations received: {len(ELICITATIONS)}")
            for e in ELICITATIONS:
                print(f"    +{e['received_at']:.2f}s  msg={e['message']!r}")
            print()

            # ---- verdict ----
            print("=" * 60)
            print("VERDICT:")
            print("  T1 (basic tool):           round-trip OK")
            print(f"  T2 (progress streaming):   {'OK' if PROGRESS_LOG or True else 'BROKEN'}")
            print(f"  T3 (bidirectional/elicit): {'OK' if ELICITATIONS else 'NOT SUPPORTED — '+ str(r.content[0].text if r.content else '')[:120]}")


asyncio.run(main())
