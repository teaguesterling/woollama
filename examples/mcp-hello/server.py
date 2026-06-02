"""cosmic-mcp-hello — a tiny reference / test MCP server using FastMCP.

Three tools chosen to exercise the cos-fab MCP architecture's load-bearing
behaviors:

  1. hello(name)       — sanity: tool call round-trips cleanly
  2. count_to(n)       — progress notifications during a call (streaming
                         convention test; mirrors cosmic-mcp-ollama's
                         progress-typed-events shape)
  3. ask_user(question) — server→client callback via MCP elicitation
                         (the standard primitive for bidirectional flow,
                         which the lackpy → cos-fab tool-callback pattern
                         would need)

Run with:
    fastmcp run server.py:mcp --transport stdio
"""
from __future__ import annotations

import asyncio

from fastmcp import Context, FastMCP

mcp = FastMCP("cosmic-mcp-hello")


@mcp.tool()
def hello(name: str = "world") -> str:
    """Return a greeting. Sanity-check tool — verifies the call path works."""
    return f"Hello, {name}!"


@mcp.tool()
async def count_to(ctx: Context, n: int = 5, delay_ms: int = 200) -> dict:
    """Count from 1 to n, emitting an MCP progress notification per step.

    Tests the streaming convention: each step is a `notifications/progress`
    that a streaming-aware client can render live. A naive client that
    ignores progress still gets the final {count, total} CallToolResult.
    """
    if n < 1 or n > 100:
        raise ValueError("n must be between 1 and 100")
    for i in range(1, n + 1):
        # FastMCP's ctx.report_progress is the canonical way to emit
        # notifications/progress with progressToken, current, total, message.
        await ctx.report_progress(
            progress=i, total=n, message=f"step {i} of {n}"
        )
        if delay_ms:
            await asyncio.sleep(delay_ms / 1000)
    return {"count": n, "total": n, "done": True}


@mcp.tool()
async def ask_user(ctx: Context, question: str) -> dict:
    """Ask the user a question via MCP elicitation. The client surfaces the
    question, collects an answer, returns it. This is the standard
    *server→client callback* in current MCP — the primitive that would
    power lackpy → cos-fab tool callbacks (with `read_file` etc.) under the
    bidirectional architecture.

    If the client doesn't support elicitation, this fails cleanly with an
    informative error — that *is* the probe result.
    """
    # FastMCP exposes elicitation via ctx.elicit(message=..., response_type=...).
    # The response_type can be a Python type (str, int, bool) or a Pydantic
    # model for structured input.
    try:
        result = await ctx.elicit(message=question, response_type=str)
        return {"asked": question, "answer": result.data, "action": result.action}
    except Exception as e:
        return {"asked": question, "error": f"{type(e).__name__}: {e}",
                "note": "client may not support elicitation"}


if __name__ == "__main__":
    mcp.run(transport="stdio")
