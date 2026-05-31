"""mcp-textops — a second example MCP server, distinct from mcp-hello, so
woollama's multi-server discovery + namespacing has something real to merge.

Three trivial text-manipulation tools (no model required, no network, no
side effects) — exists purely to demonstrate "two servers, different tool
namespaces, unified registry."

Run with:
    python server.py
"""
from __future__ import annotations

from fastmcp import FastMCP

mcp = FastMCP("mcp-textops")


@mcp.tool()
def uppercase(text: str) -> str:
    """Convert the given text to uppercase."""
    return text.upper()


@mcp.tool()
def word_count(text: str) -> int:
    """Count whitespace-separated words in the given text."""
    return len(text.split())


@mcp.tool()
def reverse_text(text: str) -> str:
    """Reverse the characters in the given text."""
    return text[::-1]


if __name__ == "__main__":
    mcp.run(transport="stdio")
