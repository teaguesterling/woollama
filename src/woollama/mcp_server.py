"""woollama as an MCP server — the outbound MCP surface (slice e).

woollama's inbound surface is OpenAI-compatible HTTP (see `router.py`). This
module projects the same machinery onto an *outbound* MCP surface so MCP
clients (Claude Desktop, the cosmic-fabric panel) can drive woollama natively:

  * each recipe        → an MCP **prompt** (prompts/list, prompts/get returns
                         the recipe's rendered system message)
  * the `chat` verb    → an MCP **tool**  (tools/call runs the orchestration
                         loop and returns only the final assistant message —
                         the same contract as /v1/chat/completions; the tool
                         loop stays hidden from the client)
  * capabilities       → advertised on `initialize`

Transport is **stdio** (slice e scope): `woollama mcp` starts the server over
stdin/stdout, which is what a client puts in its mcp.json:

    { "command": "woollama", "args": ["mcp"] }

The server reuses `router.orchestrate` — it does NOT reimplement the chat loop.
Tool dispatch routes through a long-lived `Registry`, started/stopped inside
the FastMCP lifespan so the connection-owning tasks live on the same event loop
that serves tool calls (matching `manager.ServerManager`'s loop assumptions).

Re-exporting discovered downstream tools (textops.*, hello.*) onto tools/list
is now ON (decision #3): a connecting client sees the union of every configured
server's tools (namespaced) plus the `chat` verb — woollama as an MCP
aggregator. Because a server's tools are only known once its connection is up,
re-export is a lifespan-time dynamic registration (after `registry.start_all()`)
via `_register_reexported_tools`, not a build-time concat. HTTP/SSE transport
remains a later slice.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.prompts import Prompt
from fastmcp.tools import Tool
from fastmcp.tools.tool import ToolResult
from pydantic import PrivateAttr

from . import config, recipes
from .manager import Registry, ServerManager
# NOTE: `orchestrate`/`OrchestrationError` are imported lazily inside the chat
# tool (not at module top) to break the import cycle: router.py imports this
# module to mount the MCP server onto its FastAPI app.


log = logging.getLogger("woollama.mcp_server")


def build_registry() -> Registry:
    """Build the unified tool Registry from mcp.json (servers added, NOT yet
    started — the FastMCP lifespan starts them on the serving loop)."""
    reg = Registry()
    for name, cfg in config.load_mcp_servers().items():
        reg.add(ServerManager(name, cfg["command"], cfg["args"]))
    return reg


def _recipe_prompts() -> list[Prompt]:
    """One MCP prompt per loaded recipe; rendering returns its system message.

    Snapshot of `recipes.names()` at build time. The factory closure binds
    `system` per-iteration (avoids the late-binding-loop closure bug)."""
    prompts: list[Prompt] = []
    for name in recipes.names():
        recipe = recipes.get(name)
        system = recipe["system"]

        def render(system: str = system) -> str:
            return system

        prompts.append(Prompt.from_function(
            render, name=name,
            description=f"woollama recipe '{name}' — its system prompt",
        ))
    return prompts


class _ProxyTool(Tool):
    """Re-exports a discovered downstream MCP tool onto woollama's own
    tools/list: same namespaced name (`<server>.<tool>`) and input schema,
    dispatched through the unified `Registry` — the SAME long-lived connection
    layer the chat orchestration uses, not a second client stack.

    Unlike the chat tool, this is raw passthrough — it does NOT go through
    `orchestrate`, so it owns its own failure handling (orchestrate's
    dispatch-error catch isn't reachable from here)."""

    _reg: Registry = PrivateAttr()

    @classmethod
    def build(cls, namespaced: str, description: str, schema: dict,
              reg: Registry) -> "_ProxyTool":
        # Defensive copy: `Registry.openai_tools_for` reads the same spec
        # object for the HTTP surface; don't risk FastMCP aliasing it.
        tool = cls(name=namespaced, description=description, parameters=dict(schema))
        tool._reg = reg
        return tool

    async def run(self, arguments: dict) -> ToolResult:
        try:
            result = await self._reg.dispatch(self.name, arguments)
        except Exception as e:
            raise ToolError(f"dispatch failed: {type(e).__name__}: {e}") from e
        content = list(getattr(result, "content", None) or [])
        if getattr(result, "isError", False):
            text = "\n".join(c.text for c in content if hasattr(c, "text"))
            raise ToolError(text or f"downstream tool '{self.name}' errored")
        # Pass structured output through when the downstream tool produced it
        # (e.g. a dict-returning tool) so the client gets the structured payload,
        # not just its JSON-as-text. We deliberately do NOT mirror the downstream
        # output_schema onto this tool: that would couple every call to schema
        # validation, breaking content-only results — a later refinement.
        return ToolResult(content=content,
                          structured_content=getattr(result, "structuredContent", None))


def _chat_tool(reg: Registry) -> Tool:
    """The `chat` orchestration verb — a recipe runner that hides the
    inferencer ↔ tool loop. Distinct from the re-exported passthrough tools
    (which carry a `<server>.<tool>` dotted name; `chat` has none, so no
    collision)."""

    async def chat(messages: list, recipe: str = "", model: str = "") -> str:
        """Run a woollama recipe end-to-end and return the final assistant
        message. Mirrors POST /v1/chat/completions: the internal inferencer ↔
        tool loop is hidden — the caller sees only the final answer.

        Args:
            messages: OpenAI-shaped chat messages (the system prompt is
                supplied by the recipe and prepended automatically).
            recipe: recipe name to run (e.g. "streamer"). Primary selector.
            model: optional "woollama/<recipe>" form, accepted for symmetry
                with the OpenAI surface; used only when `recipe` is empty.
        """
        from .router import OrchestrationError, orchestrate  # lazy: breaks cycle

        name = recipe or (model[len("woollama/"):]
                          if model.startswith("woollama/") else model)
        if not name:
            raise ValueError("chat requires a 'recipe' (or 'woollama/<recipe>' model)")
        rec = recipes.get(name)
        if rec is None:
            raise ValueError(f"unknown recipe '{name}'")
        try:
            resp = await orchestrate(rec, messages, reg)
        except OrchestrationError as e:
            raise ValueError(e.message) from e
        return resp["choices"][0]["message"].get("content") or ""

    return Tool.from_function(chat, name="chat")


def register_reexported_tools(mcp: FastMCP, reg: Registry) -> None:
    """Re-export every discovered downstream tool (namespaced) onto tools/list,
    so an MCP client connecting to woollama sees the union of all configured
    servers' tools plus the `chat` verb — woollama as an MCP aggregator
    (decision #3, now realized).

    This MUST run after `reg.start_all()`: a server's tools are only known once
    its connection is up. That's why it's a *dynamic* registration (FastMCP
    advertises tools.listChanged, and the tools are in place before the server
    serves its first tools/list) — NOT a build-time concat, since the registry
    isn't started when `build_server` runs. Stdio drives this from
    `build_server`'s own lifespan; the mounted HTTP path (manage_registry=False)
    drives it from router's FastAPI lifespan, which owns the shared registry."""
    count = 0
    for mgr in reg.servers.values():
        for spec in mgr.tools:
            namespaced = f"{mgr.name}.{spec.name}"
            mcp.add_tool(_ProxyTool.build(
                namespaced,
                spec.description or "",
                spec.inputSchema or {"type": "object", "properties": {}},
                reg,
            ))
            count += 1
    log.info("re-exported %d downstream tool(s): %s",
             count, reg.all_tool_names())


def build_server(registry: Registry, *, manage_registry: bool = True) -> FastMCP:
    """Construct the FastMCP server: recipe prompts + the `chat` tool.

    `manage_registry` (default True): attach a lifespan that start_all/stop_all
    the registry on the serving loop and re-exports downstream tools — used by
    the stdio (`woollama mcp`) path, which owns the registry. Set False when the
    server is MOUNTED into another app (the HTTP path): there, router's FastAPI
    lifespan owns the shared registry and calls `register_reexported_tools`
    itself, so this server must NOT also start/stop it (double-start).

    Prompts snapshot the currently-loaded recipes — callers that need a
    specific recipe set should `recipes.reload()` (with WOOLLAMA_CONFIG_DIR)
    before building."""

    lifespan = None
    if manage_registry:
        @asynccontextmanager
        async def lifespan(server: FastMCP):
            # Started here (not eagerly) so the connection-owning tasks bind to
            # the same event loop that serves tool calls — see ServerManager.
            await registry.start_all()
            # Now the downstream tools are known: re-export them onto tools/list.
            register_reexported_tools(server, registry)
            try:
                yield
            finally:
                await registry.stop_all()

    mcp = FastMCP("woollama", lifespan=lifespan)
    for prompt in _recipe_prompts():
        mcp.add_prompt(prompt)
    mcp.add_tool(_chat_tool(registry))
    return mcp


def serve() -> None:
    """Entry point for `woollama mcp`: build the registry + server and run it
    over stdio. show_banner=False keeps stdout clean for the JSON-RPC stream
    (stdout is the protocol channel; logging goes to stderr)."""
    registry = build_registry()
    server = build_server(registry)
    server.run(transport="stdio", show_banner=False)
