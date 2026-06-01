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

HTTP/SSE transport and re-exporting discovered downstream tools (textops.*,
hello.*) onto tools/list are deliberately later slices; the `_chat_tools`
builder is structured so that re-export is a one-line concat (see decision #3
in docs/slice-e-mcp-server.md).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastmcp import FastMCP
from fastmcp.prompts import Prompt
from fastmcp.tools import Tool

from . import config, recipes
from .manager import Registry, ServerManager
from .router import OrchestrationError, orchestrate


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


def _chat_tools(reg: Registry) -> list[Tool]:
    """The tools/list builder. Today returns just the `chat` orchestration
    verb; structured so re-exporting discovered downstream tools is a one-line
    concat here later (decision #3) — e.g.
        return [chat] + _reexported_registry_tools(reg)
    """

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

    return [Tool.from_function(chat, name="chat")]


def build_server(registry: Registry) -> FastMCP:
    """Construct the FastMCP server: recipe prompts + the `chat` tool, with a
    lifespan that start_all/stop_all the registry on the serving loop.

    Prompts snapshot the currently-loaded recipes — callers that need a
    specific recipe set should `recipes.reload()` (with WOOLLAMA_CONFIG_DIR)
    before building."""

    @asynccontextmanager
    async def lifespan(_server: FastMCP):
        # Started here (not eagerly) so the connection-owning tasks bind to the
        # same event loop that serves tool calls — see manager.ServerManager.
        await registry.start_all()
        log.info("registry ready: %s", registry.all_tool_names())
        try:
            yield
        finally:
            await registry.stop_all()

    mcp = FastMCP("woollama", lifespan=lifespan)
    for prompt in _recipe_prompts():
        mcp.add_prompt(prompt)
    for tool in _chat_tools(registry):
        mcp.add_tool(tool)
    return mcp


def serve() -> None:
    """Entry point for `woollama mcp`: build the registry + server and run it
    over stdio. show_banner=False keeps stdout clean for the JSON-RPC stream
    (stdout is the protocol channel; logging goes to stderr)."""
    registry = build_registry()
    server = build_server(registry)
    server.run(transport="stdio", show_banner=False)
