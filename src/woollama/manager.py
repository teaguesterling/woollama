"""Long-lived MCP server connections.

`ServerManager` owns one MCP stdio connection in a dedicated asyncio task.
Other code talks to it via `call_tool(name, args)` and `list_tools()`, which
marshal the request through an internal queue so the connection's
async-context-manager lifetime stays inside one task — sidesteps the
anyio cancel-scope error that bites when you try to enter/exit the
stdio_client across FastAPI's split lifespan startup/shutdown tasks.

`Registry` holds one `ServerManager` per configured server and resolves
namespaced tool names (`<server>.<tool>`) to the right manager.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .tooling import ToolResult, ToolSpec

log = logging.getLogger("woollama.manager")

# Default wall-clock bound for one downstream tool call, overridable with
# $WOOLLAMA_TOOL_TIMEOUT (float seconds; <= 0 disables). Matches the HTTP
# passthrough's generous 180s rather than a tight interactive budget — the
# point is that a hung downstream server bounds the TURN instead of wedging
# the connection's worker task forever.
DEFAULT_TOOL_TIMEOUT = 180.0
ENV_TOOL_TIMEOUT = "WOOLLAMA_TOOL_TIMEOUT"


def _tool_timeout() -> float | None:
    raw = os.environ.get(ENV_TOOL_TIMEOUT)
    if raw is None or not raw.strip():
        return DEFAULT_TOOL_TIMEOUT
    try:
        val = float(raw)
    except ValueError:
        log.warning("ignoring non-numeric %s=%r; using default %ss",
                    ENV_TOOL_TIMEOUT, raw, DEFAULT_TOOL_TIMEOUT)
        return DEFAULT_TOOL_TIMEOUT
    return val if val > 0 else None


class ServerManager:
    """Owns one MCP stdio connection. Tool calls marshal through a queue."""

    def __init__(self, name: str, command: str, args: list[str],
                 env: dict[str, str] | None = None):
        self.name = name           # namespace prefix used in `<name>.<tool>`
        self.command = command
        self.args = args
        # mcp.json's per-server `env` block. Forwarded to the spawned server via
        # `StdioServerParameters.env` (the SDK merges it over its safe default
        # environment) — never via argv, where a secret would show up in `ps`.
        self.env = dict(env) if env else None
        self.tools: list[Any] = []  # populated on start via list_tools
        self.tool_timeout = _tool_timeout()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._error: Exception | None = None

    async def start(self) -> None:
        """Spawn the owning task; block until the connection is ready
        (initialized + tool list cached) or fails."""
        self._task = asyncio.create_task(self._run(), name=f"mcp-mgr:{self.name}")
        await self._ready.wait()
        if self._error:
            raise self._error
        log.info("server '%s' ready; %d tools: %s",
                 self.name, len(self.tools), [t.name for t in self.tools])

    async def stop(self) -> None:
        await self._queue.put(None)
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=5)
            except asyncio.TimeoutError:
                self._task.cancel()

    async def _run(self) -> None:
        try:
            params = StdioServerParameters(command=self.command, args=self.args,
                                           env=self.env)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as sess:
                    await sess.initialize()
                    tools_result = await sess.list_tools()
                    self.tools = list(tools_result.tools)
                    self._ready.set()

                    while True:
                        item = await self._queue.get()
                        if item is None:
                            break
                        op, args, future = item
                        try:
                            # The timeout lives HERE (around the op, inside the
                            # worker), not around the caller's future: timing
                            # out the future would leave the worker stuck on
                            # the hung call, wedging every queued call behind it.
                            result = await asyncio.wait_for(
                                op(sess, *args), timeout=self.tool_timeout)
                            if not future.done():
                                future.set_result(result)
                        except asyncio.TimeoutError:
                            if not future.done():
                                future.set_exception(TimeoutError(
                                    f"call on MCP server '{self.name}' timed "
                                    f"out after {self.tool_timeout}s "
                                    f"(${ENV_TOOL_TIMEOUT} to adjust)"))
                        except Exception as e:
                            if not future.done():
                                future.set_exception(e)
        except Exception as e:
            log.exception("server '%s' failed: %s", self.name, e)
            self._error = e
            self._ready.set()  # unblock start()

    async def call_tool(self, tool_name: str, args: dict) -> Any:
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._queue.put((
            lambda sess, n, a: sess.call_tool(n, a),
            (tool_name, args),
            future,
        ))
        return await future


class Registry:
    """The unified tool registry across all configured MCP servers.

    Tool names are exposed as `<server>.<tool>` to clients. Lookups parse
    that namespace and dispatch to the owning manager."""

    def __init__(self) -> None:
        self.servers: dict[str, ServerManager] = {}

    def add(self, mgr: ServerManager) -> None:
        if mgr.name in self.servers:
            raise ValueError(f"server '{mgr.name}' already registered")
        self.servers[mgr.name] = mgr

    async def start_all(self) -> None:
        for mgr in self.servers.values():
            await mgr.start()

    async def stop_all(self) -> None:
        for mgr in reversed(list(self.servers.values())):
            await mgr.stop()

    def all_tool_names(self) -> list[str]:
        """Every tool, namespaced. For diagnostics / `/v1/models` enrichment."""
        names: list[str] = []
        for mgr in self.servers.values():
            for t in mgr.tools:
                names.append(f"{mgr.name}.{t.name}")
        return names

    def lookup_tool(self, namespaced: str):
        """Returns (manager, bare_tool_name, ToolSpec) or raises KeyError."""
        if "." not in namespaced:
            raise KeyError(f"tool name must be namespaced as '<server>.<tool>': "
                           f"got '{namespaced}'")
        server, _, bare = namespaced.partition(".")
        mgr = self.servers.get(server)
        if mgr is None:
            raise KeyError(f"unknown server '{server}' in tool '{namespaced}'")
        for t in mgr.tools:
            if t.name == bare:
                return mgr, bare, t
        raise KeyError(f"tool '{bare}' not found on server '{server}'")

    def openai_tools_for(self, allow: list[str]) -> list[dict]:
        """Translate the recipe's namespaced allow-list to the OpenAI tool
        schema array, preserving the namespaced names (so the model emits
        tool_calls with the namespaced name we can route on)."""
        out: list[dict] = []
        for namespaced in allow:
            try:
                _, _, spec = self.lookup_tool(namespaced)
            except KeyError as e:
                log.warning("recipe references unknown tool '%s': %s",
                            namespaced, e)
                continue
            out.append({
                "type": "function",
                "function": {
                    "name": namespaced,           # the namespaced name flows out
                    "description": spec.description or "",
                    "parameters": spec.inputSchema or {"type": "object", "properties": {}},
                },
            })
        return out

    async def dispatch(self, namespaced: str, args: dict, *,
                       allow: list[str] | None = None) -> Any:
        """Route a model-emitted tool_call to the owning manager.

        `allow` (when not None) is enforced HERE, at dispatch time — a name
        outside it raises `PermissionError` before anything reaches a server.
        Offer-time filtering alone is not a boundary: a model can emit any
        configured tool name, not just the ones it was offered. `allow=None`
        keeps full dispatch for the MCP aggregator surface, which re-exports
        every configured tool by design (its access control is the surface
        auth, not a recipe allow-list)."""
        if allow is not None and namespaced not in allow:
            raise PermissionError(
                f"tool '{namespaced}' is not in the active allow-list "
                f"({sorted(allow)}); refusing to dispatch")
        mgr, bare, _ = self.lookup_tool(namespaced)
        return await mgr.call_tool(bare, args)


class RegistryToolProvider:
    """Adapts a `Registry` to the server-free `core.ToolProvider` seam, so the
    core recipe loop can dispatch MCP tools without importing `manager`. It emits
    LOSSLESS `ToolSpec` / `ToolResult` (carrying the downstream tool's
    output_schema + annotations, and the call result's structuredContent + isError
    + meta) — `core.render_tool_result` is the one place that narrows them. Keeps
    `Registry.dispatch` itself unchanged (the MCP proxy path still gets the raw
    `CallToolResult`).

    `allow` is the recipe's tool allow-list, enforced at DISPATCH time (via
    `Registry.dispatch`) in Python — defense-in-depth that holds regardless of
    whether the loop driving this provider (the compiled core) only dispatches
    offered tools. A blocked dispatch raises `PermissionError`, which the core
    renders back to the model as a tool error."""

    def __init__(self, registry: "Registry",
                 allow: list[str] | None = None) -> None:
        self._reg = registry
        self._allow = list(allow) if allow is not None else None

    def tools_for(self, allow):
        out = []
        for namespaced in allow:
            try:
                _, _, spec = self._reg.lookup_tool(namespaced)
            except KeyError as e:
                log.warning("recipe references unknown tool '%s': %s", namespaced, e)
                continue
            out.append(ToolSpec(
                name=namespaced,
                schema={
                    "type": "function",
                    "function": {
                        "name": namespaced,          # namespaced name flows out
                        "description": spec.description or "",
                        "parameters": spec.inputSchema
                        or {"type": "object", "properties": {}},
                    },
                },
                source_name=namespaced,
                output_schema=getattr(spec, "outputSchema", None),
                annotations=_dump(getattr(spec, "annotations", None)),
                meta=_dump(getattr(spec, "meta", None)),
            ))
        return out

    async def dispatch(self, name: str, args: dict) -> ToolResult:
        r = await self._reg.dispatch(name, args,        # raw CallToolResult
                                     allow=self._allow)
        return ToolResult(
            blocks=list(getattr(r, "content", None) or []),
            structured=getattr(r, "structuredContent", None),
            is_error=bool(getattr(r, "isError", False)),
            meta=_dump(getattr(r, "meta", None)),
        )


def _dump(obj):
    """A pydantic model → dict (for MCP annotations/meta), else passthrough/None."""
    dump = getattr(obj, "model_dump", None)
    return dump() if callable(dump) else obj
