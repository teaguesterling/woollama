"""The tool seam — how core's recipe loop talks to tools, without pretending MCP
tools and OpenAI function-calling are the same thing.

The model on the far end of the loop speaks OpenAI function-calling: a flat
`{name, description, parameters}` advertisement in, one result back. An MCP tool is
richer (output schema, typed/multimodal result blocks, `isError`, annotations).
We reconcile them with one rule:

    lossless at the boundary, lossy only at render.

A `ToolProvider` adapter (the MCP `Registry`, or lackpy's tool layer) mirrors its
tools faithfully into `ToolSpec` / `ToolResult` — it drops nothing. A per-model
`render_tool_result` then narrows a result to what *this* target can actually
receive. The loss becomes a property of the model (a text-only model can't see an
image), explicit and pluggable, not baked into the adapter.

NON-goals of this (stateless request/response) seam: MCP elicitation, sampling,
and progress — those need an interactive path, not the loop.
"""
from __future__ import annotations

import json
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSpec:
    """A tool advertised to the model. `schema` is the ONLY thing the model reads
    (an OpenAI function-tool dict). The remaining fields carry MCP metadata through
    LOSSLESSLY for the embedder's policy / permission / UX layer — the model never
    sees them."""
    name: str                              # OpenAI-legal name the model sees
    schema: dict                           # {"type":"function","function":{name,description,parameters}}
    source_name: str | None = None         # original (namespaced/MCP) name; dispatch maps back
    output_schema: dict | None = None
    annotations: dict | None = None        # readOnly/destructive/idempotent/openWorld hints
    meta: dict | None = None


@dataclass
class ToolResult:
    """A faithful mirror of an MCP `CallToolResult`. `blocks` are the original
    content blocks (text/image/audio/resource/...), untouched; `structured` is the
    structuredContent; `is_error` is carried, never dropped (fixing the silent
    tool-failure where an errored result looked successful)."""
    blocks: list = field(default_factory=list)
    structured: dict | None = None
    is_error: bool = False
    meta: dict | None = None


@dataclass(frozen=True)
class Capabilities:
    """What a render target (an inferencer/model) can receive. Default is
    text-only; richer renderers are selected when these are set."""
    accepts_image_parts: bool = False
    accepts_audio: bool = False
    accepts_structured: bool = False


DEFAULT_CAPS = Capabilities()


class ToolProvider(Protocol):
    """The seam the recipe loop depends on. The MCP `Registry` and lackpy's tool
    layer each implement it; neither flattens — they emit lossless `ToolSpec` /
    `ToolResult`, and the loop renders per target."""

    def tools_for(self, allow: Sequence[str]) -> list[ToolSpec]: ...
    def dispatch(self, name: str, args: dict) -> Awaitable[ToolResult]: ...


def _block_text(block: Any) -> str | None:
    t = getattr(block, "text", None)
    return t if isinstance(t, str) else None


def _block_dump(block: Any) -> Any:
    dump = getattr(block, "model_dump", None)
    return dump() if callable(dump) else block


def render_tool_result(result: ToolResult, *, caps: Capabilities = DEFAULT_CAPS) -> str:
    """Render a `ToolResult` into the `tool` message content for a target.

    Phase-1 renderer (text-only — `caps` reserved for image/audio targets, added
    non-breaking later since `ToolResult` already carries every block): join the
    text blocks; if there are none, JSON-dump the raw blocks (so a structured-only
    or non-text result still reaches the model as *something*); prefix `[tool
    error]` when the result is an error so a failed tool is no longer mistaken for
    a successful empty one."""
    parts = [t for b in result.blocks if (t := _block_text(b)) is not None]
    if parts:
        body = "\n".join(parts)
    elif result.blocks:
        body = json.dumps([_block_dump(b) for b in result.blocks], default=str)
    else:
        body = ""
    if result.is_error:
        body = f"[tool error] {body}" if body else "[tool error]"
    return body
