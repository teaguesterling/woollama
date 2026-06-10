"""woollama.core — the server-free library surface.

The embeddable half of woollama: configuration, provider/model routing, the
recipe orchestration loop, and the MCP↔OpenAI tool seam. Importing this package
pulls in **no** FastAPI / uvicorn / MCP-server code — that boundary is the whole
point of the split and is enforced by `tests/test_core_is_server_free.py`.

An embedder (e.g. lackpy) implements `ToolProvider` over its own tools, builds a
recipe, and calls `orchestrate(...)` / `complete(...)`; the historical top-level
paths (`woollama.config`, `woollama.inferencers`, `woollama.recipes`,
`woollama.ollama_native`) still work via alias shims (see `docs/core-extraction.md`).
"""
from . import config, inference, inferencers, ollama_native, orchestrate, recipes  # noqa: F401
from .inference import (  # noqa: F401
    InferenceError,
    complete,
    complete_stream,
    complete_sync,
)
from .inferencers import Inferencer, InferencerError, ModelRegistry  # noqa: F401
from .orchestrate import (
    orchestrate_events,  # noqa: F401  (the `orchestrate()` drainer is core.orchestrate.orchestrate)
)
from .recipes import Recipe, make_recipe  # noqa: F401
from .tooling import (  # noqa: F401
    Capabilities,
    ToolProvider,
    ToolResult,
    ToolSpec,
    render_tool_result,
)
