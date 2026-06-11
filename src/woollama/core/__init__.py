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
import sys as _sys

from . import inference, inferencers, orchestrate  # noqa: F401  (engine — stays in core)

# Dist-split migration (docs/dist-split.md): the support modules now live at top
# level (`woollama.config` / `recipes` / `tooling` / `ollama_native`). Keep
# `woollama.core.<mod>` resolving to them during the migration — both attribute
# access (`from woollama.core import recipes`) and submodule import
# (`from woollama.core.tooling import X`) — by aliasing into sys.modules. Removed in
# the cleanup stage once all import sites are updated.
from .. import config, ollama_native, recipes, tooling  # noqa: F401
for _m in ("config", "ollama_native", "recipes", "tooling"):
    _sys.modules[f"{__name__}.{_m}"] = _sys.modules[f"woollama.{_m}"]

from .inference import (  # noqa: F401,E402
    InferenceError,
    complete,
    complete_stream,
    complete_sync,
)
from .inferencers import Inferencer, InferencerError, ModelRegistry  # noqa: F401,E402
from .orchestrate import (
    orchestrate_events,  # noqa: F401,E402  (the `orchestrate()` drainer is core.orchestrate.orchestrate)
)
from ..recipes import Recipe, make_recipe  # noqa: F401,E402
from ..tooling import (  # noqa: F401,E402
    Capabilities,
    ToolProvider,
    ToolResult,
    ToolSpec,
    render_tool_result,
)
