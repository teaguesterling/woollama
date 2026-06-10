"""woollama.core — the server-free library surface.

The embeddable half of woollama: configuration, provider/model routing, and (in
later phases) the recipe orchestration loop and the MCP↔OpenAI tool seam.
Importing this package pulls in **no** FastAPI / uvicorn / MCP-server code — that
boundary is the whole point of the split and is enforced by
`tests/test_core_is_server_free.py`.

Phase 1 of the core extraction (see `docs/core-extraction.md`) relocates the
import-clean modules here. The historical top-level paths (`woollama.config`,
`woollama.inferencers`, `woollama.recipes`, `woollama.ollama_native`) still work
via alias shims; new code should import from `woollama.core`.
"""
from . import config, inference, inferencers, ollama_native, recipes  # noqa: F401
from .inference import InferenceError, complete, complete_stream  # noqa: F401
from .inferencers import Inferencer, InferencerError  # noqa: F401
from .recipes import Recipe  # noqa: F401
