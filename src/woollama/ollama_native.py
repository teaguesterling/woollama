"""Compat shim — this module moved to ``woollama.core.ollama_native`` (core
extraction, Phase 1; see ``docs/core-extraction.md``).

It *aliases* this name to the real module so the historical
``woollama.ollama_native`` import path keeps working with identical behavior
(same module object → shared state and monkeypatch targets). New code should
import from ``woollama.core.ollama_native``; this shim is removed in the final
step.
"""
import sys as _sys

from woollama.core import ollama_native as _moved

_sys.modules[__name__] = _moved
