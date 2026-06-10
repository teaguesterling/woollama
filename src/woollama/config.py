"""Compat shim — this module moved to ``woollama.core.config`` (core extraction,
Phase 1; see ``docs/core-extraction.md``).

It *aliases* this name to the real module so the historical ``woollama.config``
import path keeps working with **identical** behavior — same module object, so
module-level state and monkeypatch targets are shared, not copied. New code
should import from ``woollama.core.config``; this shim is removed in the final
extraction step.
"""
import sys as _sys

from woollama.core import config as _moved

_sys.modules[__name__] = _moved
