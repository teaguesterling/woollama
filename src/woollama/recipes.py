"""Compat shim — this module moved to ``woollama.core.recipes`` (core extraction,
Phase 1; see ``docs/core-extraction.md``).

It *aliases* this name to the real module so the historical ``woollama.recipes``
import path keeps working with identical behavior (same module object → shared
state and monkeypatch targets; e.g. tests that ``monkeypatch.setattr(recipes,
"get", ...)`` patch the object the router actually calls). New code should import
from ``woollama.core.recipes``; this shim is removed in the final step.
"""
import sys as _sys

from woollama.core import recipes as _moved

_sys.modules[__name__] = _moved
