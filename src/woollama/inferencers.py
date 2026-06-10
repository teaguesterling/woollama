"""Compat shim — this module moved to ``woollama.core.inferencers`` (core
extraction, Phase 1; see ``docs/core-extraction.md``).

It *aliases* this name to the real module so the historical
``woollama.inferencers`` import path keeps working with identical behavior (same
module object → shared state and monkeypatch targets). New code should import
from ``woollama.core.inferencers``; this shim is removed in the final step.
"""
import sys as _sys

from woollama.core import inferencers as _moved

_sys.modules[__name__] = _moved
