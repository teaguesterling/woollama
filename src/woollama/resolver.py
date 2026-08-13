"""Pure model-resolution + eviction decision logic (server-free).

`resolve` turns a virtual/bare model id into a concrete device model id using a
snapshot of what's loaded; `needs_eviction`/`pick_eviction` decide, from a
snapshot of per-model runtime state, whether and which idle model to unload to
make room. No I/O, no async, no server imports — unit-testable in isolation and
mirrorable into the Rust oracle later (docs/superpowers/specs/2026-08-13-model-pooling-design.md).
"""
from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass


class ResolveError(Exception):
    """A virtual model could not be resolved (e.g. `default` with nothing loaded
    and no configured fallback). The router maps this to a clear client error."""


@dataclass(frozen=True)
class PoolEntry:
    """Read-only snapshot of one loaded model's runtime state, for eviction."""
    model_id: str
    in_flight: int
    queued: int
    last_used: float


def resolve(bare: str, *, virtual: dict[str, str],
            loaded: Sequence[str], default: str | None) -> str:
    """Resolve the id after `provider/` to a concrete device model id.

    - `default` -> the currently-loaded model (`loaded[0]`, MRU first); if none
      loaded, the configured fallback `default`; else raise ResolveError.
    - a `bare` present in the `virtual` alias map -> its real id.
    - anything else -> `bare` unchanged (real-id passthrough, today's behavior).
    """
    if bare == "default":
        if loaded:
            return loaded[0]
        if default:
            return default
        raise ResolveError(
            "model 'default' requested but no model is loaded and no "
            "'virtual.default' fallback is configured for this inferencer")
    if bare in virtual:
        return virtual[bare]
    return bare


def needs_eviction(loaded: Collection[str], target: str,
                   pool_max: int | None) -> bool:
    """True iff a cap is set, is reached, and `target` is not already loaded."""
    if not pool_max or pool_max <= 0:
        return False
    if target in loaded:
        return False
    return len(loaded) >= pool_max


def pick_eviction(entries: Sequence[PoolEntry]) -> str | None:
    """The LRU model among idle entries (no in-flight, empty queue), or None if
    every loaded model is busy (never evict a serving/queued model)."""
    idle = [e for e in entries if e.in_flight == 0 and e.queued == 0]
    if not idle:
        return None
    return min(idle, key=lambda e: e.last_used).model_id
