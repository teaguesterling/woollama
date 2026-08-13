from __future__ import annotations

import pytest

from woollama import resolver
from woollama.resolver import PoolEntry


def test_resolve_real_id_passthrough():
    assert resolver.resolve("Qwen/Coder", virtual={}, loaded=[], default=None) == "Qwen/Coder"


def test_resolve_default_prefers_loaded():
    assert resolver.resolve("default", virtual={"default": "Cfg"},
                            loaded=["Loaded/A", "Loaded/B"], default="Cfg") == "Loaded/A"


def test_resolve_default_falls_back_to_config_when_none_loaded():
    assert resolver.resolve("default", virtual={"default": "Cfg"},
                            loaded=[], default="Cfg") == "Cfg"


def test_resolve_default_no_loaded_no_config_raises():
    with pytest.raises(resolver.ResolveError):
        resolver.resolve("default", virtual={}, loaded=[], default=None)


def test_resolve_alias_maps_to_real_id():
    assert resolver.resolve("coder", virtual={"coder": "Qwen/Coder"},
                            loaded=[], default=None) == "Qwen/Coder"


def test_resolve_unknown_alias_returns_itself():
    assert resolver.resolve("mystery", virtual={"coder": "Qwen/Coder"},
                            loaded=["X"], default=None) == "mystery"


def test_needs_eviction_only_when_capped_full_and_target_absent():
    assert resolver.needs_eviction({"a", "b"}, "c", pool_max=2) is True
    assert resolver.needs_eviction({"a", "b"}, "a", pool_max=2) is False   # already loaded
    assert resolver.needs_eviction({"a"}, "c", pool_max=2) is False        # room
    assert resolver.needs_eviction({"a", "b"}, "c", pool_max=None) is False  # no cap
    assert resolver.needs_eviction({"a", "b"}, "c", pool_max=0) is False


def test_pick_eviction_lru_idle():
    entries = [
        PoolEntry("old", in_flight=0, queued=0, last_used=1.0),
        PoolEntry("new", in_flight=0, queued=0, last_used=9.0),
    ]
    assert resolver.pick_eviction(entries) == "old"


def test_pick_eviction_never_picks_busy():
    entries = [
        PoolEntry("serving", in_flight=1, queued=0, last_used=1.0),
        PoolEntry("queued", in_flight=0, queued=2, last_used=2.0),
    ]
    assert resolver.pick_eviction(entries) is None


def test_pick_eviction_skips_busy_returns_idle():
    entries = [
        PoolEntry("serving", in_flight=1, queued=0, last_used=1.0),
        PoolEntry("idle", in_flight=0, queued=0, last_used=5.0),
    ]
    assert resolver.pick_eviction(entries) == "idle"
