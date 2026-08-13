# Model Pooling, Request Queuing & On-Demand Loading — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make woollama device-aware — load models on demand, serialize/queue requests around a backend's real concurrency limit, and expose stable virtual model names — so a client never sees a bare 503-for-not-loaded or a wedge.

**Architecture:** Three units. A pure `resolver` module (virtual-model resolution + eviction-candidate selection, no I/O). A stateful `pool` module with `DeviceModelManager` (async actor owning loaded-model state + device I/O, modeled on `manager.ServerManager`) and `Gate` (per-model `asyncio.Semaphore` + queue-depth backpressure). The router wires them into the `/v1/chat/completions` passthrough only; the Rust `/v1/responses` core path is untouched this branch.

**Tech Stack:** Python 3.12, asyncio, httpx (async), FastAPI, pytest + pytest-asyncio (`asyncio_mode="auto"`). No Rust/maturin changes.

**Spec:** `docs/superpowers/specs/2026-08-13-model-pooling-design.md` — read it alongside this plan.

## Global Constraints

- **Backward compatible / additive only.** An inferencer with no `management_url` behaves exactly as today (stateless passthrough). Existing `inferencers.toml` files must be unaffected. Every existing test must still pass.
- **No Rust changes.** The Rust engine parses `inferencers.toml` independently (`woollama-engine/src/lib.rs:382-474`) and ignores keys it doesn't extract; there is no Python↔Rust field-parity test. The `/v1/responses` (core) path is **out of scope** this branch — pooling is wired into the Python `_passthrough` only. `pool.py`/`resolver.py` MUST NOT import `woollama.router` or FastAPI (keep `resolver` pure/server-free).
- **Device management API (verified live 2026-08-13):** base `management_url` = `http://<ip>:8800`. Auth header `Authorization: Bearer <auth_key>` (same key as the inferencer's `api_key_env`). `GET {management_url}/api/v1/models/running` → `{"object":"list","running":[<real_id>,...],"pending":[...],"instances":{...}}` — parse the top-level `"running"` list. Load/unload: `POST {management_url}/api/v1/models/{real_id}/start` and `/stop`. Real ids contain slashes (e.g. `Qwen/Qwen3-Coder-30B-A3B-Instruct`) — send them raw in the path.
- **`--parallel` default is 1** (matches the device). Per-model concurrency is `parallel`.
- **Backpressure over failure:** queue saturated (`queue_max`) or waited past `queue_timeout` → HTTP `503` with a `Retry-After` header. Device unreachable / `start` failed → HTTP `502`. Never a hang, never a bare not-loaded 503.
- **Eviction is queue-aware and conservative:** never evict a model with `in_flight > 0` or `queued > 0`; evict only the LRU idle model; only when `pool_max` is set and reached. `pool_max` unset ⇒ no cap, no eviction.
- **Test style:** async test functions need no decorator (`asyncio_mode="auto"`). Prefer real in-process threaded HTTP servers for device/upstream fakes (see `tests/test_routing.py`, `tests/test_router.py`). Do not add new dependencies.
- **Commits:** conventional-commit subject; end every commit message body with the two trailers:
  ```
  Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01MNbD4t3EqzQuXXoVZApnHu
  ```

---

### Task 1: Config keys + `Inferencer` fields

Add six additive, optional config keys and thread them through the parser and the registry merge. No behavior change yet — this task only makes the fields exist and round-trip.

**Files:**
- Modify: `src/woollama/inferencers.py` (dataclass `Inferencer` at :34-51; merge in `_registry` at :123-132)
- Modify: `src/woollama/config.py` (`load_inferencers` whitelist at :240-255)
- Test: `tests/test_inferencers.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `Inferencer` gains fields (all defaulted, frozen dataclass):
  `management_url: str | None = None`, `parallel: int = 1`, `pool_max: int | None = None`,
  `queue_max: int | None = None`, `queue_timeout: float = 30.0`,
  `virtual: dict = field(default_factory=dict)`.
- Produces: `config.load_inferencers()` now also emits, when present in the TOML, the keys
  `management_url` (str), `parallel`/`pool_max`/`queue_max` (int, not bool), `queue_timeout` (float),
  `virtual` (table of str→str). Absent keys stay absent (so the registry merge can tell unset from set).

- [ ] **Step 1: Write failing config-parse tests**

Add to `tests/test_config.py`:

```python
def test_load_inferencers_parses_pooling_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "inferencers.toml").write_text(
        '[inferencers.tiiny]\n'
        'base_url = "http://dev/v1"\n'
        'management_url = "http://dev:8800"\n'
        'parallel = 2\n'
        'pool_max = 3\n'
        'queue_max = 8\n'
        'queue_timeout = 45\n'
        'virtual = { default = "Qwen/Coder", coder = "Qwen/Coder" }\n'
    )
    spec = config.load_inferencers()["tiiny"]
    assert spec["management_url"] == "http://dev:8800"
    assert spec["parallel"] == 2
    assert spec["pool_max"] == 3
    assert spec["queue_max"] == 8
    assert spec["queue_timeout"] == 45.0
    assert spec["virtual"] == {"default": "Qwen/Coder", "coder": "Qwen/Coder"}


def test_load_inferencers_rejects_non_int_parallel(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "inferencers.toml").write_text(
        '[inferencers.x]\nbase_url="http://h/v1"\nparallel = true\n')
    import pytest
    with pytest.raises(ValueError, match="parallel"):
        config.load_inferencers()


def test_load_inferencers_rejects_non_table_virtual(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "inferencers.toml").write_text(
        '[inferencers.x]\nbase_url="http://h/v1"\nvirtual = ["a","b"]\n')
    import pytest
    with pytest.raises(ValueError, match="virtual"):
        config.load_inferencers()
```

Note `config` is already imported at the top of `tests/test_config.py`; if not, add `from woollama import config`.

- [ ] **Step 2: Run to verify they fail**

Run: `cd ~/Projects/woollama/trees/model-pooling && python -m pytest tests/test_config.py -k pooling_keys -q`
Expected: FAIL — `KeyError: 'management_url'` (keys not yet parsed).

- [ ] **Step 3: Add the whitelist+validation in `config.load_inferencers`**

In `src/woollama/config.py`, inside the `for name, entry in raw.items():` loop (currently ending ~:255), after the existing `discover` block and before `out[name] = spec`, add:

```python
        if "management_url" in entry:
            spec["management_url"] = str(entry["management_url"])
        for int_key in ("parallel", "pool_max", "queue_max"):
            if int_key in entry:
                v = entry[int_key]
                if isinstance(v, bool) or not isinstance(v, int):
                    raise ValueError(
                        f"inferencers.toml {path}: '{name}.{int_key}' must be an integer")
                spec[int_key] = v
        if "queue_timeout" in entry:
            v = entry["queue_timeout"]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(
                    f"inferencers.toml {path}: '{name}.queue_timeout' must be a number")
            spec["queue_timeout"] = float(v)
        if "virtual" in entry:
            v = entry["virtual"]
            if not isinstance(v, dict):
                raise ValueError(
                    f"inferencers.toml {path}: '{name}.virtual' must be a table")
            spec["virtual"] = {str(k): str(val) for k, val in v.items()}
```

- [ ] **Step 4: Run config tests to verify pass**

Run: `python -m pytest tests/test_config.py -q`
Expected: PASS (new + existing).

- [ ] **Step 5: Write failing registry-merge test**

Add to `tests/test_inferencers.py`:

```python
def test_registry_threads_pooling_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "inferencers.toml").write_text(
        '[inferencers.tiiny]\n'
        'base_url = "http://dev/v1"\n'
        'management_url = "http://dev:8800"\n'
        'parallel = 2\n'
        'pool_max = 3\n'
        'queue_max = 8\n'
        'queue_timeout = 45\n'
        'virtual = { default = "Qwen/Coder", coder = "Qwen/Coder" }\n'
    )
    inf = inferencers.get("tiiny")
    assert inf.management_url == "http://dev:8800"
    assert inf.parallel == 2
    assert inf.pool_max == 3
    assert inf.queue_max == 8
    assert inf.queue_timeout == 45.0
    assert inf.virtual == {"default": "Qwen/Coder", "coder": "Qwen/Coder"}


def test_registry_pooling_fields_default_when_absent():
    inf = inferencers.get("anthropic")     # a built-in, no pooling config
    assert inf.management_url is None
    assert inf.parallel == 1
    assert inf.pool_max is None
    assert inf.queue_max is None
    assert inf.queue_timeout == 30.0
    assert inf.virtual == {}
```

`inferencers` is already imported at the top of `tests/test_inferencers.py`.

- [ ] **Step 6: Run to verify failure**

Run: `python -m pytest tests/test_inferencers.py -k pooling -q`
Expected: FAIL — `AttributeError: 'Inferencer' object has no attribute 'management_url'`.

- [ ] **Step 7: Add the dataclass fields**

In `src/woollama/inferencers.py`, extend the `Inferencer` dataclass (after `model_patterns: tuple[str, ...] = ()` at :51):

```python
    # --- device-aware pooling (issue: model-pooling). All optional; absent =>
    # today's stateless passthrough. Consumed by the Python server layer
    # (pool.py / resolver.py), NOT by the Rust core's build_request. ---
    management_url: str | None = None   # device mgmt base (:8800); presence enables the pool
    parallel: int = 1                   # per-model concurrency slot size (device default 1)
    pool_max: int | None = None         # max concurrently-loaded models; None => no cap/eviction
    queue_max: int | None = None        # max queued requests per model before backpressure
    queue_timeout: float = 30.0         # seconds a request may wait before 503+Retry-After
    virtual: dict = field(default_factory=dict)  # alias -> real_id; reserved key 'default'
```

- [ ] **Step 8: Thread them through the `_registry` merge**

In `src/woollama/inferencers.py`, in the `reg[name] = Inferencer(...)` call inside the config-merge loop (:123-132), add these keyword arguments (inheriting from a `base` built-in when extended):

```python
            management_url=spec.get("management_url",
                                    base.management_url if base else None),
            parallel=spec.get("parallel", base.parallel if base else 1),
            pool_max=spec.get("pool_max", base.pool_max if base else None),
            queue_max=spec.get("queue_max", base.queue_max if base else None),
            queue_timeout=spec.get("queue_timeout",
                                   base.queue_timeout if base else 30.0),
            virtual=dict(spec.get("virtual") or (base.virtual if base else {})),
```

- [ ] **Step 9: Run the full suite**

Run: `python -m pytest tests/test_inferencers.py tests/test_config.py -q && python -m pytest -q`
Expected: all PASS (the whole suite — proves backward compatibility).

- [ ] **Step 10: Commit**

```bash
git add src/woollama/inferencers.py src/woollama/config.py tests/test_inferencers.py tests/test_config.py
git commit   # subject: "feat(config): additive pooling keys on Inferencer" + Global-Constraints trailers
```

---

### Task 2: `resolver` — pure virtual-model resolution + eviction pick

A new server-free module holding all pure decision logic. No async, no I/O, no imports beyond stdlib/dataclasses.

**Files:**
- Create: `src/woollama/resolver.py`
- Test: `tests/test_resolver.py`

**Interfaces:**
- Produces: `resolver.ResolveError(Exception)`.
- Produces: `@dataclass(frozen=True) resolver.PoolEntry(model_id: str, in_flight: int, queued: int, last_used: float)`.
- Produces: `resolver.resolve(bare: str, *, virtual: dict[str, str], loaded: Sequence[str], default: str | None) -> str`
  — `bare` is the model id after the first `/` (may itself contain slashes). Rules: `default` → `loaded[0]` if any loaded, else `default` param, else raise `ResolveError`; a `bare` present in `virtual` → its target; anything else → `bare` unchanged.
- Produces: `resolver.needs_eviction(loaded: Collection[str], target: str, pool_max: int | None) -> bool`.
- Produces: `resolver.pick_eviction(entries: Sequence[PoolEntry]) -> str | None` — LRU among idle (`in_flight==0 and queued==0`) entries; `None` if none idle.

- [ ] **Step 1: Write failing resolver tests**

Create `tests/test_resolver.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_resolver.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'woollama.resolver'`.

- [ ] **Step 3: Implement `resolver.py`**

Create `src/woollama/resolver.py`:

```python
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
```

- [ ] **Step 4: Run resolver tests**

Run: `python -m pytest tests/test_resolver.py -q`
Expected: PASS.

- [ ] **Step 5: Verify server-free**

Run: `python -c "import woollama.resolver; import sys; assert 'fastapi' not in sys.modules and 'woollama.router' not in sys.modules; print('server-free OK')"`
Expected: prints `server-free OK`.

- [ ] **Step 6: Commit**

```bash
git add src/woollama/resolver.py tests/test_resolver.py
git commit   # subject: "feat(resolver): pure virtual-model + eviction logic" + trailers
```

---

### Task 3: `DeviceModelManager` — the loaded-model actor

The stateful async actor that owns loaded-model state and does device I/O (`running`/`start`/`stop`). Tested against a real in-process fake device.

**Files:**
- Create: `src/woollama/pool.py`
- Test: `tests/test_pool.py`

**Interfaces:**
- Consumes: `resolver.PoolEntry`, `resolver.needs_eviction`, `resolver.pick_eviction`.
- Produces: `pool.DeviceError(Exception)` (→ 502), `pool.Backpressure(Exception)` with attribute `retry_after: float` (→ 503).
- Produces: `pool.DeviceModelManager`:
  - `__init__(self, management_url: str, *, headers: dict[str, str] | None = None, client: "httpx.AsyncClient | None" = None, poll_interval: float = 0.5, load_timeout: float = 120.0, retry_after: float = 5.0, clock: Callable[[], float] = time.monotonic)`
  - `async ensure_loaded(self, real_id: str, *, pool_max: int | None = None) -> None`
  - `acquire(self, real_id) / release(self, real_id) / enqueue(self, real_id) / dequeue(self, real_id)` — sync counter ops
  - `queued(self, real_id) -> int`
  - `snapshot(self) -> list[str]` — loaded ids, MRU (highest `last_used`) first
  - `async aclose(self) -> None`

- [ ] **Step 1: Write the fake device + first manager tests**

Create `tests/test_pool.py`:

```python
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from woollama import pool


class FakeDevice:
    """In-process stand-in for the device's :8800 model-management API.
    Routes: GET /api/v1/models/running, POST .../{id}/start, POST .../{id}/stop.
    `running` is a mutable set of real ids; `calls` records (verb, id)."""

    def __init__(self, running=(), fail_start=False):
        self.running = set(running)
        self.fail_start = fail_start
        self.calls: list[tuple[str, str]] = []
        self._lock = threading.Lock()
        dev = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a):
                pass

            def _json(self, status, obj):
                raw = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/api/v1/models/running":
                    with dev._lock:
                        self._json(200, {"object": "list",
                                         "running": sorted(dev.running),
                                         "pending": []})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0) or 0)
                if n:
                    self.rfile.read(n)
                p = self.path
                prefix = "/api/v1/models/"
                if p.startswith(prefix) and p.endswith("/start"):
                    mid = p[len(prefix):-len("/start")]
                    with dev._lock:
                        dev.calls.append(("start", mid))
                        if dev.fail_start:
                            self._json(500, {"error": "start failed"}); return
                        dev.running.add(mid)
                    self._json(200, {"ok": True})
                elif p.startswith(prefix) and p.endswith("/stop"):
                    mid = p[len(prefix):-len("/stop")]
                    with dev._lock:
                        dev.calls.append(("stop", mid))
                        dev.running.discard(mid)
                    self._json(200, {"ok": True})
                else:
                    self._json(404, {"error": "not found"})

        self._srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._srv.server_address[1]}"

    def close(self):
        self._srv.shutdown()


@pytest.fixture
def device():
    d = FakeDevice()
    yield d
    d.close()


async def test_ensure_loaded_starts_when_absent(device):
    mgr = pool.DeviceModelManager(device.url, poll_interval=0.01)
    await mgr.ensure_loaded("Qwen/Coder")
    assert ("start", "Qwen/Coder") in device.calls
    assert "Qwen/Coder" in device.running
    assert mgr.snapshot() == ["Qwen/Coder"]
    await mgr.aclose()


async def test_ensure_loaded_noop_when_already_loaded(device):
    device.running.add("Qwen/Coder")
    mgr = pool.DeviceModelManager(device.url, poll_interval=0.01)
    await mgr.ensure_loaded("Qwen/Coder")   # device already has it
    await mgr.ensure_loaded("Qwen/Coder")   # and again from our own state
    assert [c for c in device.calls if c[0] == "start"] == []
    await mgr.aclose()


async def test_concurrent_ensure_loaded_dedups_to_one_start(device):
    import asyncio
    mgr = pool.DeviceModelManager(device.url, poll_interval=0.01)
    await asyncio.gather(*[mgr.ensure_loaded("Qwen/Coder") for _ in range(5)])
    assert [c for c in device.calls if c == ("start", "Qwen/Coder")] == [("start", "Qwen/Coder")]
    await mgr.aclose()


async def test_start_failure_raises_device_error(device):
    device.fail_start = True
    mgr = pool.DeviceModelManager(device.url, poll_interval=0.01)
    with pytest.raises(pool.DeviceError):
        await mgr.ensure_loaded("Qwen/Coder")
    await mgr.aclose()


async def test_unreachable_device_raises_device_error():
    mgr = pool.DeviceModelManager("http://127.0.0.1:1", poll_interval=0.01,
                                  load_timeout=1.0)
    with pytest.raises(pool.DeviceError):
        await mgr.ensure_loaded("Qwen/Coder")
    await mgr.aclose()


async def test_evicts_lru_idle_at_capacity(device):
    device.running.update({"A", "B"})
    mgr = pool.DeviceModelManager(device.url, poll_interval=0.01,
                                  clock=_fake_clock())
    await mgr.ensure_loaded("A")            # last_used older
    await mgr.ensure_loaded("B")            # last_used newer
    await mgr.ensure_loaded("C", pool_max=2)   # full -> evict LRU idle (A)
    assert ("stop", "A") in device.calls
    assert "A" not in device.running
    assert "C" in device.running
    await mgr.aclose()


async def test_no_evict_when_all_busy_raises_backpressure(device):
    device.running.update({"A", "B"})
    mgr = pool.DeviceModelManager(device.url, poll_interval=0.01)
    await mgr.ensure_loaded("A")
    await mgr.ensure_loaded("B")
    mgr.acquire("A"); mgr.acquire("B")     # both serving -> not evictable
    with pytest.raises(pool.Backpressure):
        await mgr.ensure_loaded("C", pool_max=2)
    await mgr.aclose()


def _fake_clock():
    t = {"v": 0.0}
    def clock():
        t["v"] += 1.0
        return t["v"]
    return clock
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_pool.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'woollama.pool'`.

- [ ] **Step 3: Implement `DeviceModelManager` in `pool.py`**

Create `src/woollama/pool.py`:

```python
"""Device-aware runtime: the loaded-model actor + the request gate.

`DeviceModelManager` is one long-lived async object per management-capable
inferencer. It owns which models are loaded on the device and their per-model
runtime counters (in-flight, queued, last-used), loads/unloads on demand via the
device's :8800 API, and evicts an idle LRU model to make room. `Gate` (Task 4)
serializes and queues requests around it. Pure decision logic lives in
`resolver`; this module is the live I/O half and never enters the Rust core.
See docs/superpowers/specs/2026-08-13-model-pooling-design.md.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from . import resolver

log = logging.getLogger("woollama.pool")


class DeviceError(Exception):
    """Device management I/O failed (unreachable, or start/stop error). → HTTP 502."""


class Backpressure(Exception):
    """The pool cannot serve now (queue saturated, wait timed out, or capacity
    full with no idle model to evict). Carries a Retry-After hint. → HTTP 503."""

    def __init__(self, retry_after: float):
        super().__init__(f"backpressure; retry after {retry_after}s")
        self.retry_after = retry_after


@dataclass
class _Entry:
    loaded: bool
    in_flight: int
    queued: int
    last_used: float


class DeviceModelManager:
    def __init__(self, management_url: str, *,
                 headers: dict[str, str] | None = None,
                 client: httpx.AsyncClient | None = None,
                 poll_interval: float = 0.5, load_timeout: float = 120.0,
                 retry_after: float = 5.0,
                 clock: Callable[[], float] = time.monotonic):
        self._url = management_url.rstrip("/")
        self._headers = dict(headers or {})
        self._client = client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = client is None
        self._poll = poll_interval
        self._load_timeout = load_timeout
        self._retry_after = retry_after
        self._clock = clock
        self._entries: dict[str, _Entry] = {}
        self._load_lock = asyncio.Lock()

    # --- per-model counters (sync; called by the Gate) ---------------------
    def _entry(self, real_id: str) -> _Entry:
        e = self._entries.get(real_id)
        if e is None:
            e = _Entry(loaded=False, in_flight=0, queued=0, last_used=self._clock())
            self._entries[real_id] = e
        return e

    def acquire(self, real_id: str) -> None:
        self._entry(real_id).in_flight += 1

    def release(self, real_id: str) -> None:
        e = self._entries.get(real_id)
        if e:
            if e.in_flight > 0:
                e.in_flight -= 1
            e.last_used = self._clock()

    def enqueue(self, real_id: str) -> None:
        self._entry(real_id).queued += 1

    def dequeue(self, real_id: str) -> None:
        e = self._entries.get(real_id)
        if e and e.queued > 0:
            e.queued -= 1

    def queued(self, real_id: str) -> int:
        e = self._entries.get(real_id)
        return e.queued if e else 0

    def snapshot(self) -> list[str]:
        """Loaded model ids, most-recently-used first."""
        loaded = [(rid, e) for rid, e in self._entries.items() if e.loaded]
        loaded.sort(key=lambda kv: kv[1].last_used, reverse=True)
        return [rid for rid, _ in loaded]

    # --- load / evict ------------------------------------------------------
    async def ensure_loaded(self, real_id: str, *, pool_max: int | None = None) -> None:
        e = self._entries.get(real_id)
        if e and e.loaded:
            e.last_used = self._clock()
            return
        async with self._load_lock:
            e = self._entries.get(real_id)
            if e and e.loaded:
                e.last_used = self._clock()
                return
            running = await self._running()
            self._reconcile(running)
            if real_id in running:
                self._mark_loaded(real_id)
                return
            loaded_ids = {rid for rid, en in self._entries.items() if en.loaded}
            if resolver.needs_eviction(loaded_ids, real_id, pool_max):
                victim = resolver.pick_eviction([
                    resolver.PoolEntry(rid, en.in_flight, en.queued, en.last_used)
                    for rid, en in self._entries.items() if en.loaded
                ])
                if victim is None:
                    raise Backpressure(self._retry_after)
                await self._stop(victim)
                self._entries.pop(victim, None)
            await self._start(real_id)
            self._mark_loaded(real_id)

    def _mark_loaded(self, real_id: str) -> None:
        e = self._entry(real_id)
        e.loaded = True
        e.last_used = self._clock()

    def _reconcile(self, running: set[str]) -> None:
        """Fold device truth into our state: mark loaded what the device runs;
        clear the loaded flag on anything the device dropped from under us
        (keep counters — a request may still be accounted against it)."""
        for rid in running:
            self._entry(rid).loaded = True
        for rid, e in self._entries.items():
            if rid not in running:
                e.loaded = False

    async def _running(self) -> set[str]:
        try:
            r = await self._client.get(f"{self._url}/api/v1/models/running",
                                       headers=self._headers)
        except httpx.HTTPError as exc:
            raise DeviceError(f"device unreachable: {exc}") from exc
        if r.status_code != 200:
            raise DeviceError(f"running query failed: {r.status_code} {r.text[:200]}")
        try:
            return set(r.json().get("running") or [])
        except (ValueError, AttributeError) as exc:
            raise DeviceError(f"running query: bad JSON: {exc}") from exc

    async def _start(self, real_id: str) -> None:
        try:
            r = await self._client.post(
                f"{self._url}/api/v1/models/{real_id}/start", headers=self._headers)
        except httpx.HTTPError as exc:
            raise DeviceError(f"start {real_id}: unreachable: {exc}") from exc
        if r.status_code >= 400:
            raise DeviceError(f"start {real_id} failed: {r.status_code} {r.text[:200]}")
        deadline = self._clock() + self._load_timeout
        while self._clock() < deadline:
            if real_id in await self._running():
                return
            await asyncio.sleep(self._poll)
        raise DeviceError(f"start {real_id}: not running after {self._load_timeout}s")

    async def _stop(self, real_id: str) -> None:
        try:
            r = await self._client.post(
                f"{self._url}/api/v1/models/{real_id}/stop", headers=self._headers)
        except httpx.HTTPError as exc:
            raise DeviceError(f"stop {real_id}: unreachable: {exc}") from exc
        if r.status_code >= 400:
            raise DeviceError(f"stop {real_id} failed: {r.status_code} {r.text[:200]}")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
```

Note: the `clock`-based `load_timeout` loop uses the injected `clock`; with the real `time.monotonic` and a real device the poll waits real seconds. In `test_evicts_lru_idle_at_capacity` the fake clock advances on every call, so `_start`'s poll sees the model running on the first `_running()` check (the fake adds it synchronously on start) and returns before the deadline — verify this holds; if the fake-clock deadline math is too tight, pass `load_timeout=1000` in that test.

- [ ] **Step 4: Run manager tests**

Run: `python -m pytest tests/test_pool.py -q`
Expected: PASS.

- [ ] **Step 5: Verify server-free import**

Run: `python -c "import woollama.pool, sys; assert 'fastapi' not in sys.modules and 'woollama.router' not in sys.modules; print('pool server-free OK')"`
Expected: prints `pool server-free OK`.

- [ ] **Step 6: Commit**

```bash
git add src/woollama/pool.py tests/test_pool.py
git commit   # subject: "feat(pool): DeviceModelManager load/evict actor" + trailers
```

---

### Task 4: `Gate` — per-model semaphore + queue backpressure

Add the request gate to `pool.py`: serializes per model, queues, and applies backpressure. Closes the eviction race by holding a queue slot across `ensure_loaded`.

**Files:**
- Modify: `src/woollama/pool.py`
- Test: `tests/test_pool.py`

**Interfaces:**
- Consumes: `DeviceModelManager` (its `ensure_loaded`, `enqueue`/`dequeue`/`acquire`/`release`, `queued`).
- Produces: `pool.Slot` with `async release(self) -> None` (idempotent).
- Produces: `pool.Gate`:
  - `__init__(self, manager: DeviceModelManager, *, parallel: int = 1, queue_max: int | None = None, queue_timeout: float = 30.0, pool_max: int | None = None, retry_after: float = 5.0)`
  - `async enter(self, real_id: str) -> Slot` — enqueue → `ensure_loaded(pool_max=...)` → acquire semaphore within `queue_timeout` → bump in-flight; raises `Backpressure`/`DeviceError`.
  - `@asynccontextmanager async slot(self, real_id: str)` — `enter` then guaranteed `release`.

- [ ] **Step 1: Write failing gate tests**

Append to `tests/test_pool.py`:

```python
async def test_gate_serializes_per_model(device):
    import asyncio
    device.running.add("A")
    mgr = pool.DeviceModelManager(device.url, poll_interval=0.01)
    gate = pool.Gate(mgr, parallel=1)
    order = []

    async def worker(tag, hold):
        async with gate.slot("A"):
            order.append(f"enter-{tag}")
            await asyncio.sleep(hold)
            order.append(f"exit-{tag}")

    await asyncio.gather(worker("1", 0.05), worker("2", 0.0))
    # parallel=1 => the two critical sections do not interleave
    assert order in (
        ["enter-1", "exit-1", "enter-2", "exit-2"],
        ["enter-2", "exit-2", "enter-1", "exit-1"],
    )
    await mgr.aclose()


async def test_gate_queue_max_saturated_is_backpressure(device):
    import asyncio
    device.running.add("A")
    mgr = pool.DeviceModelManager(device.url, poll_interval=0.01)
    gate = pool.Gate(mgr, parallel=1, queue_max=1, queue_timeout=5.0)

    holder_ready = asyncio.Event()
    release_holder = asyncio.Event()

    async def holder():
        async with gate.slot("A"):
            holder_ready.set()
            await release_holder.wait()

    async def waiter():          # fills the single queue slot
        async with gate.slot("A"):
            pass

    h = asyncio.create_task(holder())
    await holder_ready.wait()
    w = asyncio.create_task(waiter())
    await asyncio.sleep(0.02)    # let waiter enqueue (queued == 1 == queue_max)
    with pytest.raises(pool.Backpressure):
        await gate.enter("A")    # third request: rejected immediately
    release_holder.set()
    await asyncio.gather(h, w)
    await mgr.aclose()


async def test_gate_queue_timeout_is_backpressure(device):
    import asyncio
    device.running.add("A")
    mgr = pool.DeviceModelManager(device.url, poll_interval=0.01)
    gate = pool.Gate(mgr, parallel=1, queue_timeout=0.05)

    release_holder = asyncio.Event()
    holder_ready = asyncio.Event()

    async def holder():
        async with gate.slot("A"):
            holder_ready.set()
            await release_holder.wait()

    h = asyncio.create_task(holder())
    await holder_ready.wait()
    with pytest.raises(pool.Backpressure):
        await gate.enter("A")    # waits past queue_timeout -> 503
    release_holder.set()
    await h
    await mgr.aclose()


async def test_gate_protects_serving_model_from_eviction(device):
    import asyncio
    device.running.update({"A", "B"})
    mgr = pool.DeviceModelManager(device.url, poll_interval=0.01)
    await mgr.ensure_loaded("A")
    await mgr.ensure_loaded("B")
    gate = pool.Gate(mgr, parallel=1, pool_max=2)

    ready = asyncio.Event()
    release = asyncio.Event()

    async def hold_A():
        async with gate.slot("A"):
            ready.set()
            await release.wait()

    async def hold_B():
        async with gate.slot("B"):
            await release.wait()

    ta = asyncio.create_task(hold_A())
    tb = asyncio.create_task(hold_B())
    await ready.wait()
    # both A and B are in-flight; loading C at capacity 2 must fail (nothing idle)
    with pytest.raises(pool.Backpressure):
        await gate.enter("C")
    assert ("stop", "A") not in device.calls and ("stop", "B") not in device.calls
    release.set()
    await asyncio.gather(ta, tb)
    await mgr.aclose()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_pool.py -k gate -q`
Expected: FAIL — `AttributeError: module 'woollama.pool' has no attribute 'Gate'`.

- [ ] **Step 3: Implement `Gate` + `Slot`**

Append to `src/woollama/pool.py` (add `from contextlib import asynccontextmanager` to the imports at the top):

```python
class Slot:
    """A held per-model concurrency slot. `release` is idempotent and drops both
    the in-flight ref-count and the semaphore permit."""

    def __init__(self, gate: "Gate", real_id: str):
        self._gate = gate
        self._real_id = real_id
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._gate._manager.release(self._real_id)
        self._gate._sem(self._real_id).release()


class Gate:
    def __init__(self, manager: DeviceModelManager, *, parallel: int = 1,
                 queue_max: int | None = None, queue_timeout: float = 30.0,
                 pool_max: int | None = None, retry_after: float = 5.0):
        self._manager = manager
        self._parallel = max(1, int(parallel))
        self._queue_max = queue_max
        self._queue_timeout = queue_timeout
        self._pool_max = pool_max
        self._retry_after = retry_after
        self._sems: dict[str, asyncio.Semaphore] = {}

    def _sem(self, real_id: str) -> asyncio.Semaphore:
        s = self._sems.get(real_id)
        if s is None:
            s = asyncio.Semaphore(self._parallel)
            self._sems[real_id] = s
        return s

    async def enter(self, real_id: str) -> Slot:
        """Full gating protocol for one request: reject early if the per-model
        queue is saturated; otherwise register a queue slot (which also protects
        the model from eviction), ensure it's loaded, then acquire a concurrency
        permit within queue_timeout and bump the in-flight ref-count."""
        if self._queue_max is not None and self._manager.queued(real_id) >= self._queue_max:
            raise Backpressure(self._retry_after)
        self._manager.enqueue(real_id)
        try:
            await self._manager.ensure_loaded(real_id, pool_max=self._pool_max)
            sem = self._sem(real_id)
            try:
                await asyncio.wait_for(sem.acquire(), timeout=self._queue_timeout)
            except asyncio.TimeoutError:
                raise Backpressure(self._retry_after)
        finally:
            self._manager.dequeue(real_id)
        self._manager.acquire(real_id)
        return Slot(self, real_id)

    @asynccontextmanager
    async def slot(self, real_id: str):
        s = await self.enter(real_id)
        try:
            yield s
        finally:
            await s.release()
```

Note the ordering guarantees eviction-safety: `queued > 0` holds from `enqueue` through `ensure_loaded` and the semaphore wait; the synchronous `dequeue()`→`acquire()` handoff (no `await` between them) means `in_flight > 0` takes over before any other coroutine runs — the model is continuously non-idle from enqueue through release.

- [ ] **Step 4: Run gate tests**

Run: `python -m pytest tests/test_pool.py -q`
Expected: PASS (all manager + gate tests).

- [ ] **Step 5: Commit**

```bash
git add src/woollama/pool.py tests/test_pool.py
git commit   # subject: "feat(pool): Gate per-model semaphore + backpressure" + trailers
```

---

### Task 5: Wire the pool into `_passthrough` + lifespan

Build one `(DeviceModelManager, Gate)` per management-capable inferencer at startup, and route `/v1/chat/completions` passthrough for those inferencers through resolve → gate → dispatch, mapping `Backpressure`→503+`Retry-After` and `DeviceError`→502. Non-management inferencers keep today's exact path.

**Files:**
- Modify: `src/woollama/router.py` (imports; module global `_pools`; lifespan build/teardown at :112-164; `_passthrough` at :655-677; add optional `on_close` to `_passthrough_stream` at :778-810)
- Test: `tests/test_router.py`

**Interfaces:**
- Consumes: `pool.DeviceModelManager`, `pool.Gate`, `pool.Backpressure`, `pool.DeviceError`, `resolver.resolve`; `Inferencer.management_url/parallel/pool_max/queue_max/queue_timeout/virtual` (Task 1).
- Produces: module global `router._pools: dict[str, tuple[pool.DeviceModelManager, pool.Gate]]` (empty unless a management-capable inferencer is configured).

- [ ] **Step 1: Write failing router tests**

Add to `tests/test_router.py` (top-level imports there already include `json`, `router`, `FakeRequest`, `HttpxResponseStub`):

```python
class _FakeManager:
    def __init__(self, loaded):
        self._loaded = list(loaded)
        self.ensured = []
    def snapshot(self):
        return list(self._loaded)
    async def ensure_loaded(self, real_id, *, pool_max=None):
        self.ensured.append(real_id)
    def acquire(self, real_id): pass
    def release(self, real_id): pass
    def enqueue(self, real_id): pass
    def dequeue(self, real_id): pass
    def queued(self, real_id): return 0


async def test_pooled_passthrough_resolves_default_and_forwards_real_id(monkeypatch, tmp_path):
    import httpx
    from woollama import pool, inferencers
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "inferencers.toml").write_text(
        '[inferencers.tiiny]\nbase_url="http://dev/v1"\n'
        'management_url="http://dev:8800"\nvirtual={ default = "Cfg/Fallback" }\n')

    captured = {}
    class _SpyClient:
        def __init__(self, *_a, **_kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_a): return None
        async def post(self, url, json=None, **_kw):
            captured["url"] = url; captured["body"] = json
            return HttpxResponseStub(200, {"choices": [{"message": {"content": "ok"}}]})
    monkeypatch.setattr(httpx, "AsyncClient", _SpyClient)

    mgr = _FakeManager(loaded=["Qwen/Coder"])
    gate = pool.Gate(mgr, parallel=1)
    monkeypatch.setitem(router._pools, "tiiny", (mgr, gate))
    try:
        await router.chat_completions(FakeRequest({
            "model": "tiiny/default",
            "messages": [{"role": "user", "content": "hi"}]}))
    finally:
        router._pools.pop("tiiny", None)
    assert captured["body"]["model"] == "Qwen/Coder"       # default -> loaded id
    assert mgr.ensured == ["Qwen/Coder"]


async def test_pooled_passthrough_backpressure_is_503_with_retry_after(monkeypatch, tmp_path):
    from woollama import pool
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    (tmp_path / "inferencers.toml").write_text(
        '[inferencers.tiiny]\nbase_url="http://dev/v1"\nmanagement_url="http://dev:8800"\n')

    class _BusyGate:
        async def enter(self, real_id):
            raise pool.Backpressure(7.0)
    monkeypatch.setitem(router._pools, "tiiny", (_FakeManager(loaded=["X"]), _BusyGate()))
    try:
        resp = await router.chat_completions(FakeRequest({
            "model": "tiiny/X", "messages": []}))
    finally:
        router._pools.pop("tiiny", None)
    assert resp.status_code == 503
    assert resp.headers.get("Retry-After") == "7"


async def test_non_pooled_inferencer_unchanged(monkeypatch):
    """ollama has no management_url -> stateless passthrough, prefix stripped."""
    import httpx
    captured = {}
    class _SpyClient:
        def __init__(self, *_a, **_kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_a): return None
        async def post(self, url, json=None, **_kw):
            captured["body"] = json
            return HttpxResponseStub(200, {"choices": [{"message": {"content": "ok"}}]})
    monkeypatch.setattr(httpx, "AsyncClient", _SpyClient)
    assert "ollama" not in router._pools
    await router.chat_completions(FakeRequest({
        "model": "ollama/qwen3:14b", "messages": [{"role": "user", "content": "hi"}]}))
    assert captured["body"]["model"] == "qwen3:14b"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_router.py -k "pooled or non_pooled_inferencer_unchanged" -q`
Expected: FAIL — `AttributeError: module 'woollama.router' has no attribute '_pools'`.

- [ ] **Step 3: Add imports + the `_pools` global**

In `src/woollama/router.py`, add to the `from . import (...)` block (:28-39) the names `pool` and `resolver`:

```python
from . import (
    auth,
    claude_code,
    config,
    conversations,
    core,
    inferencers,
    managed_agents,
    ollama_native,
    pool,
    recipes,
    resolver,
    responses,
)
```

Then, near the module-level `registry = Registry()` (:57), add:

```python
# One (manager, gate) per management-capable inferencer (those with a
# management_url). Populated by the lifespan; empty otherwise, in which case
# every inferencer takes the stateless passthrough below (unchanged behavior).
_pools: dict[str, tuple[pool.DeviceModelManager, pool.Gate]] = {}
```

- [ ] **Step 4: Build + tear down `_pools` in the lifespan**

In `src/woollama/router.py`, inside `lifespan` (after `register_reexported_tools(_mcp, registry)` at :128, before the conversation-store block):

```python
    # Device-aware pools: one manager+gate per inferencer that declares a
    # management_url. Reuses the inferencer's api key for the :8800 mgmt API.
    for _name, _inf in inferencers.all().items():
        if not _inf.management_url:
            continue
        try:
            _hdrs = _inf.headers()
        except inferencers.InferencerError:
            _hdrs = {}
        _mgr = pool.DeviceModelManager(_inf.management_url, headers=_hdrs)
        _gate = pool.Gate(_mgr, parallel=_inf.parallel, queue_max=_inf.queue_max,
                          queue_timeout=_inf.queue_timeout, pool_max=_inf.pool_max)
        _pools[_name] = (_mgr, _gate)
        log.info("pool ready: inferencer '%s' -> %s (parallel=%d, pool_max=%s, "
                 "queue_max=%s)", _name, _inf.management_url, _inf.parallel,
                 _inf.pool_max, _inf.queue_max)
```

And in the `finally:` at the end of the lifespan (currently `await registry.stop_all()` at :164), add before/after it:

```python
            for _mgr, _ in _pools.values():
                await _mgr.aclose()
            _pools.clear()
```

- [ ] **Step 5: Route the pooled path in `_passthrough`**

In `src/woollama/router.py`, replace the body of `_passthrough` (:655-677) so the pooled branch runs first, and the existing stateless logic is unchanged when there's no pool:

```python
async def _passthrough(body: dict) -> Response:
    """Forward `<provider>/<model>` straight to that inferencer's OpenAI-compat
    endpoint. For a management-capable inferencer (one with a pool) the request is
    resolved (virtual models), the target model is loaded on demand, and access is
    serialized/queued through the Gate; otherwise it's today's stateless relay."""
    body = dict(body)
    provider, _, bare = body["model"].partition("/")
    inf = inferencers.get(provider)        # caller verified it's known
    pooled = _pools.get(provider)
    if pooled is not None:
        return await _passthrough_pooled(inf, body, bare, *pooled)

    body["model"] = bare
    try:
        headers = inf.headers()
    except inferencers.InferencerError as e:
        return _error(str(e), "invalid_request_error", 400)
    if provider == "ollama" and ollama_native.wants_native(body):
        return await _passthrough_ollama_native(inf, body, headers)
    if body.get("stream"):
        return await _passthrough_stream(inf, body, headers)
    body["stream"] = False
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(inf.chat_url(), json=body, headers=headers)
        return JSONResponse(r.json(), status_code=r.status_code)


async def _passthrough_pooled(inf, body: dict, bare: str,
                              manager: pool.DeviceModelManager,
                              gate: pool.Gate) -> Response:
    """Resolve → load-on-demand → gate → dispatch, for a management-capable
    inferencer. Backpressure => 503+Retry-After; device errors => 502."""
    try:
        real = resolver.resolve(bare, virtual=inf.virtual,
                                loaded=manager.snapshot(),
                                default=inf.virtual.get("default"))
    except resolver.ResolveError as e:
        return _error(str(e), "invalid_request_error", 400)
    body["model"] = real
    try:
        headers = inf.headers()
    except inferencers.InferencerError as e:
        return _error(str(e), "invalid_request_error", 400)
    try:
        if body.get("stream"):
            slot = await gate.enter(real)
            try:
                return await _passthrough_stream(inf, body, headers,
                                                 on_close=slot.release)
            except BaseException:
                await slot.release()
                raise
        async with gate.slot(real):
            fwd = dict(body)
            fwd["stream"] = False
            async with httpx.AsyncClient(timeout=180) as c:
                r = await c.post(inf.chat_url(), json=fwd, headers=headers)
                return JSONResponse(r.json(), status_code=r.status_code)
    except pool.Backpressure as e:
        resp = _error("model busy; retry shortly", "server_error", 503)
        resp.headers["Retry-After"] = str(int(e.retry_after))
        return resp
    except pool.DeviceError as e:
        return _error(f"device error: {e}", "server_error", 502)
```

- [ ] **Step 6: Thread the streaming slot release through `_passthrough_stream`**

In `src/woollama/router.py`, change `_passthrough_stream`'s signature (:778) to accept an optional close hook, and call it wherever the stream/client is torn down. Replace the signature and the two teardown sites:

```python
async def _passthrough_stream(inf: inferencers.Inferencer, body: dict,
                              headers: dict, on_close=None) -> Response:
```

In the early-error branch (after `await client.aclose()` at :794), add:

```python
        if on_close is not None:
            await on_close()
```

And in the `relay()` generator's `finally` (:805-807), after `await client.aclose()`:

```python
        finally:
            await cm.__aexit__(None, None, None)
            await client.aclose()
            if on_close is not None:
                await on_close()
```

Because `Slot.release` is idempotent, the extra `await slot.release()` in `_passthrough_pooled`'s `except BaseException` is safe even if the stream already released.

- [ ] **Step 7: Run the router tests**

Run: `python -m pytest tests/test_router.py -q`
Expected: PASS (new pooled/backpressure/regression tests + all existing router tests).

- [ ] **Step 8: Run the whole suite**

Run: `python -m pytest -q`
Expected: all PASS. Confirms no regression across routing, inferencers, config, core-is-server-free.

- [ ] **Step 9: Commit**

```bash
git add src/woollama/router.py tests/test_router.py
git commit   # subject: "feat(router): pool-gated passthrough for management-capable inferencers" + trailers
```

---

## Deferred (explicitly NOT in this plan)

Per the spec's phasing — do not implement here, but named so a reviewer knows they're intentional gaps:
- `/v1/responses` (Rust core) pooling. The core path stays unchanged; `tiiny/default` and aliases resolve **only** on `/v1/chat/completions`. Wiring the same gate around the core call is a follow-on.
- Mirroring the config keys onto the Rust `Inferencer`/`Registry`. Unnecessary now: the Rust engine ignores unknown TOML keys and no parity test pins the field set.
- Explicit drain-before-evict scheduling, cross-client fairness/priority, `tiiny/auto` by request shape, multi-backend load balancing.
- Device-token refresh (headers are captured once at startup from `auth_data`); a long-lived server past `expire_time` would need a refresh hook.
- The live, env-gated integration test against a real device (single-call, mindful of `--parallel 1`). The fake-device tests are the hermetic coverage; the one thing they cannot confirm is that the device accepts a raw-slash id in the `/start` path — verify that against a real device before relying on eviction/load in production.

## Deployment note (Tiiny repo, outside this plan)

To actually enable pooling for the device, `~/.config/woollama/inferencers.toml` needs `management_url` + `virtual` on `[inferencers.tiiny]`, and `run-woollama.sh` must export the `:8800` base. That is Tiiny-side configuration, tracked separately from this woollama code change.

## Self-Review

**Spec coverage:**
- Three units — Resolver (Task 2), DeviceModelManager (Task 3), Gate (Task 4); hybrid placement (pure `resolver` vs stateful `pool`) ✓
- `ensure_loaded` + dedup + ref-count + evict-LRU-idle (Task 3) ✓
- Gate per-model semaphore, FIFO-within-model, `503`+`Retry-After` backpressure (Task 4/5) ✓
- Virtual models: `tiiny/<id>`, `tiiny/default` loaded-vs-fallback, aliases (Task 2, wired Task 5) ✓
- Queue-aware conservative eviction: never evict in-flight/queued; LRU idle; capacity-full→backpressure (Tasks 2–4) ✓
- Additive config keys threaded through dataclass + merge + parser (Task 1) ✓
- Data flow resolve→ensure_loaded→acquire→dispatch→release (Task 5 `_passthrough_pooled`) ✓
- Error table: not-loaded→load; saturated/timeout→503; start-fail→502; unreachable→502; capacity-full-no-idle→503; never-evict-in-use (Tasks 3–5) ✓
- Testing: pure Resolver/eviction tests, fake-device async tests, backward-compat suite run; live integration test deferred with rationale ✓
- MVP scope (passthrough path first, one management inferencer) ✓; `/v1/responses` + Rust mirror deferred with rationale ✓

**Placeholder scan:** none — every step carries concrete code and a concrete run command.

**Type consistency:** `resolve(bare, *, virtual, loaded, default)`, `PoolEntry(model_id, in_flight, queued, last_used)`, `needs_eviction(loaded, target, pool_max)`, `pick_eviction(entries)`, `DeviceModelManager` counter/`ensure_loaded`/`snapshot` signatures, `Gate.enter`/`slot`, `Slot.release`, and `router._pools` shape are used identically across Tasks 2→5. Device JSON parsed as top-level `running` list everywhere.
