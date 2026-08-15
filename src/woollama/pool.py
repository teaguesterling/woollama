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
import re
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import httpx

from . import config, resolver

if TYPE_CHECKING:
    from .inferencers import Inferencer

log = logging.getLogger("woollama.pool")


def _ok(status_code: int) -> bool:
    """True for any 2xx. The single success predicate for all three device
    endpoints (running/start/stop) -- keep them consistent."""
    return 200 <= status_code < 300


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


class DeviceBackend(Protocol):
    """A pluggable device-management transport: however a `DeviceModelManager` talks
    to its inferencer to discover/load/unload models. `RestBackend` is the built-in
    implementation for Tiiny's REST shape; later work adds config-defined REST
    protocols and other adapters behind this same seam."""

    async def list_loaded(self) -> set[str]: ...
    async def load(self, real_id: str) -> None: ...
    async def unload(self, real_id: str) -> None: ...


_METHOD_TOKEN = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def _resolve_method(configured: str | None, default: str) -> str:
    """Method parsed case-insensitively from an optional config override,
    falling back to `default` when unset OR unparseable (an invalid method
    string in config shouldn't fail startup; it just loses the override --
    logged so the typo isn't silently invisible). Mirrors Rust's
    `resolve_method` (pool.rs)."""
    if configured is None:
        return default
    m = configured.upper()
    if not _METHOD_TOKEN.match(m):
        log.warning("management_protocols: invalid HTTP method %r, falling back to %s",
                    configured, default)
        return default
    return m


def _get_dotted(v, path: str):
    """Dotted-path lookup into a parsed JSON value (`""` => the value itself;
    `"a.b"` => `v["a"]["b"]`) -- how `RestBackend.list_loaded` finds the
    running-models array/object inside an arbitrary config-defined response
    shape. Mirrors Rust's `get_dotted` (pool.rs)."""
    if path == "":
        return v
    cur = v
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


@dataclass(frozen=True)
class _CompiledEndpoint:
    """One HTTP call (`running`/`start`/`stop`) fully resolved against a base
    URL and default headers, but still carrying an `{id}` placeholder in
    `url`/`body`/header values -- substituted per-call by `render` (there's no
    id yet at construction time for `start`/`stop`, and never one for
    `running`). Mirrors Rust's `CompiledEndpoint` (pool.rs)."""
    method: str
    url: str
    body: str | None
    headers: dict[str, str] = field(default_factory=dict)

    def render(self, real_id: str | None) -> tuple[str, str, str | None, dict[str, str]]:
        def sub(s: str) -> str:
            return s.replace("{id}", real_id) if real_id is not None else s
        url = sub(self.url)
        body = sub(self.body) if self.body is not None else None
        headers = {k: sub(v) for k, v in self.headers.items()}
        return self.method, url, body, headers


def _compile_endpoint(base: str, default_headers: dict[str, str],
                       spec: config.EndpointSpec, default_method: str) -> _CompiledEndpoint:
    """Build a `_CompiledEndpoint` from a config `EndpointSpec`: substitute
    `{base}` into `url`/`body`/header values now (known at construction time),
    merge `spec.headers` OVER `default_headers` (an endpoint header key
    overrides the shared Bearer auth -- keyed CASE-INSENSITIVELY, since HTTP
    header names are; both maps are folded to lowercase keys so e.g. an
    endpoint header keyed `authorization` overrides a default `Authorization`,
    never sending both), and default `content-type: application/json` when a
    `body` is present and no `content-type` header was set either way. Mirrors
    Rust's `compile_endpoint` (pool.rs) -- INCLUDING the case-insensitive merge
    fix and the skip-just-the-offending-inferencer behavior it enables."""
    def sub_base(s: str) -> str:
        return s.replace("{base}", base)
    method = _resolve_method(spec.method, default_method)
    url = sub_base(spec.url)
    body = sub_base(spec.body) if spec.body is not None else None
    headers: dict[str, str] = {k.lower(): v for k, v in default_headers.items()}
    for k, v in spec.headers.items():
        headers[k.lower()] = sub_base(v)
    if body is not None and "content-type" not in headers:
        headers["content-type"] = "application/json"
    return _CompiledEndpoint(method=method, url=url, body=body, headers=headers)


class RestBackend:
    """The `DeviceBackend` for config-defined (and Tiiny's built-in) REST
    device-management shapes: three HTTP calls (list-loaded/start/stop), each
    independently templated (`RestBackend.from_spec`). `RestBackend.tiiny` is
    the Tiiny preset, expressed as a `from_spec` call with Tiiny's built-in
    endpoints. Mirrors Rust's `RestBackend` (pool.rs)."""

    def __init__(self, *, client: httpx.AsyncClient, owns_client: bool,
                 running: _CompiledEndpoint, start: _CompiledEndpoint, stop: _CompiledEndpoint,
                 running_path: str | None, running_id_field: str | None,
                 poll_interval: float, load_timeout: float,
                 clock: Callable[[], float]):
        self._client = client
        self._owns_client = owns_client
        self._running = running
        self._start = start
        self._stop = stop
        self._running_path = running_path
        self._running_id_field = running_id_field
        self._poll = poll_interval
        self._load_timeout = load_timeout
        self._clock = clock

    @classmethod
    def from_spec(cls, base_url: str, *,
                   default_headers: dict[str, str] | None = None,
                   running: config.EndpointSpec, start: config.EndpointSpec,
                   stop: config.EndpointSpec,
                   poll_interval: float = 0.5, load_timeout: float = 120.0,
                   client: httpx.AsyncClient | None = None,
                   clock: Callable[[], float] = time.monotonic) -> "RestBackend":
        """Build a `RestBackend` from a config `ProtocolSpec.Rest`'s three
        endpoints. `{base}` (-> `base_url` trimmed of a trailing `/`) is
        substituted into every `url`/`body`/header value now; `{id}` is
        substituted per-call (see `_CompiledEndpoint.render`). Per-op method
        defaults: GET for `running`, POST for `start`/`stop` (an explicit
        `method` on the spec overrides). No further `${VAR}` expansion happens
        here -- that already ran once over the whole `inferencers.toml` text
        at config-load time (`config._expand_env`), matching Rust's
        equally-upfront `expand_env` pass over the TOML text before parsing."""
        base = base_url.rstrip("/")
        owns_client = client is None
        client = client or httpx.AsyncClient(timeout=30.0)
        headers = dict(default_headers or {})
        return cls(
            client=client, owns_client=owns_client,
            running=_compile_endpoint(base, headers, running, "GET"),
            start=_compile_endpoint(base, headers, start, "POST"),
            stop=_compile_endpoint(base, headers, stop, "POST"),
            running_path=running.path, running_id_field=running.id_field,
            poll_interval=poll_interval, load_timeout=load_timeout, clock=clock,
        )

    @classmethod
    def tiiny(cls, management_url: str, *,
              headers: dict[str, str] | None = None,
              client: httpx.AsyncClient | None = None,
              poll_interval: float = 0.5, load_timeout: float = 120.0,
              clock: Callable[[], float] = time.monotonic) -> "RestBackend":
        """The Tiiny device-management REST shape (`GET {base}/api/v1/models/
        running`, `POST .../{id}/start`, `POST .../{id}/stop`), expressed as a
        `from_spec` call with Tiiny's built-in endpoints -- the preset every
        `management_protocol` resolution falls back to when an inferencer
        names none (or names `"tiiny"` explicitly)."""
        running = config.EndpointSpec(url="{base}/api/v1/models/running", path="running")
        start = config.EndpointSpec(url="{base}/api/v1/models/{id}/start")
        stop = config.EndpointSpec(url="{base}/api/v1/models/{id}/stop")
        return cls.from_spec(management_url, default_headers=headers,
                              running=running, start=start, stop=stop,
                              poll_interval=poll_interval, load_timeout=load_timeout,
                              client=client, clock=clock)

    async def _call(self, endpoint: _CompiledEndpoint, real_id: str | None) -> httpx.Response:
        """Issue one templated call: apply `endpoint.render(real_id)`'s
        method/url/body/headers to `self._client` and send it."""
        method, url, body, headers = endpoint.render(real_id)
        kwargs: dict = {"headers": headers}
        if body is not None:
            kwargs["content"] = body.encode()
        return await self._client.request(method, url, **kwargs)

    async def list_loaded(self) -> set[str]:
        try:
            r = await self._call(self._running, None)
        except httpx.HTTPError as exc:
            raise DeviceError(f"device unreachable: {exc}") from exc
        if not _ok(r.status_code):
            raise DeviceError(f"running query failed: {r.status_code} {r.text[:200]}")
        try:
            v = r.json()
        except ValueError as exc:
            raise DeviceError(f"running query: bad JSON: {exc}") from exc
        # `_get_dotted` returning `None` (key/path absent) is normal and means
        # "no running models" -- the tiiny back-compat case: a device response
        # with no "running" key at all (e.g. `{}`) must still resolve to an
        # empty set, not an error. But a path that IS present and resolves to
        # something other than a list is a config-typo signal (the author
        # pointed `path` at the wrong field/shape) -- that gets a loud
        # `DeviceError` naming the path and what it actually found, instead of
        # silently treating it as "no models" and surfacing a much more
        # confusing `load_timeout`-expiry "not running" error downstream.
        path = self._running_path or ""
        found = _get_dotted(v, path)
        if found is None:
            items = []
        elif isinstance(found, list):
            items = found
        else:
            raise DeviceError(
                f"running query: path '{path}' is present but not an array: "
                f"{str(found)[:200]}")
        running: set[str] = set()
        for item in items:
            if self._running_id_field:
                if isinstance(item, dict):
                    val = item.get(self._running_id_field)
                    if isinstance(val, str):
                        running.add(val)
            elif isinstance(item, str):
                running.add(item)
        return running

    async def load(self, real_id: str) -> None:
        try:
            r = await self._call(self._start, real_id)
        except httpx.HTTPError as exc:
            raise DeviceError(f"start {real_id}: unreachable: {exc}") from exc
        if not _ok(r.status_code):
            raise DeviceError(f"start {real_id} failed: {r.status_code} {r.text[:200]}")
        deadline = self._clock() + self._load_timeout
        while self._clock() < deadline:
            if real_id in await self.list_loaded():
                return
            await asyncio.sleep(self._poll)
        raise DeviceError(f"start {real_id}: not running after {self._load_timeout}s")

    async def unload(self, real_id: str) -> None:
        try:
            r = await self._call(self._stop, real_id)
        except httpx.HTTPError as exc:
            raise DeviceError(f"stop {real_id}: unreachable: {exc}") from exc
        if not _ok(r.status_code):
            raise DeviceError(f"stop {real_id} failed: {r.status_code} {r.text[:200]}")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


_RESERVED_PROTOCOL_NAMES = ("tiiny", "ollama")


def check_reserved_protocol_names(protocols: dict[str, config.ProtocolSpec]) -> None:
    """Warn (once, up front) if a config `[management_protocols.<name>]` block
    reuses a RESERVED built-in name (`tiiny`/`ollama`). That block is always
    shadowed by the built-in of the same name (see `build_backend`), so
    silently ignoring it would hide a config mistake. Mirrors Rust's
    reserved-name loop in `PoolRegistry.from_registry` (pool.rs): warn-and-
    ignore, not warn-and-use -- the built-in always wins."""
    for reserved in _RESERVED_PROTOCOL_NAMES:
        if reserved in protocols:
            log.warning(
                "management_protocols: config block '[management_protocols.%s]' "
                "is shadowed by the built-in '%s' protocol and will be ignored",
                reserved, reserved)


def build_backend(inf: "Inferencer", protocols: dict[str, config.ProtocolSpec], *,
                   headers: dict[str, str] | None = None,
                   poll_interval: float = 0.5, load_timeout: float = 120.0,
                   client: httpx.AsyncClient | None = None) -> "DeviceBackend | None":
    """Resolve one inferencer's `management_protocol` (default `"tiiny"` when
    unset) to a `DeviceBackend`: the built-in `"tiiny"` REST preset, a
    config-defined `[management_protocols.<name>]` REST shape (`RestBackend
    .from_spec`), or -- for `"ollama"` (built-in or a config block with
    `kind = "ollama"`) -- `None` (Task 4 lands `OllamaBackend`; until then this
    is a clearly-logged not-yet-implemented skip, not a hard failure, so it
    composes with the same skip-just-this-inferencer policy as an unresolvable
    name).

    Caller contract: `None` means "skip this ONE inferencer's pool" -- a
    `log.warning` naming the inferencer and protocol has already been emitted;
    the caller must NOT let this fail every other inferencer's pool. Mirrors
    the (already-merged, final-reviewed) Rust `PoolRegistry.from_registry`
    per-inferencer `match` (pool.rs), which fixed exactly that
    degrade-ALL-pools bug during review -- do not reproduce it here.

    `inf.management_url` must be set (the caller only invokes this for
    management-capable inferencers)."""
    name = inf.management_protocol or "tiiny"
    hdrs = headers or {}
    if name == "tiiny":
        return RestBackend.tiiny(inf.management_url, headers=hdrs,
                                  poll_interval=poll_interval, load_timeout=load_timeout,
                                  client=client)
    if name == "ollama":
        log.warning(
            "management_protocols: inferencer '%s': built-in 'ollama' protocol "
            "is not yet implemented -- skipping this inferencer's pool (other "
            "inferencers are unaffected)", inf.name)
        return None
    spec = protocols.get(name)
    if spec is None:
        log.warning(
            "management_protocols: inferencer '%s': unknown management_protocol "
            "%r -- skipping this inferencer (its device pool is disabled; other "
            "inferencers are unaffected)", inf.name, name)
        return None
    if isinstance(spec, config.RestProtocolSpec):
        return RestBackend.from_spec(inf.management_url, default_headers=hdrs,
                                      running=spec.running, start=spec.start, stop=spec.stop,
                                      poll_interval=poll_interval, load_timeout=load_timeout,
                                      client=client)
    # isinstance(spec, config.OllamaProtocolSpec): same not-yet-implemented
    # skip as the built-in "ollama" name above (Task 4 replaces both arms).
    log.warning(
        "management_protocols: inferencer '%s': protocol %r has kind=\"ollama\", "
        "which is not yet implemented -- skipping this inferencer's pool (other "
        "inferencers are unaffected)", inf.name, name)
    return None


class DeviceModelManager:
    def __init__(self, backend: DeviceBackend, *,
                 retry_after: float = 5.0,
                 clock: Callable[[], float] = time.monotonic):
        self._backend = backend
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
            running = await self._backend.list_loaded()
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
                # Close the fast-path race window *before* yielding on device
                # I/O: flip loaded off synchronously (no await between the
                # pick and this line) so a concurrent ensure_loaded(victim)
                # can no longer take the pre-lock fast path on stale "still
                # loaded" truth while we're mid-teardown -- it must now block
                # on _load_lock (which we hold) and re-check after we're done.
                self._entries[victim].loaded = False
                await self._backend.unload(victim)
                # Only discard the victim's bookkeeping if nothing referenced
                # it while the stop was in flight (a racer's enqueue()/
                # acquire() land directly on the entry, with no lock). Never
                # silently drop a nonzero in_flight/queued count -- leave the
                # entry as unloaded so the next ensure_loaded reloads it.
                ve = self._entries.get(victim)
                if ve is not None and ve.in_flight == 0 and ve.queued == 0:
                    self._entries.pop(victim, None)
            await self._backend.load(real_id)
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

    async def aclose(self) -> None:
        await self._backend.aclose()


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
        permit within queue_timeout and bump the in-flight ref-count.

        Note: `queue_timeout` bounds only the semaphore acquisition below, NOT
        the preceding `ensure_loaded` await -- time spent waiting on an
        in-progress model load (which serializes on the manager's global
        `_load_lock` and can poll up to `load_timeout`, default 120s) is
        unbounded by `queue_timeout` and is governed separately by
        `load_timeout`."""
        if self._queue_max is not None and self._manager.queued(real_id) >= self._queue_max:
            raise Backpressure(self._retry_after)
        self._manager.enqueue(real_id)
        try:
            await self._manager.ensure_loaded(real_id, pool_max=self._pool_max)
            sem = self._sem(real_id)
            try:
                # queue_timeout applies only to this acquire; see the docstring
                # note above about the load wait above not being bounded by it.
                await asyncio.wait_for(sem.acquire(), timeout=self._queue_timeout)
            except asyncio.TimeoutError:
                raise Backpressure(self._retry_after) from None
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
