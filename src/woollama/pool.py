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
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx

from . import resolver

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
                # Close the fast-path race window *before* yielding on device
                # I/O: flip loaded off synchronously (no await between the
                # pick and this line) so a concurrent ensure_loaded(victim)
                # can no longer take the pre-lock fast path on stale "still
                # loaded" truth while we're mid-teardown -- it must now block
                # on _load_lock (which we hold) and re-check after we're done.
                self._entries[victim].loaded = False
                await self._stop(victim)
                # Only discard the victim's bookkeeping if nothing referenced
                # it while the stop was in flight (a racer's enqueue()/
                # acquire() land directly on the entry, with no lock). Never
                # silently drop a nonzero in_flight/queued count -- leave the
                # entry as unloaded so the next ensure_loaded reloads it.
                ve = self._entries.get(victim)
                if ve is not None and ve.in_flight == 0 and ve.queued == 0:
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
        if not _ok(r.status_code):
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
        if not _ok(r.status_code):
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
        if not _ok(r.status_code):
            raise DeviceError(f"stop {real_id} failed: {r.status_code} {r.text[:200]}")

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


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
