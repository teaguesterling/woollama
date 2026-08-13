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
        # Round-2 addition: an opt-in stop-delay gate for race tests. When
        # `block_stop` is True, the /stop handler signals `stop_started` and
        # then blocks on `stop_release` before completing -- lets a test
        # deterministically land other work while a stop is in flight. Off by
        # default, so it never affects the original (verbatim) tests above.
        self.block_stop = False
        self.stop_started = threading.Event()
        self.stop_release = threading.Event()
        # Round-3 additions (opt-in, default off so the tests above are
        # unaffected): device-error + poll-timeout branches.
        #   fail_running      -> GET /running returns HTTP 500.
        #   running_bad_json  -> GET /running returns HTTP 200 with a
        #                        non-JSON body.
        #   start_no_register -> POST .../start returns 200 but does NOT
        #                        add the id to `running` (so a poll loop
        #                        never sees it -> load-timeout path).
        #   fail_stop         -> POST .../stop returns HTTP 500 and leaves
        #                        the id in `running` (stop failed; nothing
        #                        was actually torn down on the device).
        self.fail_running = False
        self.running_bad_json = False
        self.start_no_register = False
        self.fail_stop = False
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
                        if dev.fail_running:
                            self._json(500, {"error": "running failed"})
                            return
                        if dev.running_bad_json:
                            raw = b"not json"
                            self.send_response(200)
                            self.send_header("Content-Type", "application/json")
                            self.send_header("Content-Length", str(len(raw)))
                            self.end_headers()
                            self.wfile.write(raw)
                            return
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
                            self._json(500, {"error": "start failed"})
                            return
                        if not dev.start_no_register:
                            dev.running.add(mid)
                    self._json(200, {"ok": True})
                elif p.startswith(prefix) and p.endswith("/stop"):
                    mid = p[len(prefix):-len("/stop")]
                    if dev.block_stop:
                        dev.stop_started.set()
                        dev.stop_release.wait()
                    with dev._lock:
                        dev.calls.append(("stop", mid))
                        if dev.fail_stop:
                            self._json(500, {"error": "stop failed"})
                            return
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
    mgr.acquire("A")
    mgr.acquire("B")     # both serving -> not evictable
    with pytest.raises(pool.Backpressure):
        await mgr.ensure_loaded("C", pool_max=2)
    await mgr.aclose()


def _fake_clock():
    t = {"v": 0.0}
    def clock():
        t["v"] += 1.0
        return t["v"]
    return clock


# --- Round-2 regression: eviction mid-decision race ------------------------
#
# resolver.pick_eviction() snapshots "idle" (in_flight==0, queued==0) at the
# instant the evictor decides on a victim. But the evictor then *awaits* the
# device's /stop call, and during that await a racer can legitimately land a
# enqueue()/ensure_loaded()/acquire() sequence (the Gate's real call pattern)
# against the very model being torn down. Two invariants must hold across
# that window:
#   1. The racer's ensure_loaded() must not take the pre-lock fast path on
#      stale "still loaded" truth while the device is mid-teardown -- it must
#      block until the evictor is done, then re-check/reload for real.
#   2. The evictor's post-stop cleanup must never silently discard the
#      racer's in_flight/queued bookkeeping just because it happened to land
#      on the (about-to-be-popped) entry while the stop was in flight.
async def test_eviction_race_does_not_strand_or_lose_racer(device):
    import asyncio

    device.running.update({"A", "B"})
    device.block_stop = True
    mgr = pool.DeviceModelManager(device.url, poll_interval=0.01)
    try:
        await mgr.ensure_loaded("A")            # last_used older -> LRU victim
        await mgr.ensure_loaded("B")            # last_used newer

        evict_task = asyncio.create_task(mgr.ensure_loaded("C", pool_max=2))
        # Wait until the evictor's stop("A") request has actually landed on
        # the device and is being held there -- deterministic sync on real
        # state, no sleep-based timing guess.
        while not device.stop_started.is_set():
            await asyncio.sleep(0)

        # A racer arrives while A's stop() is in flight, mirroring the
        # Gate's real call pattern (enqueue before ensure_loaded, acquire
        # after).
        async def racer():
            mgr.enqueue("A")
            await mgr.ensure_loaded("A")
            mgr.dequeue("A")
            mgr.acquire("A")

        racer_task = asyncio.create_task(racer())
        await asyncio.sleep(0)   # let the racer run up to its blocking point

        device.stop_release.set()   # let the fake device finish stopping A
        await evict_task

        # Invariant 2: the evictor's cleanup must not have silently
        # discarded A's entry (and the racer's bookkeeping on it) just
        # because the stop was in flight when the pick was made.
        assert "A" in mgr._entries, "victim entry silently discarded mid-race"

        await racer_task

        # Invariant 1: the racer must end up bound to a genuinely
        # (re)loaded model, never a phantom / torn-down one.
        assert "A" in mgr.snapshot()
        assert "A" in device.running
        assert mgr.queued("A") == 0        # racer's dequeue balanced its enqueue
        assert mgr._entries["A"].in_flight == 1   # racer's acquire landed cleanly
    finally:
        # Always release the httpx connection pool, even on assertion
        # failure -- otherwise a still-open HTTP/1.1 keep-alive connection
        # leaves the single-threaded fake device blocked in
        # handle_one_request() forever, hanging fixture teardown (device.close()
        # -> HTTPServer.shutdown() waits for serve_forever() to notice the
        # shutdown flag, which it can't while stuck reading that socket).
        device.stop_release.set()   # in case we failed before releasing it
        await mgr.aclose()


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


# --- Round-3: device-error branches + Slot.release idempotency -------------

async def test_running_query_non_2xx_raises_device_error(device):
    """GET /running returning a non-2xx status (device-side failure, not a
    transport error) must surface as DeviceError, not propagate a raw HTTP
    status or silently treat it as "nothing running" (pool.py:172-173)."""
    device.fail_running = True
    mgr = pool.DeviceModelManager(device.url, poll_interval=0.01)
    try:
        with pytest.raises(pool.DeviceError):
            await mgr.ensure_loaded("Qwen/Coder")
    finally:
        await mgr.aclose()


async def test_running_query_bad_json_raises_device_error(device):
    """GET /running returning HTTP 200 with a non-JSON body must surface as
    DeviceError, not an uncaught ValueError from `.json()` (pool.py:176-177)."""
    device.running_bad_json = True
    mgr = pool.DeviceModelManager(device.url, poll_interval=0.01)
    try:
        with pytest.raises(pool.DeviceError):
            await mgr.ensure_loaded("Qwen/Coder")
    finally:
        await mgr.aclose()


async def test_start_poll_timeout_raises_device_error(device):
    """POST /start succeeds (200) but the model never shows up in /running --
    the post-start poll loop must give up at `load_timeout` and raise
    DeviceError with a "not running" message (pool.py:191-192), not hang or
    return silently."""
    device.start_no_register = True
    mgr = pool.DeviceModelManager(device.url, poll_interval=0.01, load_timeout=0.05)
    try:
        with pytest.raises(pool.DeviceError, match="not running"):
            await mgr.ensure_loaded("Qwen/Coder")
    finally:
        await mgr.aclose()


async def test_stop_failure_raises_device_error(device):
    """Evicting an idle LRU victim whose /stop call fails (non-2xx) must raise
    DeviceError (pool.py:200-201) and must NOT drop the victim from the
    device's running set -- the device never actually tore it down, so our
    bookkeeping must not claim otherwise."""
    device.running.update({"A", "B"})
    device.fail_stop = True
    mgr = pool.DeviceModelManager(device.url, poll_interval=0.01, clock=_fake_clock())
    try:
        await mgr.ensure_loaded("A")            # last_used older -> LRU victim
        await mgr.ensure_loaded("B")            # last_used newer
        with pytest.raises(pool.DeviceError):
            await mgr.ensure_loaded("C", pool_max=2)   # full -> evict A -> stop fails
        assert "A" in device.running, "stop failed; device state must be unchanged"
    finally:
        await mgr.aclose()


async def test_slot_release_is_idempotent(device):
    """`Slot.release` must be safe to call twice: only the FIRST call may drop
    the in-flight ref-count and release the semaphore permit (pool.py:218-219).
    A double-release would over-release the semaphore, letting two holders in
    under parallel=1 -- the invariant the streaming on_close path relies on
    (on_close can, in principle, race a caller's own cleanup)."""
    import asyncio

    device.running.add("X")
    mgr = pool.DeviceModelManager(device.url, poll_interval=0.01)
    gate = pool.Gate(mgr, parallel=1)
    try:
        slot = await gate.enter("X")
        await slot.release()
        await slot.release()   # must be a no-op, not a second semaphore release

        order = []

        async def worker(tag, hold):
            async with gate.slot("X"):
                order.append(f"enter-{tag}")
                await asyncio.sleep(hold)
                order.append(f"exit-{tag}")

        await asyncio.gather(worker("1", 0.02), worker("2", 0.0))
        # parallel=1 => still serialized. A double-release above would have
        # bumped the semaphore to 2 permits, letting both workers hold at once
        # (an interleaved order would appear here).
        assert order in (
            ["enter-1", "exit-1", "enter-2", "exit-2"],
            ["enter-2", "exit-2", "enter-1", "exit-1"],
        )
    finally:
        await mgr.aclose()
