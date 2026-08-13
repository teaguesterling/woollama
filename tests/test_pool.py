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
