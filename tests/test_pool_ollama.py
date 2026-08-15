"""Task 4: `OllamaBackend` -- the `DeviceBackend` for Ollama's native
management API (`GET /api/ps` list, `POST /api/generate` load/unload), and
its wiring into `pool.build_backend` (the built-in `"ollama"` name, and a
config `[management_protocols.<name>]` block with `kind = "ollama"`).

`OllamaMock` is a minimal in-process stand-in for `ollama serve`'s management
surface: `GET /api/ps` reflects a `loaded` set built purely from observed
`/api/generate` calls (`keep_alive: 0` numeric => evict; anything else =>
load), and records every `/api/generate` body verbatim so tests can assert
on exactly what was sent (model id, presence/absence/value of `keep_alive`).
Mirrors `RestMock` (tests/test_pool_protocols.py) adapted to Ollama's single
combined load/unload endpoint.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from woollama import config, pool, router
from woollama.inferencers import Inferencer


class OllamaMock:
    """In-process stand-in for Ollama's `:11434` management surface.
    `generate_calls` records every `/api/generate` JSON body in request
    order (raw -- including whether `keep_alive` was present at all, and its
    exact value). `loaded` is derived state: a `/api/generate` body with
    `keep_alive == 0` (numeric) evicts its `model`; any other call (missing
    `keep_alive`, or a non-zero value like `"5m"`) loads it -- so `GET
    /api/ps` reflects exactly what `OllamaBackend.load`/`unload` just did,
    the same way the real device would."""

    def __init__(self, *, fail_generate: bool = False, fail_ps: bool = False):
        self.loaded: set[str] = set()
        self.generate_calls: list[dict] = []
        self.fail_generate = fail_generate
        self.fail_ps = fail_ps
        self._lock = threading.Lock()
        mock = self

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
                if self.path == "/api/ps":
                    with mock._lock:
                        if mock.fail_ps:
                            self._json(500, {"error": "ps failed"})
                            return
                        models = [{"name": m} for m in sorted(mock.loaded)]
                    self._json(200, {"models": models})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(n) if n else b""
                if self.path != "/api/generate":
                    self._json(404, {"error": "not found"})
                    return
                try:
                    data = json.loads(raw.decode() or "{}")
                except ValueError:
                    data = {}
                with mock._lock:
                    mock.generate_calls.append(data)
                    if mock.fail_generate:
                        self._json(500, {"error": "generate failed"})
                        return
                    model = data.get("model")
                    # keep_alive == 0 (numeric, exactly what unload() sends)
                    # evicts; anything else (absent, or a duration string) loads.
                    if isinstance(model, str):
                        if data.get("keep_alive") == 0:
                            mock.loaded.discard(model)
                        else:
                            mock.loaded.add(model)
                self._json(200, {"done": True})

        self._srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._srv.server_address[1]}"

    def close(self):
        self._srv.shutdown()


def _device_inferencer(name: str, management_url: str, *,
                        management_protocol: str | None = None) -> Inferencer:
    return Inferencer(name=name, base_url="http://device.example/v1",
                       management_url=management_url,
                       management_protocol=management_protocol)


def _fake_clock():
    t = {"v": 0.0}

    def clock():
        t["v"] += 1.0
        return t["v"]
    return clock


# --- built-in "ollama" name: warm-up load (no keep_alive) + list_loaded ----

async def test_builtin_ollama_ensure_loaded_sends_warmup_generate_without_keep_alive():
    mock = OllamaMock()
    try:
        inf = _device_inferencer("device", mock.url, management_protocol="ollama")
        backend = pool.build_backend(inf, {})
        assert isinstance(backend, pool.OllamaBackend)
        mgr = pool.DeviceModelManager(backend)
        try:
            await mgr.ensure_loaded("qwen3")

            assert len(mock.generate_calls) == 1
            call = mock.generate_calls[0]
            assert call["model"] == "qwen3"
            assert "keep_alive" not in call, (
                "the keyless built-in 'ollama' protocol must omit keep_alive "
                "entirely so Ollama's own default applies")

            assert mock.loaded == {"qwen3"}
            assert await backend.list_loaded() == {"qwen3"}
            assert mgr.snapshot() == ["qwen3"]
        finally:
            await mgr.aclose()
    finally:
        mock.close()


async def test_list_loaded_parses_multiple_models_from_api_ps():
    mock = OllamaMock()
    mock.loaded = {"qwen3", "llama3"}
    try:
        backend = pool.OllamaBackend(mock.url)
        try:
            assert await backend.list_loaded() == {"qwen3", "llama3"}
        finally:
            await backend.aclose()
    finally:
        mock.close()


# --- eviction: victim gets /api/generate with keep_alive: 0 (numeric) ------

async def test_eviction_sends_keep_alive_zero_numeric_to_victim():
    mock = OllamaMock()
    try:
        inf = _device_inferencer("device", mock.url, management_protocol="ollama")
        backend = pool.build_backend(inf, {})
        mgr = pool.DeviceModelManager(backend, clock=_fake_clock())
        try:
            await mgr.ensure_loaded("a", pool_max=1)   # last_used older
            await mgr.ensure_loaded("b", pool_max=1)   # full -> evict LRU idle (a)

            assert mgr.snapshot() == ["b"]
            assert mock.loaded == {"b"}
            victim_calls = [c for c in mock.generate_calls if c.get("model") == "a"]
            assert any(c.get("keep_alive") == 0 for c in victim_calls), (
                "the evicted victim must get an /api/generate call with "
                "numeric keep_alive: 0")
        finally:
            await mgr.aclose()
    finally:
        mock.close()


# --- config kind="ollama" forwards its own keep_alive -----------------------

async def test_config_kind_ollama_forwards_configured_keep_alive_on_load():
    mock = OllamaMock()
    try:
        protocols = {"oll": config.OllamaProtocolSpec(keep_alive="5m")}
        inf = _device_inferencer("device", mock.url, management_protocol="oll")
        backend = pool.build_backend(inf, protocols)
        assert isinstance(backend, pool.OllamaBackend)
        mgr = pool.DeviceModelManager(backend)
        try:
            await mgr.ensure_loaded("m1")
            assert mock.generate_calls[-1]["model"] == "m1"
            assert mock.generate_calls[-1]["keep_alive"] == "5m"
        finally:
            await mgr.aclose()
    finally:
        mock.close()


# --- direct unit coverage of the OllamaBackend wire shapes ------------------

async def test_load_directly_posts_generate_with_model_and_keep_alive():
    mock = OllamaMock()
    try:
        backend = pool.OllamaBackend(mock.url, keep_alive="10m")
        try:
            await backend.load("m1")
            assert mock.generate_calls == [{"model": "m1", "keep_alive": "10m"}]
        finally:
            await backend.aclose()
    finally:
        mock.close()


async def test_unload_directly_posts_generate_with_numeric_keep_alive_zero():
    mock = OllamaMock()
    mock.loaded = {"m1"}
    try:
        backend = pool.OllamaBackend(mock.url)
        try:
            await backend.unload("m1")
            assert mock.generate_calls == [{"model": "m1", "keep_alive": 0}]
            assert mock.loaded == set()
        finally:
            await backend.aclose()
    finally:
        mock.close()


async def test_generate_failure_raises_device_error():
    mock = OllamaMock(fail_generate=True)
    try:
        backend = pool.OllamaBackend(mock.url)
        try:
            try:
                await backend.load("m1")
            except pool.DeviceError:
                pass
            else:
                raise AssertionError("expected DeviceError on a non-2xx /api/generate")
        finally:
            await backend.aclose()
    finally:
        mock.close()


async def test_list_loaded_failure_raises_device_error():
    mock = OllamaMock(fail_ps=True)
    try:
        backend = pool.OllamaBackend(mock.url)
        try:
            try:
                await backend.list_loaded()
            except pool.DeviceError:
                pass
            else:
                raise AssertionError("expected DeviceError on a non-2xx /api/ps")
        finally:
            await backend.aclose()
    finally:
        mock.close()


async def test_unreachable_ollama_device_raises_device_error():
    backend = pool.OllamaBackend("http://127.0.0.1:1")
    try:
        try:
            await backend.list_loaded()
        except pool.DeviceError:
            pass
        else:
            raise AssertionError("expected DeviceError for an unreachable device")
    finally:
        await backend.aclose()


# --- lifespan wiring: kind="ollama" resolves to a real pool ----------------

async def test_lifespan_builds_pool_for_kind_ollama_inferencer(monkeypatch, tmp_path):
    mock = OllamaMock()
    try:
        monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("WOOLLAMA_STATE_DIR", str(tmp_path / "state"))
        (tmp_path / "mcp.json").write_text('{"mcpServers": {}}')
        (tmp_path / "inferencers.toml").write_text(
            '[inferencers.dev]\n'
            'base_url="http://dev/v1"\n'
            f'management_url="{mock.url}"\n'
            'management_protocol="oll"\n'
            '\n'
            '[management_protocols.oll]\n'
            'kind="ollama"\n'
            'keep_alive="5m"\n')

        saved_path = router.conversation_store._path
        assert "dev" not in router._pools
        try:
            async with router.lifespan(router.app):
                assert "dev" in router._pools, \
                    "a kind=\"ollama\" management_protocol must build a real pool"
                mgr, _gate = router._pools["dev"]
                assert isinstance(mgr._backend, pool.OllamaBackend)
                assert mgr._backend._keep_alive == "5m"
            assert router._pools == {}, "lifespan teardown must close and clear every pool"
        finally:
            router._pools.clear()
            router.conversation_store._path = saved_path
    finally:
        mock.close()


async def test_lifespan_builds_pool_for_builtin_ollama_name(monkeypatch, tmp_path):
    mock = OllamaMock()
    try:
        monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("WOOLLAMA_STATE_DIR", str(tmp_path / "state"))
        (tmp_path / "mcp.json").write_text('{"mcpServers": {}}')
        (tmp_path / "inferencers.toml").write_text(
            '[inferencers.dev]\n'
            'base_url="http://dev/v1"\n'
            f'management_url="{mock.url}"\n'
            'management_protocol="ollama"\n')

        saved_path = router.conversation_store._path
        assert "dev" not in router._pools
        try:
            async with router.lifespan(router.app):
                assert "dev" in router._pools
                mgr, _gate = router._pools["dev"]
                assert isinstance(mgr._backend, pool.OllamaBackend)
                assert mgr._backend._keep_alive is None
            assert router._pools == {}
        finally:
            router._pools.clear()
            router.conversation_store._path = saved_path
    finally:
        mock.close()
