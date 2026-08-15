"""Task 3: config-driven `RestBackend.from_spec` + `management_protocol`
resolution in the lifespan.

Test A drives a fully config-defined ("custom") REST protocol end to end
through `ensure_loaded`, asserting the mock observed the configured start
call (method, path, body, header) and that the manager's view of loaded
models matches the mock. Test B is the back-compat path (no
`management_protocol` => tiiny preset). Test C asserts an unresolvable
`management_protocol` name is isolated to the offending inferencer --
`router.lifespan` skips (and warns about) just that one inferencer, while a
sibling inferencer with a valid protocol is still pooled normally. Test D
covers the case-insensitive header-merge fix: an endpoint header keyed
`authorization` (lowercase) must override the default Bearer auth, sending
exactly one `authorization` header line, never both.

`RestMock` mirrors woollama-server's `tests/common/mod.rs` `spawn_rest` fixture
(configurable running-response shape, id-in-path or id-in-body start/stop,
full recorded-request headers including every line for a header-count
assertion), adapted to Python/httpx -- extending the pattern established by
`tests/test_pool.py`'s `FakeDevice` (which only serves Tiiny's fixed shape).
"""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from woollama import config, pool, router
from woollama.inferencers import Inferencer

_UNSET = object()


class RestMock:
    """Configurable in-process REST device fixture. Serves a `running` list
    (bare id strings, or objects keyed by `running_id_key`) at
    `running_route`, and load/unload at `start_route`/`stop_route` -- the id
    either embedded in the route via an `{id}` placeholder, or read from a
    JSON request-body field (`id_body_field`). Records every inbound request:
    method, path, a last-value-wins `headers` dict, an ordered `headers_all`
    list of every header line (so a case-insensitive-merge bug that sends
    both `Authorization` and `authorization` as two wire lines is
    detectable), and the raw body text."""

    def __init__(self, *,
                 running_route: str = "/api/v1/models/running",
                 start_route: str = "/api/v1/models/{id}/start",
                 stop_route: str = "/api/v1/models/{id}/stop",
                 running_field: str = "running",
                 running_id_key: str | None = None,
                 id_body_field: str | None = None,
                 running_non_array: object = _UNSET,
                 fail_start: bool = False, fail_stop: bool = False):
        self.loaded: set[str] = set()
        self.requests: list[dict] = []
        self.running_route = running_route
        self.start_route = start_route
        self.stop_route = stop_route
        self.running_field = running_field
        self.running_id_key = running_id_key
        self.id_body_field = id_body_field
        # When set (not _UNSET), the `running` response serves this value
        # (deliberately not a list) instead of the normal loaded-ids array --
        # for testing RestBackend's present-but-non-array `path` error.
        self.running_non_array = running_non_array
        self.fail_start = fail_start
        self.fail_stop = fail_stop
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

            def _record(self, body: bytes) -> None:
                headers_all = [(k.lower(), v) for k, v in self.headers.items()]
                with mock._lock:
                    mock.requests.append({
                        "method": self.command,
                        "path": self.path,
                        "headers": dict(headers_all),
                        "headers_all": headers_all,
                        "body": body.decode() if body else "",
                    })

            def do_GET(self):
                self._record(b"")
                if self.path == mock.running_route:
                    with mock._lock:
                        if mock.running_non_array is not _UNSET:
                            payload = mock.running_non_array
                        else:
                            ids = sorted(mock.loaded)
                            payload = ([{mock.running_id_key: i} for i in ids]
                                       if mock.running_id_key else ids)
                        body = {mock.running_field: payload} if mock.running_field else payload
                    self._json(200, body)
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(n) if n else b""
                self._record(raw)
                mid = mock._match(self.path, raw, mock.start_route)
                if mid is not None:
                    with mock._lock:
                        if mock.fail_start:
                            self._json(500, {"error": "start failed"})
                            return
                        mock.loaded.add(mid)
                    self._json(200, {"ok": True})
                    return
                mid = mock._match(self.path, raw, mock.stop_route)
                if mid is not None:
                    with mock._lock:
                        if mock.fail_stop:
                            self._json(500, {"error": "stop failed"})
                            return
                        mock.loaded.discard(mid)
                    self._json(200, {"ok": True})
                    return
                self._json(404, {"error": "not found"})

        self._srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    def _match(self, path: str, raw_body: bytes, route: str) -> str | None:
        """Match `path`/`raw_body` against a configured start/stop route,
        returning the id if it matches -- mirrors the Rust fixture's
        `resolve_id`/`match_path_id`."""
        if self.id_body_field:
            if path != route:
                return None
            try:
                data = json.loads(raw_body.decode() or "{}")
            except ValueError:
                return None
            val = data.get(self.id_body_field)
            return val if isinstance(val, str) else None
        idx = route.find("{id}")
        if idx == -1:
            return path if path == route else None
        prefix, suffix = route[:idx], route[idx + 4:]
        if len(path) < len(prefix) + len(suffix) or not path.startswith(prefix) or not path.endswith(suffix):
            return None
        mid = path[len(prefix):len(path) - len(suffix)]
        return mid or None

    def requests_to(self, path_substr: str) -> list[dict]:
        with self._lock:
            return [r for r in self.requests if path_substr in r["path"]]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._srv.server_address[1]}"

    def close(self):
        self._srv.shutdown()


def header_count(req: dict, name: str) -> int:
    """How many times `name` (case-insensitive) appears as a header line on
    a recorded request -- catches a case-insensitive-merge bug that sends
    both the default and the endpoint override as two separate wire lines
    instead of one overriding the other."""
    name = name.lower()
    return sum(1 for k, _ in req["headers_all"] if k == name)


def _device_inferencer(name: str, management_url: str, *,
                        management_protocol: str | None = None,
                        api_key_env: str | None = None) -> Inferencer:
    return Inferencer(name=name, base_url="http://device.example/v1",
                       api_key_env=api_key_env, management_url=management_url,
                       management_protocol=management_protocol)


# --- Test A: custom, config-defined REST protocol --------------------------

async def test_build_backend_resolves_custom_protocol_and_drives_ensure_loaded():
    mock = RestMock(running_route="/status", start_route="/models/load",
                     stop_route="/models/unload", running_field="data",
                     running_id_key="id", id_body_field="model")
    try:
        running = config.EndpointSpec(url="{base}/status", path="data", id_field="id")
        start = config.EndpointSpec(url="{base}/models/load", method="POST",
                                     body='{"model": "{id}"}',
                                     headers={"X-Custom-Auth": "secret-token"})
        stop = config.EndpointSpec(url="{base}/models/unload", method="POST",
                                    body='{"model": "{id}"}')
        protocols = {"custom": config.RestProtocolSpec(running=running, start=start, stop=stop)}
        inf = _device_inferencer("device", mock.url, management_protocol="custom")

        backend = pool.build_backend(inf, protocols, poll_interval=0.01)
        assert backend is not None
        mgr = pool.DeviceModelManager(backend)
        try:
            await mgr.ensure_loaded("m1")

            loads = mock.requests_to("/models/load")
            assert len(loads) == 1, "exactly one start request"
            assert loads[0]["method"] == "POST"
            assert loads[0]["path"] == "/models/load"
            assert loads[0]["body"] == '{"model": "m1"}'
            assert loads[0]["headers"].get("x-custom-auth") == "secret-token"

            assert mock.loaded == {"m1"}
            assert mgr.snapshot() == ["m1"]
        finally:
            await mgr.aclose()
    finally:
        mock.close()


# --- Test B: back-compat (no management_protocol => tiiny preset) ----------

async def test_build_backend_back_compat_defaults_to_tiiny():
    mock = RestMock()   # defaults already match the Tiiny REST shape
    try:
        inf = _device_inferencer("device", mock.url, management_protocol=None)
        backend = pool.build_backend(inf, {}, poll_interval=0.01)
        assert backend is not None
        mgr = pool.DeviceModelManager(backend)
        try:
            await mgr.ensure_loaded("m1")
            assert mock.loaded == {"m1"}
            assert mgr.snapshot() == ["m1"]
        finally:
            await mgr.aclose()
    finally:
        mock.close()


# --- Test C: unknown protocol name is isolated to the offending inferencer -

async def test_lifespan_isolates_unknown_protocol_to_its_own_inferencer(monkeypatch, tmp_path, caplog):
    """A typo'd `management_protocol` on one inferencer must not disable
    pooling for every other device inferencer. `router.lifespan` (via
    `pool.build_backend`) skips (with a warning) only the offending
    inferencer -- absent from `router._pools` -- while a sibling inferencer
    with a resolvable protocol (here, the default `tiiny`) is still pooled
    normally."""
    monkeypatch.setenv("WOOLLAMA_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("WOOLLAMA_STATE_DIR", str(tmp_path / "state"))
    (tmp_path / "mcp.json").write_text('{"mcpServers": {}}')
    (tmp_path / "inferencers.toml").write_text(
        '[inferencers.bad]\nbase_url="http://dev/v1"\n'
        'management_url="http://bad.example:8800"\n'
        'management_protocol="nope"\n'
        '\n'
        '[inferencers.good]\nbase_url="http://dev/v1"\n'
        'management_url="http://good.example:8800"\n')

    saved_path = router.conversation_store._path
    assert "bad" not in router._pools and "good" not in router._pools
    try:
        with caplog.at_level(logging.WARNING, logger="woollama.pool"):
            async with router.lifespan(router.app):
                assert "bad" not in router._pools, \
                    "an inferencer with an unresolvable management_protocol must be skipped"
                assert "good" in router._pools, \
                    "a sibling inferencer with a resolvable management_protocol must still get a pool"
        assert router._pools == {}
        assert any("bad" in rec.message and "nope" in rec.message for rec in caplog.records), \
            "the warning must name both the offending inferencer and the unresolved protocol"
    finally:
        router._pools.clear()
        router.conversation_store._path = saved_path


# --- Test D: endpoint header overrides default auth case-insensitively -----

async def test_build_backend_endpoint_header_overrides_default_auth_case_insensitively(monkeypatch):
    monkeypatch.setenv("WOOLLAMA_TEST_DEVICE_TOKEN", "default-secret")
    mock = RestMock(running_route="/status", start_route="/models/{id}/start",
                     stop_route="/models/{id}/stop", running_field="running")
    try:
        running = config.EndpointSpec(url="{base}/status", path="running")
        start = config.EndpointSpec(
            url="{base}/models/{id}/start", method="POST",
            # Lowercase key, deliberately differing in case from the
            # `Authorization` header `Inferencer.headers()` produces -- must
            # still override, not coexist.
            headers={"authorization": "endpoint-secret"})
        stop = config.EndpointSpec(url="{base}/models/{id}/stop", method="POST")
        protocols = {"custom-auth": config.RestProtocolSpec(running=running, start=start, stop=stop)}

        inf = _device_inferencer("device", mock.url, management_protocol="custom-auth",
                                  api_key_env="WOOLLAMA_TEST_DEVICE_TOKEN")
        hdrs = inf.headers()
        assert hdrs == {"Authorization": "Bearer default-secret"}

        backend = pool.build_backend(inf, protocols, headers=hdrs, poll_interval=0.01)
        assert backend is not None
        mgr = pool.DeviceModelManager(backend)
        try:
            await mgr.ensure_loaded("m1")

            starts = mock.requests_to("/start")
            assert len(starts) == 1, "exactly one start request"
            assert header_count(starts[0], "authorization") == 1, (
                "exactly one authorization header line must reach the device -- not "
                "both the default Bearer and the endpoint override")
            assert starts[0]["headers"]["authorization"] == "endpoint-secret", (
                "the endpoint's own header must win over the default Bearer auth")
        finally:
            await mgr.aclose()
    finally:
        mock.close()


# --- Reserved-name shadow warning -------------------------------------------

def test_check_reserved_protocol_names_warns_on_shadowed_builtin(caplog):
    protocols = {
        "tiiny": config.RestProtocolSpec(
            running=config.EndpointSpec(url="{base}/x", path=""),
            start=config.EndpointSpec(url="{base}/y"),
            stop=config.EndpointSpec(url="{base}/z")),
    }
    with caplog.at_level(logging.WARNING, logger="woollama.pool"):
        pool.check_reserved_protocol_names(protocols)
    assert any("tiiny" in rec.message and "shadowed" in rec.message for rec in caplog.records)


def test_check_reserved_protocol_names_silent_when_no_reserved_names():
    protocols = {
        "custom": config.RestProtocolSpec(
            running=config.EndpointSpec(url="{base}/x", path=""),
            start=config.EndpointSpec(url="{base}/y"),
            stop=config.EndpointSpec(url="{base}/z")),
    }
    # Must not raise; nothing to assert on caplog since there's nothing to warn about.
    pool.check_reserved_protocol_names(protocols)


# --- The "ollama" arm: T4 fills this in; T3 must ship green on its own -----
#
# Per the task-3 ruling, resolving "ollama" (the built-in name, or any config
# block with kind="ollama") is left as a clearly-logged skip -- NOT a hard
# raise/whole-registry failure -- so it composes with the same
# skip-just-this-inferencer policy Test C exercises for an unresolvable name.
# Task 4 replaces both `build_backend` arms below with a real `OllamaBackend`.

def test_build_backend_builtin_ollama_name_is_skip_not_implemented(caplog):
    inf = _device_inferencer("device", "http://device.example:8800", management_protocol="ollama")
    with caplog.at_level(logging.WARNING, logger="woollama.pool"):
        backend = pool.build_backend(inf, {})
    assert backend is None
    assert any("device" in rec.message and "ollama" in rec.message for rec in caplog.records)


def test_build_backend_config_kind_ollama_is_skip_not_implemented(caplog):
    protocols = {"oll": config.OllamaProtocolSpec(keep_alive="5m")}
    inf = _device_inferencer("device", "http://device.example:8800", management_protocol="oll")
    with caplog.at_level(logging.WARNING, logger="woollama.pool"):
        backend = pool.build_backend(inf, protocols)
    assert backend is None
    assert any("device" in rec.message and "oll" in rec.message for rec in caplog.records)


# --- A present-but-non-array running path is a clear DeviceError -----------

async def test_ensure_loaded_errors_clearly_when_running_path_resolves_to_non_array():
    """A config `path` that resolves to a PRESENT-but-non-array value (e.g. a
    typo landing on an object or scalar instead of the loaded-models array)
    must surface as a clear `DeviceError` naming the path -- not silently
    fall through to "no models running" (which would otherwise surface
    downstream as a much more confusing `load_timeout`-expiry "not running"
    error on the very next load)."""
    mock = RestMock(running_route="/status", start_route="/models/{id}/start",
                     stop_route="/models/{id}/stop", running_field="data",
                     running_non_array={"oops": "not an array"})
    try:
        running = config.EndpointSpec(url="{base}/status", path="data")
        start = config.EndpointSpec(url="{base}/models/{id}/start", method="POST")
        stop = config.EndpointSpec(url="{base}/models/{id}/stop", method="POST")
        protocols = {"bad-shape": config.RestProtocolSpec(running=running, start=start, stop=stop)}
        inf = _device_inferencer("device", mock.url, management_protocol="bad-shape")
        backend = pool.build_backend(inf, protocols, poll_interval=0.01)
        assert backend is not None

        mgr = pool.DeviceModelManager(backend)
        try:
            try:
                await mgr.ensure_loaded("m1")
            except pool.DeviceError as exc:
                assert "data" in str(exc), f"error should name the offending path, got: {exc}"
            else:
                raise AssertionError("expected DeviceError for a present-but-non-array running path")
        finally:
            await mgr.aclose()
    finally:
        mock.close()
