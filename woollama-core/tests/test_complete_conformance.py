"""Conformance tests for the Rust `woollama_core.complete` — it must behave like
`woollama.core.complete` (Python), which is the oracle. We assert the same things
the Python hermetic suite asserts (request shape, routing, params, auth, errors),
against a threaded mock HTTP server.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import woollama_core as wc


class _Mock(BaseHTTPRequestHandler):
    record: dict = {}

    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Mock.record = {"path": self.path, "body": body,
                        "auth": self.headers.get("Authorization")}
        payload = ({"message": {"content": "sized"}} if self.path == "/api/chat"
                   else {"choices": [{"message": {"content": "hi"}}]})
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def base_url():
    srv = HTTPServer(("127.0.0.1", 0), _Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    finally:
        srv.shutdown()


MSGS = [{"role": "user", "content": "x"}]


def test_v1_path_routing_params_auth(base_url):
    out = wc.complete("openai/gpt-x", MSGS, base_url=base_url,
                      api_key="sk-x", params={"temperature": 0.2})
    assert out == "hi"
    rec = _Mock.record
    assert rec["path"] == "/v1/chat/completions"
    assert rec["auth"] == "Bearer sk-x"                       # per-call key override
    assert rec["body"] == {"model": "gpt-x", "messages": MSGS,
                           "stream": False, "temperature": 0.2}  # params -> top level


def test_options_go_under_options_key(base_url):
    wc.complete("openai/gpt-x", MSGS, base_url=base_url, api_key="k",
                options={"foo": 1})
    assert _Mock.record["body"]["options"] == {"foo": 1}


def test_ollama_num_ctx_routes_native(base_url):
    out = wc.complete("ollama/qwen", MSGS, base_url=base_url,
                      options={"num_ctx": 8192}, params={"temperature": 0.5})
    assert out == "sized"
    rec = _Mock.record
    assert rec["path"] == "/api/chat"                          # native, not /v1
    # temperature merges into options on the native path
    assert rec["body"]["options"] == {"num_ctx": 8192, "temperature": 0.5}


def test_unknown_provider_raises(base_url):
    with pytest.raises(wc.InferenceError):
        wc.complete("nope/m", MSGS, base_url=base_url)


def test_missing_key_raises_fast(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(wc.InferenceError, match="OPENAI_API_KEY"):
        wc.complete("openai/gpt-x", MSGS)        # no api_key, no env -> fail fast


def test_provider_names():
    assert wc.provider_names() == ["ollama", "anthropic", "openai",
                                   "groq", "together", "openrouter"]
