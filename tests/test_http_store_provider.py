"""conv-7 / issue #2 — the REST (HTTP) conversation-store provider.

`HttpStoreProvider` is the second `ConversationStoreProvider` implementation
(sibling to `McpStoreProvider`), proving the seam is transport-agnostic. woollama
holds NO bytes — every op is one HTTP request to an external store
(examples/rest-convstore). These tests cover, hermetically:

  - the provider maps create/get/append/delete to the right (method, path, body),
    minting the thread id itself and PUTting it;
  - the router's `_http_store_call` wrapper parses JSON, treats 204/empty as None,
    and wraps any non-2xx / transport failure as a clean OrchestrationError(502).

The full real-server round-trip (rest-convstore + ollama) lives in
test_integration.py.
"""
from __future__ import annotations

import pytest

from woollama import conversations, router

# --- provider maps to the right HTTP calls ------------------------------------

async def test_provider_maps_ops_to_http_calls():
    """HttpStoreProvider delegates each op to the injected call with the right
    (method, path, body); create mints a uuid hex id and PUTs /threads/{id}."""
    seen: list = []

    async def call(method, path, body):
        seen.append((method, path, body))
        if method == "GET":
            return [{"role": "user", "content": "hi"}]
        return None

    p = conversations.HttpStoreProvider(call)
    tid = await p.create()
    assert seen[0][0] == "PUT" and seen[0][1] == f"/threads/{tid}" and seen[0][2] is None
    assert all(ch in "0123456789abcdef" for ch in tid)   # uuid4().hex

    assert await p.get(tid) == [{"role": "user", "content": "hi"}]
    await p.append(tid, [{"role": "user", "content": "x"}])
    await p.delete(tid)
    assert seen[1:] == [
        ("GET", f"/threads/{tid}", None),
        ("PATCH", f"/threads/{tid}", [{"role": "user", "content": "x"}]),
        ("DELETE", f"/threads/{tid}", None),
    ]


async def test_provider_get_none_body_is_empty_list():
    """A None (204/empty) body from get() reads as [] — never None into the
    backend's `prior + messages` assembly."""
    async def call(method, path, body):
        return None
    assert await conversations.HttpStoreProvider(call).get("t") == []


# --- the router wrapper: parse + 502 on failure -------------------------------

class _Resp:
    def __init__(self, status_code=200, content=b"x", payload=None, raise_exc=None):
        self.status_code = status_code
        self.content = content
        self._payload = payload
        self._raise = raise_exc

    def raise_for_status(self):
        if self._raise is not None:
            raise self._raise

    def json(self):
        return self._payload


class _Client:
    """Fake httpx.AsyncClient capturing the request and returning a canned resp
    (or raising)."""
    last: dict = {}

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    async def request(self, method, url, json=None):
        _Client.last = {"method": method, "url": url, "json": json}
        resp = _Client.resp
        if isinstance(resp, Exception):
            raise resp
        return resp


async def test_http_store_call_parses_json(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    _Client.resp = _Resp(200, content=b"[]", payload=[{"role": "user", "content": "hi"}])
    call = router._http_store_call("http://store:9000/")
    out = await call("GET", "/threads/abc", None)
    assert out == [{"role": "user", "content": "hi"}]
    # base_url trailing slash trimmed; path joined.
    assert _Client.last["url"] == "http://store:9000/threads/abc"


async def test_http_store_call_204_is_none(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    _Client.resp = _Resp(204, content=b"")
    call = router._http_store_call("http://store:9000")
    assert await call("PUT", "/threads/abc", None) is None


async def test_http_store_call_wraps_non2xx_as_502(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    _Client.resp = _Resp(
        500, raise_exc=httpx.HTTPStatusError("boom", request=None, response=None))
    call = router._http_store_call("http://store:9000")
    with pytest.raises(router.OrchestrationError) as ei:
        await call("GET", "/threads/abc", None)
    assert ei.value.status == 502


async def test_http_store_call_wraps_transport_error_as_502(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    _Client.resp = httpx.ConnectError("refused")
    call = router._http_store_call("http://store:9000")
    with pytest.raises(router.OrchestrationError) as ei:
        await call("GET", "/threads/abc", None)
    assert ei.value.status == 502


# --- end-to-end: a flaky HTTP store surfaces as 502 through /v1/responses ------

class FakeRequest:
    def __init__(self, body: dict):
        self._body = body

    async def json(self) -> dict:
        return self._body


async def test_flaky_http_store_surfaces_as_502_through_responses(monkeypatch):
    """A store-backed turn over an HttpStoreProvider whose endpoint refuses must
    surface as 502 via _responses_stateful, not a 500."""
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    _Client.resp = httpx.ConnectError("refused")
    monkeypatch.setattr(router, "conversation_store", conversations.ConversationStore())
    backend = conversations.StoreBackedBackend(
        conversations.STORE_BACKEND_NAME,
        conversations.HttpStoreProvider(router._http_store_call("http://store:9000")),
        router.complete_stateless)
    monkeypatch.setattr(conversations, "BACKENDS",
                        {**conversations.BACKENDS,
                         conversations.STORE_BACKEND_NAME: backend})
    r = await router.responses_create(FakeRequest({
        "model": "ollama/qwen3", "input": "hi", "store": True}))
    assert r.status_code == 502
