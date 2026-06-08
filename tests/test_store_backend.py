"""conv-7 / issue #2 — the store-backed (BYO-inference) conversation backend.

woollama-side mechanism only: an external `ConversationStoreProvider` owns the
transcript while woollama assembles prior history + the new turn, runs the
STATELESS inferencer, and writes the turn back (design-doc §3.1, §10). The store
provider here is an in-memory fake (the REAL provider — fabric — is pending its
contract); the inferencer is a fake completer (the router injects
`complete_stateless` in production). These prove: assemble→complete→append,
recall across turns via reassembly, `history`/`items`, delete, the routing gate
(non-claude models stay stateless until a provider is registered), and that an
inference failure surfaces cleanly (not a 500).
"""
from __future__ import annotations

import json

from woollama import conversations, router


class FakeStore:
    """In-memory ConversationStoreProvider — stands in for the (pending) fabric
    provider. woollama never owns this in production; it's a test double for the
    external owner of the bytes."""

    def __init__(self) -> None:
        self.threads: dict[str, list[dict]] = {}
        self._n = 0

    async def create(self) -> str:
        self._n += 1
        tid = f"thread_{self._n}"
        self.threads[tid] = []
        return tid

    async def get(self, thread_id: str) -> list[dict]:
        return list(self.threads[thread_id])

    async def append(self, thread_id: str, messages: list[dict]) -> None:
        self.threads[thread_id].extend(messages)

    async def delete(self, thread_id: str) -> None:
        self.threads.pop(thread_id, None)


def make_completer(seen: list):
    """A fake inferencer: records (model, messages, options) it's asked to complete
    and echoes the last user message so recall is observable."""
    async def complete(model: str, messages: list[dict], *, options=None) -> str:
        seen.append((model, list(messages), options))
        last = messages[-1].get("content", "") if messages else ""
        return f"echo:{last}"
    return complete


class FakeRequest:
    def __init__(self, body: dict):
        self._body = body

    async def json(self) -> dict:
        return self._body


def setup(monkeypatch, *, complete=None):
    """Fresh handle table + a store-backed backend (over a fake store) registered
    under the gate name. monkeypatch on BACKENDS auto-restores, so the default
    stateless behavior other tests rely on is never leaked into."""
    store = FakeStore()
    seen: list = []
    monkeypatch.setattr(router, "conversation_store", conversations.ConversationStore())
    backend = conversations.StoreBackedBackend(
        conversations.STORE_BACKEND_NAME, store, complete or make_completer(seen))
    monkeypatch.setattr(conversations, "BACKENDS",
                        {**conversations.BACKENDS,
                         conversations.STORE_BACKEND_NAME: backend})
    return store, seen


# --- routing gate -------------------------------------------------------------

def test_backend_for_model_gated_on_registration(monkeypatch):
    # Default: no store provider registered → non-claude models stay stateless.
    assert conversations.backend_for_model("ollama/qwen3") is None
    assert conversations.backend_for_model("woollama/streamer") is None
    # Claude backends are unaffected by the store gate.
    assert conversations.backend_for_model("claude-code/haiku") == "claude-resume"
    # Once a store backend is registered, non-claude models route to it.
    setup(monkeypatch)
    assert conversations.backend_for_model("ollama/qwen3") == conversations.STORE_BACKEND_NAME
    assert conversations.backend_for_model("woollama/streamer") == conversations.STORE_BACKEND_NAME
    assert conversations.backend_for_model("claude-agent/opus") == "managed-agents"


def test_register_store_backend_inserts(monkeypatch):
    # Mutating-the-global path: monkeypatch BACKENDS to a fresh copy so the real
    # register() call is exercised without leaking.
    monkeypatch.setattr(conversations, "BACKENDS", dict(conversations.BACKENDS))
    conversations.register_store_backend(FakeStore(), make_completer([]))
    assert conversations.STORE_BACKEND_NAME in conversations.BACKENDS


# --- backend behavior (unit) --------------------------------------------------

async def test_send_turn_assembles_completes_appends(monkeypatch):
    store, seen = setup(monkeypatch)
    backend = conversations.BACKENDS[conversations.STORE_BACKEND_NAME]
    conv = router.conversation_store.create(conversations.STORE_BACKEND_NAME, "ollama/qwen3")

    out = await backend.send_turn(conv, [{"role": "user", "content": "hi"}])
    assert out == "echo:hi"
    assert conv.native_id == "thread_1"           # store minted the thread
    # The completer saw exactly the new turn (no prior history on turn 1).
    assert seen[-1] == ("ollama/qwen3", [{"role": "user", "content": "hi"}], None)
    # The turn (user + assistant) was written back to the store.
    assert store.threads["thread_1"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "echo:hi"}]


async def test_recall_across_turns_via_reassembly(monkeypatch):
    store, seen = setup(monkeypatch)
    backend = conversations.BACKENDS[conversations.STORE_BACKEND_NAME]
    conv = router.conversation_store.create(conversations.STORE_BACKEND_NAME, "ollama/qwen3")

    await backend.send_turn(conv, [{"role": "user", "content": "first"}])
    await backend.send_turn(conv, [{"role": "user", "content": "second"}])
    # Turn 2's completion was given the FULL prior transcript + the new input —
    # that reassembly is how a stateless inferencer "remembers".
    model, msgs, _ = seen[-1]
    assert [m["content"] for m in msgs] == ["first", "echo:first", "second"]
    # Same thread reused (not re-created).
    assert conv.native_id == "thread_1" and len(store.threads) == 1


async def test_history_and_delete(monkeypatch):
    store, _ = setup(monkeypatch)
    backend = conversations.BACKENDS[conversations.STORE_BACKEND_NAME]
    conv = router.conversation_store.create(conversations.STORE_BACKEND_NAME, "ollama/qwen3")
    assert await backend.history(conv) == []        # no thread yet
    await backend.send_turn(conv, [{"role": "user", "content": "ping"}])
    assert await backend.history(conv) == [
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "echo:ping"}]
    await backend.delete(conv)
    assert conv.status == "dead" and conv.native_id not in store.threads


# --- through the router surfaces ----------------------------------------------

async def test_stateful_responses_routes_to_store_backend(monkeypatch):
    setup(monkeypatch)
    r = await router.responses_create(FakeRequest({
        "model": "ollama/qwen3", "input": "hello", "store": True}))
    body = json.loads(r.body)
    cid = body["conversation"]["id"]
    assert body["output"][0]["content"][0]["text"] == "echo:hello"
    conv = router.conversation_store.get(cid)
    assert conv.backend == conversations.STORE_BACKEND_NAME and conv.native_id is not None


async def test_conversations_items_served_for_store_backend(monkeypatch):
    setup(monkeypatch)
    created = json.loads((await router.conversations_create(
        FakeRequest({"model": "ollama/qwen3"}))).body)
    cid = created["id"]
    assert created["backend"] == conversations.STORE_BACKEND_NAME
    await router.responses_create(FakeRequest({
        "model": "ollama/qwen3", "input": "ping", "conversation": cid}))
    items = await router.conversations_items(cid)
    assert items.status_code == 200
    data = json.loads(items.body)["data"]
    assert [d["role"] for d in data] == ["user", "assistant"]
    assert data[0]["content"][0]["text"] == "ping"


async def test_store_backed_ollama_turn_honors_num_ctx(monkeypatch):
    """The #1↔#2 seam closed: a stateful store-backed ollama turn threads
    options.num_ctx through send_turn → complete_stateless → the native /api/chat
    (which honors num_ctx). Registers the REAL complete_stateless and mocks httpx
    to capture where the inference call lands."""
    import httpx

    store = FakeStore()
    monkeypatch.setattr(router, "conversation_store", conversations.ConversationStore())
    backend = conversations.StoreBackedBackend(
        conversations.STORE_BACKEND_NAME, store, router.complete_stateless)
    monkeypatch.setattr(conversations, "BACKENDS",
                        {**conversations.BACKENDS,
                         conversations.STORE_BACKEND_NAME: backend})

    posts: list = []

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        async def post(self, url, json=None, **k):
            posts.append((url, json))
            # native /api/chat reply shape
            return type("R", (), {"status_code": 200,
                                  "json": lambda self: {"message": {"content": "ok"},
                                                        "done": True}})()
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    r = await router.responses_create(FakeRequest({
        "model": "ollama/qwen3", "input": "hi", "store": True,
        "options": {"num_ctx": 16384}}))
    assert json.loads(r.body)["output"][0]["content"][0]["text"] == "ok"
    url, sent = posts[-1]
    assert url.endswith("/api/chat")                 # native, not /v1
    assert sent["options"]["num_ctx"] == 16384


async def test_inference_failure_surfaces_cleanly_not_500(monkeypatch):
    """A store-backed turn runs inference via complete_stateless, which raises
    OrchestrationError. _responses_stateful must surface it (here 502 with the
    upstream payload), not let it escape as an unhandled 500."""
    async def boom(model, messages, *, options=None):
        raise router.OrchestrationError("inferencer down", "server_error", 502,
                                        payload={"error": "down"})
    setup(monkeypatch, complete=boom)
    r = await router.responses_create(FakeRequest({
        "model": "ollama/qwen3", "input": "hi", "store": True}))
    assert r.status_code == 502
    assert json.loads(r.body) == {"error": "down"}
