"""Attach-by-external-key — the cosmic-fabric smoother.

A caller (fabric) drives turns by its OWN key (e.g. a cosmic `sessionName`) and
never holds a woollama conversation id: woollama owns the durable `key → conv_id`
map. First use creates the handle; later uses with the same key continue it.
These cover the store-level alias index (+ durability), the idempotent
`/v1/conversations` create, and the create-or-attach `/v1/responses` path.
"""
from __future__ import annotations

import json

from woollama import conversations, router


class FakeStore:
    def __init__(self):
        self.threads, self._n = {}, 0

    async def create(self):
        self._n += 1
        tid = f"t{self._n}"
        self.threads[tid] = []
        return tid

    async def get(self, tid):
        return list(self.threads[tid])

    async def append(self, tid, msgs):
        self.threads[tid].extend(msgs)

    async def delete(self, tid):
        self.threads.pop(tid, None)


class FakeRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def _completer():
    async def complete(model, messages, *, options=None):
        last = messages[-1].get("content", "") if messages else ""
        return f"echo:{last}"
    return complete


def _setup(monkeypatch):
    monkeypatch.setattr(router, "conversation_store", conversations.ConversationStore())
    backend = conversations.StoreBackedBackend(
        conversations.STORE_BACKEND_NAME, FakeStore(), _completer())
    monkeypatch.setattr(conversations, "BACKENDS",
                        {**conversations.BACKENDS,
                         conversations.STORE_BACKEND_NAME: backend})


# --- store-level alias index --------------------------------------------------

def test_alias_create_then_attach():
    store = conversations.ConversationStore()
    conv, created = store.get_or_create_by_alias("sess-1", "store-backed", "ollama/x")
    assert created and conv.key == "sess-1"
    again, created2 = store.get_or_create_by_alias("sess-1", "store-backed", "ollama/x")
    assert not created2 and again.id == conv.id
    assert store.by_alias("sess-1") is conv
    assert store.by_alias("nope") is None


def test_alias_survives_reload_and_remove(tmp_path):
    path = tmp_path / "c.json"
    store = conversations.ConversationStore(path)
    conv = store.create("store-backed", "ollama/x", key="sess-1")

    # The key→conv_id index is rebuilt from the persisted conv.key on reload.
    reloaded = conversations.ConversationStore(path)
    assert reloaded.by_alias("sess-1").id == conv.id
    # Removing the conversation drops the alias too.
    reloaded.remove(conv.id)
    assert reloaded.by_alias("sess-1") is None
    assert conversations.ConversationStore(path).by_alias("sess-1") is None


# --- /v1/conversations idempotent create-by-key -------------------------------

async def test_conversations_create_by_key_is_idempotent(monkeypatch):
    _setup(monkeypatch)
    first = await router.conversations_create(FakeRequest(
        {"model": "ollama/qwen3", "key": "sess-42"}))
    assert first.status_code == 201
    cid = json.loads(first.body)["id"]
    # Same key again → the SAME conversation, 200 (attach), not a duplicate.
    second = await router.conversations_create(FakeRequest(
        {"model": "ollama/qwen3", "key": "sess-42"}))
    assert second.status_code == 200
    assert json.loads(second.body)["id"] == cid


# --- /v1/responses create-or-attach by key ------------------------------------

async def test_responses_by_key_creates_then_recalls(monkeypatch):
    _setup(monkeypatch)
    # Turn 1 by key only (no conversation id) → creates the handle.
    r1 = await router.responses_create(FakeRequest(
        {"model": "ollama/qwen3", "key": "sess-7", "input": "first"}))
    b1 = json.loads(r1.body)
    cid = b1["conversation"]["id"]
    assert b1["output"][0]["content"][0]["text"] == "echo:first"

    # Turn 2 by the SAME key → attaches to the same conversation (same id).
    r2 = await router.responses_create(FakeRequest(
        {"model": "ollama/qwen3", "key": "sess-7", "input": "second"}))
    assert json.loads(r2.body)["conversation"]["id"] == cid
    # And it's the store-backed stateful path (the prior turn is in the store).
    conv = router.conversation_store.get(cid)
    assert conv.key == "sess-7"
    items = await router.conversations_items(cid)
    roles = [d["role"] for d in json.loads(items.body)["data"]]
    assert roles.count("user") == 2


async def test_responses_by_key_501_when_no_backend(monkeypatch):
    # No store provider registered → non-claude models have no stateful backend,
    # so attach-by-key 501s like the bare-new path (not a 500).
    monkeypatch.setattr(router, "conversation_store", conversations.ConversationStore())
    r = await router.responses_create(FakeRequest(
        {"model": "ollama/qwen3", "key": "sess-x", "input": "hi"}))
    assert r.status_code == 501
