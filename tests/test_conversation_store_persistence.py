"""Durable conversation handle table — conv_ids survive a woollama restart.

The handle table is ROUTING state (conv_id → backend + native_id), not the
transcript (backends/stores own that), so persisting it respects "woollama never
owns conversation state". These tests prove a fresh ConversationStore over the
same path recovers the handles, response-id back-references, and the backend's
native_id — and that a stale in-flight 'busy' is reset on load. A pathless store
stays purely in-memory (the default the rest of the suite relies on).
"""
from __future__ import annotations

from woollama import conversations


def test_handles_survive_a_restart(tmp_path):
    path = tmp_path / "conversations.json"
    store = conversations.ConversationStore(path)
    conv = store.create("store-backed", "ollama/qwen3", title="t", metadata={"k": "v"})
    conv.native_id = "thread-42"          # the backend's own id (set on first turn)
    store.record_response(conv, "resp_abc")   # persists native_id + the resp mapping

    # A fresh process: new store, same file. The handle must come back intact.
    reloaded = conversations.ConversationStore(path)
    got = reloaded.get(conv.id)
    assert got is not None
    assert got.backend == "store-backed" and got.model == "ollama/qwen3"
    assert got.native_id == "thread-42"   # routing → the store thread survives
    assert got.title == "t" and got.metadata == {"k": "v"}
    assert got.response_ids == ["resp_abc"]
    # previous_response_id still resolves to the conversation after restart.
    assert reloaded.by_response("resp_abc") is got


def test_enable_persistence_loads_existing(tmp_path):
    path = tmp_path / "conversations.json"
    seed = conversations.ConversationStore(path)
    conv = seed.create("claude-resume", "claude-code/haiku")

    # The router pattern: a pathless module-level store, made durable at startup.
    store = conversations.ConversationStore()
    store.enable_persistence(path)
    assert store.get(conv.id) is not None


def test_busy_status_reset_on_load(tmp_path):
    path = tmp_path / "conversations.json"
    store = conversations.ConversationStore(path)
    conv = store.create("store-backed", "ollama/qwen3")
    conv.status = "busy"                   # simulate a crash mid-turn
    store.record_response(conv, "resp_x")  # persist the busy status

    reloaded = conversations.ConversationStore(path)
    # A turn can't be in flight after a restart → busy is stale, reset to idle.
    assert reloaded.get(conv.id).status == "idle"


def test_remove_is_persisted(tmp_path):
    path = tmp_path / "conversations.json"
    store = conversations.ConversationStore(path)
    conv = store.create("store-backed", "ollama/qwen3")
    store.record_response(conv, "resp_y")
    store.remove(conv.id)

    reloaded = conversations.ConversationStore(path)
    assert reloaded.get(conv.id) is None
    assert reloaded.by_response("resp_y") is None


def test_pathless_store_writes_nothing(tmp_path):
    store = conversations.ConversationStore()       # no path → in-memory
    store.create("store-backed", "ollama/qwen3")
    assert list(tmp_path.iterdir()) == []           # nothing persisted


def test_corrupt_file_starts_empty_not_crash(tmp_path):
    path = tmp_path / "conversations.json"
    path.write_text("{not json")
    store = conversations.ConversationStore(path)    # must not raise
    assert store.list() == []
