"""conv-1b — stateful /v1/responses via the claude-resume backend.

Proves handle routing without a real `claude`: the subprocess seam
(`claude_code._invoke`) is patched, so these assert that woollama mints a
conversation handle, captures the backend's session_id on the first turn, and
RESUMES that same session (`--resume <sid>`) when the conversation is continued —
by handle or by previous_response_id. The live counterpart is the opt-in
claude-code integration test.
"""
from __future__ import annotations

import json

from woollama import claude_code, conversations, router


class FakeRequest:
    def __init__(self, body: dict):
        self._body = body

    async def json(self) -> dict:
        return self._body


def _result(text: str, sid: str) -> bytes:
    """A `claude -p --output-format json` result event carrying a session_id."""
    return json.dumps([
        {"type": "result", "subtype": "success", "is_error": False,
         "result": text, "session_id": sid},
    ]).encode()


def patch_claude(monkeypatch, sid: str = "sid-1"):
    """Patch the subprocess seam; record each invocation's args. The result
    always reports `sid` (claude resume returns the SAME session_id)."""
    calls: list[list[str]] = []

    async def fake_invoke(args, env, cwd, timeout):
        calls.append(list(args))
        return 0, _result(f"turn{len(calls)}", sid), b""

    monkeypatch.setattr(claude_code, "_invoke", fake_invoke)
    return calls


def fresh_store(monkeypatch):
    store = conversations.ConversationStore()
    monkeypatch.setattr(router, "conversation_store", store)
    return store


def fresh_stored(monkeypatch, tmp_path):
    """A `stored` backend on its own tmp duckdb file (the module singleton)."""
    store = conversations.StoredStore(str(tmp_path / "conv.duckdb"))
    monkeypatch.setattr(conversations, "_stored", store)
    return store


def fake_completion(monkeypatch, reply: str = "ok"):
    """Patch the stateless completion the `stored` backend replays through; record
    the (model, messages) it was called with so tests can assert replay."""
    seen: list[tuple[str, list[dict]]] = []

    async def fake(model, messages):
        seen.append((model, list(messages)))
        return reply

    monkeypatch.setattr(router, "complete_stateless", fake)
    return seen


# ---------------------------------------------------------------------------
# Handle table (unit)
# ---------------------------------------------------------------------------

def test_store_create_get_and_response_mapping():
    store = conversations.ConversationStore()
    conv = store.create("claude-resume", "claude-code/haiku")
    assert conv.id.startswith("conv_") and conv.native_id is None
    assert store.get(conv.id) is conv
    store.record_response(conv, "resp_1")
    assert store.by_response("resp_1") is conv
    assert store.by_response("resp_unknown") is None


def test_backend_for_model():
    assert conversations.backend_for_model("claude-code/haiku") == "claude-resume"
    # Every other model has no native session → server-owned `stored` backend.
    assert conversations.backend_for_model("ollama/x") == "stored"
    assert conversations.backend_for_model("woollama/streamer") == "stored"


# ---------------------------------------------------------------------------
# Stateful /v1/responses — create then continue (handle routing)
# ---------------------------------------------------------------------------

async def test_stateful_create_captures_session_then_resumes_by_conversation(monkeypatch):
    calls = patch_claude(monkeypatch, sid="sid-xyz")
    fresh_store(monkeypatch)

    # Turn 1 — store:true with a claude-code model creates a conversation.
    r1 = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "hello", "store": True}))
    b1 = json.loads(r1.body)
    conv_id = b1["conversation"]["id"]         # response carries a conversation object
    assert conv_id.startswith("conv_")
    assert b1["output"][0]["content"][0]["text"] == "turn1"
    assert "--resume" not in calls[0]          # first turn STARTS the session

    # Turn 2 — attach by conversation id (a bare string) → RESUMES the session.
    r2 = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "again", "conversation": conv_id}))
    b2 = json.loads(r2.body)
    assert b2["conversation"]["id"] == conv_id
    assert "--resume" in calls[1]
    assert calls[1][calls[1].index("--resume") + 1] == "sid-xyz"


async def test_stateful_continue_via_previous_response_id(monkeypatch):
    calls = patch_claude(monkeypatch, sid="sid-prev")
    fresh_store(monkeypatch)

    r1 = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "hi", "store": True}))
    resp_id = json.loads(r1.body)["id"]

    r2 = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "more",
        "previous_response_id": resp_id}))
    b2 = json.loads(r2.body)
    # chained onto the SAME conversation, resuming its session
    assert b2["conversation"]["id"] == json.loads(r1.body)["conversation"]["id"]
    assert "--resume" in calls[1] and calls[1][calls[1].index("--resume") + 1] == "sid-prev"


async def test_stateful_response_parses_in_openai_sdk(monkeypatch):
    from openai.types.responses import Response
    patch_claude(monkeypatch, sid="s1")
    fresh_store(monkeypatch)
    r = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "hi", "store": True}))
    parsed = Response.model_validate(json.loads(r.body))
    assert parsed.output_text == "turn1"
    assert parsed.conversation.id == json.loads(r.body)["conversation"]["id"]


# ---------------------------------------------------------------------------
# Routing errors
# ---------------------------------------------------------------------------

async def test_stateful_non_claude_model_routes_to_stored(monkeypatch, tmp_path):
    """A non-claude model with store:true now gets a server-owned `stored`
    conversation (no more 501) — and woollama replays its transcript each turn."""
    fresh_store(monkeypatch)
    fresh_stored(monkeypatch, tmp_path)
    seen = fake_completion(monkeypatch, reply="banana")

    r1 = await router.responses_create(FakeRequest({
        "model": "ollama/qwen3", "input": "remember banana", "store": True}))
    b1 = json.loads(r1.body)
    conv_id = b1["conversation"]["id"]
    assert b1["output"][0]["content"][0]["text"] == "banana"
    # turn 1 replayed just the new user message (empty history)
    assert [m["content"] for m in seen[0][1]] == ["remember banana"]

    # turn 2 attaches by conversation → history (turn 1 user + assistant) replays.
    r2 = await router.responses_create(FakeRequest({
        "model": "ollama/qwen3", "input": "what?", "conversation": conv_id}))
    assert json.loads(r2.body)["conversation"]["id"] == conv_id
    assert [m["content"] for m in seen[1][1]] == [
        "remember banana", "banana", "what?"]


async def test_stored_backend_error_surfaces_orchestration_error(monkeypatch, tmp_path):
    """A bad model name on a stored turn raises OrchestrationError inside the
    backend; the stateful path maps it (here a 400), not a 500."""
    fresh_store(monkeypatch)
    fresh_stored(monkeypatch, tmp_path)
    r = await router.responses_create(FakeRequest({
        "model": "bogusprovider/x", "input": "hi", "store": True}))
    assert r.status_code == 400


async def test_stateful_prev_response_belonging_to_other_conversation_is_400(monkeypatch):
    patch_claude(monkeypatch)
    fresh_store(monkeypatch)
    # Two independent conversations.
    a = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "a", "store": True}))
    b = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "b", "store": True}))
    conv_a = json.loads(a.body)["conversation"]["id"]
    resp_b = json.loads(b.body)["id"]
    # prev belongs to B but we name conversation A → conflict.
    r = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "x",
        "conversation": conv_a, "previous_response_id": resp_b}))
    assert r.status_code == 400


async def test_stateful_backend_error_is_502(monkeypatch):
    fresh_store(monkeypatch)

    async def boom(args, env, cwd, timeout):
        return 1, b"", b"claude exploded"
    monkeypatch.setattr(claude_code, "_invoke", boom)

    r = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "hi", "store": True}))
    assert r.status_code == 502
    assert "claude-resume backend" in json.loads(r.body)["error"]["message"]


# ---------------------------------------------------------------------------
# /v1/conversations — discovery / attach / teardown (conv-2)
# ---------------------------------------------------------------------------

async def test_conversations_create_lists_and_retrieves(monkeypatch):
    from openai.types.conversations import Conversation
    fresh_store(monkeypatch)

    r = await router.conversations_create(FakeRequest({
        "model": "claude-code/haiku", "title": "demo", "metadata": {"k": "v"}}))
    assert r.status_code == 201
    obj = json.loads(r.body)
    assert obj["backend"] == "claude-resume" and obj["status"] == "idle"
    assert obj["title"] == "demo" and obj["metadata"] == {"k": "v"}
    Conversation.model_validate(obj)        # parses as an OpenAI Conversation
    cid = obj["id"]

    listing = json.loads((await router.conversations_list()).body)
    assert listing["object"] == "list"
    assert cid in [c["id"] for c in listing["data"]]

    got = await router.conversations_get(cid)
    assert json.loads(got.body)["id"] == cid


async def test_conversations_create_requires_model(monkeypatch):
    fresh_store(monkeypatch)
    r = await router.conversations_create(FakeRequest({"title": "no model"}))
    assert r.status_code == 400


async def test_conversations_create_non_claude_model_uses_stored(monkeypatch, tmp_path):
    fresh_store(monkeypatch)
    fresh_stored(monkeypatch, tmp_path)
    r = await router.conversations_create(FakeRequest({"model": "ollama/qwen3"}))
    assert r.status_code == 201
    assert json.loads(r.body)["backend"] == "stored"


async def test_conversations_get_unknown_is_404(monkeypatch):
    fresh_store(monkeypatch)
    r = await router.conversations_get("conv_nope")
    assert r.status_code == 404


async def test_conversations_items_is_501_for_known_conversation(monkeypatch):
    fresh_store(monkeypatch)
    created = json.loads((await router.conversations_create(
        FakeRequest({"model": "claude-code/haiku"}))).body)
    r = await router.conversations_items(created["id"])
    assert r.status_code == 501            # transcript = driver slice
    assert (await router.conversations_items("conv_nope")).status_code == 404


async def test_conversations_delete_removes_handle_and_workdir(monkeypatch, tmp_path):
    store = fresh_store(monkeypatch)
    conv = store.create("claude-resume", "claude-code/haiku")
    workdir = tmp_path / "wd"
    workdir.mkdir()
    conv.workdir = str(workdir)

    r = await router.conversations_delete(conv.id)
    body = json.loads(r.body)
    assert body["deleted"] is True and body["object"] == "conversation.deleted"
    assert store.get(conv.id) is None              # handle forgotten
    assert not workdir.exists()                     # backend tore down its workdir
    assert (await router.conversations_delete(conv.id)).status_code == 404


async def test_create_via_endpoint_then_continue_via_responses(monkeypatch):
    """End-to-end handle reuse: a conversation created on /v1/conversations is
    driven by /v1/responses attaching to it."""
    calls = patch_claude(monkeypatch, sid="sid-c2")
    fresh_store(monkeypatch)
    created = json.loads((await router.conversations_create(
        FakeRequest({"model": "claude-code/haiku"}))).body)
    cid = created["id"]

    r = await router.responses_create(FakeRequest({
        "model": "claude-code/haiku", "input": "hi", "conversation": cid}))
    assert json.loads(r.body)["conversation"]["id"] == cid
    assert len(calls) == 1 and "--resume" not in calls[0]   # first turn starts it
    # the listed conversation now reflects the turn (updated_at advanced/recorded)
    got = json.loads((await router.conversations_get(cid)).body)
    assert got["status"] == "idle"


# ---------------------------------------------------------------------------
# stored backend — transcript items, delete, rehydration (conv-5)
# ---------------------------------------------------------------------------

async def test_stored_items_returns_transcript(monkeypatch, tmp_path):
    fresh_store(monkeypatch)
    fresh_stored(monkeypatch, tmp_path)
    fake_completion(monkeypatch, reply="pong")
    cid = json.loads((await router.responses_create(FakeRequest({
        "model": "ollama/qwen3", "input": "ping", "store": True}))).body)["conversation"]["id"]

    r = await router.conversations_items(cid)
    assert r.status_code == 200
    data = json.loads(r.body)["data"]
    assert [(i["role"], i["content"][0]["text"]) for i in data] == [
        ("user", "ping"), ("assistant", "pong")]
    # part types follow the role (input_text vs output_text)
    assert data[0]["content"][0]["type"] == "input_text"
    assert data[1]["content"][0]["type"] == "output_text"


async def test_stored_delete_clears_persistence(monkeypatch, tmp_path):
    store = fresh_store(monkeypatch)
    stored = fresh_stored(monkeypatch, tmp_path)
    fake_completion(monkeypatch, reply="x")
    cid = json.loads((await router.responses_create(FakeRequest({
        "model": "ollama/qwen3", "input": "hi", "store": True}))).body)["conversation"]["id"]
    assert await stored.load_messages(cid)          # persisted

    r = await router.conversations_delete(cid)
    assert json.loads(r.body)["deleted"] is True
    assert store.get(cid) is None                    # handle forgotten
    assert await stored.load_messages(cid) == []     # rows gone from duckdb


async def test_stored_conversations_rehydrate(monkeypatch, tmp_path):
    """conv-5's headline behaviour: a stored conversation persisted by one
    ConversationStore is recovered into a fresh one by `rehydrate_stored` — the
    exact glue the lifespan runs at startup, so attach survives a restart."""
    fresh_store(monkeypatch)
    stored = fresh_stored(monkeypatch, tmp_path)
    fake_completion(monkeypatch, reply="y")
    cid = json.loads((await router.responses_create(FakeRequest({
        "model": "ollama/qwen3", "input": "hi", "store": True}))).body)["conversation"]["id"]

    # Simulate a restart: brand-new in-memory store, same duckdb file, run the
    # real startup helper (not a hand-rolled loop) against it.
    revived = conversations.ConversationStore()
    n = await conversations.rehydrate_stored(revived, stored)
    assert n == 1
    assert revived.get(cid) is not None
    assert revived.get(cid).backend == "stored"
    # and the rehydrated handle still reads its transcript from duckdb
    assert await stored.load_messages(cid)


async def test_stored_items_validate_as_openai_conversation_item_list(monkeypatch, tmp_path):
    """The items wire shape is held to the same SDK bar as the rest of conv-* —
    a stored transcript parses as the SDK's ConversationItemList."""
    from openai.types.conversations import ConversationItemList
    fresh_store(monkeypatch)
    fresh_stored(monkeypatch, tmp_path)
    fake_completion(monkeypatch, reply="pong")
    cid = json.loads((await router.responses_create(FakeRequest({
        "model": "ollama/qwen3", "input": "ping", "store": True}))).body)["conversation"]["id"]
    body = json.loads((await router.conversations_items(cid)).body)
    ConversationItemList.model_validate(body)        # real SDK parse
