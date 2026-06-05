# Conversations & Responses — design (stateful surface for woollama)

Status: **in progress.** Decisions locked 2026-06-02. **conv-1a shipped
2026-06-04**: `POST /v1/responses` stateless subset (`store:false`) — a
Responses-shaped superset of /v1/chat/completions, routed by `model` identically
(`router.py:responses_create`, shaping in `responses.py`), verified against the
real `openai` SDK (`.responses.create` → `.output_text`). **conv-1b shipped
2026-06-04**: in-memory handle table + the `claude-resume` backend + `store:true`
/ `conversation` / `previous_response_id` routing (`conversations.py`,
`router._responses_stateful`), live-verified via the `openai` SDK. Still to do
from §8: `/v1/conversations` listing (conv-2), the Rust driver + claude-tmux
backend (gated on §6 spikes), the interactive `requires_action` path, the duckdb
`stored` backend, and cosmic-fabric wiring.

## The principle

**woollama routes conversation *handles*; the backends own the *state*.** Many
tools already maintain conversation state (Claude Code sessions, a future
duckdb thread store, …). woollama should not become a conversation database — it
should hand out stable `conversation_id`s and route each to whatever backend
owns that conversation's bytes. The Responses/Conversations API is a *thin
routing shape* over heterogeneous stateful backends, not a store woollama
builds.

Corollary decisions:
- **Keep it a SEPARATE surface.** `/v1/responses` + `/v1/conversations` are
  stateful; `/v1/chat/completions` stays stateless. The router stays a router
  for everything that doesn't opt in.
- **No new wire format.** Adopt the OpenAI Responses + Conversations shapes
  (every OpenAI SDK and cosmic-fabric can speak them). Only the *cross-backend
  handle routing* is woollama's own contribution.
- **Heavy/fragile session-driving logic lives OUTSIDE the router**, in a
  separate Rust package (the "session driver" — see §4). woollama (Python) stays
  thin; the driver owns tmux, send-keys, jsonl tailing, and detection.

## Architecture

```
cosmic-fabric / OpenAI client
        │  /v1/responses, /v1/conversations  (stateful, OpenAI-shaped)
        ▼
   woollama (router)
        │  ConversationBackend interface (§3) — routes conversation_id → backing
        ├─▶ stateless         (store=false; caller owns history; today's model)
        ├─▶ claude-resume      (delegated; `claude --resume <sid>`, non-interactive)
        ├─▶ claude-tmux        (delegated, LIVE + interactive) ──HTTP/SSE──▶ session driver (Rust)
        │                                                          owns: tmux, send-keys (Esc/Enter),
        │                                                          jsonl tail, turn/pending detection
        └─▶ stored             (server-owned; duckdb thread)   [later]
```

## 1. External API — `/v1/responses`

Stateful counterpart of chat-completions. Routing by `model` is unchanged
(`woollama/<recipe>`, `claude-code/<model>`, `ollama/<model>`).

Request:
```jsonc
POST /v1/responses
{
  "model": "woollama/<recipe>",
  "input": "..." | [ {role, content}, ... ],
  "conversation": "conv_abc",        // optional: attach to an existing conversation
  "previous_response_id": "resp_x",  // optional: chain off a prior turn (fork point)
  "store": true,                     // false → stateless (no backing created)
  "stream": false
}
```
Response:
```jsonc
{
  "id": "resp_123",
  "conversation": "conv_abc",
  "status": "completed",             // | "requires_action" | "incomplete" | "failed"
  "output": [ { "type": "message", "role": "assistant", "content": [ ... ] } ],
  "required_action": null            // populated when status == requires_action (see §5)
}
```
`store: false` and no `conversation` → behaves exactly like chat-completions
(stateless passthrough), so the surface is a superset.

## 2. External API — `/v1/conversations` (discovery + attach)

This is what cosmic-fabric binds to: list existing conversations, pick one,
drive it.

```
POST   /v1/conversations            { "backend": "claude-tmux" | "claude-resume" | "stored",
                                       "model": "...", "metadata": {...} }   -> {id, status}
GET    /v1/conversations            -> [ {id, backend, status, title, updated_at}, ... ]
GET    /v1/conversations/{id}        -> {id, backend, status, ...}
GET    /v1/conversations/{id}/items  -> the transcript (messages)
DELETE /v1/conversations/{id}        -> end / kill the backing
```
`status` ∈ `idle | busy | awaiting_input | dead`. `awaiting_input` is the
attach-time signal that a live session is blocked on a question (§5).

## 3. Internal seam — the `ConversationBackend` interface

woollama-side abstraction; each backend implements it. woollama stays thin —
all backends are small adapters; the hard one (claude-tmux) is just an HTTP
client to the Rust driver.

```
create() -> conversation_id
send_turn(id, input) -> Response            # may resolve to requires_action
history(id) -> [messages]
poll(id) -> status (+ pending question if awaiting_input)
answer(id, answer | control_key) -> Response   # resolve requires_action / send Esc, Enter, …
delete(id)
```

## 4. The session driver (separate Rust package)

Owns everything fragile. Exposed to woollama as a **local HTTP service with SSE**
for streaming turn output (language-agnostic boundary; woollama is a thin httpx
client; also lines up with the future streaming roadmap item). woollama may
spawn-and-manage it (like it does MCP servers) or connect to a configured URL.

Driver responsibilities (NOT in the router):
- tmux session lifecycle (`new-session -d` running `claude`, kill).
- The **interaction driver**: the send-keys state machine that knows Claude
  Code's TUI modes — submit, **Escape** to interrupt, answer-an-AskUserQuestion,
  dismiss. (This is the Esc/Enter fragility; it belongs here, isolated.)
- jsonl tailing of the session transcript (`~/.claude/projects/<enc>/<sid>.jsonl`).
- **Turn-complete detection** and **pending-question detection** (the load-bearing
  signals — see §6).

Driver API (mirrors the backend interface):
```
POST   /sessions                  { model, system?, cwd? }   -> {session_id, jsonl_path}
POST   /sessions/{id}/turns       { input }  -> SSE: assistant events …, then
                                                {done: completed | requires_action(question)}
GET    /sessions/{id}/transcript  -> messages (parsed from jsonl)
GET    /sessions/{id}/status      -> idle | busy | awaiting_input(+question) | dead
POST   /sessions/{id}/answer      { answer } | { control: "escape" | "enter" }
DELETE /sessions/{id}
```
(Working name TBD — something in the herding vein fits the llama theme, e.g.
"drover": it drives the session. Name it later.)

Why HTTP/SSE and not MCP: conversations are long-lived stateful *resources* with
streaming output and interrupt semantics — a poor fit for MCP's tool-call shape.
A purpose-built REST+SSE service is cleaner, and keeps the driver usable
independently of woollama.

## 5. Interactive turns — pending questions

A live Claude session can pause on an `AskUserQuestion` or a permission prompt.
Map it to the existing Responses primitive:

- Turn pauses → Response `status: "requires_action"`, `required_action`:
  ```jsonc
  { "type": "ask_user", "question": { /* the AskUserQuestion payload */ } }
  ```
- Client answers by continuing the conversation: `POST /v1/responses` with the
  same `conversation` and the answer as `input` → woollama sees the conversation
  is `awaiting_input` and routes to `backend.answer(...)` (→ driver send-keys) →
  the next Response.
- cosmic-fabric renders `required_action` as a question UI; the user's choice
  flows back the same path. This is the eventual attach-and-converse UX.

## 6. Spikes to settle FIRST (owned by the driver; run outside a nested Claude session)

These crashed when attempted from inside this Claude Code session; run them in a
plain terminal before building the claude-tmux backend:

1. **Live-session jsonl shape + turn-complete signal** — what event marks "done"
   for a live (non-`-p`) session? Same shape as `-p` stream-json?
2. **Pending-question signal** — trigger an AskUserQuestion; what appears in the
   jsonl/pane, and how is it answered deterministically via send-keys?
3. **send-keys reliability** — the exact Escape/Enter discipline that reliably
   submits / interrupts / answers without races.

## 7. Concept mapping

| OpenAI Responses | woollama | backing |
|---|---|---|
| `response.id` | a turn | a turn in the session |
| `conversation` | routable handle | tmux session / `--resume` id / duckdb thread |
| `previous_response_id` | chain / fork point | append vs. fork a new session |
| `store: false` | stateless | none (caller owns history) |
| `status: requires_action` | awaiting_input | Claude paused on AskUserQuestion |

## 8. Build sequence (sequence the risk)

1. **`/v1/responses` subset + `claude-resume` backend** — proves handle-routing +
   the Responses shape against the EASY (non-interactive) backend. No tmux.
   - [x] **conv-1a** — `/v1/responses` stateless subset (`store:false`); the
     Responses wire shape, SDK-verified. No backend/handle table yet.
   - [x] **conv-1b SHIPPED 2026-06-04** — in-memory handle table
     (`conversations.py`: `conv_id → {backend, native_id, workdir}`; resp_id is
     the chain key; one `asyncio.Lock` writer per conversation) + `claude-resume`
     backend + `store:true` / `conversation` / `previous_response_id` routing in
     `router._responses_stateful`. Verified live via the `openai` SDK (create →
     continue → recalled the codeword).
     Resume facts (claude 2.1.163, headless — no hang; the §6 hang is the
     INTERACTIVE TUI, not `-p`): `claude -p --output-format json` → the `result`
     event carries `session_id`; `claude --resume <sid> -p` continues, recalls
     context, returns the SAME `session_id`. **Load-bearing gotcha the live test
     caught:** Claude scopes sessions BY PROJECT (cwd), so all turns of a
     conversation MUST run in the same dir — each conversation pins a stable
     `workdir` (a fresh empty temp dir, cleaned on delete in a later slice).
     `--resume` continues from the session TIP (no fork-from-earlier-turn
     primitive), so `previous_response_id` CHAINS off the conversation; true
     forking is later. Handle table is in-memory → a restart loses sid mappings.
2. **`/v1/conversations` listing + delete** — discovery/attach surface.
   **conv-2 SHIPPED 2026-06-04**: `POST` (create handle; backend derived from
   `model`), `GET` (list), `GET /{id}`, `DELETE /{id}` (backend teardown +
   forget handle), all over the in-memory handle table; objects parse as OpenAI
   Conversation + woollama routing extras (`backend`/`status`/`title`). Live
   CRUD verified. `GET /{id}/items` (transcript) is a deliberate 501 — reading a
   backend's transcript is the session-driver's job (slice 3+).
3. **Session driver (Rust) + claude-tmux backend** — the live backing (gated on
   the §6 spikes). The hard infra, isolated in its own package.
4. **`requires_action` / interactive answer path** — §5; the interaction driver.
5. **duckdb `stored` backend** — server-owned conversations (no backend owns).
6. **cosmic-fabric wiring** — when that UX returns.

## 9. Risk flags

- The TUI driver is the fragile part — isolated in Rust on purpose; treat its
  reliability (Esc/Enter races) as the project's main risk.
- One writer per conversation — woollama serializes turns per `conversation_id`.
- This is the "woollama becomes stateful" shift — contained to the new surface;
  chat-completions stays stateless.
- Don't half-implement the Responses spec — minimal subset only (create /
  continue / read / fork / requires_action).
