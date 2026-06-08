# Conversations & Responses — design (stateful surface for woollama)

Status: **in progress.** Decisions locked 2026-06-02. **conv-1a shipped
2026-06-04**: `POST /v1/responses` stateless subset (`store:false`) — a
Responses-shaped superset of /v1/chat/completions, routed by `model` identically
(`router.py:responses_create`, shaping in `responses.py`), verified against the
real `openai` SDK (`.responses.create` → `.output_text`). **conv-1b shipped
2026-06-04**: in-memory handle table + the `claude-resume` backend + `store:true`
/ `conversation` / `previous_response_id` routing (`conversations.py`,
`router._responses_stateful`), live-verified via the `openai` SDK. **conv-2
shipped 2026-06-04**: `/v1/conversations` create/list/get/delete. **conv-5
(duckdb `stored` backend) shipped 2026-06-05 then REVERTED 2026-06-06** — it made
woollama OWN conversation storage (an embedded duckdb), which violates the
principle below; woollama must never be the store. Models with no state-owning
backend are now stateless (`store:false`); see §8 item 5. Still to do from §8:
a state-owning backend for non-claude models that DEFERS to an external owner —
the leading candidate is **Managed Agents** (§8 item 7, Anthropic owns the
session); plus the Rust driver + claude-tmux backend (gated on §6 spikes), the
interactive `requires_action` path, and cosmic-fabric wiring.

## The principle

**woollama routes conversation *handles*; the backends own the *state*.** Many
systems already maintain conversation state (Claude Code sessions in `~/.claude`,
Anthropic Managed Agents sessions, …). woollama should **never** become a
conversation database — it hands out stable `conversation_id`s and routes each to
whatever backend owns that conversation's bytes. The Responses/Conversations API
is a *thin routing shape* over heterogeneous stateful backends, not a store
woollama builds. **(Learned the hard way: conv-5 added an embedded duckdb store
and was reverted — woollama may proxy/retrieve a transcript or create one in
another system, but it does not store in its own.)** When no backend owns the
state, the turn is stateless and the caller owns history — woollama does not
fabricate a store.

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
        └─▶ managed-agents     (Anthropic-hosted; /v1/sessions)  [conv-6, §8.7]
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
POST   /v1/conversations            { "backend": "claude-resume" | "claude-tmux" | "managed-agents",
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

### 3.1 Two backend kinds — and the one invariant they share

Every state-owning backend **defers the transcript bytes to an external owner**;
woollama only holds the `conv_id → {backend, native_id}` handle. They split by
*who runs the inference loop*:

- **Native-loop backends** — the owner runs the loop AND inference, so `send_turn`
  delegates the whole turn and woollama just routes the handle:
  - `claude-resume` — owner = the Claude CLI session (bytes in `~/.claude`'s JSONL).
  - `managed-agents` — owner = Anthropic's hosted session (conv-6).
  - `claude-tmux` (future) — owner = the live Claude TUI via the Rust driver.

- **Store-only (BYO-inference) backends** — the owner holds the bytes but does NOT
  run inference, so woollama does the **assembly + inference** itself:
  `history ← store.get(native_id)` → prepend to the new input → call the
  **stateless** inferencer (e.g. `ollama/<model>`, honoring `num_ctx` via the
  native path) → `store.append(native_id, turn)`. This is the family that makes
  non-claude models stateful (issue #2) — see §10.

The invariant in both: **woollama is never the store.** A store-only backend is
parameterized by a pluggable *conversation-store provider* (§10); fabric is the
first provider, but the seam is provider-agnostic so an MCP conversation-store, or
even a JSONL reader mirroring claude-resume's model, can drop in later without
woollama ever owning bytes.

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

**SHIPPED 2026-06-07 via the managed-agents backend** (ahead of the tmux driver).
A hosted CMA session pauses when the model calls a client-side custom tool
(`ask_user`, declared on the agent): `agent.custom_tool_use` fires and the session
idles with `stop_reason.type == "requires_action"`. woollama maps that to the
Responses primitive below and resumes by returning the answer as a
`user.custom_tool_result`. The claude-tmux driver (a live TUI pausing on a real
`AskUserQuestion`) will map onto the SAME primitive later.

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
| `conversation` | routable handle | tmux session / `--resume` id / managed-agents session |
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
4. **`requires_action` / interactive answer path** — §5. **SHIPPED 2026-06-07 via
   the managed-agents backend** (the `ask_user` custom tool); the claude-tmux
   driver will reuse the same Responses primitive.
5. **duckdb `stored` backend** — **SHIPPED 2026-06-05 (conv-5), then REVERTED
   2026-06-06.** It made woollama OWN conversation storage: an embedded duckdb at
   `$XDG_DATA_HOME/woollama/conversations.duckdb` that persisted the transcript
   and replayed it through `complete_stateless`. That directly contradicts §1 —
   *woollama must never be the store.* Reverted in full (the duckdb dep,
   `StoredStore`/`StoredBackend`, startup rehydration, and the
   `backend_for_model → stored` default). **The replacement is a NON-decision:**
   a model with no state-owning backend is **stateless** (`store:false`, the
   caller owns history — exactly as the Anthropic Messages API is stateless), so
   `/v1/responses` with `store:true` and `/v1/conversations` create both return a
   clean 501 for such models. If non-claude models need *stateful* conversations
   later, the answer is a backend that DEFERS to an external owner woollama is a
   *client* to — e.g. a "conversation store" MCP server, or Managed Agents
   (item 7) — never woollama's own embedded DB.
6. **cosmic-fabric wiring** — when that UX returns.
7. **`managed-agents` backend (Claude-hosted stateful sessions)** — **SHIPPED
   2026-06-07 (conv-6).** A `ConversationBackend` that defers conversation state
   to **Anthropic's Managed Agents** API (`/v1/agents` + `/v1/sessions`, beta
   `managed-agents-2026-04-01`).

   *Shipped scope:* namespace `claude-agent/<model>` → backend `managed-agents`
   (`conversations.ManagedAgentsBackend`, SDK wrapper in `managed_agents.py`). One
   TOOL-LESS agent per model, created lazily + cached on the backend instance
   (never per session); a single shared environment, created once; a session per
   conversation, created on the first turn. `send_turn` streams events to
   `session.status_idle` and collects the `agent.message` text (sending only the
   NEW turn — Anthropic owns prior history). **`history` IS implemented** (parses
   `events.list` → transcript items), so `/v1/conversations/{id}/items` serves the
   transcript here — the first backend for which it does (claude-resume still
   501s). `delete` → `sessions.delete`. Hermetic tests mock the SDK seam
   (`managed_agents._client`); the live gate is `@needs_anthropic` (PAID).
   *Deferred (unchanged from below):* recipe→agent MCP mapping, vaults, file/repo
   resources, the `requires_action` interactive path. *Known limit:* the
   in-memory handle table means a restart orphans live (billed) sessions, and each
   fresh process re-creates its per-model agent — the `ant`-YAML / reuse-by-name
   control plane is the eventual fix.
   The purest embodiment of "backends own state" — Anthropic literally hosts the
   session, the loop, and a per-session container; woollama just routes the
   handle. **An alternative to slices 3/4** (the Rust claude-tmux driver +
   interactive path): it delivers a Claude-hosted, stateful, tool-running,
   interruptible session WITHOUT the §6 terminal-blocked spikes — the hard infra
   is Anthropic's, not ours.

   *Auth/transport:* `ANTHROPIC_API_KEY` (NOT subscription — distinct from the
   keyless `claude-resume`/`claude-tmux` paths) over the `anthropic` SDK; the SDK
   sets the beta header. New routing namespace, e.g. `claude-agent/<model>`, so
   `backend_for_model` maps it here.

   *Interface mapping (§3) — clean:*
   - `create()` → `sessions.create(agent=<agent_id>, environment_id=<env_id>)`;
     the `session_id` is the handle's `native_id`.
   - `send_turn(id, input)` → `sessions.events.send(user.message)` then stream
     events to `session.status_idle`, collecting `agent.message` text → final
     answer. (CMA streams natively → maps onto woollama's SSE orchestration.)
   - `history(id)` → `sessions.events.list()` parsed into transcript items
     (`responses.item_object`) — Anthropic owns the bytes; woollama RETRIEVES.
   - `delete(id)` → `sessions.delete()` (or `archive`).
   - `poll`/`answer` (requires_action, §5) → CMA's `session.status_idle` with
     `stop_reason: requires_action` (a `user.tool_confirmation` /
     `user.custom_tool_result` is pending) maps directly onto the Responses
     `requires_action` path. **This is how the interactive path (slice 4) can
     ship without the tmux driver.**

   *Recipes → agents (the load-bearing setup/runtime split):* a CMA **agent** is
   a persisted, versioned, REUSABLE config (model + system prompt + tools),
   created ONCE — never per-conversation (the documented anti-pattern). woollama
   maps a **recipe → one CMA agent** (the recipe's `system` + `tools` become the
   agent config), created lazily and cached, keyed by a hash of the recipe so a
   recipe edit creates a new agent version. A single **environment** is created
   once. Each conversation is then a **session** referencing that agent. A
   recipe's MCP `tools` can map onto CMA `mcp_servers` + a `mcp_toolset` (with
   credentials in a vault), or onto the built-in `agent_toolset` — so this is
   also a richer **executor** than claude-code delegation (Anthropic hosts the
   tool sandbox).

   *Interactive `requires_action` — SHIPPED 2026-06-07 (§5).* The agent carries
   one client-side custom tool, `ask_user`; when the model calls it the session
   idles with `stop_reason: requires_action`, woollama returns a Responses
   `requires_action` (the tool input is the question), and a continuing turn
   resumes via `user.custom_tool_result`. Custom tools are client-executed, so
   this adds no container provisioning. Hermetically tested (pause→answer
   round-trip, exact tool_use_id, the answer/send_turn routing discriminator); the
   live gate is best-effort (the model must *choose* to call `ask_user`).

   *Scope/tradeoffs:* needs an API key (cost, not subscription); beta API. Still
   deferred: multiagent, outcomes/rubrics, file/repo resources, vault-credentialed
   MCP, recipe→agent MCP mapping. The shipped agent is otherwise tool-less (just
   `ask_user`), so plain Q&A provisions no container.

8. **Store-only backend for non-claude models (issue #2)** — **woollama-side
   mechanism IMPLEMENTED 2026-06-07 behind an un-wired seam; fabric provider +
   contract pending.** The first BYO-inference backend (§3.1, §10): makes
   `ollama/<model>` (and recipes) stateful by deferring the transcript to an
   external store provider and doing assembly + stateless inference woollama-side.
   `ConversationStoreProvider` protocol + `StoreBackedBackend` + routing gate +
   clean error path shipped and tested; no provider ships by default (non-claude
   models stay stateless). Decision: **defer to fabric / the cosmic-fabricd
   session daemon, behind a provider-agnostic store seam** so an MCP
   conversation-store or a JSONL reader can drop in later. The protocol shape is a
   *provisional proposal* fed back to cosmic-fabric (issue #2), not a settled
   contract. See §10.

## 9. Risk flags

- The TUI driver is the fragile part — isolated in Rust on purpose; treat its
  reliability (Esc/Enter races) as the project's main risk.
- One writer per conversation — woollama serializes turns per `conversation_id`.
- woollama holds **handles, not state**. It routes a `conversation_id` to a
  backend that owns the bytes (or runs the turn statelessly); it does NOT store
  transcripts in its own system. The in-memory handle table is just the
  routing map. (conv-5 briefly broke this with an embedded duckdb; reverted.)
- Don't half-implement the Responses spec — minimal subset only (create /
  continue / read / fork / requires_action).

## 10. Pluggable conversation stores — the BYO-inference family (issue #2)

**Goal.** Make `ollama/<model>` (and recipe) conversations stateful through
`/v1/responses` + `/v1/conversations`, so cosmic-fabric can route its session
chat path through woollama instead of bifurcating (claude→woollama,
local→fabric). The agreed target architecture: *woollama is the inference
backbone; fabric sits behind woollama as a pattern source; cosmic-fabricd thins
toward a desktop-session daemon.*

**Why the obvious candidates don't fit.**
- *Managed Agents* (conv-6) pins inference to a **Claude** model — it cannot run
  a local ollama model, so it can't make an ollama session stateful.
- *Ollama itself* has **no server-side sessions** (verified, ollama 0.24.0):
  `/api/chat` is stateless; the `/api/generate` `context` token-id array is
  caller-held, generate-only, opaque (not a readable transcript), and reload-
  fragile — so ollama is not a state owner.
- *An embedded woollama store* is the conv-5 violation — out by principle.

**The decision (2026-06-07).** Defer the transcript to **fabric / the
cosmic-fabricd session daemon** (where these sessions' bytes already live), behind
a **provider-agnostic conversation-store seam** so the choice of owner is
pluggable. woollama stays a router/client to the store; it never holds bytes.

### 10.1 The store-provider seam — IMPLEMENTED (woollama side)

A small protocol woollama is a *client* to (mirrors how claude-resume is a client
to Claude's JSONL and managed-agents to Anthropic's session API). Implemented as
`conversations.ConversationStoreProvider` (named to avoid clashing with the
`ConversationStore` handle table, which is routing state, not transcript bytes):

```
ConversationStoreProvider:                        # PROVISIONAL contract (see 10.2)
    create()                  -> thread_id        # owner mints the thread
    get(thread_id)            -> [messages]       # the transcript (woollama RETRIEVES)
    append(thread_id, turn)   -> None             # user + assistant messages of one turn
    delete(thread_id)         -> None
```

The store-only `StoreBackedBackend` (§3.1) composes a provider with a stateless
inferencer (injected as `complete`, not imported, to avoid a conversations↔router
cycle — the router passes `complete_stateless`):

```
send_turn(conv, input):
    if conv.native_id is None: conv.native_id = store.create()
    prior   = store.get(conv.native_id)            # bytes owned by the provider
    answer  = complete(conv.model, prior + input)  # stateless inference (woollama ASSEMBLY)
    store.append(conv.native_id, input + [answer]) # write the turn back
    return answer
history(conv): return store.get(conv.native_id)    # /items works for free
```

`native_id` = the provider's thread key (a fabric `sessionName`). One writer per
conversation (existing per-conv lock). Reuses conv-6's handle-table scaffolding
verbatim — only the owner of the bytes differs. Routing: `backend_for_model`
returns the store backend for any non-claude model **iff** a provider has been
wired in via `register_store_backend` — none ships by default, so non-claude
models stay stateless until a provider exists (the existing `ollama→501` test is
the no-regression gate). Hermetically tested with an in-memory fake provider
(`tests/test_store_backend.py`): assemble→complete→append, cross-turn recall via
reassembly, `/items`, delete, the routing gate, and that an inference failure
surfaces cleanly (not a 500).

**#1 ↔ #2 seam — CLOSED (2026-06-07):** the `/v1/responses` request's `options`
(e.g. `num_ctx`) are threaded through `send_turn` → the injected
`complete_stateless`, which now routes ollama through the native `/api/chat` when
`num_ctx` is present — so a store-backed (and plain stateless) ollama turn sizes
its context too. (`complete_stateless`'s recipe/orchestrate branch is unaffected;
num_ctx applies to direct ollama models.)

### 10.2 First provider: fabric / cosmic-fabricd — PENDING (the contract proposal)

The woollama-side mechanism (10.1) is done; what remains is a concrete
`ConversationStoreProvider` for fabric. The `create/get/append/delete` shape above
is woollama's **proposed read/append contract** — fabric hasn't agreed it yet.
**Coordination needed (cosmic-fabric side):** fabric must expose, to woollama, a
session **read** (transcript by `sessionName`) and **append** (one turn) mapping
onto that Protocol — today fabric owns `sessionName` sessions but this consume
surface needs agreeing (transport: the owner-only UDS woollama already serves on,
or a fabric endpoint woollama calls). Once agreed, the provider is a thin adapter
+ a `register_store_backend(provider, complete_stateless)` call at startup; then
cosmic-fabric maps a cosmic session name → a woollama `conversation_id` and drives
turns via `/v1/responses` (`store:true`/`conversation`). The proposal has been
fed back to cosmic-fabric (issue #2) as the "confirm" step.

### 10.3 Future providers (pluggable, not now)

- **MCP conversation-store** — woollama as an MCP client to a server exposing
  get/append/create/delete (most general; works for any provider). The seam above
  is deliberately shaped so this is a drop-in.
- **JSONL reader** — read a claude-resume-style on-disk transcript as a provider
  (read-only history for a native-loop owner).

### 10.4 Scope / status

**Woollama-side mechanism IMPLEMENTED + hermetically tested (2026-06-07), behind
an un-wired seam.** The protocol, the `StoreBackedBackend`, the routing gate, and
the clean error path all exist and pass; no provider ships, so runtime behavior is
unchanged (non-claude models stay stateless). What remains is cross-repo and
deliberately not guessed: (1) **the contract** — the `create/get/append/delete`
shape is woollama's *proposal* to fabric, not an agreed contract; (2) **the fabric
provider** — a thin adapter once that's settled; (3) **cosmic-fabric wiring**
(part of the last v1.0 gate). This stages #2 as a *visible proposal that invites
correction* rather than a silent guess baked into a hard-to-revert backend.
Tracked as issue #2 / roadmap conv-7; build-sequence item §8.8.
