# woollama roadmap & status

Single source of truth for *what's built, what's next, and in what order.*
Updated 2026-06-03. Detailed history: [`build-log.md`](build-log.md). Target
design: [`architecture.md`](architecture.md). v1.0 gate:
[`rust-transition.md`](rust-transition.md).

woollama is a **router** between OpenAI-/MCP-speaking clients and OpenAI-/MCP-
speaking backends. It owns routing and composition (recipes); it does not own
inference or tools. Still the **Python prototype** — Rust is v1.0 (see gate).

## Shipped

| Capability | Where | Slice |
|---|---|---|
| OpenAI HTTP surface (`/v1/models`, `/v1/chat/completions`) | `router.py` | a |
| `ollama/<model>` pass-through | `router.py` | a |
| Recipe orchestration (hidden chat-loop) | `router.py:orchestrate` | a |
| Config files (`mcp.json`, `recipes.toml`) | `config.py` | b |
| Multi-MCP-server discovery + unified tool registry; long-lived connections | `manager.py` | c |
| Broad test suite (unit + opt-in integration) | `tests/` | d |
| **woollama AS an MCP server** — recipes→prompts, `chat` verb→tool (stdio) | `mcp_server.py` | e |
| **MCP aggregator** — re-export every downstream tool, namespaced | `mcp_server.py` | f |
| Recipe **allow-list boundary** (a recipe can't dispatch out-of-list tools) | `router.py:orchestrate` | f |
| Routing demo + hermetic routing matrix | `examples/routing_demo.py`, `tests/test_routing.py` | f |
| **MCP over Streamable HTTP**, mounted on one port (`/v1/*` + `/mcp`) | `router.py` | h |
| **Claude Code** as a (tool-less) inference backend (keyless) | `claude_code.py` | i |
| **OpenAI-compat inferencer seam** (multi-backend router) | `inferencers.py` | j |
| Cloud providers: anthropic, openai, groq, together, openrouter + ollama | `inferencers.py` | j, k |
| **Config-file inferencers** (`inferencers.toml`) — any OpenAI-compat backend | `config.py`, `inferencers.py` | k |
| **Streaming passthrough** — `stream:true` on `<provider>/<model>` relays upstream SSE verbatim | `router.py:_passthrough_stream` | streaming-1 |
| **Streaming orchestration** — `stream:true` on `woollama/<recipe>` streams the answer as OpenAI SSE; tool turns stay hidden. Core loop is now one async generator (`orchestrate_events`); `orchestrate` is a thin drainer | `router.py` | streaming-2 |
| **MCP progress events** — the `chat` tool emits a `ctx.info` notification per tool call/result during the hidden loop (live progress; return value unchanged) | `mcp_server.py`, `router.py` | streaming-3 |
| **Unix socket alongside HTTP loopback** — one app on a UDS (`$XDG_RUNTIME_DIR/woollama.sock`, mode 0600) + the loopback TCP port | `binding.py`, `__main__.py` | unix-socket |
| **`/v1/responses` (stateless subset)** — OpenAI Responses-shaped superset of chat-completions (`store:false`), SDK-verified | `responses.py`, `router.py` | conv-1a |
| **Stateful `/v1/responses`** — handle table routes `conversation_id` → backend; `claude-resume` backend (`store:true`/`conversation`/`previous_response_id`); live-verified | `conversations.py`, `router.py` | conv-1b |
| **`/v1/conversations`** — discovery/attach: create, list, get, delete (handle table; OpenAI Conversation shape + routing extras) | `router.py`, `conversations.py` | conv-2 |
| **`managed-agents` backend** — `claude-agent/<model>` → Anthropic Managed Agents (hosted session owns state); implements `history` so `/items` serves the transcript | `managed_agents.py`, `conversations.py` | conv-6 |
| Lint-clean (`ruff check .`) | tree-wide | — |

Surfaces today: `/v1/chat/completions` (pass-through AND `woollama/<recipe>`
orchestration, both with `stream:true` → OpenAI SSE), `/v1/responses` (stateless
subset + stateful via the `claude-resume` and `managed-agents` backends — OpenAI
Responses shape; non-claude models are stateless-only, `store:false`),
`/v1/conversations` (create/list/get/delete + `items` for managed-agents),
`/v1/models`,
`/v1/tools`, `/mcp` (Streamable HTTP),
and `woollama mcp` (stdio) — served on BOTH a Unix socket
(`$XDG_RUNTIME_DIR/woollama.sock`) and the loopback TCP port.

## Open tracks (recommended order)

1. ~~**Streaming**~~ — ✅ DONE (all three slices). OpenAI SSE out + MCP progress
   events. Reshaped `orchestrate` into one async generator without forking the
   loop. Highest value for the cosmic-fabric panel. Slices:
   - [x] **streaming-1: passthrough SSE** — `stream:true` on `<provider>/<model>`
     relays the upstream stream verbatim (`router.py:_passthrough_stream`).
   - [x] **streaming-2: orchestration SSE** — `stream:true` on `woollama/<recipe>`
     streams the answer as OpenAI SSE; tool-call JSON/results stay hidden and the
     per-turn `finish_reason`/`[DONE]` are swallowed (one synthesized terminator).
     The core loop is now the async generator `orchestrate_events`; `orchestrate`
     is a thin drainer (single source of truth preserved). Product note: in
     streaming mode every turn's *content* is surfaced (continuous assistant
     message), so it can show more prose than the non-streaming path, which
     returns only the final turn — a deliberate, documented divergence.
   - [x] **streaming-3: MCP progress events** — the `chat` tool emits a
     `ctx.info` notification per tool call/result during the hidden loop, so a
     connected MCP client sees live progress through the tool turns. The tool's
     return value (the final answer) is unchanged. The shared loop now also
     yields `tool_call`/`tool_result` events; HTTP adapters ignore them.
2. ~~**Unix socket transport**~~ — ✅ DONE. One uvicorn server binds the app to
   a UDS (`$XDG_RUNTIME_DIR/woollama.sock`, mode 0600 — a connectable socket can
   spend the router's keys) alongside the loopback TCP port (`binding.py`,
   `__main__.py`). Verified live: both surfaces serve; cleanup on shutdown.
3. **Conversations / Responses** (stateful surface) — scoped in
   [`conversations-api-design.md`](conversations-api-design.md). woollama routes
   conversation *handles*; backends own state (incl. a live Claude-in-tmux
   session driven by a separate Rust package). Build order is in that doc.
   - [x] **conv-1a** — `/v1/responses` stateless subset (`store:false`); the
     OpenAI Responses wire shape, verified live via the `openai` SDK.
   - [x] **conv-1b** — in-memory handle table (`conversation_id` → backend +
     claude `session_id` + stable workdir; one writer per conversation) +
     `claude-resume` backend + `store:true` / `conversation` /
     `previous_response_id` routing. Live-verified (create → resume → recall).
   - [x] **conv-2** — `/v1/conversations` create / list / get / delete (the
     discovery + attach + teardown surface). Live CRUD verified; the full e2e
     journey (create → discover → two-turn recall → `items` 501 → delete) is
     live-green against `claude-resume` post-revert (2026-06-07, 17s).
   - [~] **conv-5** — duckdb `stored` backend: SHIPPED 2026-06-05, **REVERTED
     2026-06-06**. It made woollama own conversation storage (embedded duckdb),
     contradicting the design principle (woollama routes handles, never owns
     state). Non-claude models are now **stateless-only** (`store:false`; a clean
     501 on `store:true`). Stateful conversations for them, if ever needed, must
     defer to an EXTERNAL owner (a conversation-store MCP server, or Managed
     Agents) — not woollama's own DB. See conversations-api-design.md §8.5.
   - [x] **conv-6 — `managed-agents` backend** SHIPPED 2026-06-07 (design-doc
     §8.7): defers conversation state to Anthropic's `/v1/agents` +
     `/v1/sessions`. Namespace `claude-agent/<model>`; one tool-less agent per
     model (cached, reused), a session per conversation; `send_turn` streams to
     idle, `delete` → `sessions.delete`. The purest "backend owns state" — and
     the FIRST backend to implement `history`, so `/items` serves the transcript
     (claude-resume still 501s). Needs an API key (paid, not subscription).
     Hermetic-tested (SDK seam mocked) + SDK signatures introspected against
     `anthropic==0.107.1` + **live round-trip verified 2026-06-07** (15s: create →
     two-turn recall → `/items` 200 → delete). Deferred: recipe→agent MCP mapping,
     vaults, outcomes/multiagent.
   - [x] **conv-8 — interactive `requires_action` path** SHIPPED 2026-06-07
     (design-doc §5) via the managed-agents backend, WITHOUT the §6-blocked tmux
     driver: the agent carries an `ask_user` custom tool; calling it idles the
     session with `stop_reason: requires_action` → woollama returns a Responses
     `requires_action` (the question rides `required_action`), and a continuing
     turn resumes via `user.custom_tool_result`. Hermetic round-trip (pause→answer,
     exact tool_use_id, the answer/send_turn routing discriminator); live gate is
     best-effort (the model must choose to call ask_user) — paid, written-not-run.
   - [ ] conv-3/4 — the Rust session driver + claude-tmux backend (gated on the
     §6 INTERACTIVE spikes — these genuinely hang nested, unlike `-p`); maps the
     LIVE-TUI pause onto the same `requires_action` primitive (now shipped via
     managed-agents); cosmic-fabric wiring.
   - [~] **conv-7 — store-only backend for non-claude models** (issue #2):
     woollama-side mechanism IMPLEMENTED 2026-06-07 (design-doc §10) behind an
     UN-WIRED seam; fabric provider + contract pending. `ConversationStoreProvider`
     protocol + `StoreBackedBackend` (assemble prior history → stateless inference
     → append turn) + routing gate + clean error path, hermetically tested
     (`tests/test_store_backend.py`, 9). No provider ships by default, so non-claude
     models stay stateless (the `ollama→501` test is the no-regression gate). The
     **#1↔#2 seam is CLOSED**: request `options` (num_ctx) thread through
     `send_turn`→`complete_stateless`, which routes ollama native — so stateful
     ollama turns size their context (live-verified on the stateless /v1/responses
     path; /api/ps shows the requested ctx).
     Managed Agents was ruled out (pins inference to Claude); ollama has no native
     sessions (verified). Remaining (cross-repo, not guessed): the fabric
     read/append contract — woollama's `create/get/append/delete` proposal was fed
     back to cosmic-fabric (#2) — then a thin fabric provider + the wiring.
4. **Rust port (v1.0)** — last, once the design freezes. See the gate.

Smaller follow-ons (not blocking):
- ~~**Honor `num_ctx` for ollama**~~ (#1) — ✅ DONE 2026-06-07. `ollama/<model>`
  passthrough with `options.num_ctx` routes to ollama's native `/api/chat`
  (which honors it; `/v1` ignores it), translating request + response (stream +
  non-stream) back to the OpenAI shape (`ollama_native.py`). Live-verified via
  `/api/ps` (context_length=16384). Tools + num_ctx stay on `/v1` (documented).
- Config-file-driven inferencers shipped; could add more built-in clouds
  (deepseek/xai/mistral) — but config already covers them.
- ~~A pre-commit / CI hook so `ruff` actually gates~~ — ✅ DONE. GitHub Actions
  CI (`.github/workflows/ci.yml`) runs `ruff check .` + the hermetic suite on
  push/PR (3.11 + 3.12); an opt-in `.pre-commit-config.yaml` mirrors the lint
  gate locally. Lint only (no `ruff format` — the tree is hand-wrapped, `E501`
  ignored).
- ~~`output_schema` pass-through on re-exported proxy tools~~ — ✅ DONE. The
  aggregator now mirrors each downstream tool's `output_schema` onto its
  re-exported proxy, so it's advertised on `tools/list` and enforced on results
  (safe: a downstream that declares a schema has already validated its own
  output before woollama forwards it — confirmed live via `hello.count_to`). A
  non-conforming downstream surfaces a clear output-validation error (the
  faithful-proxy choice), covered by a hermetic test.
- ~~Tool DELEGATION to Claude Code (Claude owns the loop, runs a recipe's MCP
  tools)~~ — ✅ DONE. A `claude-code/<model>` recipe WITH a non-empty `tools`
  list now delegates: Claude Code owns the agentic loop and calls the recipe's
  allow-listed tools itself (`claude_code.run_delegated`), woollama returns the
  final answer. Option B (config containment): woollama writes a per-recipe
  `--mcp-config` with ONLY the servers the allow-list references + `--allowedTools`
  with ONLY those tools — a HARD boundary (spike-verified: `dontAsk` denies
  anything unlisted), on top of the slice-i built-in lockdown. The child env now
  also strips `CLAUDE_CODE*`/`CLAUDECODE` (nested-harness contamination, found via
  the spike). Hermetic unit + routing tests cover the construction/boundary; the
  positive + adversarial *live* gate is plain-terminal-only (see below).

## Pending verifications (need a real terminal + creds; can't run nested)

- **Claude Code tool-lockdown** (slice i): the runtime safety boundary is
  verified by construction + unit tests, NOT live. Run in a plain terminal:
  `WOOLLAMA_TEST_CLAUDE_CODE=1 uv run --extra dev pytest tests/test_integration.py -m integration -k claude_code`
  (checks a real completion works AND neither a shell-exec nor a file-read
  prompt-injection succeeds).
- ~~**Tool delegation** (executor)~~ — ✅ VERIFIED + HARDENED. The lockdown is
  `--tools ""` (an allow-list of built-in tools set to NONE) — robust against
  whatever tools a deployment's global config / plugins enable, which a deny-list
  can't enumerate and `dontAsk` doesn't deny (it only hard-denies tools that
  *prompt*; auto-approved extras like Skill/Workflow slipped through the old
  deny-list). `ENABLE_TOOL_SEARCH=false` keeps the recipe's MCP tools reachable
  (they load upfront once the built-in ToolSearch is gone). Verified at the event
  level through woollama's real code: the delegated Claude exposes ONLY the
  recipe's MCP server tools (no built-ins, no LSP, no harness tools), the
  out-of-list tool is HARD-denied, the delegated tool runs, and a shell-exec
  attempt is refused (Bash is absent, not merely denied). Because `--tools ""`
  strips the harness, the live gate now runs trustworthily even nested:
  `WOOLLAMA_TEST_CLAUDE_CODE=1 uv run --extra dev pytest tests/test_integration.py -m integration -k delegation`
  The **adversarial safety pass** (the executor's flagged prerequisite) is also
  done — review fixed a provider-key env leak (child env is now an allow-list), a
  host-settings undercut (`--setting-sources project`), and a tool-name comma
  injection; SQL/argv/JSON-injection + path surfaces verified safe; same-server
  sibling-tool denial now has a live test. See docs/build-log.md (2026-06-06).
- **Anthropic (and other cloud) live round-trips** (slices j/k): routing/auth
  is unit-tested on the emit side + doc-confirmed (tools supported); the live
  round-trip is unverified without keys. With `ANTHROPIC_API_KEY` set:
  `uv run --extra dev pytest tests/test_integration.py -m integration -k anthropic`
- ~~**managed-agents backend live round-trip**~~ (conv-6) — ✅ VERIFIED
  2026-06-07 against the real Managed Agents API (`claude-agent/haiku`, 15s):
  create (backend `managed-agents`) → two-turn recall proving Anthropic resumed
  the hosted session → `/items` 200 serving the transcript → session delete →
  404. **PAID + creates persistent account objects** (a per-session container +
  a per-model agent that survives session delete). Re-run (launch WITH the
  `agents` extra so the server subprocess has the SDK):
  `uv run --extra dev --extra agents pytest tests/test_integration.py -m integration -k managed_agents_conversation_journey_live`
- ~~**Streaming orchestration against real Ollama**~~ (slice streaming-2): ✅
  VERIFIED 2026-06-04 against real Ollama (`qwen3:14b-iq4xs`) — fragmented
  tool_call SSE deltas reassemble, the tool loop stays hidden, and the answer
  streams with one terminator (`test_orchestrated_recipe_streams_final_answer_hiding_tool_loop`,
  alongside the non-streaming + two-provider + MCP chat live tests). Re-run:
  `uv run --extra dev pytest tests/test_integration.py -m integration -k "stream or orchestrat or two_provider"`

## v1.0 (Rust) gate — progress

From [`rust-transition.md`](rust-transition.md), criterion #2 ("Python surface
covers the v1.0 feature set"):

- [x] real config files (`recipes.toml` + `mcp.json`)
- [x] multi-MCP-server discovery + unified tool registry
- [x] the Anthropic backend
- [x] woollama-as-MCP-server side
- [x] long-lived MCP connections (was the criterion-#4 latency concern)
- [x] streaming on both sides (OpenAI SSE out — passthrough + orchestration; MCP
      progress events on the `chat` tool)
- [x] Unix socket alongside HTTP loopback
- [ ] the panel-confirm round-trip equivalent (the conversations surface +
      cosmic-fabric consuming it)

Criteria #1 (architecture stable), #3 (a real consumer — cosmic-fabric actively
using it), #4 (a specific Python limit biting) are not yet all met → keep
iterating Python.
