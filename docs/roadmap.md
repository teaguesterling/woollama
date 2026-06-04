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
| **Unix socket alongside HTTP loopback** — one app on a UDS (`$XDG_RUNTIME_DIR/woollama.sock`, mode 0600) + the loopback TCP port | `binding.py`, `__main__.py` | unix-socket |
| Lint-clean (`ruff check .`) | tree-wide | — |

Surfaces today: `/v1/chat/completions` (+ pass-through, with `stream:true`),
`/v1/models`, `/v1/tools`, `/mcp` (Streamable HTTP), and `woollama mcp` (stdio)
— served on BOTH a Unix socket (`$XDG_RUNTIME_DIR/woollama.sock`) and the
loopback TCP port.

## Open tracks (recommended order)

1. **Streaming** — OpenAI SSE out + MCP progress events. The biggest remaining
   UX item and the largest change (reshapes `orchestrate` + both surfaces from
   request/response to streaming). Highest value for the cosmic-fabric panel.
   Done in slices:
   - [x] **streaming-1: passthrough SSE** — `stream:true` on `<provider>/<model>`
     relays the upstream stream verbatim (`router.py:_passthrough_stream`).
   - [ ] **streaming-2: orchestration SSE** — stream the `woollama/<recipe>` final
     turn; tool-call turns stay invisible. Make the core loop an async generator
     so `orchestrate` stays the single source of truth (its docstring forbids a
     forked loop); non-streaming `orchestrate` drains it for the final dict.
   - [ ] **streaming-3: MCP progress events** — progress notifications on the MCP
     `chat` tool during tool turns.
2. ~~**Unix socket transport**~~ — ✅ DONE. One uvicorn server binds the app to
   a UDS (`$XDG_RUNTIME_DIR/woollama.sock`, mode 0600 — a connectable socket can
   spend the router's keys) alongside the loopback TCP port (`binding.py`,
   `__main__.py`). Verified live: both surfaces serve; cleanup on shutdown.
3. **Conversations / Responses** (stateful surface) — scoped in
   [`conversations-api-design.md`](conversations-api-design.md). woollama routes
   conversation *handles*; backends own state (incl. a live Claude-in-tmux
   session driven by a separate Rust package). Build order is in that doc.
4. **Rust port (v1.0)** — last, once the design freezes. See the gate.

Smaller follow-ons (not blocking):
- Config-file-driven inferencers shipped; could add more built-in clouds
  (deepseek/xai/mistral) — but config already covers them.
- A pre-commit / CI hook so `ruff` actually gates (it's configured but not run
  by pytest; it drifted once and was cleaned in `7da4b52`).
- `output_schema` pass-through on re-exported proxy tools (currently content +
  structuredContent only).
- Tool DELEGATION to Claude Code (Claude owns the loop, runs a recipe's MCP
  tools) — a separate "executor" concept; needs its own adversarial safety pass.

## Pending verifications (need a real terminal + creds; can't run nested)

- **Claude Code tool-lockdown** (slice i): the runtime safety boundary is
  verified by construction + unit tests, NOT live. Run in a plain terminal:
  `WOOLLAMA_TEST_CLAUDE_CODE=1 uv run --extra dev pytest tests/test_integration.py -m integration -k claude_code`
  (checks a real completion works AND neither a shell-exec nor a file-read
  prompt-injection succeeds).
- **Anthropic (and other cloud) live round-trips** (slices j/k): routing/auth
  is unit-tested on the emit side + doc-confirmed (tools supported); the live
  round-trip is unverified without keys. With `ANTHROPIC_API_KEY` set:
  `uv run --extra dev pytest tests/test_integration.py -m integration -k anthropic`

## v1.0 (Rust) gate — progress

From [`rust-transition.md`](rust-transition.md), criterion #2 ("Python surface
covers the v1.0 feature set"):

- [x] real config files (`recipes.toml` + `mcp.json`)
- [x] multi-MCP-server discovery + unified tool registry
- [x] the Anthropic backend
- [x] woollama-as-MCP-server side
- [x] long-lived MCP connections (was the criterion-#4 latency concern)
- [ ] streaming on both sides
- [x] Unix socket alongside HTTP loopback
- [ ] the panel-confirm round-trip equivalent (the conversations surface +
      cosmic-fabric consuming it)

Criteria #1 (architecture stable), #3 (a real consumer — cosmic-fabric actively
using it), #4 (a specific Python limit biting) are not yet all met → keep
iterating Python.
