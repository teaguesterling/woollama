# Changelog

## Unreleased

- **Cloud models discoverable in `GET /v1/models`** (#3): each inferencer can opt
  in via `inferencers.toml` — a static `models = [...]` list and/or live
  `discover = true` that queries the provider's own `/v1/models`, filtered by
  `model_patterns` (fnmatch globs, e.g. `["claude-*", "gpt-4*"]`) so a huge
  catalog can be narrowed. Built-in cloud providers surface nothing until
  configured (no regression; ollama still auto-discovers its local catalog).
  Config now merges over built-ins field-by-field, so you can add `models` to
  `anthropic` without restating its `base_url`. Closes #3.

## v0.3.0 — 2026-06-07

Still the Python prototype (v1.0 is the Rust rewrite). Conversation-surface
release: a second state-owning backend (Managed Agents), the interactive
pause/answer path, streaming `/v1/responses`, ollama context-window control, and
the woollama side of a pluggable store backend. Authoritative live status is
`docs/roadmap.md`.

- **Streaming `/v1/responses`** (conv-1a streaming): a stateless `stream:true`
  turn now emits OpenAI **Responses SSE** (`response.created` →
  `output_text.delta`* → `response.completed`), sourcing deltas from a recipe
  (`orchestrate_events`, tool turns hidden) or a plain inferencer's chat SSE. The
  emitted frames validate against the `openai` SDK event models. Stateful
  streaming stays deferred (400). See `docs/conversations-api-design.md` §1.
- **Interactive `requires_action` path** (conv-8) via the managed-agents backend
  — without the tmux driver. The hosted agent carries an `ask_user` custom tool;
  when it's called the session pauses, woollama returns a Responses
  `status:"requires_action"` carrying the question, and continuing the
  conversation with the answer resumes it (`user.custom_tool_result`). The
  `requires_action` response is a documented superset of the OpenAI Responses
  shape. See `docs/conversations-api-design.md` §5.
- **Store-backed conversation backend** (#2, woollama-side mechanism): a
  store-only / BYO-inference backend that makes non-claude models (ollama, cloud,
  recipes) stateful by deferring the transcript to an external
  `ConversationStoreProvider` and doing assembly + stateless inference
  woollama-side — woollama never owns the bytes. Ships behind an **un-wired seam**
  (no provider by default, so those models stay stateless until one is
  registered); the provider contract is a *provisional proposal* to fabric. See
  `docs/conversations-api-design.md` §10. Stateful/store-backed ollama turns
  honor `num_ctx` too (request `options` thread through to `complete_stateless`,
  which routes ollama native — the #1↔#2 seam, closed and live-verified on the
  stateless `/v1/responses` path).
- **Ollama `num_ctx` honored** (#1): `ollama/<model>` passthrough requests that
  ask for a context size (`options.num_ctx`) now route to ollama's native
  `/api/chat` (which honors it) instead of the OpenAI-compat `/v1` endpoint
  (which silently ignores it), translating the request and the response (stream
  + non-stream) back to the OpenAI shape. Requests without `num_ctx`, and those
  with `tools`, stay on `/v1` unchanged. Live-verified: `/api/ps` reports the
  requested context.

## v0.2.0 — 2026-06-07

Still the Python prototype (v1.0 is the Rust rewrite — see
`docs/rust-transition.md`); these are committed slice-by-slice (see
`docs/build-log.md`). This resolves essentially every "queued for v0.2"
limitation listed under v0.1.0 below. Authoritative live status is
`docs/roadmap.md`.

### Surfaces

- **Streaming on both sides.** `stream:true` on `<provider>/<model>` relays the
  upstream SSE verbatim; on `woollama/<recipe>` it streams the answer as OpenAI
  SSE with the tool loop hidden (one async generator, `orchestrate_events`). The
  MCP `chat` tool emits a progress notification per tool call/result.
- **Stateful surface** (`docs/conversations-api-design.md`): `/v1/responses`
  (stateless subset + stateful) and `/v1/conversations` (create/list/get/delete
  + transcript `items`), in the OpenAI Responses/Conversations shape. woollama
  routes conversation *handles*; backends own the state.
- **MCP over Streamable HTTP** at `/mcp`, mounted on the same port as `/v1/*`,
  plus the stdio `woollama mcp` server. **Aggregator**: every downstream tool is
  re-exported namespaced, now carrying its `output_schema`; recipes become MCP
  prompts; a `chat` verb runs orchestration.
- **Unix socket** at `$XDG_RUNTIME_DIR/woollama.sock` (mode 0600) served
  alongside the loopback TCP port — the default for local MCP clients.
- `/v1/tools` introspection endpoint.

### Backends & routing

- **Multi-backend inferencer seam**: anthropic, openai, groq, together,
  openrouter built in, plus any OpenAI-compatible endpoint via `inferencers.toml`
  (e.g. self-hosted vLLM).
- **Claude Code** as a keyless inference backend (subscription auth), tool-less
  AND as an **executor** (tool delegation): a `claude-code` recipe with tools
  lets Claude own the agentic loop and call the recipe's allow-listed MCP tools
  itself, contained by a per-recipe `--mcp-config` + `--allowedTools`.
- **Conversation backends** (woollama routes handles; backends own state):
  - `claude-resume` (`claude --resume`, for `claude-code/<model>`) — the native
    Claude session owns the bytes; keyless/subscription.
  - `managed-agents` (Anthropic Managed Agents, for `claude-agent/<model>`) —
    Anthropic hosts the session + container; `ANTHROPIC_API_KEY` (paid). The
    first backend to implement transcript retrieval, so
    `/v1/conversations/{id}/items` serves it. (In the `agents` optional extra.)
  - Models with no state-owning backend are stateless (`store:false`). (A duckdb
    `stored` backend was briefly added and reverted — woollama does not store
    conversations in its own system; it routes handles to backends that own the
    state.)

### Platform

- **Multi-MCP-server discovery + unified tool registry** with long-lived
  connections (replaces per-request subprocess spawning).
- **File-driven config**: `mcp.json`, `recipes.toml`, `inferencers.toml`
  (`${VAR}` expansion; inferencers merge over built-ins).
- **Recipe allow-list** enforced as a security boundary (in the orchestration
  loop AND in delegation).
- **CI**: GitHub Actions runs `ruff check` + the hermetic suite on Python
  3.11/3.12; opt-in pre-commit hook mirrors the lint gate.
- **Documentation site**: MkDocs (Material) over the existing Markdown docs,
  published on ReadTheDocs at <https://woollama.readthedocs.io/>.

## v0.1.0 — 2026-05-31

First public version. **Working router; Python prototype, not production.**
Architecture validated end-to-end; v0.2 will harden, configure, and expand
the prototype. **v1.0 is a Rust rewrite** once the architecture stabilizes —
see `docs/rust-transition.md` for the explicit criteria.

### What v0.1 does

- **OpenAI-compatible HTTP surface** at `/v1/models` and `/v1/chat/completions`.
- **Model namespace routing**:
  - `ollama/<name>` — pure pass-through to local Ollama at `localhost:11434`
  - `woollama/<recipe>` — orchestrated chat-loop using the named recipe
- **One bundled example recipe** (`woollama/streamer`) demonstrating
  pattern + tools + inferencer composition.
- **MCP tool dispatch** via per-request stdio connection to the bundled
  hello server (`examples/mcp-hello/server.py`).
- **Ephemeral local-only binding** — random free loopback port at startup,
  persisted to `$XDG_RUNTIME_DIR/woollama.addr` for client discovery. Never
  binds to `0.0.0.0` without explicit `WOOLLAMA_ADDRESS` override.
- **Smoke tests** that don't require Ollama or network.

### Design ideas validated

- MCP + OpenAI compose as complementary standards without extension
- The model namespace (`<provider>/<name>`) is a sufficient addressing
  scheme for raw / pattern / variant / recipe model kinds
- Recipe orchestration is invisible to OpenAI clients — they get one final
  answer; the chat-loop happens inside the router
- Ephemeral local binding works for the OpenAI SDK out of the box (clients
  read the addr-file)

### Known limitations / queued for v0.2

- **No streaming** on either side (non-streaming round-trips only)
- **One hardcoded recipe** — real `~/.config/woollama/recipes.toml` to follow
- **One MCP server** — multi-server discovery + unified tool registry to follow
- **Ollama only** — Anthropic / OpenAI / vLLM via OpenAI-compat to follow
- **No Unix socket** transport — HTTP loopback only
- **woollama as MCP server** to its own clients is not yet implemented
- **No CI** — manual smoke tests; pytest config added but no GitHub Actions yet
- **Per-request MCP subprocess** is correct but slow; connection pooling to follow

### Origin

woollama is the rewrite of an architecture co-designed in [cosmic-fabric](
https://github.com/teaguesterling/cosmic-fabric), which remains as a frontend
client. The full design context lives in `docs/architecture.md` and
`docs/naming.md`.
