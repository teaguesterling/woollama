# Changelog

## Unreleased

**Surface authentication + fail-closed binding.** The HTTP surfaces (`/v1/*` and
the mounted `/mcp`) are now access-controlled: with no token configured, only
*local* peers (loopback TCP, the 0600 Unix socket) are served; a non-loopback
`WOOLLAMA_ADDRESS` refuses to start unless `WOOLLAMA_TOKEN` is set; with a token
set, every TCP request must send `Authorization: Bearer <token>` (the Unix
socket stays exempt — its file mode is the credential). The default loopback,
no-token workflow is unchanged.

- **Recipe allow-list enforced at dispatch time (Python).** `Registry.dispatch`
  now takes the active allow-list and refuses a tool outside it; the recipe
  loop's `RegistryToolProvider` carries the recipe's `tools` list, so the
  boundary holds in Python independent of the core's offer-time filtering. The
  MCP aggregator surface (which re-exports every configured tool by design) is
  unchanged and gated by surface auth.
- **`mcp.json` `env` now reaches the spawned server** (via
  `StdioServerParameters.env`, merged over the SDK's safe default environment)
  instead of being parsed and dropped — no more `${VAR}`-into-argv workaround
  that leaked values into `ps`.
- **Downstream MCP tool calls are time-bounded** (`WOOLLAMA_TOOL_TIMEOUT`,
  default 180s): a hung server bounds the turn and no longer wedges the
  connection's worker for subsequent calls.
- **Managed-agents environments default to `limited` networking** (least
  privilege for the tool-less agent); `WOOLLAMA_AGENT_NETWORKING=unrestricted`
  restores the previous behavior.
- The durable conversation handle table (`conversations.json`) is written
  owner-only (0600).

## v0.7.0 — 2026-06-30

**Breaking:** `GET /w1/patterns` `variables` changes shape — bare name strings
become objects (`{name, default?, choices?, description?}`). Clients that read only
the variable `name` (e.g. cosmic-fabric's `WoollamaClient`) are unaffected; any
client that consumed `variables` as a `string[]` must update.

- **Vision (image input) for fabric patterns.** A `/w1/patterns/{name}/run` whose
  `input` carries an OpenAI `image_url` content part is dispatched to fabric's
  one-shot CLI (`fabric --attachment=…`, user text on stdin) — fabric's REST
  `/chat` has no attachment field. `http(s)://` image URLs pass through; `data:`
  URLs are decoded to a temp file (cleaned up after). Needs a vision-capable
  `model` (e.g. `ollama/llama3.2-vision`); one image per run (fabric `-a` is
  single-attachment); non-streaming (a `stream:true` request still gets the OpenAI
  SSE shape). As a byproduct, array-`content` messages no longer drop their text on
  the fabric REST path (previously `content` arrays were ignored entirely).
- **Native multimodal (`image_url`) confirmed.** A NATIVE recipe bound to (or
  `model`-overridden with) a vision model accepts `image_url` content with no
  special handling — the engine already forwards the messages array verbatim to
  ollama's OpenAI-compatible endpoint. Works on `/w1/…/run` and via
  `/v1/chat/completions` as `woollama/<recipe>`. Locked with a regression test; no
  engine change (it stays parity-locked).
- **Variable-metadata overlay for `/w1/` patterns.** Native recipes can annotate
  their `{{var}}` tokens with a `default`, `choices`, and `description` via an
  optional `[recipes.<name>.variables.<var>]` table in `recipes.toml`. `GET
  /w1/patterns` now surfaces `variables` as objects (`{name, default?, choices?,
  description?}`, absent fields omitted) instead of bare name strings; `default`s
  are applied wherever a recipe renders — `/w1/.../render`, `/w1/.../run`, and the
  MCP `prompts/get` surface — when the caller omits a variable (caller-supplied
  wins; `choices` is advisory, not enforced); `description` carries across to the
  MCP prompt argument. fabric-library patterns are unaffected (still `[]`).

## v0.6.0 — 2026-06-22

**Pattern templating (`/w1/`) + the fabric backend.** woollama can now own prompt
templating and front a full fabric deployment, behind a pluggable backend seam.

- **`/w1/` — woollama-native pattern templating.** A namespace parallel to `/v1/`:
  `GET /w1/patterns` (discovery), `POST /w1/patterns/{name}/render` (substitute
  `{{vars}}` without running), `POST /w1/patterns/{name}/run` (render then infer →
  an OpenAI completion/SSE). Patterns *are* recipes — a recipe whose `system`
  carries `{{var}}` tokens — plus an optional fabric-style `[patterns]` directory
  scan. Substitution is byte-compatible with fabric's (a dumb `{{k}}`→value
  replace); the engine never sees a `{{var}}`.
- **MCP prompts.** Recipes are exposed as MCP prompts on `/mcp`; their `{{var}}`
  tokens become prompt arguments and `prompts/get` renders them.
- **The fabric backend.** An optional `fabric` key in `mcp.json`: woollama spawns +
  supervises a `fabric --serve` (managed; reuse + graceful-kill) or routes to an
  external one (`url`). It **merges** fabric's ~250-pattern library into
  `/w1/patterns` (a `recipes.toml` recipe wins on a name collision) and
  **reverse-proxies fabric's REST verbatim at `/fabric/*`** (SSE, advanced
  `context`/`strategy`/`language`/`search`, and vision all pass through). On the
  `/w1` path, fabric's native SSE is translated to/from the OpenAI shape.
- **`PatternBackend` plugin seam.** Additional non-OpenAI prompt/inference systems
  plug in behind one trait + a single composition root; native recipes stay the
  built-in core, and the fabric backend is the reference impl. See
  `docs/extending.md`.
- **Self-healing fabric.** The pattern cache re-sources on a TTL
  (`WOOLLAMA_FABRIC_REFRESH_SECS`, default 60s — fabric hot-reloads its pattern
  dir) and after every respawn; a dead/hung **managed** fabric is respawned on the
  same address and the request retried once (single-flight, kill-before-rebind).
  `url` mode re-probes but never respawns a process it doesn't own.
- **Version family realigned to 0.6.0** across `woollama-engine`,
  `woollama-server`, `woollama-core`, and the `woollama` Python dist (the Rust
  crates had lagged at 0.5.0, the wheels at 0.5.3).

Docs: `docs/patterns.md` (the `/w1/` + `/fabric/` reference), `docs/extending.md`
(adding a backend), `docs/configuration.md` (the `fabric` key + resilience).

## v0.5.0 — 2026-06-14

**The Rust cutover.** woollama is now the Rust daemon **`woollamad`** (the
`woollama-server` crate), and it's **published**: `cargo install woollama-server`
installs the daemon, and `pip install woollama` pulls the pure-Python package plus
the native `woollama-core` engine wheel. The Python implementation in
`src/woollama/` is kept as the reference server and differential-test oracle, not
deleted. Authoritative live status is `docs/roadmap.md`.

*(Shipped across 0.5.0–0.5.3: 0.5.0 = crates.io publish of `woollama-engine` +
`woollama-server`; 0.5.1–0.5.3 = PyPI wheel publish of `woollama` +
`woollama-core` and fixes to the cross-platform wheel CI, notably the
manylinux-aarch64 build.)*

- **`woollamad`, the Rust router.** The full router surface ported to Rust on
  `woollama-engine` (pure engine) + `axum` + `rmcp`: OpenAI-compatible HTTP
  (`/v1/models`, `/v1/chat/completions` passthrough + recipe orchestration +
  streaming, `/v1/responses`, `/v1/conversations`), the MCP aggregator at `/mcp`
  and over stdio (`woollamad mcp`), the claude-code executor, stateful
  conversations (claude-resume / store-backed / managed-agents), and `/v1/models`
  discovery. Binds a unix socket (`$XDG_RUNTIME_DIR/woollama.sock`) + the loopback
  TCP port, same as the Python server.
- **Verified by a differential oracle.** The Python live integration suite runs
  against either implementation (`WOOLLAMA_TEST_CMD`), with `woollamad` the
  default target. Real behavioral divergences were caught and fixed (MCP
  capability advertisement, the `chat` tool's structured output, tool-level vs
  JSON-RPC error semantics).
- **Published.** crates.io: `woollama-engine` + `woollama-server` 0.5.0. PyPI:
  `woollama` + `woollama-core` 0.5.3, with cross-platform wheels (manylinux
  x86_64 + aarch64, musllinux, macOS x86_64 + arm64, Windows; cp311/312/313) +
  sdist, built by `.github/workflows/wheels.yml` (maturin-action).
- **TLS:** the engine's HTTP client moved from native-tls (OpenSSL) to **rustls**
  (system trust store via native-roots), so the native wheels build without a
  system OpenSSL and cross-compile to aarch64.

## v0.4.0 — 2026-06-10

Still the Python prototype (v1.0 is the Rust rewrite). The big shift: woollama is
now **embeddable as a library** — a server-free `woollama.core` other Python
projects import for model management — alongside an external conversation-store
family that makes non-claude models stateful without woollama ever owning bytes.
Authoritative live status is `docs/roadmap.md`.

- **Embeddable `woollama.core` library.** The model-management core — config +
  provider/model routing (`complete`/`complete_stream`, per-call `api_key`/
  `base_url`), `ModelRegistry`, recipes, and the recipe orchestration loop — is
  extracted into a server-free `woollama.core` subpackage so other projects embed
  it instead of running a sidecar. The FastAPI/MCP router now layers on top; the
  boundary is enforced by a test (importing `woollama.core` pulls in no
  FastAPI/uvicorn/MCP). The MCP↔OpenAI tool seam is explicit and lossless
  (`ToolProvider`/`ToolSpec`/`ToolResult` + a per-model renderer; carries MCP
  `isError`, fixing a silent tool-failure). Design: `docs/core-extraction.md`.
  **Note:** the old top-level module paths (`woollama.config`,
  `woollama.inferencers`, `woollama.recipes`, `woollama.ollama_native`) are gone —
  import from `woollama.core` (the server's public surface — the CLI + HTTP — is
  unchanged).
- **Pluggable conversation stores** (#2): non-claude models (ollama, cloud,
  recipes) become stateful through `/v1/responses` + `/v1/conversations` via an
  **external** store woollama is only a client to — it never owns transcript
  bytes (the conv-5 principle). Two reference providers prove the seam is
  transport-agnostic: an **MCP** store (`examples/mcp-convstore`) and a **REST**
  file store (`examples/rest-convstore`, persistent). Selected by the
  `conversationStore` key in `mcp.json` (a server name, or `{type:"mcp"|"http"}`);
  unset ⇒ stateless (no behavior change). A flaky store surfaces as a clean `502`.
- **Durable conversation handle table.** The `conversation_id → backend + native
  id` routing table is persisted (`$XDG_STATE_HOME/woollama/conversations.json`),
  so a client's conversation id keeps resolving across a woollama restart. Routing
  state only — never transcripts.
- **Attach by external key.** `POST /v1/conversations` and `/v1/responses` accept
  a caller-owned `key` (e.g. a session name): create-or-attach, idempotent — the
  caller drives turns by its own key and keeps no `key → id` map of its own.
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
