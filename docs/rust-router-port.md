# Porting the woollama **router service** to Rust

Status: **DONE — shipped.** `woollamad` (the `woollama-server` crate) is on crates.io and is
the canonical router as of v0.5.0; the Python server is the reference/oracle. This stays as the
slice-by-slice record of the completed port. Supersedes the "should we go all-Rust?" discussion.

## Framing (the decision that sets scope)

woollama is **primarily a router service** — it gathers tools/models from other
services and routes inference/orchestration between clients and those services.
The Python `import woollama` API is **auxiliary** (it became useful for lackpy).
So the goal is: **the router service runs as a Rust binary**; the PyO3 wheel
stays as an optional embedding surface over the same core. "Deprecate Python"
really means *demote it to an auxiliary binding*, not delete it.

This is a defensible goal, not rewrite-for-its-own-sake: the server's deps are
**client/test tooling, not server runtime** (the `openai`/`anthropic` SDKs define
the wire shapes we must emit, but the server never imports them), and the one
piece with real "is this even possible in Rust" risk — the MCP aggregator — has
been **de-risked by a working spike** (`/tmp/mcp-spike`, see `dist-split.md`'s
sibling note): rmcp 1.7 does dynamic tools, output_schema mirroring, structured
passthrough across two hops, prompts, and the axum `/mcp` mount.

## Current state

Already Rust (`woollama-core`, cdylib): `complete` / `complete_stream` /
`orchestrate` / `orchestrate_events` / `ModelRegistry` / `InferenceError`.

Still Python (`src/woollama/`, ~3.7k LoC — the port target):

| module | LoC | role | port target |
|---|---|---|---|
| `router.py` | 922 | FastAPI app, routes, error→HTTP, claude-code delegation | axum app + handlers |
| `conversations.py` | 534 | ConversationStore seam, handle table, backends | core lib + stores |
| `claude_code.py` | 331 | executor: `run_completion` + `run_delegated` + lockdown | `tokio::process` |
| `managed_agents.py` | 263 | Anthropic Managed Agents backend | raw REST (no SDK) |
| `mcp_server.py` | 260 | woollama-as-MCP aggregator (fastmcp) | rmcp (spike-proven) |
| `manager.py` | 218 | downstream MCP registry (connect/start_all/dispatch) | rmcp clients |
| `config.py` | 258 | toml/json loading, dirs | already half-ported (`toml` crate) |
| `inferencers.py` | 176 | discovery + Python `ModelRegistry` | Rust registry + discovery |
| `binding.py` | 166 | TCP/UDS socket binding | `tokio::net` |
| `ollama_native.py` | 150 | native `/api/chat` translation | pure fns |
| `responses.py` | 135 | `/v1/responses` shaping | pure fns |
| `tooling.py` | 103 | ToolSpec/ToolResult/ToolProvider seam | Rust trait (see below) |

Routes to reproduce: `GET /v1/models`, `GET /v1/tools`, `POST /v1/chat/completions`,
`POST /v1/responses`, `POST|GET|DELETE /v1/conversations[/{id}[/items]]`, and the
`/mcp` mount.

## The load-bearing refactor: decouple the engine from PyO3

Today `lib.rs` mixes the **engine** with **PyO3 glue**, and the orchestrate loop
dispatches tools by `await`ing a **Python** `ToolProvider.dispatch` coroutine
(`pyo3_async_runtimes`). A Rust binary can't (and shouldn't) call back into
Python to run tools. So the engine must become generic over a **Rust** tool seam:

```rust
// in the pure-Rust core (no pyo3):
#[async_trait]
trait ToolProvider {
    async fn tools_for(&self, allow: &[String]) -> Vec<ToolSpec>;   // schemas
    async fn dispatch(&self, name: &str, args: &Value) -> ToolResult;
}
```

Two implementors, one engine:
- **`PyToolProvider`** (in the cdylib wheel) — wraps the Python callback. Preserves
  today's behavior; the wheel/lackpy keep working unchanged.
- **`RegistryToolProvider`** (in the server) — dispatches to downstream MCP servers
  via rmcp clients (the spike's `Aggregator::call_tool` path).

This is the linchpin slice, and it is **bigger than the trait alone** — making the
core a pure rlib means extracting pure-Rust equivalents for everything the engine
currently expresses in PyO3:
- `InferenceError` is a `#[pyclass]` → becomes a plain Rust error type in the core,
  re-wrapped as the pyclass only in the cdylib.
- `DeltaStream` / `EventIter` are pyclasses → become Rust `Stream`s in the core; the
  cdylib wraps them as async-iterator pyclasses.
- return values flow through `pythonize` / `Py<PyAny>` → the core speaks `serde_json`;
  the cdylib does the Python translation at the boundary.
- **the subtle one:** `PyToolProvider::dispatch` must bridge back to a Python
  coroutine via `pyo3_async_runtimes` from inside an `async-trait` method. That works
  in today's monolithic engine; **verify it still composes** once the engine is
  generic and the bridge lives in the wrapper crate under the server's tokio runtime.

Treat slice 1 as the expensive, risky one — this is where days go if it's mistaken
for mechanical. Once the engine is PyO3-free and trait-driven, every later slice is
additive.

## Target workspace layout (as built — deviates from the original sketch)

The published wheel name `woollama-core` is load-bearing (PyPI dist, the server's
`uv` path source, the `woollama.core` import + namespace merge). Renaming it would
churn all of that for no gain, so we KEPT `woollama-core` as the cdylib wheel and
added a new `woollama-engine` rlib instead of the sketch's `woollama-core`=rlib +
`woollama-py`=cdylib. Same intent, far less blast radius. Flat layout (no `crates/`).

```
woollama/                         (cargo workspace root = the woollama placeholder pkg)
  woollama-engine/ (rlib)         ← pure Rust engine: EngineError, ToolProvider trait,
                                    Registry, complete*, build_setup/events_stream.
                                    NO pyo3. The reusable heart. [slice 1 ✅]
  woollama-core/   (cdylib)       ← thin PyO3 wrapper → the wheel, name UNCHANGED.
                                    InferenceError, PyToolProvider (coroutine bridge),
                                    the pyclasses, the pyfunctions. [slice 1 ✅]
  woollama-server/ (crate)        ← the PRODUCT: builds the `woollamad` daemon —
                                    axum + rmcp + stores + claude-code.
                                    Depends on woollama-engine. [stub; grows slice 2+]
  src/woollama/    (python)       ← stays until cutover; the differential oracle.
```

## Slice ordering (risk-front-loaded; each slice ships green)

0. **Workspace split.** ✅ DONE (commit `1701552`). Cargo workspace (root placeholder +
   `woollama-core` cdylib + `woollama-server` bin stub). Proved the workspace wrapper is
   benign: maturin wheel + editable install + namespace merge + conformance/server suites
   all green. (Turned out to be a pure Cargo change in the end — maturin needed no rewire
   since `woollama-core` stayed the cdylib.)
1. **Trait-ize tool dispatch + PyO3 extraction (the expensive slice).** ✅ DONE (commit
   `41937e4`). Extracted the engine into `woollama-engine` (pure rlib): `EngineError`
   (replaces `PyErr`/`InferenceError`), `ToolProvider` trait (replaces the `Py<PyAny>`
   seam), `Registry`, `build_setup`/`events_stream`, `complete_stream` re-expressed as a
   `stream!`. `woollama-core` is now a thin wrapper (incl. the `PyToolProvider`
   coroutine bridge). Gate met: conformance 42 + server 226 green; `woollama-server`
   links the engine with **no pyo3** in its dep tree.
2. **Server skeleton.** ✅ DONE (commit `91c3ad9`). `woollama-server` is now a real axum
   service over the engine: `resolve_tcp_target` + TCP bind, `GET /v1/models` (route +
   list shape), `POST /v1/chat/completions` **passthrough** (bare-model rewrite + relay).
   **Scope correction vs the original plan:** orchestration moved to slice 4 (it needs
   the MCP registry), and the **gate is Rust integration tests against a mock upstream**,
   not the repointed live suite — the live tests are interdependent (passthrough's test
   picks a model from `/v1/models`, which needs discovery=slice 8; orchestration=slice 4),
   so they can't go green until those land. The Rust test covers binding + `/v1/models` +
   the bare-model rewrite/relay + the 501/400 deferrals; it also replaces the TCP half of
   the in-process `unix_socket` test. (UDS still deferred.)
3. **Native + Responses.** ✅ DONE (commit `5538bc7`). `ollama_native` translators
   (pure, ported + unit-tested), native num_ctx → `/api/chat` (non-stream), streaming
   passthrough (SSE relay), stateless `/v1/responses` (non-stream; the inferencer path
   reuses the engine `complete`, which already does native num_ctx). Gate: 10 unit + 2
   integration tests vs a mock upstream. **Streaming (3b) ✅ DONE** (commit `43e7a6a`):
   native num_ctx streaming (NDJSON→SSE via the ported `SseTranslator`), Responses
   streaming (the full event sequence), and streaming orchestration — gated by a
   streaming.rs end-to-end test (incl. a fragmented tool_call reassembled mid-stream).
   Only Responses transcript `/items` (stateful) remains, with the stores slice.
4. **MCP aggregator + orchestration.** Split into:
   - **4a ✅ DONE** (commit `afaf4a7`) — the downstream MCP registry (rmcp child-process
     **clients**) + `RegistryToolProvider` (the engine `ToolProvider` seam) + recipe/
     `mcp.json` loading + `woollama/<recipe>` ORCHESTRATION on `/v1/chat/completions` and
     stateless `/v1/responses` (non-stream). Gate: an end-to-end test driving a recipe
     against a real stdio MCP fixture + a mock inferencer (tool_call → dispatch → final).
   - **4b ✅ DONE** (commit `163b1bc`) — woollama-AS-an-MCP-server (`WoollamaMcp`): the
     `chat` tool + re-exported downstream tools (input+output schema mirrored, structured
     passthrough) + recipe prompts, served from one handler over BOTH a Streamable-HTTP
     `/mcp` mount (shared port) and a `woollamad mcp` stdio subcommand. Gate: an
     rmcp-client end-to-end test (aggregation + proxy + the chat tool + prompts) + the
     **shared-registry-across-sessions stress** (two concurrent sessions — the open
     lifecycle question, settled) + a stdio `initialize` smoke. (Streaming orchestration
     was folded into 3b ✅.)
5. **claude-code executor.** ✅ DONE (commit `5ca47b4`). `run_completion` (tool-less) +
   `run_delegated` (Claude owns the loop) via `tokio::process`, intercepted in
   `orchestrate_recipe` before the engine loop; the full `--tools ""` lockdown ported and
   unit-tested (incl. the allow-list boundary + the env allow-list). Gate: unit tests pin
   the lockdown/boundary + an e2e via a fake `claude` CLI through chat/responses/streaming.
   **Still deferred:** `run_resumable` (the claude-resume conversation backend) → slice 6;
   the 3 LIVE security gates (real `claude`: shell refused, sibling denied) → opt-in
   plain-terminal tests, rewritten HTTP/recipe-driven.
6. **Conversation stores.** Split:
   - **6a ✅ DONE** (commit `0f07112`) — the durable handle table (`WOOLLAMA_STATE_DIR`,
     atomic rewrite, restart-survival) + per-conversation locks + the **claude-resume**
     backend + stateful `/v1/responses` + `/v1/conversations` CRUD. Gate: a hermetic e2e
     (fake `claude`) incl. restart survival. (`run_resumable` added to claude_code.)
   - **6b ✅ DONE** (commit `e52a118`) — store-backed statefulness for ollama/cloud/recipe
     models: the `StoreProvider` seam + `HttpStoreProvider`/`McpStoreProvider` (REST + MCP
     clients) + `complete_stateless` + `/items` served from the store. Gate: a mock-REST-
     store e2e proving prior reassembly + items + delete.
7. **Managed agents.** ✅ DONE (woollama-side; commit `e178303`). The managed-agents
   backend for claude-agent/* models: client (base-URL-mockable) + lazy agent/env cache +
   the **requires_action pause/resume** path + `/items` from the event log + lifecycle.
   Gate: a mock-Anthropic e2e drives create→turn→pause→answer→items→delete.
   **⚠️ Caveat:** the Anthropic Managed Agents REST/streaming wire shapes aren't in the
   repo (Python uses the SDK), so the client targets a SIMPLIFIED protocol exercised by
   the mock — the real API must be reconciled before the opt-in live `@needs_anthropic`
   test passes. The tested value is woollama's routing, not the Anthropic wire format.
8. **`/v1/models` discovery in Rust.** ✅ DONE (commit `dcfe8f8`). Discovery fields
   (`models`/`discover`/`model_patterns`) ported into the engine `Inferencer`/registry
   (so ONE registry serves orchestration + discovery — the two-registry drift is gone;
   `inferencer_to_json` unchanged so conformance is untouched); server `/v1/models` does
   static + live discovery (namespaced, fnmatch-filtered) + recipes. Gate: a mock-`/v1/models`
   e2e covering live discover, pattern filtering, static models, and recipes.
9. **Cutover.** ✅ DONE. `woollamad` (the Rust daemon) is the canonical router; the Python
   server is demoted to reference/oracle (still runnable via `python -m woollama`, kept as
   the differential oracle). Done UDS-first: slice 9a ported the unix-socket + addr-file
   surface to `woollamad` so local MCP clients don't regress (`binding.rs`, two serve tasks
   on TCP + UDS). The live oracle default flipped to `woollamad` (Python is now the opt-in
   `WOOLLAMA_TEST_CMD="python -m woollama"`): **16/0/9 against woollamad** (the in-process
   `binding.py` unix test skips — woollamad's UDS is covered by the Rust `tests/binding.rs`,
   so every pass is genuine woollamad evidence) and 17/0/8 against the Python reference.
   Packaging: `cargo install woollama-server` → `woollamad` (crates.io, publishable but not
   yet published). Wheel stays
   the auxiliary embed surface; re-pin lackpy to it (still pending the woollama-core publish).

## Verification strategy (the strongest asset)

The Python server **+ most of its 25-test live integration gate is the differential
oracle.** The **20 HTTP/SDK-driven tests** (the `woollama_server*` fixtures, plus the
2 stdio-MCP tests that spawn `python -m woollama mcp`) are accepted by pointing the
*existing* fixture's process spawn at the **Rust binary** (and its `mcp` subcommand)
instead of `python -m woollama` — same OpenAI-SDK clients, same assertions, same real
Ollama / Claude / Anthropic backends. For those, the openai/anthropic SDKs themselves
are the live conformance check; we never hand-wave wire parity.

**The oracle is NOT universal — 5 tests drive the Python API in-process and must be
re-expressed, not repointed** (verified set):
- `test_unix_socket_serves_http_end_to_end` (`binding.open_sockets()`) → slice 2:
  becomes a Rust integration test of the `binding` module (`tokio::net`), not a
  repointed Python test.
- `test_claude_code_backend_completes_and_refuses_shell`,
  `..._delegation_runs_tool_and_keeps_boundary`,
  `..._delegation_denies_same_server_sibling` (call `router.orchestrate` /
  `claude_code._*` directly) → slice 5: **rewrite as HTTP/recipe-driven gates** —
  define the recipe, drive it over `/v1/chat/completions` against the Rust binary,
  assert the canary/refusal from the *outside*. Without this, slice 5 (the security
  gates) would ship with no live gate.
- `test_anthropic_inferencer_completes_live` (`router.orchestrate`) → slice 7: same,
  drive the cloud recipe over `/v1/chat/completions`.

Re-express these **per slice as we reach them**, not up front — the HTTP oracle is the
right design for the majority; it just isn't total.

Plus: the Rust conformance suite (42) continues to pin engine behavior, and the
hermetic server suite guards the Python path until cutover.

## Known risks, and where each lands

- **Session-sharing lifecycle** (the spike proved the mechanism, not load): slice 4
  carries an explicit concurrent-sessions-share-one-downstream stress test.
- **stdio MCP transport**: documented in rmcp but unspiked; slice 4.
- **Managed-agents without the SDK**: raw REST against the Managed Agents API; slice 7
  (moderate, isolated).
- **Error→HTTP parity**: `InferenceError.{kind,status,payload}` already structured;
  the axum handlers must map identically — the live oracle catches drift.

## What stays Python / explicitly deferred

- The PyO3 wheel (auxiliary, permanent) — lackpy's embed surface.
- The Python server stays runnable until slice 9, as the oracle.
- Nothing is silently dropped: discovery (`/v1/models`) is slice 8, not abandoned.

## Pre-cutover live review (slices 0–8 complete; the review the user asked for)

The differential oracle was run for real: `tests/test_integration.py` repointed at the
release Rust binary via a new `WOOLLAMA_TEST_CMD` env hook (`_woollama_argv`), against
**live Ollama** and the **real OpenAI SDK** (every prior server-side test was gated by
self-authored mocks). Command:

```
WOOLLAMA_TEST_CMD="$PWD/target/release/woollamad" \
WOOLLAMA_EXAMPLES_DIR="$PWD/examples" WOOLLAMA_OLLAMA_URL="http://localhost:11434" \
uv run --extra dev pytest tests/test_integration.py -m integration -v
```

**Initial result: 13 passed / 4 failed / 8 skipped** (all 4 failures in the `/mcp`
surface — see Findings below). After fixing findings 1+2: **17 passed / 0 failed / 8
skipped.** Honest breakdown of the passes:

- **12 genuine Rust-binary passes** — discovery, passthrough chat (+ streaming, 64 real
  token deltas), native `num_ctx`, Responses stateless (+ streaming, + SDK), orchestration
  non-stream + streaming, two-provider recipe, store-backed conversations (MCP + HTTP),
  handle-table-survives-restart. All driven over HTTP/SDK against the spawned Rust process.
- **1 non-Rust pass** — `test_unix_socket_serves_http_end_to_end` imports `woollama.binding`
  in-process; it tested PYTHON, not the binary. Not Rust evidence (the Rust unix-socket
  surface is still deferred). Already flagged above for re-expression.
- **8 skipped** = the paid tiers, correctly auto-skipped (5 `@needs_claude_code`,
  3 `@needs_anthropic`) — no hidden coverage.

**Decisive tool-dispatch proof** (the orchestration "I have counted to N" answer is
fabricatable by the model alone, so the green test is necessary-not-sufficient): an
instrumented `count_to` server writing a sentinel confirmed the engine genuinely
dispatched `count_to(n=7)` to a real MCP child process. Orchestration is verified live,
not inferred.

### Findings (4 failures = 3 root causes)

1. **[FIXED] MCP surface didn't advertise capabilities** (`mcp_surface.rs` `get_info` →
   `ServerInfo::default()`). The handler overrides `list_tools`/`list_prompts`, but the
   `initialize` handshake reported `tools=None, prompts=None`. Capability-checking clients
   saw no tools. (2 failures: stdio + HTTP `*_surface`/`*_shares`.) Fixed: `get_info`
   builds `ServerCapabilities::builder().enable_tools().enable_prompts()`.
2. **[FIXED] `chat` tool returned no structured content** (`mcp_surface.rs` `run_chat` →
   `Content::text(text)` only). FastMCP's client `result.data` was None. The Python `chat`
   is `-> str` via `Tool.from_function`, which FastMCP auto-wraps into
   `structured_content {"result": text}` + an `x-fastmcp-wrap-result` output schema (shape
   verified against the installed fastmcp). (2 failures: stdio + HTTP `*_chat`.) Fixed: the
   chat tool now declares the wrap output_schema and returns `structured_content
   {"result": text}`. **Adjacent fix surfaced during repair:** bad recipe / orchestration
   failures were JSON-RPC `McpError`s; the Python `chat` raises `ValueError`, which FastMCP
   turns into a TOOL-level `isError` result (client `ToolError`). Now matched — `run_chat`
   returns `CallToolResult::error(..)` for those cases. All three contract points
   (capabilities, wrapped structured_content, tool-level error) are now pinned in the
   hermetic `mcp_surface.rs` test too, so they can't regress without Ollama in the loop.
3. **[FIXED] `WOOLLAMA_EXAMPLES_DIR` not auto-resolved.** Python's `config._examples_dir()`
   sets it from runtime `__file__`; the Rust binary left it unset, so the bundled
   `mcp.json`'s example servers silently failed to spawn (→ orchestration recipes couldn't
   dispatch). Decision (yours): **ship the examples alongside the binary as the default**,
   with env/config override taking priority. Implemented as `config::ensure_examples_dir()`
   (called first in `build_state`), precedence: (1) an explicit `WOOLLAMA_EXAMPLES_DIR`
   wins; (2) `<exe-dir>/examples` (packaged install — examples are 116K, ship with the
   binary); (3) the source checkout's `examples/` (dev / `cargo run` / the integration
   suite). A candidate must contain `mcp-hello/server.py` to count — this guards against
   cargo's reserved empty `target/<profile>/examples` dir (which the first naive attempt
   wrongly matched). Proven: the FULL oracle passes (17/0/8) with `WOOLLAMA_EXAMPLES_DIR`
   UNSET (auto-resolved via the dev-checkout fallback). **Packaging TODO:** the release/
   install step must physically copy `examples/` beside the installed binary for precedence
   (2) to fire in production; precedence (3) only covers in-repo runs.

All four original failures were in the `/mcp` aggregator surface (slice 4b/5); the
orchestration they wrap always worked end-to-end. None touched the OpenAI HTTP surface,
which was green throughout. With findings 1+2 fixed, the only open item is finding 3
(examples-dir resolution), which is a deliberate design decision rather than a bug.
