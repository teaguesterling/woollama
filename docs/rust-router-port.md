# Porting the woollama **router service** to Rust

Status: **plan** (not started). Supersedes the "should we go all-Rust?" discussion.

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
  woollama-server/ (bin)          ← the PRODUCT: axum + rmcp + stores + claude-code.
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
   integration tests vs a mock upstream. **Carved out to slice 3b:** native num_ctx
   STREAMING (NDJSON→SSE, the `sse_translator`) + Responses streaming/`items` — each a
   clear 501 today.
4. **MCP aggregator (productionize the spike).** `manager` (downstream registry as
   rmcp clients, `start_all`) + `mcp_server` (`/mcp` over **stdio AND** Streamable-HTTP,
   re-export + schema mirroring). **Includes the one open lifecycle question:** a load
   pass on the shared-downstream-across-sessions factory. Gate: the 4 MCP live tests.
5. **claude-code executor.** `run_completion` + `run_delegated` + the `--tools ""`
   lockdown, via `tokio::process`. Gate: the 2 SDK-driven tests (claude-resume,
   conversations journey) repointed, **plus the 3 security gates rewritten as
   HTTP/recipe-driven** (canary/refusal asserted from outside) — **in a plain terminal**.
6. **Conversation stores.** ConversationStore seam, durable handle table
   (`WOOLLAMA_STATE_DIR`), claude-resume + store-backed backends (MCP + REST clients).
   Gate: the store-backed + restart-survival live tests.
7. **Managed agents.** Anthropic Managed Agents backend via raw REST. Gate: the 2
   SDK-driven anthropic tests (managed-agents journey, requires_action) repointed,
   **plus the anthropic-inferencer gate rewritten as HTTP/recipe-driven**.
8. **`/v1/models` discovery in Rust.** Static `models` + live `discover` +
   `model_patterns`; collapses the two-registry drift (the named deferred item).
9. **Cutover.** `woollama` entrypoint = the Rust binary; Python server retired to
   reference. Wheel stays as the auxiliary embed surface; re-pin lackpy to it.

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
