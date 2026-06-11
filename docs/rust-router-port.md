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

This is the linchpin slice. Once the engine is PyO3-free and trait-driven, every
later slice is additive.

## Target workspace layout

```
woollama/                         (cargo workspace)
  crates/
    woollama-core/   (rlib)       ← pure Rust engine: types, complete*, orchestrate*,
                                    ModelRegistry, ToolProvider trait, InferenceError.
                                    NO pyo3. The reusable heart.
    woollama-py/     (cdylib)     ← thin PyO3 wrapper → the wheel (AUXILIARY).
                                    PyToolProvider + the #[pyclass]/#[pymodule] glue.
    woollama-server/ (bin)        ← the PRODUCT: axum + rmcp + stores + claude-code.
                                    Depends on woollama-core.
  src/woollama/      (python)     ← stays until cutover; becomes the differential oracle.
```

(Today's single cdylib splits into core-rlib + py-cdylib; the `crate-type` change
is slice 0.)

## Slice ordering (risk-front-loaded; each slice ships green)

0. **Workspace split.** `woollama-core` → pure rlib; `woollama-py` cdylib wraps it.
   No behavior change. Gate: conformance (42) + server suite green, wheel imports.
1. **Trait-ize tool dispatch.** Introduce the Rust `ToolProvider` trait; engine
   generic over it; `PyToolProvider` preserves the Python path. Gate: same suites green.
2. **Server skeleton.** `woollama-server` bin: axum, `binding` (TCP/UDS), `/v1/models`
   (static+registry), `/v1/chat/completions` **passthrough** + **orchestration** via
   the core. Gate: the **live integration gate run against the Rust binary** (below).
3. **Native + Responses.** `ollama_native` translation, `/v1/responses` (stateless,
   `complete_stateless`), num_ctx native routing. Gate: the num_ctx + responses live tests.
4. **MCP aggregator (productionize the spike).** `manager` (downstream registry as
   rmcp clients, `start_all`) + `mcp_server` (`/mcp` over **stdio AND** Streamable-HTTP,
   re-export + schema mirroring). **Includes the one open lifecycle question:** a load
   pass on the shared-downstream-across-sessions factory. Gate: the 4 MCP live tests.
5. **claude-code executor.** `run_completion` + `run_delegated` + the `--tools ""`
   lockdown, via `tokio::process`. Gate: the 5 claude-code live tests (incl. the 3
   security gates) — **in a plain terminal**.
6. **Conversation stores.** ConversationStore seam, durable handle table
   (`WOOLLAMA_STATE_DIR`), claude-resume + store-backed backends (MCP + REST clients).
   Gate: the store-backed + restart-survival live tests.
7. **Managed agents.** Anthropic Managed Agents backend via raw REST. Gate: the 3
   anthropic live tests.
8. **`/v1/models` discovery in Rust.** Static `models` + live `discover` +
   `model_patterns`; collapses the two-registry drift (the named deferred item).
9. **Cutover.** `woollama` entrypoint = the Rust binary; Python server retired to
   reference. Wheel stays as the auxiliary embed surface; re-pin lackpy to it.

## Verification strategy (the strongest asset)

The Python server **+ its 25-test live integration gate is the differential
oracle.** Every server-facing slice is accepted by pointing the *existing*
`test_integration.py` fixture's process spawn at the **Rust binary** instead of
`python -m woollama` — same OpenAI-SDK clients, same assertions, same real Ollama /
Claude / Anthropic backends. A slice is "done" when its slice of that suite passes
against the Rust binary identically. This means we never hand-wave wire parity:
the openai/anthropic SDKs themselves are the conformance check, run live.

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
