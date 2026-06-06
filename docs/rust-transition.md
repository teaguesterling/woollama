# Rust transition criteria

woollama is **a Rust program at v1.0.** The Python implementation in
`src/woollama/` is a prototype used to iterate the architecture quickly while
the design surface is in flux. This doc captures *when* we stop iterating
Python and start the Rust rewrite — to prevent the prototype from becoming
the de facto production system by inertia.

## When we transition

The Rust port begins when **all four** of these are true:

1. **The architecture is stable.** No more major design questions like "MCP
   extension vs. cos-fab tool" or "where does substitution live" — the
   answers have settled and the doc captures them.

2. **The Python surface covers the v1.0 feature set.** Concretely (progress as
   of 2026-06-06 — kept in sync with [`roadmap.md`](roadmap.md)'s gate
   checklist; the two MUST agree):
   - [x] real config file (`recipes.toml` + `mcp.json`, plus `inferencers.toml`)
   - [x] multi-MCP-server discovery + unified tool registry
   - [x] long-lived MCP connections (was the criterion-#4 latency concern)
   - [x] the Anthropic backend (and the generic OpenAI-compat inferencer seam)
   - [x] the woollama-as-MCP-server side (stdio + Streamable HTTP)
   - [x] streaming on both sides (OpenAI SSE out — passthrough + orchestration;
         MCP progress events on the `chat` tool)
   - [x] Unix socket alongside HTTP loopback
   - [ ] the panel-confirm round-trip equivalent. The stateful Conversations
         surface itself IS shipped (`/v1/responses` + `/v1/conversations`,
         claude-resume + duckdb `stored` backends); this stays open on the other
         half — **cosmic-fabric actually consuming it**.

3. **There is a real consumer.** Cosmic-fabric panel (or another real
   client) is actively using woollama through its OpenAI/MCP surfaces. The
   API has survived contact with a real user.

4. **A specific limit of Python is biting.** Either: subprocess-per-MCP-call
   latency is meaningful in real workloads (we should pool connections in
   Python first if so), OR memory/perf in a long-lived router is a problem
   that profiling confirms isn't a Python-code issue but a Python-runtime
   issue.

If any of 1–3 isn't true, we keep iterating Python. If 1–3 are true but 4
isn't, the question becomes "is now the right time" — defaulting to "stay
Python a while longer."

## What the Rust port preserves

- **The public surface** is identical. OpenAI endpoints at `/v1/...`, MCP at
  `/mcp` (Streamable HTTP) and stdio, same model namespace and recipe
  resolution. Existing clients see no behavior change.
- **The architectural decisions** in `docs/architecture.md` carry over
  verbatim — they were the point.
- **The downstream MCP servers** are external (their own processes) and
  don't change.
- **The `examples/` and `docs/`** stay as-is.

## What the Rust port replaces

- `src/woollama/*.py` → `src/*.rs`
- `pyproject.toml` → `Cargo.toml` (already a placeholder)
- `uv sync` → `cargo build --release`
- `python -m woollama` → `woollama` binary

## What the Rust port lets us do that Python doesn't

- **Single binary** that ships without an interpreter. Easier to install,
  fewer ways to break.
- **No GIL constraints** on concurrent MCP + HTTP fan-out under load.
- **Direct integration with cosmic-fabric's Rust panel** without a process
  boundary (could be library use or subprocess, but the option exists).
- **Smaller resident memory** for a long-running router.

## Choice of Rust crates (when we get there)

- **`axum`** for the HTTP server (current ecosystem leader; clean async)
- **`reqwest`** for the OpenAI-compat client (de facto Rust HTTP client)
- **`rmcp`** for the MCP client (Anthropic's official Rust SDK; pin a version)
- **`tokio`** for async runtime
- **`serde` / `serde_json`** for wire formats
- **`clap`** for CLI
- **`tracing`** for structured logs

## What the Python prototype contributes to the port

- A **working reference implementation** that the Rust code can be tested
  against (port the smoke tests; require the Rust version to pass the same
  tests as the Python one)
- **Concrete shapes** for the recipe model, the chat-loop, the binding
  conventions — the Rust types follow these
- **Real measurements** of where the prototype is slow, so the Rust port
  targets known pain points
- **The bundled MCP-hello example** stays the same — both Python and Rust
  routers can use it for smoke tests

## The honest principle

The Python prototype exists to **learn fast**. Every additional feature
landed in Python that turns out to need a different shape in Rust is wasted
work. So:

- Don't optimize the Python prototype for performance or production
  hardening — those investments belong in Rust.
- Don't add features to the Python prototype that aren't needed to *learn
  the right shape*.
- When in doubt about whether a feature is "learning the shape" or "doing
  production work," choose Rust.

The danger is iterating Python long enough that throwing it away gets
emotionally expensive. This doc exists so we don't have that conversation
when the time comes — we already had it.
