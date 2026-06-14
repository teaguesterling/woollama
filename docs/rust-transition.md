# Rust transition criteria — COMPLETED

> **Status: DONE (v0.5.x).** The transition described here happened. The router is
> now the Rust daemon **`woollamad`** (the `woollama-server` crate), published to
> crates.io (`cargo install woollama-server`) and PyPI (`pip install woollama` +
> the native `woollama-core` wheel). The Python implementation in `src/woollama/`
> is kept as the reference server and **differential-test oracle** — the live
> integration suite runs against either, which is how the Rust port was verified.
> This doc is retained as the record of the criteria we set for *when* to move
> (all the gating ones were met; the one open follow-on is cosmic-fabric
> consuming the surface) and what the port preserved vs. replaced.

woollama is now **a Rust program (`woollamad`).** The Python implementation in
`src/woollama/` was the prototype used to iterate the architecture quickly while
the design surface was in flux; it is now the oracle. This doc captured *when* we
stop iterating Python and start the Rust rewrite — to prevent the prototype from
becoming the de facto production system by inertia.

## When we transition (the criteria — all gating ones now met)

The Rust port was gated on **all four** of these being true:

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
         `claude-resume` backend — woollama routes handles, backends own state;
         non-owning models are stateless); this stays open on the other
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

## What the Rust port changed (as built)

- The router moved to a Cargo **workspace**: `woollama-engine` (the pure engine,
  no PyO3) + `woollama-core` (the PyO3 wheel that provides `woollama.core`) +
  `woollama-server` (which builds the **`woollamad`** daemon).
- The canonical run command is the **`woollamad`** binary
  (`cargo install woollama-server`, or `cargo build --release`), not
  `python -m woollama`.
- `src/woollama/*.py` is **kept**, not deleted — it's the reference server and
  differential-test oracle; `python -m woollama` (and `pip install woollama`)
  still run it.

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
