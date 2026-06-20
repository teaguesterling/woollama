# Extending woollama — pattern backends

woollama's `/w1/` pattern surface is **pluggable**. The native path (recipes
dispatched through `woollama-engine`) is built in; any *additional* source of
patterns — a prompt library, another templating service, a non-OpenAI inference
system — plugs in behind one trait: **`PatternBackend`**. The bundled
[fabric backend](patterns.md#the-fabric-backend)
is the reference implementation.

This guide is for contributors adding a backend to `woollama-server`. It is a
**compile-time plugin** model (a Rust trait + registration), not a dynamically
loaded module — adding a backend means adding a module and recompiling.

## The shape

- **Native recipes stay the built-in core** (`lib.rs`). They need the engine,
  the MCP registry, and the inferencer set, so they aren't a plugin — making them
  one would be churn for no clarity. This asymmetry is deliberate.
- **Additional backends are plugins**: each implements `PatternBackend`, is
  constructed in one composition root, and lives in `AppState.pattern_backends`.
  The request handlers iterate that list generically — **adding a backend never
  touches a handler**, and no backend name appears in `lib.rs`'s logic.
- **Dispatch order is deterministic**: a native recipe wins on a name collision,
  then backends in registration order.

```
request → /w1 handler (lib.rs, generic)
            |- native recipe?            -> engine path        (built-in core)
            +- for b in pattern_backends -> b.has(name)? b.run(name, body)  (plugins)
```

## The trait

`woollama-server/src/pattern_backend.rs`:

```rust
#[async_trait::async_trait]
pub trait PatternBackend: Send + Sync {
    /// Stable id (e.g. "fabric"). Also the mount prefix for the optional proxy.
    fn id(&self) -> &str;

    /// Patterns offered, for /w1/patterns discovery + /v1/models.
    fn list(&self) -> Vec<PatternInfo>;            // { name, variables, source }

    /// Does this backend serve `name`?
    fn has(&self, name: &str) -> bool;

    /// Rendered system text with `variables` substituted (woollama appends the
    /// user input). None if the backend doesn't know the pattern.
    async fn render(&self, name: &str, variables: &Map<String, Value>) -> Option<String>;

    /// Run `name` -> an OpenAI chat-completion Response (or OpenAI SSE when the
    /// body sets stream:true). Only called after has(name) is true.
    async fn run(&self, name: &str, body: &Value) -> Response;

    // --- optional, with sensible defaults --------------------------------
    fn v1_addressable(&self) -> bool { true }   // advertise in /v1/models?
    fn proxies(&self) -> bool { false }         // mount a /{id}/* proxy?
    async fn proxy(&self, method, path_and_query, content_type, body) -> Response { 501 }
    async fn shutdown(&self) {}                  // graceful cleanup (kill a child, ...)
}
```

The trait speaks **only woollama's terms** — pattern name, a variables map, a
rendered system string, an OpenAI `Response`. That is the contract you must keep:
**no backend-specific concept may appear in the trait.** Everything about *your*
system (its endpoints, request body shape, streaming format, auth) stays inside
your `impl`.

### Method notes

- **`list`** returns `PatternInfo { name, variables, source }`. Scanning variable
  names can be expensive (a big library) — returning `variables: []` and resolving
  them on render/run is fine, as the fabric backend does.
- **`render`** returns the *system* text only; woollama appends the user input and
  formats the `/w1/.../render` response. Reuse `crate::config::render_system(sys,
  vars)` for byte-compatible `{{var}}` substitution.
- **`run`** owns its translation to/from OpenAI shape. For SSE, reuse the helpers
  `crate::{chat_chunk, chatcmpl_id, now_secs, take_line, sse_response}`.
- **`v1_addressable`** gates `/v1/models` so it stays honest: return `false` if a
  pattern can't actually run via `/v1/chat/completions` (which has no per-call
  model slot). `/w1/patterns` lists the backend regardless.
- **`proxies` + `proxy`** are optional. If you return `proxies() == true`,
  woollama mounts `/{id}` and `/{id}/{*rest}` and forwards matching requests to
  your `proxy`, which streams the response back (use
  `crate::pattern_backend::stream_reqwest` to pipe a `reqwest::Response`). Reserved
  prefixes (`v1`, `w1`, `mcp`) are skipped.

## Adding a backend — the recipe

### 1. Write the module

`woollama-server/src/mybackend.rs`:

```rust
pub struct MyBackend { /* client, cached names, config ... */ }

#[async_trait::async_trait]
impl crate::pattern_backend::PatternBackend for MyBackend {
    fn id(&self) -> &str { "mybackend" }
    fn list(&self) -> Vec<crate::pattern_backend::PatternInfo> { /* ... */ }
    fn has(&self, name: &str) -> bool { /* ... */ }
    async fn render(&self, name: &str, vars: &serde_json::Map<String, serde_json::Value>)
        -> Option<String> { /* fetch system + render_system */ }
    async fn run(&self, name: &str, body: &serde_json::Value) -> axum::response::Response {
        /* translate body -> your API -> OpenAI completion/SSE */
    }
}

/// Per-backend registration entry point (see pattern_backend::register_all).
pub async fn register(backends: &mut Vec<std::sync::Arc<dyn crate::pattern_backend::PatternBackend>>) {
    match crate::config::load_mybackend_config() {
        Ok(Some(cfg)) => if let Some(b) = MyBackend::connect(cfg).await { backends.push(b); },
        Ok(None) => {}
        Err(e) => eprintln!("woollamad: mybackend config error: {e}"),
    }
}
```

### 2. Declare the module and register it

In `lib.rs`, add `mod mybackend;` (the module-tree declaration — the *only* place
`lib.rs` names your backend). Then add **one line** to the composition root in
`pattern_backend.rs`:

```rust
pub async fn register_all() -> Vec<Arc<dyn PatternBackend>> {
    let mut backends: Vec<Arc<dyn PatternBackend>> = Vec::new();
    crate::fabric::register(&mut backends).await;
    crate::mybackend::register(&mut backends).await;   // <- add this
    backends
}
```

That's it — `build_state` calls `register_all`, and the handlers pick your backend
up generically. No handler, route, or `/v1`/`/w1` code changes.

## The config boundary — read this

**Config selects and locates; code defines the protocol.** This split is
load-bearing:

- **Config** (your loader, reading mcp.json or similar): *which* backends are
  active, managed-vs-external, address/URL, default model, credentials *by env var
  name*.
- **NOT config**: endpoint paths, request/response body shapes, SSE event types,
  auth scheme. Those are your backend's **protocol** and live in code. Trying to
  express a wire format in config is a rabbit hole — don't.

> **Why backends don't go in `inferencers.toml`.** That file is consumed by
> `woollama-engine`, which is OpenAI-compatible and **parity-locked**, and whose
> loader *requires* every `[inferencers.*]` entry to have a `base_url` (it errors
> otherwise). A non-OpenAI backend there breaks config load. Put your backend's
> config under its own top-level key in **mcp.json** (mirroring `conversationStore`
> and `fabric`), read by a loader in `config.rs`.

## Lifecycle, if you spawn a process

The fabric backend supervises a child (`fabric --serve`). The pattern there is
worth copying: **reuse + graceful-kill**.

- Spawn **detached** (no `kill_on_drop`) and **persist the address**, so a
  woollamad restart *reuses* the live child instead of orphaning it and re-paying
  startup. A readiness probe reclaims a still-running one.
- Kill it only on **graceful shutdown**, via your `shutdown()` impl — `main.rs`
  calls `backend.shutdown()` on every registered backend.
- Your child is the user's own trusted tool, so it inherits the full environment
  (it may need provider keys). This is the opposite of downstream **MCP** servers,
  which are env-scrubbed because they're untrusted.

## Checklist

- [ ] `impl PatternBackend` — no backend-specific concept leaks into the trait.
- [ ] `render` reuses `config::render_system`; `run` reuses the SSE helpers.
- [ ] `v1_addressable()` honest (don't advertise un-runnable `/v1` models).
- [ ] Config under a top-level mcp.json key, **not** `[inferencers.*]`.
- [ ] If you spawn a process: reuse + graceful-kill, with `shutdown()`.
- [ ] One line in `register_all`; `mod` declared in `lib.rs`; handlers untouched.
- [ ] Hermetic test with a mock of your backend (see `tests/fabric_proxy.rs` and
      `tests/fabric_patterns.rs`); suite + `clippy -D warnings` green.

See also: [Pattern templating](patterns.md) · [Configuration](configuration.md).
